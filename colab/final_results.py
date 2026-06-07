from __future__ import annotations

import csv
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Windows consoles default to cp1252 which can't encode the em-dash / Δ / − characters
# we print in headers. Force utf-8 so the script doesn't crash mid-run on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


ARCHS = ("olmoe", "deepseek_moe", "qwen_moe")
STAGES = ("baseline", "cont", "align")
LANGS = ("en", "de", "nl", "lu")
RUNS = Path("runs")
OUT_DIR = RUNS
PACK_DIR = RUNS / "thesis_pack"
SUMMARY_MD = RUNS / "thesis_summary.md"


def jload(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def eval_path(arch: str, stage: str) -> Path:
    return RUNS / arch / "eval" / f"eval_{stage}.json"


def routing_dir(arch: str) -> Path:
    return RUNS / arch / "routing_analysis"


def eval_dir(arch: str) -> Path:
    return RUNS / arch / "eval"


def coverage() -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for arch in ARCHS:
        out[arch] = {
            "model1": (RUNS / arch / "checkpoints" / "model1_baseline" / "train_state.json").exists(),
            "model2": (RUNS / arch / "checkpoints" / "model2_cont" / "train_state.json").exists(),
            "model3": (RUNS / arch / "checkpoints" / "model3_align" / "train_state.json").exists(),
            "eval": all(eval_path(arch, st).exists() for st in STAGES),
            "multiseed": (eval_dir(arch) / "comparison_metrics_multiseed_aggregate.csv").exists(),
        }
    return out


def headline_table() -> list[str]:
    lines = [
        "| arch | stage | ppl_en | ppl_de | ppl_nl | ppl_lu | bpc_lu | bleu | chrf | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arch in ARCHS:
        for st in STAGES:
            p = eval_path(arch, st)
            if not p.exists():
                lines.append(f"| `{arch}` | {st} | — | — | — | — | — | — | — | — |")
                continue
            d = jload(p)
            pp = d["ppl"]
            bpc = d.get("bpc", {}) or {}
            lx = d["luxgen"]
            lines.append(
                f"| `{arch}` | **{st}** | "
                f"{pp['en']:.3f} | {pp['de']:.3f} | {pp['nl']:.3f} | {pp['lu']:.3f} | "
                f"{(bpc.get('lu') if bpc.get('lu') is not None else float('nan')):.3f} | "
                f"{lx.get('bleu', 0):.3f} | {lx.get('chrf', 0):.3f} | {lx.get('num_samples')} |"
            )
    return lines


def alignment_effect() -> tuple[list[str], list[tuple[str, float, float, float]]]:
    lines = [
        "| arch | Δppl_lu | Δbpc_lu | Δbleu | Δchrf | n | verdict |",
        "|---|---:|---:|---:|---:|---:|:--|",
    ]
    rows: list[tuple[str, float, float, float]] = []
    for arch in ARCHS:
        c_path, a_path = eval_path(arch, "cont"), eval_path(arch, "align")
        if not (c_path.exists() and a_path.exists()):
            lines.append(f"| `{arch}` | — | — | — | — | — | _missing_ |")
            continue
        c, a = jload(c_path), jload(a_path)
        dlu = a["ppl"]["lu"] - c["ppl"]["lu"]
        c_bpc, a_bpc = (c.get("bpc") or {}).get("lu"), (a.get("bpc") or {}).get("lu")
        dbpc = (a_bpc - c_bpc) if (a_bpc is not None and c_bpc is not None) else float("nan")
        db = a["luxgen"].get("bleu", 0) - c["luxgen"].get("bleu", 0)
        dch = a["luxgen"].get("chrf", 0) - c["luxgen"].get("chrf", 0)
        n = a["luxgen"].get("num_samples", 0)
        if dlu < -0.5 or (db > 1.0 and n >= 30):
            verdict = "**A** helps LU"
        elif dlu > 0.5 and not (db > 0 and n >= 30):
            verdict = "**C** hurts LU"
        else:
            verdict = "B neutral"
        lines.append(
            f"| `{arch}` | {dlu:+.3f} | {dbpc:+.3f} | {db:+.3f} | {dch:+.3f} | {n} | {verdict} |"
        )
        rows.append((arch, dlu, db, dch))
    return lines, rows


def plot_alignment_effect(rows: list[tuple[str, float, float, float]]) -> Path | None:
    if not rows:
        return None
    archs = [r[0] for r in rows]
    x = np.arange(len(archs))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (label, idx, color) in zip(
        axes,
        [
            ("Δppl_lu (lower better)", 1, "C0"),
            ("Δbleu (higher better)", 2, "C1"),
            ("Δchrf (higher better)", 3, "C2"),
        ],
    ):
        ax.bar(x, [r[idx] for r in rows], color=color)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(archs, rotation=15)
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = OUT_DIR / "final_alignment_effect_by_arch.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_similarity_overlay(pair: str = "en-de") -> Path | None:
    fig, axes = plt.subplots(1, len(ARCHS), figsize=(5 * len(ARCHS), 4))
    if len(ARCHS) == 1:
        axes = [axes]
    plotted_any = False
    for ax, arch in zip(axes, ARCHS):
        ax.set_title(f"{arch} — routing similarity ({pair})")
        for st in STAGES:
            p = routing_dir(arch) / f"{st}_routing_metrics.json"
            if not p.exists():
                continue
            sims = jload(p).get("pairwise_similarity", {}).get(pair, {})
            if not sims:
                continue
            layers = sorted(int(k) for k in sims.keys())
            cs = [sims[str(L)]["cosine"] for L in layers]
            ax.plot(layers, cs, marker="o", label=st)
            plotted_any = True
        ax.set_xlabel("layer")
        ax.set_ylabel("cosine similarity")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    out = OUT_DIR / f"final_similarity_{pair}_by_arch.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out if plotted_any else None


def plot_jsd_grid() -> Path | None:
    fig, axes = plt.subplots(len(ARCHS), len(STAGES), figsize=(4 * len(STAGES), 4 * len(ARCHS)))
    plotted_any = False
    for r, arch in enumerate(ARCHS):
        for c, st in enumerate(STAGES):
            ax = axes[r, c] if len(ARCHS) > 1 else axes[c]
            path = routing_dir(arch) / f"{st}_lang_jsd_matrix.csv"
            if not path.exists():
                ax.set_title(f"{arch} / {st}: missing")
                ax.axis("off")
                continue
            with path.open() as f:
                rows = list(csv.reader(f))
            col_labels = rows[0][1:]
            row_labels = [r0[0] for r0 in rows[1:]]
            M = np.array([[float(x) for x in r0[1:]] for r0 in rows[1:]])
            im = ax.imshow(M, aspect="auto")
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)
            ax.set_title(f"{arch} / {st}")
            plt.colorbar(im, ax=ax, fraction=0.046)
            plotted_any = True
    plt.tight_layout()
    out = OUT_DIR / "final_jsd_grid_arch_x_stage.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out if plotted_any else None


def read_aggregate(arch: str) -> dict[str, dict[str, str]]:
    p = eval_dir(arch) / "comparison_metrics_multiseed_aggregate.csv"
    if not p.exists():
        return {}
    with p.open() as f:
        return {row["model_type"]: row for row in csv.DictReader(f)}


def plot_multiseed_metric(metric_key: str, label: str) -> Path | None:
    aggs = {arch: read_aggregate(arch) for arch in ARCHS}
    if not any(aggs.values()):
        return None
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.25
    x = np.arange(len(STAGES))
    for i, arch in enumerate(ARCHS):
        means = [
            float(aggs[arch][st][f"{metric_key}_mean"]) if aggs[arch].get(st) else 0.0
            for st in STAGES
        ]
        stds = [
            float(aggs[arch][st][f"{metric_key}_std"]) if aggs[arch].get(st) else 0.0
            for st in STAGES
        ]
        ax.bar(x + (i - 1) * width, means, width=width, yerr=stds, capsize=4, label=arch)
    ax.set_xticks(x)
    ax.set_xticklabels(list(STAGES))
    ax.set_ylabel(label)
    ax.set_title(f"LuxGen {label} (mean ± std across seeds 42/43/44)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = OUT_DIR / f"final_luxgen_{label.lower()}_multiseed.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def qualitative_samples(per_stage: int = 2) -> list[str]:
    out: list[str] = []
    for arch in ARCHS:
        out.append(f"### {arch}")
        for st in STAGES:
            p = eval_dir(arch) / f"luxgen_samples_{st}.jsonl"
            if not p.exists():
                continue
            out.append(f"**{st}**")
            with p.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= per_stage:
                        break
                    r = json.loads(line)
                    out.append(f"- prompt: `{r['prompt']}`")
                    out.append(f"  - pred: {r['prediction'][:140]}")
                    out.append(f"  - ref:  {r['reference'][:120]}")
        out.append("")
    return out


def write_summary(headline: list[str], align: list[str], samples: list[str]) -> Path:
    parts = [
        "# OLMoE Cross-Lingual Routing Alignment — Thesis Summary",
        "",
        f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_",
        "",
        "## Headline (single-seed)",
        "",
        *headline,
        "",
        "## Alignment effect (align − cont)",
        "",
        *align,
        "",
        "## Multi-seed aggregate (LuxGen, mean ± std)",
        "",
        "| arch | stage | bleu mean ± std | chrf mean ± std |",
        "|---|---|---|---|",
    ]
    for arch in ARCHS:
        ag = read_aggregate(arch)
        for st in STAGES:
            r = ag.get(st)
            if not r:
                parts.append(f"| `{arch}` | {st} | — | — |")
                continue
            bm, bs = float(r["luxgen_bleu_mean"]), float(r["luxgen_bleu_std"])
            cm, cs = float(r["luxgen_chrf_mean"]), float(r["luxgen_chrf_std"])
            parts.append(f"| `{arch}` | {st} | {bm:.3f} ± {bs:.3f} | {cm:.3f} ± {cs:.3f} |")
    parts += [
        "",
        "## Cross-architecture figures",
        "",
        "- `runs/final_alignment_effect_by_arch.png`",
        "- `runs/final_similarity_en-de_by_arch.png`",
        "- `runs/final_jsd_grid_arch_x_stage.png`",
        "- `runs/final_luxgen_bleu_multiseed.png`",
        "- `runs/final_luxgen_chrf_multiseed.png`",
        "",
        "## Qualitative samples",
        "",
        *samples,
    ]
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(parts), encoding="utf-8")
    return SUMMARY_MD


def pack_artifacts() -> tuple[Path, int]:
    if PACK_DIR.exists():
        shutil.rmtree(PACK_DIR)
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    candidates: list[Path] = []
    for arch in ARCHS:
        ev_dir = eval_dir(arch)
        rt_dir = routing_dir(arch)
        candidates += [ev_dir / f"eval_{st}.json" for st in STAGES]
        candidates += [ev_dir / f"luxgen_samples_{st}.jsonl" for st in STAGES]
        candidates += [
            ev_dir / "comparison_metrics.csv",
            ev_dir / "comparison_metrics_multiseed_per_seed.csv",
            ev_dir / "comparison_metrics_multiseed_aggregate.csv",
            ev_dir / "thesis_evaluation_report.md",
            ev_dir / "thesis_evaluation_report_multiseed.md",
        ]
        candidates += [rt_dir / f"{st}_routing_metrics.json" for st in STAGES]
        candidates += [rt_dir / f"{st}_lang_jsd_matrix.csv" for st in STAGES]
        candidates += [rt_dir / f"{st}_expert_load.csv" for st in STAGES]
        candidates += [rt_dir / f"{st}_jsd_matrix_heatmap.png" for st in STAGES]
        candidates += [rt_dir / f"{st}_similarity_vs_layer.png" for st in STAGES]
        candidates += [rt_dir / f"{st}_heatmap_{lang}.png" for st in STAGES for lang in LANGS]
        for sub in ("model1_baseline", "model2_cont", "model3_align"):
            candidates.append(RUNS / arch / "checkpoints" / sub / "train_state.json")
    candidates += [
        OUT_DIR / "final_alignment_effect_by_arch.png",
        OUT_DIR / "final_similarity_en-de_by_arch.png",
        OUT_DIR / "final_jsd_grid_arch_x_stage.png",
        OUT_DIR / "final_luxgen_bleu_multiseed.png",
        OUT_DIR / "final_luxgen_chrf_multiseed.png",
        SUMMARY_MD,
    ]
    copied = 0
    for path in candidates:
        if not path.exists():
            continue
        rel = path.relative_to(RUNS)
        dst = PACK_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        copied += 1
    archive = shutil.make_archive(str(RUNS / "thesis_pack"), "zip", PACK_DIR)
    return Path(archive), copied


def main() -> int:
    cov = coverage()
    print("=== coverage ===")
    for arch, flags in cov.items():
        print(f"  {arch:>13}: " + "  ".join(f"{k}={'Y' if v else 'N'}" for k, v in flags.items()))

    print("\n=== headline (3 archs × 3 stages) ===")
    head = headline_table()
    for line in head:
        print(line)

    print("\n=== alignment effect (align − cont) ===")
    align_lines, align_rows = alignment_effect()
    for line in align_lines:
        print(line)

    figs = [
        plot_alignment_effect(align_rows),
        plot_similarity_overlay(),
        plot_jsd_grid(),
        plot_multiseed_metric("luxgen_bleu", "BLEU"),
        plot_multiseed_metric("luxgen_chrf", "ChrF"),
    ]
    print("\n=== figures ===")
    for f in figs:
        print(f"  {'wrote' if f else 'skipped'} {f if f else '(no data)'}")

    samples = qualitative_samples()
    summary = write_summary(head, align_lines, samples)
    print(f"\n=== summary ===")
    print(f"  wrote {summary}")

    archive, copied = pack_artifacts()
    print(f"  packed {copied} files into {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
