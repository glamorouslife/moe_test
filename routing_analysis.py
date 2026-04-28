from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    ALIGN_LAYERS,
    ANALYSIS_DIR,
    LANGS,
    ROUTING_ANALYSIS_MAX_BATCHES,
    ROUTING_TOP_K,
    ensure_run_directories,
)


try:
    from transformers.models.olmoe.modeling_olmoe import OlmoeTopKRouter
except Exception:  # pragma: no cover - depends on transformers internals
    OlmoeTopKRouter = None


_ROUTING_CACHE: dict[int, list[dict[str, Any]]] = {}
_CURRENT_LANG_TAGS: list[str] | None = None


def clear_routing_cache() -> None:
    _ROUTING_CACHE.clear()


def set_current_lang_tags(lang_tags: list[str] | None) -> None:
    global _CURRENT_LANG_TAGS
    _CURRENT_LANG_TAGS = lang_tags


def snapshot_routing_cache() -> dict[int, list[dict[str, Any]]]:
    return {layer: list(entries) for layer, entries in _ROUTING_CACHE.items()}


def _parse_layer_idx(module_name: str) -> int | None:
    match = re.search(r"layers\.(\d+)\.", module_name)
    if match is None:
        return None
    return int(match.group(1))


def _reshape_router_logits(router_logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    if router_logits.dim() == 3:
        return router_logits

    if router_logits.dim() == 2:
        bt, experts = router_logits.shape
        if batch_size <= 0:
            batch_size = 1
        if bt % batch_size != 0:
            batch_size = 1
        tokens = bt // batch_size
        return router_logits.view(batch_size, tokens, experts)

    raise ValueError(f"Unsupported router logits shape: {tuple(router_logits.shape)}")


def attach_routing_hooks(
    model: torch.nn.Module,
    layers: list[int] | tuple[int, ...],
    detach: bool = True,
    store_topk: bool = True,
) -> list[Any]:
    layer_set = set(int(layer) for layer in layers)
    handles: list[Any] = []

    def should_track(module: torch.nn.Module) -> bool:
        if OlmoeTopKRouter is not None and isinstance(module, OlmoeTopKRouter):
            return True
        name = module.__class__.__name__.lower()
        return "router" in name and "topk" in name

    for module_name, module in model.named_modules():
        if not should_track(module):
            continue

        layer_idx = _parse_layer_idx(module_name)
        if layer_idx is None or layer_idx not in layer_set:
            continue

        def hook(_module, _inputs, outputs, tracked_layer: int = layer_idx):
            if isinstance(outputs, tuple):
                router_logits = outputs[0]
            else:
                router_logits = outputs

            if not isinstance(router_logits, torch.Tensor):
                return

            if _CURRENT_LANG_TAGS is None or len(_CURRENT_LANG_TAGS) == 0:
                batch_size = 1
                lang_tags = ["unknown"]
            else:
                batch_size = len(_CURRENT_LANG_TAGS)
                lang_tags = list(_CURRENT_LANG_TAGS)

            probs = F.softmax(_reshape_router_logits(router_logits, batch_size).float(), dim=-1)

            if detach:
                probs_to_store = probs.detach().cpu()
            else:
                probs_to_store = probs

            entry: dict[str, Any] = {
                "probs": probs_to_store,
                "lang_tags": lang_tags,
            }

            if store_topk:
                top_k = min(ROUTING_TOP_K, probs.shape[-1])
                _, top_idx = torch.topk(probs, k=top_k, dim=-1)
                entry["topk_idx"] = top_idx.detach().cpu() if detach else top_idx

            _ROUTING_CACHE.setdefault(tracked_layer, []).append(entry)

        handles.append(module.register_forward_hook(hook))

    return handles


def remove_routing_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def routing_cache_to_sentence_distributions(
    routing_cache: dict[int, list[dict[str, Any]]],
    target_lang: str | None = None,
    layers: list[int] | tuple[int, ...] | None = None,
) -> dict[int, torch.Tensor]:
    selected_layers = None if layers is None else set(int(x) for x in layers)
    by_layer: dict[int, torch.Tensor] = {}

    for layer, entries in routing_cache.items():
        if selected_layers is not None and layer not in selected_layers:
            continue

        sentence_vectors: list[torch.Tensor] = []
        for entry in entries:
            probs: torch.Tensor = entry["probs"]
            tags: list[str] = list(entry.get("lang_tags", ["unknown"] * probs.shape[0]))
            if len(tags) != probs.shape[0]:
                tags = ["unknown"] * probs.shape[0]

            for sample_idx, sample_lang in enumerate(tags):
                if target_lang is not None and sample_lang != target_lang:
                    continue
                sentence_vectors.append(probs[sample_idx].mean(dim=0))

        if sentence_vectors:
            by_layer[layer] = torch.stack(sentence_vectors, dim=0)

    return by_layer


def _normalize_distribution(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.float()
    return vec / vec.sum().clamp_min(1e-12)


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = _normalize_distribution(p)
    q = _normalize_distribution(q)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log()))
    kl_qm = torch.sum(q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log()))
    return 0.5 * (kl_pm + kl_qm)


def compute_expert_load(
    routing_cache: dict[int, list[dict[str, Any]]],
    lang: str,
) -> dict[int, torch.Tensor]:
    sent_dists = routing_cache_to_sentence_distributions(routing_cache, target_lang=lang)
    load_by_layer: dict[int, torch.Tensor] = {}

    for layer, values in sent_dists.items():
        mean_dist = _normalize_distribution(values.mean(dim=0))
        load_by_layer[layer] = mean_dist.detach().cpu()

    return load_by_layer


def compute_routing_similarity(
    routing_cache: dict[int, list[dict[str, Any]]],
    lang_a: str,
    lang_b: str,
) -> dict[int, dict[str, float]]:
    dist_a = routing_cache_to_sentence_distributions(routing_cache, target_lang=lang_a)
    dist_b = routing_cache_to_sentence_distributions(routing_cache, target_lang=lang_b)

    output: dict[int, dict[str, float]] = {}
    for layer in sorted(set(dist_a.keys()) & set(dist_b.keys())):
        mean_a = _normalize_distribution(dist_a[layer].mean(dim=0))
        mean_b = _normalize_distribution(dist_b[layer].mean(dim=0))

        cosine = float(F.cosine_similarity(mean_a.unsqueeze(0), mean_b.unsqueeze(0)).item())
        jsd = float(_js_divergence(mean_a, mean_b).item())
        output[layer] = {"cosine": cosine, "jsd": jsd}

    return output


def compute_routing_entropy(
    routing_cache: dict[int, list[dict[str, Any]]],
    lang: str,
) -> dict[int, float]:
    entropy_by_layer: dict[int, float] = {}

    for layer, entries in routing_cache.items():
        entropies: list[float] = []
        for entry in entries:
            probs: torch.Tensor = entry["probs"].float()
            tags: list[str] = list(entry.get("lang_tags", ["unknown"] * probs.shape[0]))
            if len(tags) != probs.shape[0]:
                tags = ["unknown"] * probs.shape[0]

            token_entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            for sample_idx, sample_lang in enumerate(tags):
                if sample_lang == lang:
                    entropies.append(float(token_entropy[sample_idx].mean().item()))

        if entropies:
            entropy_by_layer[layer] = float(sum(entropies) / len(entropies))

    return entropy_by_layer


def _save_expert_load_csv(
    expert_load: dict[str, dict[int, torch.Tensor]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for lang, by_layer in expert_load.items():
        for layer, vec in by_layer.items():
            for expert_idx, value in enumerate(vec.tolist()):
                rows.append(
                    {
                        "lang": lang,
                        "layer": layer,
                        "expert": expert_idx,
                        "load": value,
                    }
                )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lang", "layer", "expert", "load"])
        writer.writeheader()
        writer.writerows(rows)


def _try_plot_heatmaps(expert_load: dict[str, dict[int, torch.Tensor]], prefix: str) -> list[str]:
    created_paths: list[str] = []
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return created_paths

    for lang, by_layer in expert_load.items():
        if not by_layer:
            continue

        layers = sorted(by_layer.keys())
        matrix = torch.stack([by_layer[layer] for layer in layers], dim=0).numpy()

        plt.figure(figsize=(12, 4))
        plt.imshow(matrix, aspect="auto")
        plt.colorbar(label="expert load")
        plt.xlabel("expert index")
        plt.ylabel("layer")
        plt.yticks(range(len(layers)), layers)
        plt.title(f"Expert load heatmap ({lang})")
        plt.tight_layout()
        out_path = ANALYSIS_DIR / f"{prefix}_heatmap_{lang}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        created_paths.append(str(out_path))

    return created_paths


def _try_plot_similarity(
    pairwise_similarity: dict[str, dict[int, dict[str, float]]],
    prefix: str,
) -> str | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    plt.figure(figsize=(10, 4))
    plotted = False

    for pair_name, values in pairwise_similarity.items():
        if not values:
            continue
        layers = sorted(values.keys())
        cosine_vals = [values[layer]["cosine"] for layer in layers]
        plt.plot(layers, cosine_vals, marker="o", label=pair_name)
        plotted = True

    if not plotted:
        plt.close()
        return None

    plt.xlabel("layer")
    plt.ylabel("cosine similarity")
    plt.title("Routing similarity vs layer")
    plt.legend()
    plt.tight_layout()
    out_path = ANALYSIS_DIR / f"{prefix}_similarity_vs_layer.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def _try_plot_jsd_matrix_heatmap(jsd_matrix_csv_path: Path, output_path: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    if not jsd_matrix_csv_path.exists():
        return None

    with jsd_matrix_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        return None

    col_labels = rows[0][1:]
    row_labels: list[str] = []
    matrix: list[list[float]] = []

    for row in rows[1:]:
        if len(row) < 2:
            continue
        row_labels.append(str(row[0]))
        matrix.append([float(val) for val in row[1:]])

    if not matrix:
        return None

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="JSD")
    plt.xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title("Language Routing JSD Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def visualize_saved_routing(output_prefix: str) -> dict[str, Any]:
    ensure_run_directories()

    metrics_path = ANALYSIS_DIR / f"{output_prefix}_routing_metrics.json"
    jsd_matrix_csv_path = ANALYSIS_DIR / f"{output_prefix}_lang_jsd_matrix.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Routing metrics file not found for '{output_prefix}': {metrics_path}"
        )

    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    expert_load_raw = payload.get("expert_load", {})
    pairwise_raw = payload.get("pairwise_similarity", {})

    expert_load: dict[str, dict[int, torch.Tensor]] = {}
    for lang, by_layer in expert_load_raw.items():
        layer_map: dict[int, torch.Tensor] = {}
        for layer_str, values in by_layer.items():
            layer_map[int(layer_str)] = torch.tensor(values, dtype=torch.float32)
        expert_load[str(lang)] = layer_map

    pairwise_similarity: dict[str, dict[int, dict[str, float]]] = {}
    for pair_name, by_layer in pairwise_raw.items():
        layer_map: dict[int, dict[str, float]] = {}
        for layer_str, vals in by_layer.items():
            layer_map[int(layer_str)] = {
                "cosine": float(vals.get("cosine", 0.0)),
                "jsd": float(vals.get("jsd", 0.0)),
            }
        pairwise_similarity[str(pair_name)] = layer_map

    heatmap_paths = _try_plot_heatmaps(expert_load, prefix=output_prefix)
    similarity_plot_path = _try_plot_similarity(pairwise_similarity, prefix=output_prefix)
    jsd_heatmap_path = _try_plot_jsd_matrix_heatmap(
        jsd_matrix_csv_path=jsd_matrix_csv_path,
        output_path=ANALYSIS_DIR / f"{output_prefix}_jsd_matrix_heatmap.png",
    )

    return {
        "metrics_json_path": str(metrics_path),
        "jsd_matrix_csv_path": str(jsd_matrix_csv_path),
        "heatmap_paths": heatmap_paths,
        "similarity_plot_path": similarity_plot_path,
        "jsd_heatmap_path": jsd_heatmap_path,
    }


def analyze_model_routing(
    model: torch.nn.Module,
    datasets_by_lang: dict[str, Any],
    max_batches_per_lang: int = ROUTING_ANALYSIS_MAX_BATCHES,
    output_prefix: str = "model",
) -> dict[str, Any]:
    ensure_run_directories()
    clear_routing_cache()

    hooks = attach_routing_hooks(model, ALIGN_LAYERS, detach=True, store_topk=True)
    device = next(model.parameters()).device

    try:
        model.eval()
        with torch.no_grad():
            for lang, dataset_like in datasets_by_lang.items():
                if isinstance(dataset_like, DataLoader):
                    loader = dataset_like
                else:
                    from data_and_eval import collate_lm_batch

                    loader = DataLoader(dataset_like, batch_size=1, collate_fn=collate_lm_batch)

                for batch_idx, batch in enumerate(loader, start=1):
                    if batch_idx > max_batches_per_lang:
                        break

                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    tags = [lang] * int(input_ids.shape[0])

                    set_current_lang_tags(tags)
                    _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_router_logits=True,
                    )
    finally:
        set_current_lang_tags(None)
        remove_routing_hooks(hooks)

    cache = snapshot_routing_cache()

    expert_load: dict[str, dict[int, torch.Tensor]] = {
        lang: compute_expert_load(cache, lang) for lang in LANGS
    }
    entropy: dict[str, dict[int, float]] = {
        lang: compute_routing_entropy(cache, lang) for lang in LANGS
    }

    pairwise_similarity: dict[str, dict[int, dict[str, float]]] = {}
    for lang_a, lang_b in (("en", "de"), ("en", "nl"), ("de", "nl"), ("en", "lu")):
        pair_name = f"{lang_a}-{lang_b}"
        pairwise_similarity[pair_name] = compute_routing_similarity(cache, lang_a, lang_b)

    summary = {
        "expert_load": {
            lang: {
                str(layer): vec.tolist() for layer, vec in by_layer.items()
            }
            for lang, by_layer in expert_load.items()
        },
        "entropy": {
            lang: {str(layer): value for layer, value in by_layer.items()}
            for lang, by_layer in entropy.items()
        },
        "pairwise_similarity": {
            pair: {
                str(layer): vals for layer, vals in by_layer.items()
            }
            for pair, by_layer in pairwise_similarity.items()
        },
    }

    analysis_json = ANALYSIS_DIR / f"{output_prefix}_routing_metrics.json"
    with analysis_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    expert_load_csv_path = ANALYSIS_DIR / f"{output_prefix}_expert_load.csv"
    _save_expert_load_csv(expert_load, expert_load_csv_path)
    heatmap_paths = _try_plot_heatmaps(expert_load, prefix=output_prefix)
    similarity_plot_path = _try_plot_similarity(pairwise_similarity, prefix=output_prefix)

    # Optional language distance matrix from JSD over aligned layers.
    jsd_matrix_path = ANALYSIS_DIR / f"{output_prefix}_lang_jsd_matrix.csv"
    with jsd_matrix_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lang"] + LANGS)

        for lang_a in LANGS:
            row = [lang_a]
            for lang_b in LANGS:
                if lang_a == lang_b:
                    row.append(0.0)
                    continue

                pair_key = f"{lang_a}-{lang_b}"
                reverse_pair_key = f"{lang_b}-{lang_a}"
                pair_vals = pairwise_similarity.get(pair_key) or pairwise_similarity.get(reverse_pair_key) or {}
                jsd_vals = [v["jsd"] for _, v in sorted(pair_vals.items())]
                row.append(float(sum(jsd_vals) / len(jsd_vals)) if jsd_vals else 0.0)
            writer.writerow(row)

    jsd_heatmap_path = _try_plot_jsd_matrix_heatmap(
        jsd_matrix_csv_path=jsd_matrix_path,
        output_path=ANALYSIS_DIR / f"{output_prefix}_jsd_matrix_heatmap.png",
    )

    summary["metrics_json_path"] = str(analysis_json)
    summary["expert_load_csv_path"] = str(expert_load_csv_path)
    summary["jsd_matrix_csv_path"] = str(jsd_matrix_path)
    summary["heatmap_paths"] = heatmap_paths
    summary["similarity_plot_path"] = similarity_plot_path
    summary["jsd_heatmap_path"] = jsd_heatmap_path
    return summary


__all__ = [
    "attach_routing_hooks",
    "clear_routing_cache",
    "set_current_lang_tags",
    "snapshot_routing_cache",
    "routing_cache_to_sentence_distributions",
    "compute_expert_load",
    "compute_routing_similarity",
    "compute_routing_entropy",
    "analyze_model_routing",
    "visualize_saved_routing",
    "remove_routing_hooks",
]
