from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.automatic_dynamic_shapes = False  # type: ignore[attr-defined]
        try:
            torch._C._dynamo.eval_frame._set_lru_cache(False)  # type: ignore[attr-defined]
        except Exception:
            pass
except Exception:
    pass

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional runtime dependency in non-training flows
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self._enabled = False

        def add_scalar(self, *args, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None

from config import (
    ALIGN_DISTANCE,
    ALIGN_LAMBDA,
    ALIGN_LAYERS,
    BASE_MODEL_DIR,
    BASE_MODEL_NAME,
    CHECKPOINT_DIR,
    EVAL_DIR,
    GEN_MAX_NEW_TOKENS,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    LEARNING_RATE_ALIGN,
    LEARNING_RATE_CONT,
    LOG_EVERY,
    LOW_VRAM_TRAIN_LM_HEAD,
    LU_EVAL_SPLIT,
    LU_HELDOUT_RATIO,
    MAX_STEPS_MODEL2,
    MAX_STEPS_MODEL3,
    MODEL2_CKPT_DIR,
    MODEL3_CKPT_DIR,
    MODEL_RUNTIME,
    SAVE_EVERY,
    SEED,
    SEQUENCE_LENGTH,
    TENSORBOARD_DIR,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    ensure_model_local,
    ensure_run_directories,
    set_global_seed,
)
from data_and_eval import (
    build_eval_mono_datasets,
    build_mono_train_loader,
    compute_perplexity,
    evaluate_luxgen,
    load_lu_eval_datasets,
    load_parallel_datasets,
)
from routing_analysis import (
    analyze_model_routing,
    attach_routing_hooks,
    clear_routing_cache,
    remove_routing_hooks,
    routing_cache_to_sentence_distributions,
    set_current_lang_tags,
    snapshot_routing_cache,
    visualize_saved_routing,
)


def _checkpoint_ready(path: Path) -> bool:
    if not path.exists():
        return False
    if not (path / "config.json").exists():
        return False
    if not (path / "tokenizer.json").exists():
        return False

    has_weights = bool(list(path.glob("model*.safetensors"))) or bool(
        list(path.glob("pytorch_model*.bin"))
    )
    has_index = (path / "model.safetensors.index.json").exists()
    return has_weights or has_index


def _save_checkpoint(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ckpt_dir: Path,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    train_state = {
        "step": step,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if metadata is not None:
        train_state.update(metadata)

    with (ckpt_dir / "train_state.json").open("w", encoding="utf-8") as f:
        json.dump(train_state, f, indent=2)


def _next_batch(loader, iterator):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def _prepare_model_for_low_vram_training(
    model: torch.nn.Module,
    mode: str,
    enable_gradient_checkpointing: bool = True,
) -> dict[str, float]:
    # Reduce activation memory during backward.
    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # On consumer GPUs, full-parameter optimizer state for this model is too large.
    # Train routers + lm_head so end-to-end runs can complete on available hardware.
    trainable_keywords = ["router"]
    if LOW_VRAM_TRAIN_LM_HEAD:
        trainable_keywords.append("lm_head")

    trainable_keywords_tuple = tuple(trainable_keywords)
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        is_trainable = any(keyword in name.lower() for keyword in trainable_keywords_tuple)
        param.requires_grad = is_trainable
        if is_trainable:
            trainable_params += param.numel()

    # Hard fallback in case architecture naming differs.
    if trainable_params == 0:
        for name, param in model.named_parameters():
            if "lm_head" in name.lower():
                param.requires_grad = True
                trainable_params += param.numel()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ratio = (100.0 * trainable_params / max(total_params, 1))
    print(
        f"[LowVRAM:{mode}] trainable={trainable_params} / total={total_params} "
        f"({ratio:.3f}%)"
    )
    return {
        "trainable_params": float(trainable_params),
        "total_params": float(total_params),
        "trainable_percent": float(ratio),
    }


def load_base_model(download_if_missing: bool = True) -> tuple[torch.nn.Module, AutoTokenizer]:
    model_path = ensure_model_local(download_if_missing=download_if_missing)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        dtype=MODEL_RUNTIME.dtype,
    ).to(MODEL_RUNTIME.device)

    return model, tokenizer


def _first_non_finite_parameter_name(model: torch.nn.Module) -> str | None:
    for name, param in model.named_parameters():
        tensor = param.detach()
        if tensor.numel() == 0:
            continue
        if not bool(torch.isfinite(tensor).all().item()):
            return name
    return None


def _load_model_from_checkpoint(ckpt_dir: Path) -> tuple[torch.nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_dir),
        local_files_only=True,
        dtype=MODEL_RUNTIME.dtype,
    ).to(MODEL_RUNTIME.device)

    bad_param = _first_non_finite_parameter_name(model)
    if bad_param is not None:
        raise RuntimeError(
            f"Checkpoint at {ckpt_dir} contains non-finite values (first detected: {bad_param})."
        )

    return model, tokenizer


def load_or_init_model(model_type: str) -> tuple[torch.nn.Module, AutoTokenizer, str]:
    mode = model_type.lower()

    if mode == "baseline":
        model, tokenizer = load_base_model(download_if_missing=True)
        return model, tokenizer, BASE_MODEL_NAME

    if mode == "cont":
        if _checkpoint_ready(MODEL2_CKPT_DIR):
            try:
                model, tokenizer = _load_model_from_checkpoint(MODEL2_CKPT_DIR)
                return model, tokenizer, str(MODEL2_CKPT_DIR)
            except Exception as exc:
                print(f"[Warn] Invalid Model 2 checkpoint, falling back to base model: {exc}")

        model, tokenizer = load_base_model(download_if_missing=True)
        return model, tokenizer, BASE_MODEL_NAME

    if mode == "align":
        if _checkpoint_ready(MODEL3_CKPT_DIR):
            try:
                model, tokenizer = _load_model_from_checkpoint(MODEL3_CKPT_DIR)
                return model, tokenizer, str(MODEL3_CKPT_DIR)
            except Exception as exc:
                print(f"[Warn] Invalid Model 3 checkpoint, trying fallback: {exc}")

        if _checkpoint_ready(MODEL2_CKPT_DIR):
            try:
                model, tokenizer = _load_model_from_checkpoint(MODEL2_CKPT_DIR)
                return model, tokenizer, str(MODEL2_CKPT_DIR)
            except Exception as exc:
                print(f"[Warn] Invalid Model 2 checkpoint, falling back to base model: {exc}")

        model, tokenizer = load_base_model(download_if_missing=True)
        return model, tokenizer, BASE_MODEL_NAME

    raise ValueError(f"Unsupported model_type '{model_type}'.")


def _to_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def lm_loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p / p.sum().clamp_min(1e-12)
    q = q / q.sum().clamp_min(1e-12)
    m = 0.5 * (p + q)

    kl_pm = torch.sum(p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log()))
    kl_qm = torch.sum(q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log()))
    return 0.5 * (kl_pm + kl_qm)


def alignment_loss_fn(
    routing_stats_langs: dict[str, dict[int, torch.Tensor]],
    align_layers: list[int],
    align_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    langs = list(routing_stats_langs.keys())
    if len(langs) < 2:
        device = torch.device(MODEL_RUNTIME.device)
        return torch.tensor(0.0, device=device), {"alignment_raw": 0.0}

    distances: list[torch.Tensor] = []
    for idx_a in range(len(langs)):
        for idx_b in range(idx_a + 1, len(langs)):
            lang_a = langs[idx_a]
            lang_b = langs[idx_b]
            stats_a = routing_stats_langs[lang_a]
            stats_b = routing_stats_langs[lang_b]

            for layer in align_layers:
                if layer not in stats_a or layer not in stats_b:
                    continue

                dist_a = stats_a[layer].mean(dim=0)
                dist_b = stats_b[layer].mean(dim=0)

                if ALIGN_DISTANCE == "cosine":
                    distance = 1.0 - F.cosine_similarity(
                        dist_a.unsqueeze(0),
                        dist_b.unsqueeze(0),
                    ).squeeze(0)
                else:
                    distance = _js_divergence(dist_a, dist_b)

                distances.append(distance)

    if not distances:
        device = next(iter(routing_stats_langs.values()))[next(iter(align_layers))].device
        return torch.tensor(0.0, device=device), {"alignment_raw": 0.0}

    raw = torch.stack(distances).mean()
    return align_lambda * raw, {"alignment_raw": float(raw.detach().item())}


def smoke_test(max_new_tokens: int = 20) -> None:
    set_global_seed(SEED)
    model, tokenizer = load_base_model(download_if_missing=True)

    print("=== Smoke Test ===")
    print(f"Model: {BASE_MODEL_NAME}")
    print(f"Device: {MODEL_RUNTIME.device}")
    print(f"Dtype: {MODEL_RUNTIME.dtype}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"GPU: {props.name}")
        print(f"GPU total memory: {total_gb:.2f} GB")
        print(f"GPU allocated memory: {allocated_gb:.2f} GB")
        print(f"GPU reserved memory: {reserved_gb:.2f} GB")
    else:
        print("CUDA not available; running on CPU.")

    prompts = {
        "en": "Luxembourg has three official languages:",
        "de": "Luxemburg hat drei Amtssprachen:",
        "nl": "Luxemburg heeft drie officiele talen:",
        "lu": "Lebuerg huet dr ai offiziell Sproochen:",
    }

    model.eval()
    with torch.no_grad():
        for lang, prompt in prompts.items():
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            enc = {k: v.to(MODEL_RUNTIME.device) for k, v in enc.items()}
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"[{lang}] {text}")


def train_model_2_cont_pretrain(max_steps: int | None = None) -> Path:
    ensure_run_directories()
    set_global_seed(SEED)

    model, tokenizer, init_source = load_or_init_model("cont")
    low_vram_stats = _prepare_model_for_low_vram_training(model, mode="cont")
    model.train()

    train_loader = build_mono_train_loader(tokenizer)
    train_iter = iter(train_loader)

    steps = MAX_STEPS_MODEL2 if max_steps is None else max_steps

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lr = LEARNING_RATE_CONT
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=WEIGHT_DECAY, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=max(steps, WARMUP_STEPS + 1),
    )

    writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR / "model2_cont"))

    optimizer.zero_grad(set_to_none=True)

    for step in range(1, steps + 1):
        batch, train_iter = _next_batch(train_loader, train_iter)

        input_ids = batch["input_ids"].to(MODEL_RUNTIME.device)
        attention_mask = batch["attention_mask"].to(MODEL_RUNTIME.device)
        labels = batch["labels"].to(MODEL_RUNTIME.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        lm_loss = outputs.loss
        if not torch.isfinite(lm_loss):
            raise RuntimeError(
                f"Non-finite lm_loss detected at Model2 step={step}. "
                "Training aborted to avoid saving a corrupted checkpoint."
            )

        loss = lm_loss / GRAD_ACCUM_STEPS
        loss.backward()

        if step % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        writer.add_scalar("train/model2_lm_loss", float(lm_loss.detach().item()), step)
        writer.add_scalar("train/model2_lr", float(scheduler.get_last_lr()[0]), step)

        if step % LOG_EVERY == 0:
            print(f"[Model2] step={step} lm_loss={lm_loss.item():.4f}")

        if step % SAVE_EVERY == 0:
            _save_checkpoint(
                model,
                tokenizer,
                MODEL2_CKPT_DIR,
                step=step,
                metadata={
                    "init_source": init_source,
                    "objective": "lm_only",
                    "low_vram": low_vram_stats,
                },
            )

    _save_checkpoint(
        model,
        tokenizer,
        MODEL2_CKPT_DIR,
        step=steps,
        metadata={
            "init_source": init_source,
            "objective": "lm_only",
            "low_vram": low_vram_stats,
        },
    )

    writer.close()
    return MODEL2_CKPT_DIR


def train_model_3_align(max_steps: int | None = None) -> Path:
    ensure_run_directories()
    set_global_seed(SEED)

    model, tokenizer, init_source = load_or_init_model("align")
    low_vram_stats = _prepare_model_for_low_vram_training(
        model,
        mode="align",
        enable_gradient_checkpointing=False,
    )
    model.train()

    mono_loader = build_mono_train_loader(tokenizer)
    mono_iter = iter(mono_loader)

    parallel_loaders = load_parallel_datasets(tokenizer)
    parallel_iters = {pair: iter(loader) for pair, loader in parallel_loaders.items()}

    steps = MAX_STEPS_MODEL3 if max_steps is None else max_steps

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lr = LEARNING_RATE_ALIGN
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=WEIGHT_DECAY, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=max(steps, WARMUP_STEPS + 1),
    )

    hooks = attach_routing_hooks(model, ALIGN_LAYERS, detach=False, store_topk=False)
    writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR / "model3_align"))

    optimizer.zero_grad(set_to_none=True)

    try:
        for step in range(1, steps + 1):
            mono_batch, mono_iter = _next_batch(mono_loader, mono_iter)

            input_ids = mono_batch["input_ids"].to(MODEL_RUNTIME.device)
            attention_mask = mono_batch["attention_mask"].to(MODEL_RUNTIME.device)
            labels = mono_batch["labels"].to(MODEL_RUNTIME.device)

            lm_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            lm_loss = lm_outputs.loss
            if not torch.isfinite(lm_loss):
                raise RuntimeError(
                    f"Non-finite lm_loss detected at Model3 step={step}. "
                    "Training aborted to avoid saving a corrupted checkpoint."
                )

            pair_name = random.choice(list(parallel_loaders.keys()))
            pair_batch, pair_iter = _next_batch(parallel_loaders[pair_name], parallel_iters[pair_name])
            parallel_iters[pair_name] = pair_iter

            lang_a = str(pair_batch["lang_a"])
            lang_b = str(pair_batch["lang_b"])

            ids_a = pair_batch[lang_a].to(MODEL_RUNTIME.device)
            att_a = pair_batch[f"attention_{lang_a}"].to(MODEL_RUNTIME.device)
            ids_b = pair_batch[lang_b].to(MODEL_RUNTIME.device)
            att_b = pair_batch[f"attention_{lang_b}"].to(MODEL_RUNTIME.device)

            clear_routing_cache()
            set_current_lang_tags([lang_a] * int(ids_a.shape[0]))
            _ = model(
                input_ids=ids_a,
                attention_mask=att_a,
                use_cache=False,
                output_router_logits=True,
            )
            cache_a = snapshot_routing_cache()

            clear_routing_cache()
            set_current_lang_tags([lang_b] * int(ids_b.shape[0]))
            _ = model(
                input_ids=ids_b,
                attention_mask=att_b,
                use_cache=False,
                output_router_logits=True,
            )
            cache_b = snapshot_routing_cache()
            set_current_lang_tags(None)

            stats_a = routing_cache_to_sentence_distributions(
                cache_a,
                target_lang=lang_a,
                layers=ALIGN_LAYERS,
            )
            stats_b = routing_cache_to_sentence_distributions(
                cache_b,
                target_lang=lang_b,
                layers=ALIGN_LAYERS,
            )

            align_loss, align_meta = alignment_loss_fn(
                routing_stats_langs={lang_a: stats_a, lang_b: stats_b},
                align_layers=ALIGN_LAYERS,
                align_lambda=ALIGN_LAMBDA,
            )

            if not torch.isfinite(align_loss):
                raise RuntimeError(
                    f"Non-finite align_loss detected at Model3 step={step}. "
                    "Training aborted to avoid saving a corrupted checkpoint."
                )

            total_loss = (lm_loss + align_loss) / GRAD_ACCUM_STEPS
            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"Non-finite total_loss detected at Model3 step={step}. "
                    "Training aborted to avoid saving a corrupted checkpoint."
                )

            total_loss.backward()

            if step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            writer.add_scalar("train/model3_lm_loss", float(lm_loss.detach().item()), step)
            writer.add_scalar("train/model3_align_loss", float(align_loss.detach().item()), step)
            writer.add_scalar(
                "train/model3_alignment_raw",
                float(align_meta["alignment_raw"]),
                step,
            )
            writer.add_scalar("train/model3_total_loss", float((lm_loss + align_loss).detach().item()), step)
            writer.add_scalar("train/model3_lr", float(scheduler.get_last_lr()[0]), step)

            if step % LOG_EVERY == 0:
                print(
                    "[Model3] "
                    f"step={step} lm_loss={lm_loss.item():.4f} "
                    f"align_loss={align_loss.item():.4f} pair={pair_name}"
                )

            if step % SAVE_EVERY == 0:
                _save_checkpoint(
                    model,
                    tokenizer,
                    MODEL3_CKPT_DIR,
                    step=step,
                    metadata={
                        "init_source": init_source,
                        "objective": "lm_plus_alignment",
                        "align_lambda": ALIGN_LAMBDA,
                        "align_layers": ALIGN_LAYERS,
                        "low_vram": low_vram_stats,
                    },
                )
    finally:
        set_current_lang_tags(None)
        clear_routing_cache()
        remove_routing_hooks(hooks)

    _save_checkpoint(
        model,
        tokenizer,
        MODEL3_CKPT_DIR,
        step=steps,
        metadata={
            "init_source": init_source,
            "objective": "lm_plus_alignment",
            "align_lambda": ALIGN_LAMBDA,
            "align_layers": ALIGN_LAYERS,
            "low_vram": low_vram_stats,
        },
    )

    writer.close()
    return MODEL3_CKPT_DIR


def evaluate_model(
    model_type: str,
    seed: int | None = None,
    output_tag: str | None = None,
    include_routing: bool = True,
) -> dict[str, Any]:
    ensure_run_directories()
    run_seed = SEED if seed is None else int(seed)
    set_global_seed(run_seed)

    tag = output_tag or model_type

    model, tokenizer, source = load_or_init_model(model_type)
    used_fallback = model_type in {"cont", "align"} and source == BASE_MODEL_NAME

    mono_eval = build_eval_mono_datasets(tokenizer, max_chunks_per_lang=4_000)
    lu_ppl_dataset, luxgen_eval_loader = load_lu_eval_datasets(tokenizer)

    ppl_results: dict[str, float | None] = {}
    ppl_errors: dict[str, str] = {}

    ppl_inputs = {
        "en": mono_eval["en"],
        "de": mono_eval["de"],
        "nl": mono_eval["nl"],
        "lu": lu_ppl_dataset,
    }
    for lang, dataset in ppl_inputs.items():
        try:
            ppl_value = compute_perplexity(model, dataset, lang=lang)
            finite_value = _to_finite_float(ppl_value)
            if finite_value is None:
                raise RuntimeError(f"Non-finite perplexity returned ({ppl_value}).")
            ppl_results[lang] = finite_value
        except Exception as exc:
            ppl_results[lang] = None
            ppl_errors[lang] = str(exc)
            print(f"[Eval:{model_type}] PPL failed for {lang}: {exc}")

    luxgen_metrics = evaluate_luxgen(model, luxgen_eval_loader, output_tag=tag)

    routing_metrics: dict[str, Any] = {}
    if include_routing:
        routing_metrics = analyze_model_routing(
            model,
            datasets_by_lang={
                "en": mono_eval["en"],
                "de": mono_eval["de"],
                "nl": mono_eval["nl"],
                "lu": lu_ppl_dataset,
            },
            output_prefix=tag,
        )

    summary = {
        "model_type": model_type,
        "seed": run_seed,
        "output_tag": tag,
        "loaded_from": source,
        "evaluation_protocol": {
            "lu_eval_split": LU_EVAL_SPLIT,
            "lu_heldout_ratio": float(LU_HELDOUT_RATIO),
        },
        "ppl": ppl_results,
        "luxgen": luxgen_metrics,
        "routing": {
            "metrics_json_path": routing_metrics.get("metrics_json_path"),
            "expert_load_csv_path": routing_metrics.get("expert_load_csv_path"),
            "jsd_matrix_csv_path": routing_metrics.get("jsd_matrix_csv_path"),
            "heatmap_paths": routing_metrics.get("heatmap_paths", []),
            "similarity_plot_path": routing_metrics.get("similarity_plot_path"),
            "jsd_heatmap_path": routing_metrics.get("jsd_heatmap_path"),
        },
    }
    if not include_routing:
        summary["routing"] = {"status": "skipped"}

    if ppl_errors:
        summary["ppl_errors"] = ppl_errors
    if used_fallback:
        summary["checkpoint_fallback"] = (
            "Used base model because cont/align checkpoint was unavailable or invalid."
        )

    summary["plots"] = _try_plot_eval_summary(summary, model_type=tag)

    out_path = EVAL_DIR / f"eval_{tag}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def _try_plot_eval_summary(summary: dict[str, Any], model_type: str) -> dict[str, str | None]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {
            "ppl_plot_path": None,
            "luxgen_plot_path": None,
        }

    ensure_run_directories()
    plot_paths: dict[str, str | None] = {
        "ppl_plot_path": None,
        "luxgen_plot_path": None,
    }

    ppl = summary.get("ppl", {})
    if isinstance(ppl, dict) and ppl:
        pairs = [(str(lang), _to_finite_float(value)) for lang, value in ppl.items()]
        pairs = [(lang, value) for lang, value in pairs if value is not None]
        if pairs:
            langs = [lang for lang, _ in pairs]
            values = [value for _, value in pairs]

            plt.figure(figsize=(7, 4))
            plt.bar(langs, values)
            plt.ylabel("perplexity")
            plt.xlabel("language")
            plt.title(f"{model_type.upper()} PPL by Language")
            plt.tight_layout()

            ppl_path = EVAL_DIR / f"{model_type}_ppl.png"
            plt.savefig(ppl_path, dpi=150)
            plt.close()
            plot_paths["ppl_plot_path"] = str(ppl_path)

    luxgen = summary.get("luxgen", {})
    if isinstance(luxgen, dict):
        keys = [
            k
            for k in ("bleu", "chrf", "token_f1_fallback")
            if k in luxgen and _to_finite_float(luxgen.get(k)) is not None
        ]
        if keys:
            values = [float(luxgen[k]) for k in keys]

            plt.figure(figsize=(7, 4))
            plt.bar(keys, values)
            plt.ylabel("score")
            plt.xlabel("metric")
            plt.title(f"{model_type.upper()} LuxGen Metrics")
            plt.tight_layout()

            lux_path = EVAL_DIR / f"{model_type}_luxgen_metrics.png"
            plt.savefig(lux_path, dpi=150)
            plt.close()
            plot_paths["luxgen_plot_path"] = str(lux_path)

    return plot_paths


def generate_visualizations(eval_model: str) -> dict[str, Any]:
    ensure_run_directories()

    model_types = ["baseline", "cont", "align"] if eval_model == "all" else [eval_model]
    report: dict[str, Any] = {}

    for model_type in model_types:
        eval_json_path = EVAL_DIR / f"eval_{model_type}.json"
        if not eval_json_path.exists():
            report[model_type] = {
                "status": "missing_eval_json",
                "eval_json_path": str(eval_json_path),
            }
            continue

        with eval_json_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        eval_plots = _try_plot_eval_summary(summary, model_type=model_type)

        try:
            routing_plots = visualize_saved_routing(output_prefix=model_type)
        except FileNotFoundError as exc:
            routing_plots = {
                "status": "missing_routing_metrics",
                "error": str(exc),
            }

        report[model_type] = {
            "status": "ok",
            "eval_json_path": str(eval_json_path),
            "eval_plots": eval_plots,
            "routing_plots": routing_plots,
        }

    out_path = EVAL_DIR / "visualization_manifest.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"manifest_path": str(out_path), "models": report}, indent=2))
    return report


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.4f}"
    return "n/a"


def _safe_sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    if not math.isfinite(var) or var < 0:
        return None
    return float(math.sqrt(var))


def _mean_std_ci95(values: list[float | None]) -> dict[str, float | int | None]:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    n = len(finite)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None}

    mean = float(sum(finite) / n)
    std = _safe_sample_std(finite)
    if std is None:
        ci95 = None
    else:
        ci95 = float(1.96 * std / math.sqrt(n))
    return {"n": n, "mean": mean, "std": std, "ci95": ci95}


def _try_plot_cross_model_comparison(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np
    except Exception:
        return {
            "ppl_comparison_plot_path": None,
            "luxgen_comparison_plot_path": None,
        }

    if not rows:
        return {
            "ppl_comparison_plot_path": None,
            "luxgen_comparison_plot_path": None,
        }

    paths: dict[str, str | None] = {
        "ppl_comparison_plot_path": None,
        "luxgen_comparison_plot_path": None,
    }

    model_labels = [str(row["model_type"]) for row in rows]
    ppl_langs = ["en", "de", "nl", "lu"]

    x = np.arange(len(ppl_langs))
    width = 0.8 / max(len(rows), 1)

    plt.figure(figsize=(9, 4))
    for idx, row in enumerate(rows):
        offset = (idx - (len(rows) - 1) / 2.0) * width
        values = [
            (_to_finite_float(row.get(f"ppl_{lang}")) if row.get(f"ppl_{lang}") is not None else None)
            for lang in ppl_langs
        ]
        values = [val if val is not None else float("nan") for val in values]
        plt.bar(x + offset, values, width=width, label=model_labels[idx])

    plt.xticks(x, ppl_langs)
    plt.ylabel("perplexity")
    plt.xlabel("language")
    plt.title("Cross-Model PPL Comparison")
    plt.legend()
    plt.tight_layout()

    ppl_plot_path = EVAL_DIR / "comparison_ppl_by_language.png"
    plt.savefig(ppl_plot_path, dpi=150)
    plt.close()
    paths["ppl_comparison_plot_path"] = str(ppl_plot_path)

    luxgen_keys = [
        key
        for key in ("luxgen_bleu", "luxgen_chrf", "luxgen_token_f1_fallback")
        if any(key in row and row[key] is not None for row in rows)
    ]
    if not luxgen_keys:
        return paths

    x = np.arange(len(luxgen_keys))
    plt.figure(figsize=(9, 4))
    for idx, row in enumerate(rows):
        offset = (idx - (len(rows) - 1) / 2.0) * width
        values = [
            (_to_finite_float(row.get(key)) if row.get(key) is not None else 0.0)
            for key in luxgen_keys
        ]
        values = [0.0 if val is None else val for val in values]
        plt.bar(x + offset, values, width=width, label=model_labels[idx])

    labels = [key.replace("luxgen_", "") for key in luxgen_keys]
    plt.xticks(x, labels)
    plt.ylabel("score")
    plt.xlabel("metric")
    plt.title("Cross-Model LuxGen Comparison")
    plt.legend()
    plt.tight_layout()

    lux_plot_path = EVAL_DIR / "comparison_luxgen_metrics.png"
    plt.savefig(lux_plot_path, dpi=150)
    plt.close()
    paths["luxgen_comparison_plot_path"] = str(lux_plot_path)
    return paths


def build_thesis_evaluation_report(eval_model: str = "all") -> dict[str, Any]:
    ensure_run_directories()

    requested_models = ["baseline", "cont", "align"] if eval_model == "all" else [eval_model]
    summaries: dict[str, dict[str, Any]] = {}

    for model_type in requested_models:
        eval_json_path = EVAL_DIR / f"eval_{model_type}.json"
        if not eval_json_path.exists():
            continue
        with eval_json_path.open("r", encoding="utf-8") as f:
            summaries[model_type] = json.load(f)

    if not summaries:
        raise FileNotFoundError(
            f"No eval summaries found in {EVAL_DIR}. Run --mode eval first."
        )

    rows: list[dict[str, Any]] = []
    ordered_models = [m for m in requested_models if m in summaries]

    for model_type in ordered_models:
        summary = summaries[model_type]
        ppl = summary.get("ppl", {})
        luxgen = summary.get("luxgen", {})
        row = {
            "model_type": model_type,
            "loaded_from": summary.get("loaded_from"),
            "eval_json_path": str(EVAL_DIR / f"eval_{model_type}.json"),
            "luxgen_samples_path": luxgen.get("samples_path"),
            "ppl_en": _to_finite_float(ppl.get("en")),
            "ppl_de": _to_finite_float(ppl.get("de")),
            "ppl_nl": _to_finite_float(ppl.get("nl")),
            "ppl_lu": _to_finite_float(ppl.get("lu")),
            "luxgen_bleu": _to_finite_float(luxgen.get("bleu")),
            "luxgen_chrf": _to_finite_float(luxgen.get("chrf")),
            "luxgen_token_f1_fallback": (
                float(luxgen.get("token_f1_fallback"))
                if "token_f1_fallback" in luxgen
                else None
            ),
        }
        ppl_values = [
            row["ppl_en"],
            row["ppl_de"],
            row["ppl_nl"],
            row["ppl_lu"],
        ]
        row["ppl_mean"] = _finite_mean(ppl_values)
        rows.append(row)

    baseline_row = next((row for row in rows if row["model_type"] == "baseline"), None)
    deltas_vs_baseline: list[dict[str, Any]] = []
    if baseline_row is not None:
        for row in rows:
            if row["model_type"] == "baseline":
                continue
            deltas_vs_baseline.append(
                {
                    "model_type": row["model_type"],
                    "delta_ppl_mean": (
                        float(row["ppl_mean"] - baseline_row["ppl_mean"])
                        if row["ppl_mean"] is not None and baseline_row["ppl_mean"] is not None
                        else None
                    ),
                    "delta_ppl_lu": (
                        float(row["ppl_lu"] - baseline_row["ppl_lu"])
                        if row["ppl_lu"] is not None and baseline_row["ppl_lu"] is not None
                        else None
                    ),
                    "delta_bleu": (
                        float(row["luxgen_bleu"] - baseline_row["luxgen_bleu"])
                        if row["luxgen_bleu"] is not None and baseline_row["luxgen_bleu"] is not None
                        else None
                    ),
                    "delta_chrf": (
                        float(row["luxgen_chrf"] - baseline_row["luxgen_chrf"])
                        if row["luxgen_chrf"] is not None and baseline_row["luxgen_chrf"] is not None
                        else None
                    ),
                }
            )

    csv_path = EVAL_DIR / "comparison_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_type",
                "loaded_from",
                "eval_json_path",
                "luxgen_samples_path",
                "ppl_en",
                "ppl_de",
                "ppl_nl",
                "ppl_lu",
                "ppl_mean",
                "luxgen_bleu",
                "luxgen_chrf",
                "luxgen_token_f1_fallback",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_paths = _try_plot_cross_model_comparison(rows)

    significance_template = {
        "status": "template_only_single_seed",
        "observed_runs_per_model": {row["model_type"]: 1 for row in rows},
        "recommended_protocol": {
            "seeds_per_model": 3,
            "ppl": "report mean/std and 95% CI across seeds",
            "luxgen": "paired bootstrap over sample-level metrics (>=10k resamples)",
            "routing": "layerwise paired statistical test across seeds",
        },
        "note": "Current report is single-run and should not be used for significance claims.",
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": rows,
        "deltas_vs_baseline": deltas_vs_baseline,
        "csv_path": str(csv_path),
        "plot_paths": plot_paths,
        "significance_template": significance_template,
        "protocol": {
            "lu_eval_split": LU_EVAL_SPLIT,
            "lu_heldout_ratio": float(LU_HELDOUT_RATIO),
        },
    }

    json_path = EVAL_DIR / "thesis_evaluation_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Thesis Evaluation Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- LU eval split: {LU_EVAL_SPLIT}",
        f"- LU heldout ratio: {LU_HELDOUT_RATIO:.2f}",
        "",
        "## Model Comparison",
        "",
        "| model | ppl_en | ppl_de | ppl_nl | ppl_lu | ppl_mean | bleu | chrf |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| "
            f"{row['model_type']} | "
            f"{_format_metric(row['ppl_en'])} | "
            f"{_format_metric(row['ppl_de'])} | "
            f"{_format_metric(row['ppl_nl'])} | "
            f"{_format_metric(row['ppl_lu'])} | "
            f"{_format_metric(row['ppl_mean'])} | "
            f"{_format_metric(row['luxgen_bleu'])} | "
            f"{_format_metric(row['luxgen_chrf'])} |"
        )

    md_lines.extend(
        [
            "",
            "## Significance And Variance Template",
            "",
            "Single-run metrics are reported. Complete thesis claims should use multi-seed results.",
            "Recommended minimum: 3 seeds per model and paired bootstrap for LuxGen metrics.",
            "",
            f"- Comparison CSV: {csv_path}",
            f"- Comparison JSON: {json_path}",
            f"- PPL comparison plot: {plot_paths.get('ppl_comparison_plot_path')}",
            f"- LuxGen comparison plot: {plot_paths.get('luxgen_comparison_plot_path')}",
        ]
    )

    md_path = EVAL_DIR / "thesis_evaluation_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    print(json.dumps(report, indent=2))
    return report


def run_evaluation_suite(eval_model: str) -> None:
    if eval_model == "all":
        for model_type in ("baseline", "cont", "align"):
            evaluate_model(model_type)
        build_thesis_evaluation_report(eval_model="all")
    else:
        evaluate_model(eval_model)


def run_multi_seed_evaluation_suite(
    eval_model: str,
    seeds: list[int],
    include_routing: bool = False,
) -> dict[str, Any]:
    ensure_run_directories()

    model_types = ["baseline", "cont", "align"] if eval_model == "all" else [eval_model]
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        for model_type in model_types:
            tag = f"{model_type}_seed{seed}"
            summary = evaluate_model(
                model_type,
                seed=seed,
                output_tag=tag,
                include_routing=include_routing,
            )
            ppl = summary.get("ppl", {})
            luxgen = summary.get("luxgen", {})
            row = {
                "seed": int(seed),
                "model_type": model_type,
                "output_tag": tag,
                "eval_json_path": str(EVAL_DIR / f"eval_{tag}.json"),
                "ppl_en": _to_finite_float(ppl.get("en")),
                "ppl_de": _to_finite_float(ppl.get("de")),
                "ppl_nl": _to_finite_float(ppl.get("nl")),
                "ppl_lu": _to_finite_float(ppl.get("lu")),
                "luxgen_bleu": _to_finite_float(luxgen.get("bleu")),
                "luxgen_chrf": _to_finite_float(luxgen.get("chrf")),
                "luxgen_token_f1_fallback": _to_finite_float(luxgen.get("token_f1_fallback")),
            }
            row["ppl_mean"] = _finite_mean(
                [row["ppl_en"], row["ppl_de"], row["ppl_nl"], row["ppl_lu"]]
            )
            rows.append(row)

    per_seed_csv_path = EVAL_DIR / "comparison_metrics_multiseed_per_seed.csv"
    with per_seed_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "model_type",
                "output_tag",
                "eval_json_path",
                "ppl_en",
                "ppl_de",
                "ppl_nl",
                "ppl_lu",
                "ppl_mean",
                "luxgen_bleu",
                "luxgen_chrf",
                "luxgen_token_f1_fallback",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows: list[dict[str, Any]] = []
    for model_type in model_types:
        model_rows = [r for r in rows if r["model_type"] == model_type]
        aggregate_rows.append(
            {
                "model_type": model_type,
                "num_seeds": len(model_rows),
                "ppl_mean": _mean_std_ci95([r.get("ppl_mean") for r in model_rows]),
                "ppl_en": _mean_std_ci95([r.get("ppl_en") for r in model_rows]),
                "ppl_de": _mean_std_ci95([r.get("ppl_de") for r in model_rows]),
                "ppl_nl": _mean_std_ci95([r.get("ppl_nl") for r in model_rows]),
                "ppl_lu": _mean_std_ci95([r.get("ppl_lu") for r in model_rows]),
                "luxgen_bleu": _mean_std_ci95([r.get("luxgen_bleu") for r in model_rows]),
                "luxgen_chrf": _mean_std_ci95([r.get("luxgen_chrf") for r in model_rows]),
                "luxgen_token_f1_fallback": _mean_std_ci95(
                    [r.get("luxgen_token_f1_fallback") for r in model_rows]
                ),
            }
        )

    aggregate_csv_path = EVAL_DIR / "comparison_metrics_multiseed_aggregate.csv"
    with aggregate_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_type",
                "num_seeds",
                "ppl_mean_mean",
                "ppl_mean_std",
                "ppl_mean_ci95",
                "ppl_lu_mean",
                "ppl_lu_std",
                "ppl_lu_ci95",
                "luxgen_bleu_mean",
                "luxgen_bleu_std",
                "luxgen_bleu_ci95",
                "luxgen_chrf_mean",
                "luxgen_chrf_std",
                "luxgen_chrf_ci95",
            ],
        )
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(
                {
                    "model_type": row["model_type"],
                    "num_seeds": row["num_seeds"],
                    "ppl_mean_mean": row["ppl_mean"]["mean"],
                    "ppl_mean_std": row["ppl_mean"]["std"],
                    "ppl_mean_ci95": row["ppl_mean"]["ci95"],
                    "ppl_lu_mean": row["ppl_lu"]["mean"],
                    "ppl_lu_std": row["ppl_lu"]["std"],
                    "ppl_lu_ci95": row["ppl_lu"]["ci95"],
                    "luxgen_bleu_mean": row["luxgen_bleu"]["mean"],
                    "luxgen_bleu_std": row["luxgen_bleu"]["std"],
                    "luxgen_bleu_ci95": row["luxgen_bleu"]["ci95"],
                    "luxgen_chrf_mean": row["luxgen_chrf"]["mean"],
                    "luxgen_chrf_std": row["luxgen_chrf"]["std"],
                    "luxgen_chrf_ci95": row["luxgen_chrf"]["ci95"],
                }
            )

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_model": eval_model,
        "seeds": [int(s) for s in seeds],
        "include_routing": bool(include_routing),
        "protocol": {
            "lu_eval_split": LU_EVAL_SPLIT,
            "lu_heldout_ratio": float(LU_HELDOUT_RATIO),
        },
        "per_seed_csv_path": str(per_seed_csv_path),
        "aggregate_csv_path": str(aggregate_csv_path),
        "aggregate": aggregate_rows,
    }

    json_path = EVAL_DIR / "thesis_evaluation_report_multiseed.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Thesis Multi-Seed Evaluation Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Eval model: {eval_model}",
        f"- Seeds: {', '.join(str(s) for s in report['seeds'])}",
        f"- Include routing: {include_routing}",
        f"- LU eval split: {LU_EVAL_SPLIT}",
        f"- LU heldout ratio: {LU_HELDOUT_RATIO:.2f}",
        "",
        "## Aggregate Metrics (mean +- std, 95% CI)",
        "",
        "| model | n | ppl_mean | ppl_lu | bleu | chrf |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        ppl_mean = row["ppl_mean"]
        ppl_lu = row["ppl_lu"]
        bleu = row["luxgen_bleu"]
        chrf = row["luxgen_chrf"]
        md_lines.append(
            "| "
            f"{row['model_type']} | "
            f"{row['num_seeds']} | "
            f"{_format_metric(ppl_mean['mean'])} +- {_format_metric(ppl_mean['std'])} "
            f"(CI {_format_metric(ppl_mean['ci95'])}) | "
            f"{_format_metric(ppl_lu['mean'])} +- {_format_metric(ppl_lu['std'])} "
            f"(CI {_format_metric(ppl_lu['ci95'])}) | "
            f"{_format_metric(bleu['mean'])} +- {_format_metric(bleu['std'])} "
            f"(CI {_format_metric(bleu['ci95'])}) | "
            f"{_format_metric(chrf['mean'])} +- {_format_metric(chrf['std'])} "
            f"(CI {_format_metric(chrf['ci95'])}) |"
        )

    md_lines.extend(
        [
            "",
            f"- Per-seed CSV: {per_seed_csv_path}",
            f"- Aggregate CSV: {aggregate_csv_path}",
            f"- JSON report: {json_path}",
        ]
    )

    md_path = EVAL_DIR / "thesis_evaluation_report_multiseed.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate OLMoE routing-alignment thesis models. "
            "Modes: smoke | baseline | cont | align | eval | multiseed_eval | viz | report"
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "smoke",
            "baseline",
            "cont",
            "align",
            "eval",
            "multiseed_eval",
            "viz",
            "report",
        ],
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default="all",
        choices=["baseline", "cont", "align", "all"],
        help="Used with --mode eval.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional override for training steps in cont/align mode.",
    )
    parser.add_argument(
        "--smoke-max-new-tokens",
        type=int,
        default=20,
        help="Optional generation length for smoke test.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds for --mode multiseed_eval.",
    )
    parser.add_argument(
        "--include-routing",
        action="store_true",
        help="Include routing analysis for --mode multiseed_eval (slower).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "smoke":
        smoke_test(max_new_tokens=args.smoke_max_new_tokens)
        return

    if args.mode == "baseline":
        evaluate_model("baseline")
        return

    if args.mode == "cont":
        ckpt = train_model_2_cont_pretrain(max_steps=args.max_steps)
        print(f"Model 2 checkpoint saved to: {ckpt}")
        return

    if args.mode == "align":
        ckpt = train_model_3_align(max_steps=args.max_steps)
        print(f"Model 3 checkpoint saved to: {ckpt}")
        return

    if args.mode == "eval":
        run_evaluation_suite(eval_model=args.eval_model)
        return

    if args.mode == "multiseed_eval":
        parsed_seeds = [
            int(part.strip()) for part in str(args.seeds).split(",") if part.strip()
        ]
        if not parsed_seeds:
            raise ValueError("At least one seed is required for --mode multiseed_eval.")
        run_multi_seed_evaluation_suite(
            eval_model=args.eval_model,
            seeds=parsed_seeds,
            include_routing=bool(args.include_routing),
        )
        return

    if args.mode == "viz":
        generate_visualizations(eval_model=args.eval_model)
        return

    if args.mode == "report":
        build_thesis_evaluation_report(eval_model=args.eval_model)
        return

    raise ValueError(f"Unhandled mode '{args.mode}'.")


if __name__ == "__main__":
    main()
