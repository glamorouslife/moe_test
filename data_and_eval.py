from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import IterableDataset as HfIterableDataset
from datasets import interleave_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import (
    ALIGN_LAYERS,
    ANALYSIS_DIR,
    BATCH_SIZE_MONO,
    BATCH_SIZE_PARALLEL,
    EVAL_BATCH_SIZE,
    EVAL_DIR,
    GEN_MAX_NEW_TOKENS,
    LANGS,
    LUXGEN_DATASET_CANDIDATES,
    LU_EVAL_SPLIT,
    LU_HELDOUT_RATIO,
    LU_PPL_DATASET_CANDIDATES,
    LU_PPL_SPLIT,
    MAX_LU_PPL_CHUNKS,
    MAX_LUXGEN_SAMPLES,
    MAX_MONO_CHUNKS_PER_LANG,
    MAX_PARALLEL_EXAMPLES_PER_PAIR,
    MONO_DATASET_CANDIDATES,
    MONO_SPLIT,
    NUM_WORKERS,
    PARALLEL_DATASET_CANDIDATES,
    PARALLEL_PAIRS,
    PARALLEL_MAX_LENGTH,
    PARALLEL_SPLIT,
    PPL_EVAL_MAX_BATCHES,
    SEED,
    SEQUENCE_LENGTH,
    ensure_model_local,
    ensure_run_directories,
)


WHITESPACE_RE = re.compile(r"\s+")
HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
WIKI_MARKUP_RE = re.compile(r"\[\[|\]\]|\{\{|\}\}|\[http", re.IGNORECASE)


def _clean_text(text: str) -> str:
    cleaned = HTML_RE.sub(" ", text)
    cleaned = WIKI_MARKUP_RE.sub(" ", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _valid_text(text: str, min_chars: int = 40) -> bool:
    if not text:
        return False
    if len(text) < min_chars:
        return False
    if URL_RE.search(text):
        return False
    alpha = sum(ch.isalpha() for ch in text)
    if alpha / max(len(text), 1) < 0.30:
        return False
    return True


def _try_load_stream(
    candidates: list[tuple[str, str | None]],
    split: str,
) -> tuple[str, str | None, HfIterableDataset]:
    errors: list[str] = []
    for name, cfg in candidates:
        cfg_attempts: list[str | None]
        if cfg is None:
            cfg_attempts = [None]
        else:
            cfg_attempts = [cfg]
            if "-" in cfg:
                parts = cfg.split("-")
                if len(parts) == 2:
                    reversed_cfg = f"{parts[1]}-{parts[0]}"
                    if reversed_cfg != cfg:
                        cfg_attempts.append(reversed_cfg)

        for cfg_try in cfg_attempts:
            try:
                if cfg_try is None:
                    stream = load_dataset(name, split=split, streaming=True)
                else:
                    stream = load_dataset(name, cfg_try, split=split, streaming=True)
                return name, cfg_try, stream
            except Exception as exc:  # pragma: no cover - depends on external datasets
                errors.append(f"{name}/{cfg_try}: {exc}")

    joined = "\n".join(errors[-6:])
    raise RuntimeError(
        f"Could not load any dataset candidate for split='{split}'. Recent errors:\n{joined}"
    )


def _resolve_text_field(stream: HfIterableDataset) -> str | None:
    candidates = ("text", "content", "document", "sentence", "article")
    if stream.features is not None:
        for key in candidates:
            if key in stream.features:
                return key

    for row in stream.take(1):
        for key in candidates:
            if key in row:
                return key

        # Header-less datasets sometimes store text in keys/values directly.
        # Returning None lets downstream code build text from all textual parts.
        return None

    raise RuntimeError("Stream is empty while resolving text field.")


def _extract_row_text(row: dict[str, Any], text_field: str | None) -> str:
    if text_field is not None and text_field in row and row[text_field] is not None:
        return str(row[text_field])

    pieces: list[str] = []

    for value in row.values():
        if isinstance(value, str):
            value = value.strip()
            if value:
                pieces.append(value)

    for key in row.keys():
        if isinstance(key, str):
            key = key.strip()
            if key:
                pieces.append(key)

    if not pieces:
        return ""

    # Keep order while removing duplicates.
    unique_pieces = list(dict.fromkeys(pieces))
    return " ".join(unique_pieces)


def _mono_chunk_generator(
    text_stream: HfIterableDataset,
    text_field: str | None,
    tokenizer: PreTrainedTokenizerBase,
    lang: str,
    sequence_length: int,
    max_chunks: int,
) -> Iterable[dict[str, Any]]:
    emitted = 0
    token_buffer: list[int] = []
    eos_id = tokenizer.eos_token_id

    for row in text_stream:
        raw_text = _extract_row_text(row, text_field)
        
        text = _clean_text(raw_text)
        if not _valid_text(text):
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue

        if eos_id is not None:
            token_ids.append(eos_id)

        token_buffer.extend(token_ids)

        while len(token_buffer) >= sequence_length:
            chunk = token_buffer[:sequence_length]
            token_buffer = token_buffer[sequence_length:]
            emitted += 1
            yield {
                "input_ids": chunk,
                "attention_mask": [1] * sequence_length,
                "labels": chunk.copy(),
                "lang": lang,
            }

            if emitted >= max_chunks:
                return


def load_mono_dataset(
    lang: str,
    tokenizer: PreTrainedTokenizerBase,
    split: str = MONO_SPLIT,
    max_chunks: int = MAX_MONO_CHUNKS_PER_LANG,
    sequence_length: int = SEQUENCE_LENGTH,
) -> HfIterableDataset:
    if lang not in ("en", "de", "nl"):
        raise ValueError(f"Monolingual loader expects one of en/de/nl, got '{lang}'.")

    name, cfg, text_stream = _try_load_stream(MONO_DATASET_CANDIDATES[lang], split=split)
    text_field = _resolve_text_field(text_stream)

    chunked = HfIterableDataset.from_generator(
        _mono_chunk_generator,
        gen_kwargs={
            "text_stream": text_stream,
            "text_field": text_field,
            "tokenizer": tokenizer,
            "lang": lang,
            "sequence_length": sequence_length,
            "max_chunks": max_chunks,
        },
    )

    setattr(chunked, "_source_name", name)
    setattr(chunked, "_source_config", cfg)
    return chunked


def collate_lm_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
        "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
        "lang": [str(item["lang"]) for item in batch],
    }


def build_mono_train_loader(tokenizer: PreTrainedTokenizerBase) -> DataLoader:
    mono_en = load_mono_dataset("en", tokenizer)
    mono_de = load_mono_dataset("de", tokenizer)
    mono_nl = load_mono_dataset("nl", tokenizer)

    mixed = interleave_datasets(
        [mono_en, mono_de, mono_nl],
        probabilities=[0.34, 0.33, 0.33],
        seed=SEED,
        stopping_strategy="first_exhausted",
    )

    return DataLoader(
        mixed,
        batch_size=BATCH_SIZE_MONO,
        num_workers=NUM_WORKERS,
        collate_fn=collate_lm_batch,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )


def _extract_parallel_texts(row: dict[str, Any], lang_a: str, lang_b: str) -> tuple[str, str] | None:
    if "translation" in row and isinstance(row["translation"], dict):
        text_a = row["translation"].get(lang_a)
        text_b = row["translation"].get(lang_b)
    else:
        text_a = row.get(lang_a)
        text_b = row.get(lang_b)

    if text_a is None or text_b is None:
        return None

    clean_a = _clean_text(str(text_a))
    clean_b = _clean_text(str(text_b))

    if not _valid_text(clean_a, min_chars=10) or not _valid_text(clean_b, min_chars=10):
        return None

    return clean_a, clean_b


def _parallel_pair_generator(
    pair_stream: HfIterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    lang_a: str,
    lang_b: str,
    max_examples: int,
) -> Iterable[dict[str, Any]]:
    emitted = 0
    for row in pair_stream:
        pair = _extract_parallel_texts(row, lang_a, lang_b)
        if pair is None:
            continue

        text_a, text_b = pair
        enc_a = tokenizer(
            text_a,
            truncation=True,
            padding="max_length",
            max_length=PARALLEL_MAX_LENGTH,
            return_attention_mask=True,
        )
        enc_b = tokenizer(
            text_b,
            truncation=True,
            padding="max_length",
            max_length=PARALLEL_MAX_LENGTH,
            return_attention_mask=True,
        )

        emitted += 1
        yield {
            lang_a: enc_a["input_ids"],
            lang_b: enc_b["input_ids"],
            f"attention_{lang_a}": enc_a["attention_mask"],
            f"attention_{lang_b}": enc_b["attention_mask"],
            "pair": f"{lang_a}-{lang_b}",
            "lang_a": lang_a,
            "lang_b": lang_b,
        }

        if emitted >= max_examples:
            return


def _load_parallel_pair_dataset(
    lang_a: str,
    lang_b: str,
    tokenizer: PreTrainedTokenizerBase,
    split: str = PARALLEL_SPLIT,
) -> HfIterableDataset:
    candidates = PARALLEL_DATASET_CANDIDATES[(lang_a, lang_b)]
    _, _, pair_stream = _try_load_stream(candidates, split=split)

    return HfIterableDataset.from_generator(
        _parallel_pair_generator,
        gen_kwargs={
            "pair_stream": pair_stream,
            "tokenizer": tokenizer,
            "lang_a": lang_a,
            "lang_b": lang_b,
            "max_examples": MAX_PARALLEL_EXAMPLES_PER_PAIR,
        },
    )


def collate_parallel_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lang_a = str(batch[0]["lang_a"])
    lang_b = str(batch[0]["lang_b"])

    return {
        lang_a: torch.tensor([item[lang_a] for item in batch], dtype=torch.long),
        lang_b: torch.tensor([item[lang_b] for item in batch], dtype=torch.long),
        f"attention_{lang_a}": torch.tensor(
            [item[f"attention_{lang_a}"] for item in batch], dtype=torch.long
        ),
        f"attention_{lang_b}": torch.tensor(
            [item[f"attention_{lang_b}"] for item in batch], dtype=torch.long
        ),
        "pair": batch[0]["pair"],
        "lang_a": lang_a,
        "lang_b": lang_b,
    }


def load_parallel_datasets(tokenizer: PreTrainedTokenizerBase) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for lang_a, lang_b in PARALLEL_PAIRS:
        stream = _load_parallel_pair_dataset(lang_a, lang_b, tokenizer)
        pair_name = f"{lang_a}-{lang_b}"
        loaders[pair_name] = DataLoader(
            stream,
            batch_size=BATCH_SIZE_PARALLEL,
            num_workers=NUM_WORKERS,
            collate_fn=collate_parallel_batch,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def _load_lu_text_stream(split: str = LU_PPL_SPLIT) -> HfIterableDataset:
    requested = split.strip().lower()
    if requested in {"train", "validation", "test"}:
        _, _, stream = _try_load_stream(LU_PPL_DATASET_CANDIDATES, split=requested)
        return stream

    if requested != "heldout":
        raise ValueError(
            f"Unsupported LU eval split '{split}'. Expected train/validation/test/heldout."
        )

    _, _, base_stream = _try_load_stream(LU_PPL_DATASET_CANDIDATES, split=LU_PPL_SPLIT)
    text_field = _resolve_text_field(base_stream)

    return HfIterableDataset.from_generator(
        _lu_holdout_generator,
        gen_kwargs={
            "lu_stream": base_stream,
            "text_field": text_field,
            "heldout_ratio": LU_HELDOUT_RATIO,
        },
    )


def _stable_bucket(text: str, bucket_count: int = 10_000) -> int:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) % max(bucket_count, 1)


def _is_holdout_text(text: str, heldout_ratio: float) -> bool:
    if heldout_ratio <= 0.0:
        return False
    if heldout_ratio >= 1.0:
        return True
    threshold = int(heldout_ratio * 10_000)
    return _stable_bucket(text, bucket_count=10_000) < threshold


def _lu_holdout_generator(
    lu_stream: HfIterableDataset,
    text_field: str | None,
    heldout_ratio: float,
) -> Iterable[dict[str, Any]]:
    for row in lu_stream:
        text = _clean_text(_extract_row_text(row, text_field))
        if not text:
            continue
        if _is_holdout_text(text, heldout_ratio=heldout_ratio):
            yield row


def _lu_ppl_chunk_generator(
    lu_stream: HfIterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    max_chunks: int,
) -> Iterable[dict[str, Any]]:
    text_field = _resolve_text_field(lu_stream)
    emitted = 0
    token_buffer: list[int] = []
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0

    for row in lu_stream:
        raw_text = _extract_row_text(row, text_field)
            
        text = _clean_text(raw_text)
        if not _valid_text(text, min_chars=20):
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue

        if eos_id is not None:
            ids.append(eos_id)

        token_buffer.extend(ids)

        while len(token_buffer) >= SEQUENCE_LENGTH:
            chunk = token_buffer[:SEQUENCE_LENGTH]
            token_buffer = token_buffer[SEQUENCE_LENGTH:]
            emitted += 1
            yield {
                "input_ids": chunk,
                "attention_mask": [1] * SEQUENCE_LENGTH,
                "labels": chunk.copy(),
                "lang": "lu",
            }
            if emitted >= max_chunks:
                return

    # On small held-out streams, flush the remaining tail as a masked padded chunk
    # so perplexity still has valid tokens to evaluate.
    if token_buffer and emitted < max_chunks:
        remainder = token_buffer[:SEQUENCE_LENGTH]
        attn = [1] * len(remainder)
        labels = remainder.copy()

        if len(remainder) < SEQUENCE_LENGTH:
            pad_len = SEQUENCE_LENGTH - len(remainder)
            remainder = remainder + ([pad_id] * pad_len)
            attn = attn + ([0] * pad_len)
            labels = labels + ([-100] * pad_len)

        yield {
            "input_ids": remainder,
            "attention_mask": attn,
            "labels": labels,
            "lang": "lu",
        }


def _build_luxgen_eval_rows(
    lu_stream: HfIterableDataset,
    max_samples: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    text_field = _resolve_text_field(lu_stream)

    for row in lu_stream:
        raw_text = _extract_row_text(row, text_field)
            
        text = _clean_text(raw_text)
        if not _valid_text(text, min_chars=30):
            continue

        words = text.split()
        if len(words) < 12:
            continue

        prompt = " ".join(words[:8])
        rows.append({"prompt": prompt, "reference": text, "lang": "lu"})

        if len(rows) >= max_samples:
            break

    return rows


def load_lu_eval_datasets(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[HfIterableDataset, DataLoader]:
    lu_ppl_stream = _load_lu_text_stream(split=LU_EVAL_SPLIT)

    lu_ppl_dataset = HfIterableDataset.from_generator(
        _lu_ppl_chunk_generator,
        gen_kwargs={
            "lu_stream": lu_ppl_stream,
            "tokenizer": tokenizer,
            "max_chunks": MAX_LU_PPL_CHUNKS,
        },
    )

    # Build generation eval prompts from the same LU source (LuxGen-style prompt->continuation).
    luxgen_stream = _load_lu_text_stream(split=LU_EVAL_SPLIT)
    luxgen_rows = _build_luxgen_eval_rows(luxgen_stream, max_samples=MAX_LUXGEN_SAMPLES)
    luxgen_eval_loader = DataLoader(luxgen_rows, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    return lu_ppl_dataset, luxgen_eval_loader


def build_eval_mono_datasets(
    tokenizer: PreTrainedTokenizerBase,
    max_chunks_per_lang: int = 8_000,
) -> dict[str, HfIterableDataset]:
    datasets: dict[str, HfIterableDataset] = {}
    for lang in ("en", "de", "nl"):
        # Validation split is attempted first; fallback to train stream if missing.
        try:
            datasets[lang] = load_mono_dataset(
                lang,
                tokenizer,
                split="validation",
                max_chunks=max_chunks_per_lang,
            )
        except Exception:
            datasets[lang] = load_mono_dataset(
                lang,
                tokenizer,
                split=MONO_SPLIT,
                max_chunks=max_chunks_per_lang,
            )
    return datasets


def _ensure_eval_loader(dataset_or_loader: Any) -> DataLoader:
    if isinstance(dataset_or_loader, DataLoader):
        return dataset_or_loader
    return DataLoader(
        dataset_or_loader,
        batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_lm_batch,
        pin_memory=torch.cuda.is_available(),
    )


def compute_perplexity(
    model: torch.nn.Module,
    dataset: Any,
    lang: str,
    max_batches: int = PPL_EVAL_MAX_BATCHES,
) -> float:
    loader = _ensure_eval_loader(dataset)
    device = next(model.parameters()).device

    total_nll = 0.0
    total_tokens = 0
    skipped_non_finite = 0

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            if step > max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )

            valid_tokens = int((labels != -100).sum().item())
            loss_value = float(outputs.loss.item())
            if not math.isfinite(loss_value):
                skipped_non_finite += 1
                continue

            total_nll += loss_value * max(valid_tokens, 1)
            total_tokens += max(valid_tokens, 1)

    if total_tokens == 0:
        detail = (
            f"all batches produced non-finite loss ({skipped_non_finite} skipped)"
            if skipped_non_finite > 0
            else "no valid tokens"
        )
        raise RuntimeError(f"No valid tokens available for PPL computation ({lang}): {detail}.")

    avg_nll = total_nll / total_tokens
    if not math.isfinite(avg_nll):
        raise RuntimeError(f"Non-finite average NLL during PPL computation ({lang}).")

    return float(math.exp(min(avg_nll, 50.0)))


def _simple_token_f1(pred: str, ref: str) -> float:
    pred_tokens = pred.strip().split()
    ref_tokens = ref.strip().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    overlap = len(pred_set & ref_set)

    precision = overlap / max(len(pred_set), 1)
    recall = overlap / max(len(ref_set), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_luxgen(
    model: torch.nn.Module,
    luxgen_data: DataLoader,
    output_tag: str = "eval",
) -> dict[str, Any]:
    ensure_run_directories()

    # Tokenizer is shared across Model 1/2/3, so load from local base path.
    model_path = ensure_model_local(download_if_missing=False)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    model.eval()

    predictions: list[str] = []
    references: list[str] = []
    samples: list[dict[str, str]] = []

    with torch.no_grad():
        for batch in luxgen_data:
            prompts = [str(p) for p in batch["prompt"]]
            refs = [str(r) for r in batch["reference"]]

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=SEQUENCE_LENGTH // 2,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            generated = model.generate(
                **encoded,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for prompt, pred, ref in zip(prompts, decoded, refs):
                predictions.append(pred)
                references.append(ref)
                samples.append({"prompt": prompt, "prediction": pred, "reference": ref})

    metrics: dict[str, Any] = {
        "num_samples": len(predictions),
    }

    try:
        import sacrebleu  # type: ignore

        metrics["bleu"] = float(sacrebleu.corpus_bleu(predictions, [references]).score)
        metrics["chrf"] = float(sacrebleu.corpus_chrf(predictions, [references]).score)
    except Exception:
        f1_scores = [_simple_token_f1(p, r) for p, r in zip(predictions, references)]
        metrics["bleu"] = 0.0
        metrics["chrf"] = 0.0
        metrics["token_f1_fallback"] = float(sum(f1_scores) / max(len(f1_scores), 1))

    safe_tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", output_tag)
    out_path = EVAL_DIR / f"luxgen_samples_{safe_tag}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics["eval_split"] = LU_EVAL_SPLIT
    metrics["heldout_ratio"] = float(LU_HELDOUT_RATIO)
    metrics["samples_path"] = str(out_path)
    return metrics


__all__ = [
    "load_mono_dataset",
    "load_parallel_datasets",
    "load_lu_eval_datasets",
    "build_mono_train_loader",
    "build_eval_mono_datasets",
    "compute_perplexity",
    "evaluate_luxgen",
    "collate_lm_batch",
    "collate_parallel_batch",
]
