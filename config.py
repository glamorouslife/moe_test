from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download


# =========================
# Core Thesis Configuration
# =========================
BASE_MODEL_NAME = "allenai/OLMoE-1B-7B-0125"
LANGS = ["en", "de", "nl", "lu"]

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
BASE_MODEL_DIR = MODELS_DIR / "olmoe-1b-7b-0125"

RUNS_DIR = ROOT_DIR / "runs"
CHECKPOINT_DIR = RUNS_DIR / "checkpoints"
MODEL2_CKPT_DIR = CHECKPOINT_DIR / "model2_cont"
MODEL3_CKPT_DIR = CHECKPOINT_DIR / "model3_align"
TENSORBOARD_DIR = RUNS_DIR / "tensorboard"
ANALYSIS_DIR = RUNS_DIR / "routing_analysis"
EVAL_DIR = RUNS_DIR / "eval"


# ================
# Dataset Settings
# ================
# Monolingual EN/DE/NL candidates (Wikipedia-first, then open corpora fallback).
MONO_DATASET_CANDIDATES: dict[str, list[tuple[str, str | None]]] = {
    "en": [
        ("wikimedia/wikipedia", "20231101.en"),
        ("wikipedia", "20220301.en"),
        ("open_subtitles", "en"),
    ],
    "de": [
        ("wikimedia/wikipedia", "20231101.de"),
        ("wikipedia", "20220301.de"),
        ("open_subtitles", "de"),
    ],
    "nl": [
        ("wikimedia/wikipedia", "20231101.nl"),
        ("wikipedia", "20220301.nl"),
        ("open_subtitles", "nl"),
    ],
}

# Parallel EN-DE / EN-NL / DE-NL candidates (Europarl-first, then similar fallback).
PARALLEL_PAIRS = (("en", "de"), ("en", "nl"), ("de", "nl"))
PARALLEL_DATASET_CANDIDATES: dict[tuple[str, str], list[tuple[str, str | None]]] = {
    ("en", "de"): [
        ("Helsinki-NLP/europarl", "en-de"),
        ("europarl_bilingual", "en-de"),
        ("opus100", "en-de"),
    ],
    ("en", "nl"): [
        ("Helsinki-NLP/europarl", "en-nl"),
        ("europarl_bilingual", "en-nl"),
        ("opus100", "en-nl"),
    ],
    ("de", "nl"): [
        ("Helsinki-NLP/europarl", "de-nl"),
        ("europarl_bilingual", "de-nl"),
        ("opus100", "de-nl"),
    ],
}

LU_PPL_DATASET_CANDIDATES: list[tuple[str, str | None]] = [
    ("Shdorsh/luxembourgish-lu", None),
]

LUXGEN_DATASET_CANDIDATES: list[tuple[str, str | None]] = [
    ("Shdorsh/luxembourgish-lu", None),
]


# ==================
# Training / Runtime
# ==================
MONO_SPLIT = "train"
PARALLEL_SPLIT = "train"
LU_PPL_SPLIT = "train"
LU_EVAL_SPLIT = "heldout"  # heldout | validation | test | train
LU_HELDOUT_RATIO = 0.20

SEQUENCE_LENGTH = 1024
PARALLEL_MAX_LENGTH = 256
BATCH_SIZE_MONO = 2
BATCH_SIZE_PARALLEL = 2
EVAL_BATCH_SIZE = 2
NUM_WORKERS = 0

MAX_MONO_CHUNKS_PER_LANG = 200_000
MAX_PARALLEL_EXAMPLES_PER_PAIR = 300_000
MAX_LU_PPL_CHUNKS = 20_000
MAX_LUXGEN_SAMPLES = 256

LEARNING_RATE = 1e-5
LEARNING_RATE_CONT = 2e-6
LEARNING_RATE_ALIGN = 2e-6
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
MAX_STEPS_MODEL2 = 1_000
MAX_STEPS_MODEL3 = 1_000
GRAD_ACCUM_STEPS = 1
LOG_EVERY = 10
SAVE_EVERY = 200

# Stability-first low-VRAM policy. Enable lm_head only if training is numerically stable.
LOW_VRAM_TRAIN_LM_HEAD = False

ALIGN_LAYERS = list(range(8, 21))
ALIGN_LAMBDA = 0.1
ALIGN_DISTANCE = "jsd"  # jsd | cosine

ROUTING_TOP_K = 8
ROUTING_ANALYSIS_MAX_BATCHES = 30
PPL_EVAL_MAX_BATCHES = 200
GEN_MAX_NEW_TOKENS = 80

SEED = 42
DETERMINISTIC = False
DOWNLOAD_IF_MISSING = True
USE_FP16_IF_CUDA = True


# =========================
# Base Model Local Handling
# =========================
REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
)


@dataclass(frozen=True)
class RuntimeModelConfig:
    model_name: str = BASE_MODEL_NAME
    model_dir: Path = BASE_MODEL_DIR
    use_fp16_if_cuda: bool = USE_FP16_IF_CUDA

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def dtype(self) -> torch.dtype:
        if self.device == "cuda" and self.use_fp16_if_cuda:
            return torch.float16
        return torch.float32


MODEL_RUNTIME = RuntimeModelConfig()


def ensure_run_directories() -> None:
    for path in (RUNS_DIR, CHECKPOINT_DIR, TENSORBOARD_DIR, ANALYSIS_DIR, EVAL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED, deterministic: bool = DETERMINISTIC) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def model_artifacts_present(model_dir: Path | None = None) -> bool:
    root = model_dir or MODEL_RUNTIME.model_dir
    return root.exists() and all((root / name).exists() for name in REQUIRED_MODEL_FILES)


def ensure_model_local(download_if_missing: bool | None = None) -> Path:
    should_download = DOWNLOAD_IF_MISSING if download_if_missing is None else download_if_missing
    model_dir = MODEL_RUNTIME.model_dir

    if model_artifacts_present(model_dir):
        return model_dir

    if not should_download:
        missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).exists()]
        raise FileNotFoundError(
            f"Model not ready at {model_dir}. Missing artifacts: {missing}. "
            "Set download_if_missing=True or download manually."
        )

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_RUNTIME.model_name,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )

    if not model_artifacts_present(model_dir):
        raise RuntimeError(f"Download completed but expected artifacts are missing in {model_dir}.")

    return model_dir