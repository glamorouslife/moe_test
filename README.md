# OLMoE Cross-Lingual Routing Alignment (EN-DE-NL-LU)

This repository contains a thesis implementation for cross-lingual expert routing
alignment on `allenai/OLMoE-1B-7B-0125`.

## Strict 4-File Architecture

All thesis logic is intentionally limited to these four Python files:

1. `config.py`
2. `data_and_eval.py`
3. `models_and_training.py`
4. `routing_analysis.py`

No restricted external corpus is used in this implementation.

## Install

Requirements:

- Python 3.12+
- CUDA-capable GPU recommended
- Disk space for local model checkpoints in `models/`

Using `uv`:

```powershell
uv sync
```

Using `pip`:

```powershell
python -m pip install -e .
```

## Main Entry Point

All runtime operations are controlled via:

```powershell
python models_and_training.py --mode <smoke|baseline|cont|align|eval>
```

### Commands

Smoke test (GPU + multilingual generation):

```powershell
python models_and_training.py --mode smoke
```

Model 1 baseline evaluation:

```powershell
python models_and_training.py --mode baseline
```

Model 2 continued pretraining:

```powershell
python models_and_training.py --mode cont
```

Model 3 alignment training:

```powershell
python models_and_training.py --mode align
```

Evaluate one or all models:

```powershell
python models_and_training.py --mode eval --eval-model all
```

Optional step override for training:

```powershell
python models_and_training.py --mode cont --max-steps 100
```

## File Responsibilities

- `config.py`: constants, paths, runtime config, seed setup, model download/local checks.
- `data_and_eval.py`: monolingual/parallell/LU data loaders and evaluation functions
	(PPL and LuxGen-style generation metrics).
- `models_and_training.py`: model loading, smoke test, Model 2 and Model 3 training loops,
	evaluation CLI.
- `routing_analysis.py`: router hooks, cache utilities, expert load/similarity/entropy metrics,
	routing analysis output artifacts.

## Outputs

Generated artifacts are stored under `runs/`:

- checkpoints: `runs/checkpoints/`
- tensorboard logs: `runs/tensorboard/`
- routing metrics: `runs/routing_analysis/`
- eval outputs: `runs/eval/`
