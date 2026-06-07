"""End-to-end OLMoE pipeline runner: cont -> align -> eval all + multi-seed."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OLMOE_MOE_NAME", "olmoe")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

STAGES: list[tuple[str, list[str]]] = [
    ("wipe_cont",  []),
    ("model2_cont",  ["--mode", "cont"]),
    ("eval_cont",    ["--mode", "eval", "--eval-model", "cont"]),
    ("wipe_align", []),
    ("model3_align", ["--mode", "align"]),
    ("eval_align",   ["--mode", "eval", "--eval-model", "align"]),
    ("eval_baseline", ["--mode", "eval", "--eval-model", "baseline"]),
    ("multiseed",    ["--mode", "multiseed_eval", "--eval-model", "all",
                      "--seeds", "42,43,44", "--include-routing"]),
]

ckpt_root = REPO / "runs" / "olmoe" / "checkpoints"


def wipe(name: str) -> None:
    target = {"wipe_cont": ckpt_root / "model2_cont",
              "wipe_align": ckpt_root / "model3_align"}[name]
    shutil.rmtree(target, ignore_errors=True)
    print(f"[runner] wiped {target}", flush=True)


def main() -> int:
    log = REPO / "runs" / "olmoe_e2e.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for stage, args in STAGES:
        print(f"\n[runner] === {stage} === elapsed={(time.time()-start)/60:.1f} min", flush=True)
        if not args:
            wipe(stage)
            continue
        cmd = [PYTHON, "models_and_training.py", *args]
        print(f"[runner] $ {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        print(f"[runner] {stage} rc={rc} elapsed={(time.time()-start)/60:.1f} min", flush=True)
        if rc != 0:
            print(f"[runner] FAILED at {stage}", flush=True)
            return rc
    print(f"\n[runner] DONE total={(time.time()-start)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
