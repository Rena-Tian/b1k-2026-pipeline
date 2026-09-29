#!/usr/bin/env python3
"""Keep the fine-tune checkpoint dir within the network-volume budget.

Orbax keeps the latest step plus every `keep_period` step. Each full step dir is ~50 GB
(train_state = optimizer + EMA + params, plus params for inference). This janitor:
  * keeps train_state only for the newest completed step (needed for --resume),
  * strips train_state from every older step (params + assets stay: ~12.6 GB, enough for eval),
  * deletes older steps entirely unless they are in KEEP_PARAMS_STEPS.
Run in the background: python ckpt_janitor.py <checkpoint_dir> [interval_s]
"""
import shutil, sys, time
from pathlib import Path

root = Path(sys.argv[1]); interval = int(sys.argv[2]) if len(sys.argv) > 2 else 120
KEEP_PARAMS_STEPS = {2000, 4000, 6000, 8000, 10000}

def completed_steps():
    out = []
    for d in root.iterdir():
        if d.is_dir() and d.name.isdigit() and not any(p.name.endswith("orbax-checkpoint-tmp") or ".orbax-checkpoint-tmp" in p.name for p in d.iterdir()):
            out.append(int(d.name))
    return sorted(out)

def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)

while True:
    try:
        steps = completed_steps()
        if steps:
            newest = steps[-1]
            for s in steps[:-1]:
                d = root / str(s)
                if s in KEEP_PARAMS_STEPS or s == newest:
                    ts = d / "train_state"
                    if ts.exists():
                        log(f"step {s}: dropping train_state (keeping params for eval)"); shutil.rmtree(ts, ignore_errors=True)
                else:
                    log(f"step {s}: deleting (not a sweep step, newer checkpoint exists)"); shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        log(f"error: {e!r}")
    time.sleep(interval)
