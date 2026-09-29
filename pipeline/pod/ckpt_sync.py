#!/usr/bin/env python3
"""Move params-only checkpoints from the pod's local disk to the network volume.

Training saves to LOCAL_ROOT (fast local overlay, ~40 GB free). This script copies every completed
step dir to VOL_ROOT, verifies the byte count, then deletes the local copy (keeping the newest one
so --resume/inspection still works), and on the volume keeps only the sweep steps + the newest.
Usage: ckpt_sync.py LOCAL_ROOT VOL_ROOT [interval_s]
"""
import os, shutil, subprocess, sys, time
from pathlib import Path

local_root, vol_root = Path(sys.argv[1]), Path(sys.argv[2])
interval = int(sys.argv[3]) if len(sys.argv) > 3 else 60
SWEEP = set(int(x) for x in os.environ.get("CKPT_SWEEP", "2000,4000,6000,8000,10000").split(","))

def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
def du(p): return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
def completed(root):
    if not root.exists(): return []
    return sorted(int(d.name) for d in root.iterdir() if d.is_dir() and d.name.isdigit()
                  and not any("orbax-checkpoint-tmp" in p.name for p in d.iterdir()))

while True:
    try:
        vol_root.mkdir(parents=True, exist_ok=True)
        loc = completed(local_root)
        for s in loc:
            src, dst = local_root / str(s), vol_root / str(s)
            if s not in SWEEP and s != loc[-1]:
                continue  # volume is quota-limited (200 GB): sweep steps + the newest step (crash insurance) only
            if not dst.exists() or du(dst) != du(src):
                tmp = vol_root / f"{s}.copying"
                if tmp.exists(): shutil.rmtree(tmp)
                log(f"step {s}: copying {du(src)/1e9:.1f} GB to volume")
                t = time.time(); subprocess.run(["cp", "-r", str(src), str(tmp)], check=True)
                if du(tmp) != du(src): log(f"step {s}: size mismatch after copy, retrying next round"); continue
                tmp.rename(dst); log(f"step {s}: copied in {time.time()-t:.0f}s")
        # local: keep only the newest completed step (sweep steps only once they are on the volume)
        for s in loc[:-1]:
            if s not in SWEEP or (vol_root / str(s)).exists():
                log(f"step {s}: removing local copy"); shutil.rmtree(local_root / str(s), ignore_errors=True)
        # volume: keep sweep steps + newest
        vol = completed(vol_root)
        for s in vol[:-1]:
            if s not in SWEEP:
                log(f"step {s}: removing from volume (not a sweep step)"); shutil.rmtree(vol_root / str(s), ignore_errors=True)
    except Exception as e:
        log(f"error: {e!r}")
    time.sleep(interval)
