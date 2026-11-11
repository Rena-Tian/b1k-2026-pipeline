#!/usr/bin/env bash
# Task 54 (putting_away_toys) fine-tune — task-64 recipe: 50-task generalist init, 10k steps, batch 16, ft1.
# 8 goal predicates (all inside), 11,297-step demos. Config was appended on the pod (backup config.py.pre-task54).
# Usage: POD=root@<ip> PORT=<port> ./bootstrap_task54_training.sh
set -u
POD="${POD:?set POD=root@<ip>}"; PORT="${PORT:?set PORT}"
LOG=~/eval_results/bootstrap_task54.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
sshp(){ ssh -o ConnectTimeout=25 -o BatchMode=yes -p $PORT $POD "$@" </dev/null; }

if sshp 'tmux ls 2>/dev/null | grep -q train'; then log "FATAL: a training tmux session is already running"; exit 1; fi
sshp 'mkdir -p /root/logs /root/ckpt'
sshp 'test -x /root/venv_b1k/bin/python' || { log "FATAL: /root/venv_b1k is gone"; exit 1; }
sshp 'test -f /workspace/openpi_checkpoints/winner_larchenko/behavior_50t_checkpoint/params/manifest.ocdbt' || { log "FATAL: generalist checkpoint missing"; exit 1; }
log "pod up: $(sshp 'nvidia-smi --query-gpu=name --format=csv,noheader'), volume $(sshp 'du -sh /workspace | cut -f1')"

if ! sshp 'test -d /workspace/behavior-challenge-2026/datasets/task54/data'; then
  log "downloading task-54 dataset (chunk-054, rgb only)"
  sshp 'cd /workspace && /root/venv_b1k/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"behavior-1k/2026-challenge-demos\", repo_type=\"dataset\",
  allow_patterns=[\"meta/**\",\"data/chunk-054/*\",\"videos/observation.rgb.*/chunk-054/*\"],
  local_dir=\"/workspace/behavior-challenge-2026/datasets/task54\", max_workers=8)
print(\"dataset done\")"' > /dev/null 2>> "$LOG"
fi
log "dataset: $(sshp 'du -sh /workspace/behavior-challenge-2026/datasets/task54 | cut -f1'), $(sshp 'ls /workspace/behavior-challenge-2026/datasets/task54/data/chunk-054 | wc -l') parquet files"

log "pruning non-rgb video keys from meta/info.json"
sshp 'D=/workspace/behavior-challenge-2026/datasets/task54; cp -n $D/meta/info.json $D/meta/info.upstream.json 2>/dev/null; /root/venv_b1k/bin/python -c "
import json; p=\"$D/meta/info.json\"; d=json.load(open(p))
keep={k:v for k,v in d[\"features\"].items() if not (v.get(\"dtype\")==\"video\" and not k.startswith(\"observation.rgb.\"))}
d[\"features\"]=keep; json.dump(d, open(p,\"w\"), indent=4); print(\"video keys:\", [k for k,v in keep.items() if v.get(\"dtype\")==\"video\"])"' >> "$LOG" 2>&1

log "installing assets from the generalist checkpoint"
sshp 'cd /workspace/behavior-1k-solution && A=outputs/assets/pi_behavior_b1k_2026_task54/b1k_2026_task54 && mkdir -p $A && cp -r /workspace/openpi_checkpoints/winner_larchenko/behavior_50t_checkpoint/assets/*/*/{norm_stats.json,fast_tokenizer} $A/ 2>/dev/null || cp -r /workspace/openpi_checkpoints/winner_larchenko/behavior_50t_checkpoint/assets/*/{norm_stats.json,fast_tokenizer} $A/; ls $A' >> "$LOG" 2>&1

log "sanity check (one batch) — expect tokenized_prompt [[54, s], ...]"
sshp 'cd /workspace/behavior-1k-solution && JAX_PLATFORMS=cpu /root/venv_b1k/bin/python -c "
import dataclasses, numpy as np
from b1k.training import config as c, data_loader as dl
cfg=dataclasses.replace(c.get_config(\"pi_behavior_b1k_2026_task54\"), batch_size=4, num_workers=0)
obs,act=next(iter(dl.create_behavior_data_loader(cfg, shuffle=True, num_batches=1)))
d=obs.to_dict(); print(\"tokenized_prompt\", np.asarray(d[\"tokenized_prompt\"]).tolist())
a=np.asarray(act); print(\"actions\", a.shape, \"max|norm|\", float(np.abs(a[:,:,:23]).max()))"' 2>&1 | grep -vE "Warning|pynvml|Unknown fields|INFO" | tee -a "$LOG"
if ! grep -q "tokenized_prompt \[\[54," "$LOG"; then log "FATAL: sanity check did not show task id 54 — not launching"; exit 1; fi

log "launching training (tmux train54)"
sshp 'cp /workspace/.netrc_wandb ~/.netrc 2>/dev/null; chmod 600 ~/.netrc 2>/dev/null; cd /workspace && (CKPT_SWEEP=2000,4000,6000,8000,9999 setsid nohup python3 ckpt_sync.py /root/ckpt/pi_behavior_b1k_2026_task54/ft1 /workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task54/ft1 60 > /workspace/logs/ckpt_sync_task54.log 2>&1 < /dev/null &); tmux new -d -s train54 "cd /workspace/behavior-1k-solution && B1K_PARAMS_ONLY_CHECKPOINTS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 /root/venv_b1k/bin/python -u scripts/train.py pi_behavior_b1k_2026_task54 --exp_name=ft1 --overwrite --checkpoint_base_dir=/root/ckpt --batch_size=16 --num_train_steps=10000 --save_interval=500 --keep_period=2000 --log_interval=25 --num_workers=16 2>&1 | tee -a /root/logs/train_task54_ft1.log"; sleep 3; tmux ls' | tee -a "$LOG"

log "arming the checkpoint streamer -> ~/models/task54_ft"
POD=$POD PORT=$PORT setsid nohup bash ~/eval_results/stream_task62r2.sh pi_behavior_b1k_2026_task54 task54_ft 2000 4000 6000 8000 9999 >/dev/null 2>&1 < /dev/null &
log "bootstrap complete"
