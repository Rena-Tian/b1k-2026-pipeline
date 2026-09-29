#!/usr/bin/env bash
# Retrain task 99 (sorting_books_on_shelf) as a DEDICATED single-task run.
#
# Previously it shared a 20 000-step run with tasks 56 and 78, so its task embedding received
# roughly a third of the gradient signal; 56 came out of that run at 0.263 and 99 at 0.000.
# Task 89 — one task, one run, same generalist init — produced four outright solves. That is the
# hypothesis here. Target is ONE book landed (0.1 per book, 10 graded placements), not completion:
# the eval currently reaches stage 4 of 6 and never places one.
#
# Waits for the pod to come back up — it was stopped on 2026-09-25 after the task-1 run.
set -u
POD="${POD:?set POD=root@<ip>}"; PORT="${PORT:?set PORT}"
LOG=~/winner-eval/bootstrap_task99.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
sshp(){ ssh -o ConnectTimeout=25 -o BatchMode=yes -p $PORT $POD "$@" </dev/null; }

log "waiting for the pod to become reachable (start it at console.runpod.io, pod hks46y52h8iqkg)"
until sshp true 2>/dev/null; do sleep 60; done
log "pod is up: $(sshp 'nvidia-smi --query-gpu=name --format=csv,noheader')"
log "volume: $(sshp 'du -sh /workspace | cut -f1')"

# the network volume survives a stop, but the container disk does not — recreate what lived there
sshp 'mkdir -p /root/logs /root/ckpt'
if ! sshp 'test -x /root/venv_b1k/bin/python'; then
  log "FATAL: /root/venv_b1k is gone (container disk was wiped and not rebuilt)."
  log "       The venv must be recreated before training can start — stopping here rather than guessing."
  exit 1
fi

log "syncing repo (task-99 config)"
rsync -rlt --exclude '.venv' --exclude 'outputs' --exclude '__pycache__' --exclude '*.egg-info' -e "ssh -p $PORT" ~/behavior-1k-solution/ "$POD:/workspace/behavior-1k-solution/" >> "$LOG" 2>&1

if ! sshp 'test -f /workspace/openpi_checkpoints/winner_larchenko/behavior_50t_checkpoint/params/manifest.ocdbt'; then
  log "FATAL: the 50-task generalist checkpoint is missing from the volume"; exit 1
fi
log "init checkpoint present: behavior_50t_checkpoint"

# the 3-task dataset was deleted to free 36 GB, so task 99's chunk has to come down again
if ! sshp 'test -d /workspace/behavior-challenge-2026/datasets/task99/data'; then
  log "downloading task-99 dataset (chunk-099, rgb only)"
  sshp 'cd /workspace && /root/venv_b1k/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"behavior-1k/2026-challenge-demos\", repo_type=\"dataset\",
  allow_patterns=[\"meta/**\",\"data/chunk-099/*\",\"videos/observation.rgb.*/chunk-099/*\"],
  local_dir=\"/workspace/behavior-challenge-2026/datasets/task99\", max_workers=8)
print(\"dataset done\")"' >> "$LOG" 2>&1
fi

# LeRobot refuses a dataset whose info.json lists video streams we did not download (depth/seg)
log "pruning non-rgb video keys from meta/info.json"
sshp 'D=/workspace/behavior-challenge-2026/datasets/task99; cp -n $D/meta/info.json $D/meta/info.upstream.json 2>/dev/null; /root/venv_b1k/bin/python -c "
import json; p=\"$D/meta/info.json\"; d=json.load(open(p))
keep={k:v for k,v in d[\"features\"].items() if not (v.get(\"dtype\")==\"video\" and not k.startswith(\"observation.rgb.\"))}
d[\"features\"]=keep; json.dump(d, open(p,\"w\"), indent=4); print(\"video keys:\", [k for k,v in keep.items() if v.get(\"dtype\")==\"video\"])"' >> "$LOG" 2>&1

log "installing assets from the generalist checkpoint"
sshp 'cd /workspace/behavior-1k-solution && A=outputs/assets/pi_behavior_b1k_2026_task99/b1k_2026_task99 && mkdir -p $A && cp -r /workspace/openpi_checkpoints/winner_larchenko/behavior_50t_checkpoint/assets/*/*/{norm_stats.json,fast_tokenizer} $A/ 2>/dev/null || cp -r /workspace/openpi_checkpoints/winner_larchenko/behavior_50t_checkpoint/assets/*/{norm_stats.json,fast_tokenizer} $A/; ls $A' >> "$LOG" 2>&1

log "sanity check (one batch) — expect tokenized_prompt [[99, s], ...]"
sshp 'cd /workspace/behavior-1k-solution && JAX_PLATFORMS=cpu /root/venv_b1k/bin/python -c "
import dataclasses, numpy as np
from b1k.training import config as c, data_loader as dl
cfg=dataclasses.replace(c.get_config(\"pi_behavior_b1k_2026_task99\"), batch_size=4, num_workers=0)
obs,act=next(iter(dl.create_behavior_data_loader(cfg, shuffle=True, num_batches=1)))
d=obs.to_dict(); print(\"tokenized_prompt\", np.asarray(d[\"tokenized_prompt\"]).tolist())
a=np.asarray(act); print(\"actions\", a.shape, \"max|norm|\", float(np.abs(a[:,:,:23]).max()))"' 2>&1 | grep -vE "Warning|pynvml|Unknown fields|INFO" | tee -a "$LOG"

# exp_name stays ft1 (user preference); CKPT_SWEEP ends at 9999, never 10000.
log "launching training (tmux train99)"
sshp 'cp /workspace/.netrc_wandb ~/.netrc 2>/dev/null; chmod 600 ~/.netrc 2>/dev/null; cd /workspace && (CKPT_SWEEP=2000,4000,6000,8000,9999 setsid nohup python3 ckpt_sync.py /root/ckpt/pi_behavior_b1k_2026_task99/ft1 /workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task99/ft1 60 > /workspace/logs/ckpt_sync_task99.log 2>&1 < /dev/null &); tmux new -d -s train99 "cd /workspace/behavior-1k-solution && B1K_PARAMS_ONLY_CHECKPOINTS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 /root/venv_b1k/bin/python -u scripts/train.py pi_behavior_b1k_2026_task99 --exp_name=ft1 --overwrite --checkpoint_base_dir=/root/ckpt --batch_size=16 --num_train_steps=10000 --save_interval=500 --keep_period=2000 --log_interval=25 --num_workers=16 2>&1 | tee -a /root/logs/train_task99_ft1.log"; sleep 3; tmux ls' | tee -a "$LOG"

log "arming the checkpoint streamer for task 99"
cd ~/winner-eval && POD=$POD PORT=$PORT setsid nohup bash ./stream_ckpts.sh pi_behavior_b1k_2026_task99 ft_task99 2000 4000 6000 8000 9999 >/dev/null 2>&1 < /dev/null &
log "bootstrap complete"
