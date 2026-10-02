#!/usr/bin/env bash
# Task 12 (preparing_lunch_box) fine-tune — task-1 recipe: init winner checkpoint_2 (already trained
# task 12; the winner scored 0.517 public / 0.367 hidden on it), num_tasks=50, 10k steps, batch 16, ft1.
# DO NOT RUN until the user confirms task 12 is claimed on the team tracker. Usage: POD=root@<ip> PORT=<port> ./bootstrap_task12_training.sh
set -u
POD="${POD:?set POD=root@<ip>}"; PORT="${PORT:?set PORT}"
LOG=~/winner-eval/bootstrap_task12.log
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

log "syncing repo (task-12 config)"
rsync -rlt --exclude '.venv' --exclude 'outputs' --exclude '__pycache__' --exclude '*.egg-info' -e "ssh -p $PORT" ~/behavior-1k-solution/ "$POD:/workspace/behavior-1k-solution/" >> "$LOG" 2>&1

if ! sshp 'test -f /workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params/manifest.ocdbt'; then
  log "FATAL: winner checkpoint_2 is missing from the volume"; exit 1
fi
log "init checkpoint present: checkpoint_2"

# task 12 has never been on the volume, so its chunk comes down fresh
if ! sshp 'test -d /workspace/behavior-challenge-2026/datasets/task12/data'; then
  log "downloading task-12 dataset (chunk-012, rgb only)"
  sshp 'cd /workspace && /root/venv_b1k/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"behavior-1k/2026-challenge-demos\", repo_type=\"dataset\",
  allow_patterns=[\"meta/**\",\"data/chunk-012/*\",\"videos/observation.rgb.*/chunk-012/*\"],
  local_dir=\"/workspace/behavior-challenge-2026/datasets/task12\", max_workers=8)
print(\"dataset done\")"' >> "$LOG" 2>&1
fi

# LeRobot refuses a dataset whose info.json lists video streams we did not download (depth/seg)
log "pruning non-rgb video keys from meta/info.json"
sshp 'D=/workspace/behavior-challenge-2026/datasets/task12; cp -n $D/meta/info.json $D/meta/info.upstream.json 2>/dev/null; /root/venv_b1k/bin/python -c "
import json; p=\"$D/meta/info.json\"; d=json.load(open(p))
keep={k:v for k,v in d[\"features\"].items() if not (v.get(\"dtype\")==\"video\" and not k.startswith(\"observation.rgb.\"))}
d[\"features\"]=keep; json.dump(d, open(p,\"w\"), indent=4); print(\"video keys:\", [k for k,v in keep.items() if v.get(\"dtype\")==\"video\"])"' >> "$LOG" 2>&1

log "installing assets from winner checkpoint_2"
sshp 'cd /workspace/behavior-1k-solution && A=outputs/assets/pi_behavior_b1k_2026_task12/b1k_2026_task12 && mkdir -p $A && cp -r /workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/assets/IliaLarchenko/behavior_224_rgb/{norm_stats.json,fast_tokenizer} $A/ && ls $A' >> "$LOG" 2>&1

log "sanity check (one batch) — expect tokenized_prompt [[12, s], ...]"
sshp 'cd /workspace/behavior-1k-solution && JAX_PLATFORMS=cpu /root/venv_b1k/bin/python -c "
import dataclasses, numpy as np
from b1k.training import config as c, data_loader as dl
cfg=dataclasses.replace(c.get_config(\"pi_behavior_b1k_2026_task12\"), batch_size=4, num_workers=0)
obs,act=next(iter(dl.create_behavior_data_loader(cfg, shuffle=True, num_batches=1)))
d=obs.to_dict(); print(\"tokenized_prompt\", np.asarray(d[\"tokenized_prompt\"]).tolist())
a=np.asarray(act); print(\"actions\", a.shape, \"max|norm|\", float(np.abs(a[:,:,:23]).max()))"' 2>&1 | grep -vE "Warning|pynvml|Unknown fields|INFO" | tee -a "$LOG"

# exp_name stays ft1 (user preference); CKPT_SWEEP ends at 9999, never 10000.
log "launching training (tmux train12)"
sshp 'cp /workspace/.netrc_wandb ~/.netrc 2>/dev/null; chmod 600 ~/.netrc 2>/dev/null; cd /workspace && (CKPT_SWEEP=2000,4000,6000,8000,9999 setsid nohup python3 ckpt_sync.py /root/ckpt/pi_behavior_b1k_2026_task12/ft1 /workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task12/ft1 60 > /workspace/logs/ckpt_sync_task12.log 2>&1 < /dev/null &); tmux new -d -s train12 "cd /workspace/behavior-1k-solution && B1K_PARAMS_ONLY_CHECKPOINTS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 /root/venv_b1k/bin/python -u scripts/train.py pi_behavior_b1k_2026_task12 --exp_name=ft1 --overwrite --checkpoint_base_dir=/root/ckpt --batch_size=16 --num_train_steps=10000 --save_interval=500 --keep_period=2000 --log_interval=25 --num_workers=16 2>&1 | tee -a /root/logs/train_task12_ft1.log"; sleep 3; tmux ls' | tee -a "$LOG"

log "arming the checkpoint streamer for task 12"
cd ~/winner-eval && POD=$POD PORT=$PORT setsid nohup bash ./stream_ckpts.sh pi_behavior_b1k_2026_task12 ft_task12 2000 4000 6000 8000 9999 >/dev/null 2>&1 < /dev/null &
log "bootstrap complete"
