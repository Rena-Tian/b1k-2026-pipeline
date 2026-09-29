#!/usr/bin/env bash
# Waits for the pod to come back, then sets it up and launches the task-40 fine-tune.
# Everything here is idempotent; safe to re-run.
set -u
POD="root@185.216.21.214"; PORT=35233
PROXY="f9o2yy3xq43l9a-64410cb1@ssh.runpod.io"   # stable across restarts (no scp/rsync over it)
LOG=~/winner-eval/bootstrap_task40.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
sshp(){ ssh -o ConnectTimeout=20 -o BatchMode=yes -p $PORT $POD "$@" </dev/null; }

# A restart usually changes the pod's direct IP/port. Poll the old address and the stable proxy;
# if only the proxy answers, read the new IP/port from the pod's own env vars.
log "waiting for the pod (start it at console.runpod.io); trying $POD:$PORT and the ssh.runpod.io proxy"
while true; do
  if sshp true 2>/dev/null; then log "pod reachable at $POD:$PORT"; break; fi
  out=$(ssh -tt -o ConnectTimeout=20 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$PROXY" 'echo NEWADDR=$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_22; exit' 2>/dev/null | tr -d '\r' | grep -o 'NEWADDR=[0-9.]*:[0-9]*' || true)
  if [ -n "$out" ]; then
    addr=${out#NEWADDR=}; newip=${addr%:*}; newport=${addr#*:}
    if [ -n "$newip" ] && [ -n "$newport" ]; then
      POD="root@$newip"; PORT=$newport
      printf 'POD=%s\nPORT=%s\n' "$POD" "$PORT" > ~/winner-eval/pod_address.env
      log "pod address from proxy: $POD:$PORT"
      sshp true 2>/dev/null && { log "pod reachable at $POD:$PORT"; break; }
    fi
  fi
  sleep 60
done
sshp 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /workspace | tail -1' | tee -a "$LOG"

# 1. repo (container disk is wiped on stop; /workspace survives)
log "syncing repo"
rsync -rlt --exclude '.venv' --exclude 'outputs' --exclude '__pycache__' --exclude '*.egg-info' \
  -e "ssh -p $PORT" ~/behavior-1k-solution/ "$POD:/workspace/behavior-1k-solution/" >> "$LOG" 2>&1

# 2. venv: rebuild on the network volume, then copy to local disk (fast imports)
if ! sshp 'test -x /root/venv_b1k/bin/python'; then
  log "installing env (uv sync) — this takes ~10 min"
  sshp 'cd /workspace/behavior-1k-solution && GIT_LFS_SKIP_SMUDGE=1 UV_PYTHON=3.11 uv sync' >> "$LOG" 2>&1
  log "copying venv to local disk"
  sshp 'cp -r /workspace/behavior-1k-solution/.venv /root/venv_b1k' >> "$LOG" 2>&1
fi

# 3. winner checkpoint_4 (task-40 specialist) on the pod
if ! sshp 'test -f /workspace/openpi_checkpoints/winner_larchenko/checkpoint_4/params/manifest.ocdbt'; then
  log "downloading winner checkpoint_4 to the pod"
  sshp 'cd /workspace && /root/venv_b1k/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"IliaLarchenko/behavior_submission\", allow_patterns=[\"checkpoint_4/*\",\"checkpoint_4/**\"], local_dir=\"/workspace/openpi_checkpoints/winner_larchenko\", max_workers=8)
print(\"ckpt4 done\")"' >> "$LOG" 2>&1
fi

# 4. task-40 dataset (chunk-040 of the official export; rgb videos only, no depth/seg)
if ! sshp 'test -d /workspace/behavior-challenge-2026/datasets/task40/data'; then
  log "downloading task-40 dataset (chunk-040, rgb only)"
  sshp 'cd /workspace && /root/venv_b1k/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"behavior-1k/2026-challenge-demos\", repo_type=\"dataset\",
  allow_patterns=[\"meta/**\",\"data/chunk-040/*\",\"videos/observation.rgb.*/chunk-040/*\"],
  local_dir=\"/workspace/behavior-challenge-2026/datasets/task40\", max_workers=8)
print(\"dataset done\")"' >> "$LOG" 2>&1
fi
sshp 'du -sh /workspace/behavior-challenge-2026/datasets/task40 2>/dev/null; ls /workspace/behavior-challenge-2026/datasets/task40/data 2>/dev/null | head -3' | tee -a "$LOG"

# 5. assets: reuse the winner's norm stats + FAST tokenizer
log "installing assets"
sshp 'cd /workspace/behavior-1k-solution && A=outputs/assets/pi_behavior_b1k_2026_task40/b1k_2026_task40 && mkdir -p $A && cp -r /workspace/openpi_checkpoints/winner_larchenko/checkpoint_4/assets/IliaLarchenko/behavior_224_rgb/{norm_stats.json,fast_tokenizer} $A/ && ls $A' >> "$LOG" 2>&1

# 6. one-batch sanity check
log "sanity check (one batch)"
sshp 'cd /workspace/behavior-1k-solution && JAX_PLATFORMS=cpu /root/venv_b1k/bin/python -c "
import dataclasses, numpy as np
from b1k.training import config as c, data_loader as dl
cfg=dataclasses.replace(c.get_config(\"pi_behavior_b1k_2026_task40\"), batch_size=4, num_workers=0)
obs,act=next(iter(dl.create_behavior_data_loader(cfg, shuffle=True, num_batches=1)))
d=obs.to_dict(); print(\"tokenized_prompt\", np.asarray(d[\"tokenized_prompt\"]).tolist())
a=np.asarray(act); print(\"actions\", a.shape, \"max|norm|\", np.abs(a[:,:,:23]).max().round(2))"' 2>&1 | grep -vE "Warning|pynvml|Unknown fields|INFO" | tee -a "$LOG"

# 7. launch training + checkpoint syncer
log "launching training (tmux train40)"
sshp 'mkdir -p /root/logs /root/ckpt; cd /workspace && (CKPT_SWEEP=2000,4000,6000,8000,10000 setsid nohup python3 ckpt_sync.py /root/ckpt/pi_behavior_b1k_2026_task40/ft1 /workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task40/ft1 60 > /workspace/logs/ckpt_sync_task40.log 2>&1 < /dev/null &); tmux new -d -s train40 "cd /workspace/behavior-1k-solution && B1K_PARAMS_ONLY_CHECKPOINTS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 /root/venv_b1k/bin/python -u scripts/train.py pi_behavior_b1k_2026_task40 --exp_name=ft1 --overwrite --checkpoint_base_dir=/root/ckpt --batch_size=16 --num_train_steps=10000 --save_interval=500 --keep_period=2000 --log_interval=25 --num_workers=16 2>&1 | tee -a /root/logs/train_task40_ft1.log"; sleep 3; tmux ls' | tee -a "$LOG"
log "bootstrap complete"
