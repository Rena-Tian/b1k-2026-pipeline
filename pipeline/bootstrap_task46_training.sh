#!/usr/bin/env bash
# Set up and launch the task-46 (cook_bacon) fine-tune on the running pod.
set -u
POD="root@185.216.21.214"; PORT=35233
LOG=~/winner-eval/bootstrap_task46.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
sshp(){ ssh -o ConnectTimeout=25 -o BatchMode=yes -p $PORT $POD "$@" </dev/null; }

# wait for the task-40 checkpoint fetch so we do not compete for volume space / bandwidth
while pgrep -f "fetch_task40_ckpts.s[h]" >/dev/null; do sleep 120; done
log "task-40 fetch done; volume: $(sshp 'du -sh /workspace | cut -f1')"

log "syncing repo (task-46 config)"
rsync -rlt --exclude '.venv' --exclude 'outputs' --exclude '__pycache__' --exclude '*.egg-info' -e "ssh -p $PORT" ~/behavior-1k-solution/ "$POD:/workspace/behavior-1k-solution/" >> "$LOG" 2>&1

if ! sshp 'test -f /workspace/openpi_checkpoints/winner_larchenko/checkpoint_3/params/manifest.ocdbt'; then
  log "downloading winner checkpoint_3 (covers task 46)"
  sshp 'cd /workspace && /root/venv_b1k/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"IliaLarchenko/behavior_submission\", allow_patterns=[\"checkpoint_3/*\",\"checkpoint_3/**\"], local_dir=\"/workspace/openpi_checkpoints/winner_larchenko\", max_workers=8)
print(\"ckpt3 done\")"' >> "$LOG" 2>&1
fi

if ! sshp 'test -d /workspace/behavior-challenge-2026/datasets/task46/data'; then
  log "downloading task-46 dataset (chunk-046, rgb only)"
  sshp 'cd /workspace && /root/venv_b1k/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"behavior-1k/2026-challenge-demos\", repo_type=\"dataset\",
  allow_patterns=[\"meta/**\",\"data/chunk-046/*\",\"videos/observation.rgb.*/chunk-046/*\"],
  local_dir=\"/workspace/behavior-challenge-2026/datasets/task46\", max_workers=8)
print(\"dataset done\")"' >> "$LOG" 2>&1
fi
# LeRobot refuses a dataset whose info.json lists video streams we did not download (depth/seg)
log "pruning non-rgb video keys from meta/info.json"
sshp 'D=/workspace/behavior-challenge-2026/datasets/task46; cp -n $D/meta/info.json $D/meta/info.upstream.json 2>/dev/null; /root/venv_b1k/bin/python -c "
import json; p=\"$D/meta/info.json\"; d=json.load(open(p))
keep={k:v for k,v in d[\"features\"].items() if not (v.get(\"dtype\")==\"video\" and not k.startswith(\"observation.rgb.\"))}
d[\"features\"]=keep; json.dump(d, open(p,\"w\"), indent=4); print(\"video keys:\", [k for k,v in keep.items() if v.get(\"dtype\")==\"video\"])"' >> "$LOG" 2>&1

log "installing assets"
sshp 'cd /workspace/behavior-1k-solution && A=outputs/assets/pi_behavior_b1k_2026_task46/b1k_2026_task46 && mkdir -p $A && cp -r /workspace/openpi_checkpoints/winner_larchenko/checkpoint_3/assets/IliaLarchenko/behavior_224_rgb/{norm_stats.json,fast_tokenizer} $A/ && ls $A' >> "$LOG" 2>&1

log "sanity check (one batch)"
sshp 'cd /workspace/behavior-1k-solution && JAX_PLATFORMS=cpu /root/venv_b1k/bin/python -c "
import dataclasses, numpy as np
from b1k.training import config as c, data_loader as dl
cfg=dataclasses.replace(c.get_config(\"pi_behavior_b1k_2026_task46\"), batch_size=4, num_workers=0)
obs,act=next(iter(dl.create_behavior_data_loader(cfg, shuffle=True, num_batches=1)))
d=obs.to_dict(); print(\"tokenized_prompt\", np.asarray(d[\"tokenized_prompt\"]).tolist())
a=np.asarray(act); print(\"actions\", a.shape, \"max|norm|\", float(np.abs(a[:,:,:23]).max()))"' 2>&1 | grep -vE "Warning|pynvml|Unknown fields|INFO" | tee -a "$LOG"

log "launching training (tmux train46)"
sshp 'mkdir -p /root/logs /root/ckpt; cp /workspace/.netrc_wandb ~/.netrc 2>/dev/null; chmod 600 ~/.netrc 2>/dev/null; cd /workspace && (CKPT_SWEEP=2000,4000,6000,8000,10000 setsid nohup python3 ckpt_sync.py /root/ckpt/pi_behavior_b1k_2026_task46/ft1 /workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task46/ft1 60 > /workspace/logs/ckpt_sync_task46.log 2>&1 < /dev/null &); tmux new -d -s train46 "cd /workspace/behavior-1k-solution && B1K_PARAMS_ONLY_CHECKPOINTS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 /root/venv_b1k/bin/python -u scripts/train.py pi_behavior_b1k_2026_task46 --exp_name=ft1 --overwrite --checkpoint_base_dir=/root/ckpt --batch_size=16 --num_train_steps=10000 --save_interval=500 --keep_period=2000 --log_interval=25 --num_workers=16 2>&1 | tee -a /root/logs/train_task46_ft1.log"; sleep 3; tmux ls' | tee -a "$LOG"
log "bootstrap complete"
