#!/usr/bin/env bash
# Launch the task-42 (chop_an_onion) fine-tune. The pod GPU is idle after task 10 finished.
#
# Task choice, using the two predictors that match our own results:
#   * 6400-step human demo horizon — shortest untried multi-predicate task after task 10.
#   * 4 goal predicates, two of which (parer in sink, chopping board in sink) are plain placements
#     that score partial credit even if the dicing itself never succeeds.
# Init = winner checkpoint_1, already on the pod from the task-10 bootstrap, so no 12 GB download.
set -u
POD="root@185.216.21.214"; PORT=35233
LOG=~/winner-eval/bootstrap_task42.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
sshp(){ ssh -o ConnectTimeout=25 -o BatchMode=yes -p $PORT $POD "$@" </dev/null; }

log "=== task-42 setup (volume before: $(sshp 'du -sh /workspace | cut -f1')) ==="
log "syncing repo (task-42 config)"
rsync -rlt --exclude '.venv' --exclude 'outputs' --exclude '__pycache__' --exclude '*.egg-info' -e "ssh -p $PORT" ~/behavior-1k-solution/ "$POD:/workspace/behavior-1k-solution/" >> "$LOG" 2>&1

if ! sshp 'test -f /workspace/openpi_checkpoints/winner_larchenko/checkpoint_1/params/manifest.ocdbt'; then
  log "checkpoint_1 missing, downloading"
  sshp 'cd /workspace && /root/venv_b1k/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"IliaLarchenko/behavior_submission\", allow_patterns=[\"checkpoint_1/*\",\"checkpoint_1/**\"], local_dir=\"/workspace/openpi_checkpoints/winner_larchenko\", max_workers=8)
print(\"ckpt1 done\")"' >> "$LOG" 2>&1
else
  log "checkpoint_1 already on the pod (kept from task 10)"
fi

if ! sshp 'test -d /workspace/behavior-challenge-2026/datasets/task42/data'; then
  log "downloading task-42 dataset (chunk-042, rgb only)"
  sshp 'cd /workspace && /root/venv_b1k/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"behavior-1k/2026-challenge-demos\", repo_type=\"dataset\",
  allow_patterns=[\"meta/**\",\"data/chunk-042/*\",\"videos/observation.rgb.*/chunk-042/*\"],
  local_dir=\"/workspace/behavior-challenge-2026/datasets/task42\", max_workers=8)
print(\"dataset done\")"' >> "$LOG" 2>&1
fi

# LeRobot refuses a dataset whose info.json lists video streams we did not download (depth/seg)
log "pruning non-rgb video keys from meta/info.json"
sshp 'D=/workspace/behavior-challenge-2026/datasets/task42; cp -n $D/meta/info.json $D/meta/info.upstream.json 2>/dev/null; /root/venv_b1k/bin/python -c "
import json; p=\"$D/meta/info.json\"; d=json.load(open(p))
keep={k:v for k,v in d[\"features\"].items() if not (v.get(\"dtype\")==\"video\" and not k.startswith(\"observation.rgb.\"))}
d[\"features\"]=keep; json.dump(d, open(p,\"w\"), indent=4); print(\"video keys:\", [k for k,v in keep.items() if v.get(\"dtype\")==\"video\"])"' >> "$LOG" 2>&1

log "installing assets"
sshp 'cd /workspace/behavior-1k-solution && A=outputs/assets/pi_behavior_b1k_2026_task42/b1k_2026_task42 && mkdir -p $A && cp -r /workspace/openpi_checkpoints/winner_larchenko/checkpoint_1/assets/IliaLarchenko/behavior_224_rgb/{norm_stats.json,fast_tokenizer} $A/ && ls $A' >> "$LOG" 2>&1

log "sanity check (one batch)"
sshp 'cd /workspace/behavior-1k-solution && JAX_PLATFORMS=cpu /root/venv_b1k/bin/python -c "
import dataclasses, numpy as np
from b1k.training import config as c, data_loader as dl
cfg=dataclasses.replace(c.get_config(\"pi_behavior_b1k_2026_task42\"), batch_size=4, num_workers=0)
obs,act=next(iter(dl.create_behavior_data_loader(cfg, shuffle=True, num_batches=1)))
d=obs.to_dict(); print(\"tokenized_prompt\", np.asarray(d[\"tokenized_prompt\"]).tolist())
a=np.asarray(act); print(\"actions\", a.shape, \"max|norm|\", float(np.abs(a[:,:,:23]).max()))"' 2>&1 | grep -vE "Warning|pynvml|Unknown fields|INFO" | tee -a "$LOG"

# CKPT_SWEEP ends at 9999 (= num_train_steps - 1), never 10000.
log "launching training (tmux train42)"
sshp 'mkdir -p /root/logs /root/ckpt; cp /workspace/.netrc_wandb ~/.netrc 2>/dev/null; chmod 600 ~/.netrc 2>/dev/null; cd /workspace && (CKPT_SWEEP=2000,4000,6000,8000,9999 setsid nohup python3 ckpt_sync.py /root/ckpt/pi_behavior_b1k_2026_task42/ft1 /workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task42/ft1 60 > /workspace/logs/ckpt_sync_task42.log 2>&1 < /dev/null &); tmux new -d -s train42 "cd /workspace/behavior-1k-solution && B1K_PARAMS_ONLY_CHECKPOINTS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 /root/venv_b1k/bin/python -u scripts/train.py pi_behavior_b1k_2026_task42 --exp_name=ft1 --overwrite --checkpoint_base_dir=/root/ckpt --batch_size=16 --num_train_steps=10000 --save_interval=500 --keep_period=2000 --log_interval=25 --num_workers=16 2>&1 | tee -a /root/logs/train_task42_ft1.log"; sleep 3; tmux ls' | tee -a "$LOG"
log "bootstrap complete"
