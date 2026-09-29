#!/usr/bin/env bash
# Clean the pod disks after the task-46 checkpoint fetch, then launch the task-10 fine-tune.
#
# Task 10 set_up_a_coffee_station_in_your_kitchen, init = winner checkpoint_1.
# Chosen from our own eval evidence, not the winner's (unpublished) per-task table:
#   * 6 goal predicates, all nextto/ontop placements -> partial credit available (the task-56
#     profile that gave our best score, 0.263). Single-predicate tasks (34, 78) score 0.0 even
#     for the 2025 winner.
#   * 6266-step human demo horizon: the shortest of any untried multi-predicate task. Every task
#     we have scored is under 4600; task 99 at 7739 scores 0.000.
set -u
POD="root@185.216.21.214"; PORT=35233
LOG=~/winner-eval/bootstrap_task10.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
sshp(){ ssh -o ConnectTimeout=25 -o BatchMode=yes -p $PORT $POD "$@" </dev/null; }

# never delete a checkpoint before the fetch has verified every copy
while pgrep -f "fetch_task46_ckpts.s[h]" >/dev/null; do sleep 60; done
log "fetch finished: $(ls /media/rena/Elements/openpi_checkpoints/ft_task46 | tr '\n' ' ')"

MISSING=0
for S in 9999 8000 6000 4000 2000; do
  [ -f "/media/rena/Elements/openpi_checkpoints/ft_task46/$S/params/manifest.ocdbt" ] || { log "MISSING task46 step $S locally"; MISSING=1; }
done
if [ "$MISSING" = 1 ]; then log "ABORT: not all task-46 checkpoints are on Elements; nothing deleted"; exit 1; fi

log "=== deleting pod copies (all verified byte-for-byte on Elements) ==="
log "before: volume $(sshp 'du -sh /workspace | cut -f1'), local disk $(sshp 'du -sh /root/ckpt 2>/dev/null | cut -f1')"
# 12G container-local task-46 checkpoint (already fetched + verified)
sshp 'rm -rf /root/ckpt/pi_behavior_b1k_2026_task46'
# task-46 volume checkpoints (already fetched + verified)
sshp 'rm -rf /workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task46'
# 12G duplicate of ft_3more/19999 which is on Elements
sshp 'rm -rf /workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_3more'
# 36G dataset for the finished 56/78/99 run
sshp 'rm -rf /workspace/behavior-challenge-2026/datasets/3more_tasks'
# 12G winner checkpoint_4: task 40 is trained and the checkpoint is on Elements
sshp 'rm -rf /workspace/openpi_checkpoints/winner_larchenko/checkpoint_4'
log "after:  volume $(sshp 'du -sh /workspace | cut -f1'), local disk $(sshp 'du -sh /root/ckpt 2>/dev/null | cut -f1')"

log "=== task-10 setup ==="
log "syncing repo (task-10 config)"
rsync -rlt --exclude '.venv' --exclude 'outputs' --exclude '__pycache__' --exclude '*.egg-info' -e "ssh -p $PORT" ~/behavior-1k-solution/ "$POD:/workspace/behavior-1k-solution/" >> "$LOG" 2>&1

if ! sshp 'test -f /workspace/openpi_checkpoints/winner_larchenko/checkpoint_1/params/manifest.ocdbt'; then
  log "downloading winner checkpoint_1 (covers task 10)"
  sshp 'cd /workspace && /root/venv_b1k/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"IliaLarchenko/behavior_submission\", allow_patterns=[\"checkpoint_1/*\",\"checkpoint_1/**\"], local_dir=\"/workspace/openpi_checkpoints/winner_larchenko\", max_workers=8)
print(\"ckpt1 done\")"' >> "$LOG" 2>&1
fi

if ! sshp 'test -d /workspace/behavior-challenge-2026/datasets/task10/data'; then
  log "downloading task-10 dataset (chunk-010, rgb only)"
  sshp 'cd /workspace && /root/venv_b1k/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"behavior-1k/2026-challenge-demos\", repo_type=\"dataset\",
  allow_patterns=[\"meta/**\",\"data/chunk-010/*\",\"videos/observation.rgb.*/chunk-010/*\"],
  local_dir=\"/workspace/behavior-challenge-2026/datasets/task10\", max_workers=8)
print(\"dataset done\")"' >> "$LOG" 2>&1
fi

# LeRobot refuses a dataset whose info.json lists video streams we did not download (depth/seg)
log "pruning non-rgb video keys from meta/info.json"
sshp 'D=/workspace/behavior-challenge-2026/datasets/task10; cp -n $D/meta/info.json $D/meta/info.upstream.json 2>/dev/null; /root/venv_b1k/bin/python -c "
import json; p=\"$D/meta/info.json\"; d=json.load(open(p))
keep={k:v for k,v in d[\"features\"].items() if not (v.get(\"dtype\")==\"video\" and not k.startswith(\"observation.rgb.\"))}
d[\"features\"]=keep; json.dump(d, open(p,\"w\"), indent=4); print(\"video keys:\", [k for k,v in keep.items() if v.get(\"dtype\")==\"video\"])"' >> "$LOG" 2>&1

log "installing assets"
sshp 'cd /workspace/behavior-1k-solution && A=outputs/assets/pi_behavior_b1k_2026_task10/b1k_2026_task10 && mkdir -p $A && cp -r /workspace/openpi_checkpoints/winner_larchenko/checkpoint_1/assets/IliaLarchenko/behavior_224_rgb/{norm_stats.json,fast_tokenizer} $A/ && ls $A' >> "$LOG" 2>&1

log "sanity check (one batch)"
sshp 'cd /workspace/behavior-1k-solution && JAX_PLATFORMS=cpu /root/venv_b1k/bin/python -c "
import dataclasses, numpy as np
from b1k.training import config as c, data_loader as dl
cfg=dataclasses.replace(c.get_config(\"pi_behavior_b1k_2026_task10\"), batch_size=4, num_workers=0)
obs,act=next(iter(dl.create_behavior_data_loader(cfg, shuffle=True, num_batches=1)))
d=obs.to_dict(); print(\"tokenized_prompt\", np.asarray(d[\"tokenized_prompt\"]).tolist())
a=np.asarray(act); print(\"actions\", a.shape, \"max|norm|\", float(np.abs(a[:,:,:23]).max()))"' 2>&1 | grep -vE "Warning|pynvml|Unknown fields|INFO" | tee -a "$LOG"

# NOTE: CKPT_SWEEP uses 9999, not 10000 — training ends at step 9999, and on the task-46 run the
# final checkpoint was never synced to the volume because the sweep listed 10000.
log "launching training (tmux train10)"
sshp 'mkdir -p /root/logs /root/ckpt; cp /workspace/.netrc_wandb ~/.netrc 2>/dev/null; chmod 600 ~/.netrc 2>/dev/null; cd /workspace && (CKPT_SWEEP=2000,4000,6000,8000,9999 setsid nohup python3 ckpt_sync.py /root/ckpt/pi_behavior_b1k_2026_task10/ft1 /workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task10/ft1 60 > /workspace/logs/ckpt_sync_task10.log 2>&1 < /dev/null &); tmux new -d -s train10 "cd /workspace/behavior-1k-solution && B1K_PARAMS_ONLY_CHECKPOINTS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 /root/venv_b1k/bin/python -u scripts/train.py pi_behavior_b1k_2026_task10 --exp_name=ft1 --overwrite --checkpoint_base_dir=/root/ckpt --batch_size=16 --num_train_steps=10000 --save_interval=500 --keep_period=2000 --log_interval=25 --num_workers=16 2>&1 | tee -a /root/logs/train_task10_ft1.log"; sleep 3; tmux ls' | tee -a "$LOG"
log "bootstrap complete"
