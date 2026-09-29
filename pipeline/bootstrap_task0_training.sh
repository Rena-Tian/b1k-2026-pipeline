#!/usr/bin/env bash
# Queue the task-0 (turning_on_radio) fine-tune. Waits until stream_ckpts.sh has pulled ALL of
# task 42's checkpoints to Elements and cleared the pod, so the two runs never overlap on the GPU.
#
# Why task 0, using the predictors that match our own results:
#   * 2150-step human demo horizon — the SHORTEST of all 100 tasks, two thirds shorter than task 40
#     (3238) which is our best score at 0.350. Horizon is the strongest predictor we have: every
#     task we score on is under 4600, and 7680+ (tasks 46, 99) has always returned 0.000.
#   * Goal is a single toggled_on(radio). Single-predicate goals are usually a warning sign (34, 78
#     with `attached` score 0.0 even for the 2025 winner), but task 89 needed ontop + toggled_on and
#     solved 4 of 20 outright, so this predicate is demonstrably reachable.
#   * The scene contains ONE radio and no look-alike appliance. That matters: task 46 failed partly
#     because the robot opened the oven instead of the stove, and this architecture has no language
#     conditioning to disambiguate similar objects — only a task-ID embedding.
# Init = winner checkpoint_2, which is not yet on the pod (~12 GB download).
set -u
POD="root@185.216.21.214"; PORT=35233
LOG=~/winner-eval/bootstrap_task0.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
sshp(){ ssh -o ConnectTimeout=25 -o BatchMode=yes -p $PORT $POD "$@" </dev/null; }

# 1) wait for task-42 training to end and all five of its checkpoints to reach Elements
while pgrep -f "bash \./stream_ckpts\.sh pi_behavior_b1k_2026_task42" >/dev/null; do sleep 120; done
MISSING=0
for S in 2000 4000 6000 8000 9999; do
  [ -f "/media/rena/Elements/openpi_checkpoints/ft_task42/$S/params/manifest.ocdbt" ] || { log "task42 step $S NOT on Elements"; MISSING=1; }
done
if [ "$MISSING" = 1 ]; then log "ABORT: task-42 checkpoints incomplete; not starting task 0"; exit 1; fi
log "task 42 complete: $(ls /media/rena/Elements/openpi_checkpoints/ft_task42 | tr '\n' ' ')"

log "=== task-0 setup (volume: $(sshp 'du -sh /workspace | cut -f1')) ==="
log "syncing repo (task-0 config)"
rsync -rlt --exclude '.venv' --exclude 'outputs' --exclude '__pycache__' --exclude '*.egg-info' -e "ssh -p $PORT" ~/behavior-1k-solution/ "$POD:/workspace/behavior-1k-solution/" >> "$LOG" 2>&1

if ! sshp 'test -f /workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/params/manifest.ocdbt'; then
  log "downloading winner checkpoint_2 (covers task 0)"
  sshp 'cd /workspace && /root/venv_b1k/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"IliaLarchenko/behavior_submission\", allow_patterns=[\"checkpoint_2/*\",\"checkpoint_2/**\"], local_dir=\"/workspace/openpi_checkpoints/winner_larchenko\", max_workers=8)
print(\"ckpt2 done\")"' >> "$LOG" 2>&1
fi

if ! sshp 'test -d /workspace/behavior-challenge-2026/datasets/task0/data'; then
  log "downloading task-0 dataset (chunk-000, rgb only)"
  sshp 'cd /workspace && /root/venv_b1k/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"behavior-1k/2026-challenge-demos\", repo_type=\"dataset\",
  allow_patterns=[\"meta/**\",\"data/chunk-000/*\",\"videos/observation.rgb.*/chunk-000/*\"],
  local_dir=\"/workspace/behavior-challenge-2026/datasets/task0\", max_workers=8)
print(\"dataset done\")"' >> "$LOG" 2>&1
fi

# LeRobot refuses a dataset whose info.json lists video streams we did not download (depth/seg)
log "pruning non-rgb video keys from meta/info.json"
sshp 'D=/workspace/behavior-challenge-2026/datasets/task0; cp -n $D/meta/info.json $D/meta/info.upstream.json 2>/dev/null; /root/venv_b1k/bin/python -c "
import json; p=\"$D/meta/info.json\"; d=json.load(open(p))
keep={k:v for k,v in d[\"features\"].items() if not (v.get(\"dtype\")==\"video\" and not k.startswith(\"observation.rgb.\"))}
d[\"features\"]=keep; json.dump(d, open(p,\"w\"), indent=4); print(\"video keys:\", [k for k,v in keep.items() if v.get(\"dtype\")==\"video\"])"' >> "$LOG" 2>&1

log "installing assets"
sshp 'cd /workspace/behavior-1k-solution && A=outputs/assets/pi_behavior_b1k_2026_task0/b1k_2026_task0 && mkdir -p $A && cp -r /workspace/openpi_checkpoints/winner_larchenko/checkpoint_2/assets/IliaLarchenko/behavior_224_rgb/{norm_stats.json,fast_tokenizer} $A/ && ls $A' >> "$LOG" 2>&1

log "sanity check (one batch) — expect tokenized_prompt [[0, s], ...]"
sshp 'cd /workspace/behavior-1k-solution && JAX_PLATFORMS=cpu /root/venv_b1k/bin/python -c "
import dataclasses, numpy as np
from b1k.training import config as c, data_loader as dl
cfg=dataclasses.replace(c.get_config(\"pi_behavior_b1k_2026_task0\"), batch_size=4, num_workers=0)
obs,act=next(iter(dl.create_behavior_data_loader(cfg, shuffle=True, num_batches=1)))
d=obs.to_dict(); print(\"tokenized_prompt\", np.asarray(d[\"tokenized_prompt\"]).tolist())
a=np.asarray(act); print(\"actions\", a.shape, \"max|norm|\", float(np.abs(a[:,:,:23]).max()))"' 2>&1 | grep -vE "Warning|pynvml|Unknown fields|INFO" | tee -a "$LOG"

# exp_name stays ft1 (user preference); CKPT_SWEEP ends at 9999, never 10000.
log "launching training (tmux train0)"
sshp 'mkdir -p /root/logs /root/ckpt; cp /workspace/.netrc_wandb ~/.netrc 2>/dev/null; chmod 600 ~/.netrc 2>/dev/null; cd /workspace && (CKPT_SWEEP=2000,4000,6000,8000,9999 setsid nohup python3 ckpt_sync.py /root/ckpt/pi_behavior_b1k_2026_task0/ft1 /workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task0/ft1 60 > /workspace/logs/ckpt_sync_task0.log 2>&1 < /dev/null &); tmux new -d -s train0 "cd /workspace/behavior-1k-solution && B1K_PARAMS_ONLY_CHECKPOINTS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 /root/venv_b1k/bin/python -u scripts/train.py pi_behavior_b1k_2026_task0 --exp_name=ft1 --overwrite --checkpoint_base_dir=/root/ckpt --batch_size=16 --num_train_steps=10000 --save_interval=500 --keep_period=2000 --log_interval=25 --num_workers=16 2>&1 | tee -a /root/logs/train_task0_ft1.log"; sleep 3; tmux ls' | tee -a "$LOG"

log "arming the checkpoint streamer for task 0"
cd ~/winner-eval && setsid nohup bash ./stream_ckpts.sh pi_behavior_b1k_2026_task0 ft_task0 2000 4000 6000 8000 9999 >/dev/null 2>&1 < /dev/null &
log "bootstrap complete"
