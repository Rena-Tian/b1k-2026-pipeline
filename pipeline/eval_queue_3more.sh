#!/usr/bin/env bash
# Eval queue for the 56/78/99 fine-tune: pull sweep checkpoints from the pod volume, evaluate locally.
# Order chosen for eval cost (rose ~85 min/instance, cabinet ~50, books ~38):
#   8000  -> tasks 99, 56 (3 instances)      | 19999 -> tasks 99, 56, 78 (5 instances) | 16000 -> 99, 56 (3)
set -u
POD="root@38.128.232.232"; PORT=23942
VOL=/workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_3more/ft1
LOCALCK=/root/ckpt/pi_behavior_b1k_2026_3more/ft1
LOCAL=/media/rena/Elements/openpi_checkpoints/ft_3more
LOG=~/winner-eval/eval_queue_3more.log
CFG=pi_behavior_b1k_2026_3more
mkdir -p "$LOCAL"
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
run_task(){ # step task id instances...
  local STEP=$1 TASK=$2 TID=$3; shift 3
  log "evaluating step $STEP task $TASK ($TID) on instances $*"
  POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task $TASK --task-id $TID --config $CFG --ckpt-root "$LOCAL" --step $STEP --instances "$@" >> "$LOG" 2>&1 || log "eval $TASK step $STEP FAILED"
  log "step $STEP $TASK result: $(tr -d '\n ' < ~/eval_outputs/${TASK}_ft1_step${STEP}/summary.json 2>/dev/null)"
}
fetch(){ # step
  local STEP=$1
  [ -f "$LOCAL/$STEP/params/manifest.ocdbt" ] && return 0
  log "waiting for step $STEP on the pod"
  until ssh -o ConnectTimeout=20 -p $PORT $POD "test -d $VOL/$STEP/params || test -d $LOCALCK/$STEP/params" 2>/dev/null; do
    ssh -o ConnectTimeout=20 -p $PORT $POD "tmux has-session -t train3more" 2>/dev/null || { ssh -o ConnectTimeout=20 -p $PORT $POD "test -d $VOL/$STEP/params || test -d $LOCALCK/$STEP/params" 2>/dev/null || return 1; }
    sleep 300
  done
  local SRC="$VOL/$STEP"; ssh -o ConnectTimeout=20 -p $PORT $POD "test -d $VOL/$STEP/params" 2>/dev/null || SRC="$LOCALCK/$STEP"
  log "step $STEP available; rsync from $SRC"
  until rsync -rlt --partial --info=progress2 -e "ssh -p $PORT" "$POD:$SRC/" "$LOCAL/$STEP/" >> "$LOG" 2>&1; do log "rsync failed, retrying in 60s"; sleep 60; done
}
fetch 8000 && { run_task 8000 sorting_books_on_shelf 99 0 1 2; run_task 8000 make_rose_centerpieces 56 0 1 2; }
fetch 19999 && { run_task 19999 sorting_books_on_shelf 99 0 1 2 3 4; run_task 19999 make_rose_centerpieces 56 0 1 2 3 4; run_task 19999 make_cabinet_doors 78 0 1 2 3 4; }
fetch 16000 && { run_task 16000 sorting_books_on_shelf 99 0 1 2; run_task 16000 make_rose_centerpieces 56 0 1 2; }
log "3more queue finished"
