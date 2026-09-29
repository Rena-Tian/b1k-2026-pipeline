#!/usr/bin/env bash
# Pull each fine-tune checkpoint from the pod as it appears and evaluate it locally, in order.
# Steps 500..8000: 5 instances (301-305) for speed (~45 min/instance for this task); final 10000: all 10.
set -u
POD="root@38.128.232.232"; PORT=23942
VOL=/workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task89/ft1
LOCAL=/media/rena/Elements/openpi_checkpoints/ft_task89
LOG=~/winner-eval/eval_queue_task89.log
mkdir -p "$LOCAL"
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
for STEP in 500 2000 4000 6000 8000 10000; do
  if [ -f ~/eval_outputs/installing_a_fax_machine_ft1_step${STEP}/status.json ] && grep -q '"completed"' ~/eval_outputs/installing_a_fax_machine_ft1_step${STEP}/status.json; then
    log "step $STEP already evaluated, skipping"; continue; fi
  log "waiting for step $STEP on the pod volume"
  until ssh -o ConnectTimeout=20 -p $PORT $POD "test -d $VOL/$STEP/params || test -d /root/ckpt/pi_behavior_b1k_2026_task89/ft1/$STEP/params" 2>/dev/null; do
    if ssh -o ConnectTimeout=20 -p $PORT $POD "tmux has-session -t train89" 2>/dev/null; then sleep 300; else
      # training over: if the step never appears, move on
      ssh -o ConnectTimeout=20 -p $PORT $POD "test -d $VOL/$STEP/params || test -d /root/ckpt/pi_behavior_b1k_2026_task89/ft1/$STEP/params" 2>/dev/null || { log "training ended and step $STEP absent; skipping"; continue 2; }
    fi
  done
  log "step $STEP available; rsync to $LOCAL/$STEP"
  SRC="$VOL/$STEP"; ssh -o ConnectTimeout=20 -p $PORT $POD "test -d $VOL/$STEP/params" 2>/dev/null || SRC="/root/ckpt/pi_behavior_b1k_2026_task89/ft1/$STEP"
  until rsync -rlt --partial --info=progress2 -e "ssh -p $PORT" "$POD:$SRC/" "$LOCAL/$STEP/" >> "$LOG" 2>&1; do log "rsync failed, retrying in 60s"; sleep 60; done
  INST="0 1 2 3 4"; [ "$STEP" = 10000 ] && INST="0 1 2 3 4 5 6 7 8 9"
  log "evaluating step $STEP on instances $INST"
  python3 ~/winner-eval/run_task89_ft_eval.py --step $STEP --instances $INST >> "$LOG" 2>&1 || log "eval of step $STEP FAILED (see log)"
  log "step $STEP result: $(cat ~/eval_outputs/installing_a_fax_machine_ft1_step${STEP}/summary.json 2>/dev/null | tr -d '\n ')"
done
log "queue finished"
