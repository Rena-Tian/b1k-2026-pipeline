#!/usr/bin/env bash
# After the main queue finishes: evaluate the final (9999) checkpoint on all 10 instances, then retry
# every instance the sweep lost (GPU OOM) for steps 4000/6000/8000. Runner skips finished instances.
set -u
POD="root@38.128.232.232"; PORT=23942
VOL=/workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task89/ft1
LOCAL=/media/rena/Elements/openpi_checkpoints/ft_task89
LOG=~/winner-eval/eval_queue_task89.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
# wait for the main queue to finish
while pgrep -f "eval_queue_task89.s[h]" >/dev/null; do sleep 120; done
log "main queue done; fetching final checkpoint 9999"
if [ ! -f "$LOCAL/9999/params/manifest.ocdbt" ]; then
  until rsync -rlt --partial --info=progress2 -e "ssh -p $PORT" "$POD:$VOL/9999/" "$LOCAL/9999/" >> "$LOG" 2>&1; do log "rsync failed, retrying in 60s"; sleep 60; done
else log "9999 already on the Elements drive"; fi
log "evaluating step 9999 on instances 0-9"
POLICY_DEVICE=gpu python3 ~/winner-eval/run_task89_ft_eval.py --step 9999 --instances 0 1 2 3 4 5 6 7 8 9 >> "$LOG" 2>&1
log "step 9999 result: $(tr -d '\n ' < ~/eval_outputs/installing_a_fax_machine_ft1_step9999/summary.json 2>/dev/null)"
for STEP in 4000 6000 8000; do
  log "retry pass for step $STEP (missing instances only)"
  POLICY_DEVICE=gpu python3 ~/winner-eval/run_task89_ft_eval.py --step $STEP --instances 0 1 2 3 4 >> "$LOG" 2>&1
  log "step $STEP result: $(tr -d '\n ' < ~/eval_outputs/installing_a_fax_machine_ft1_step${STEP}/summary.json 2>/dev/null)"
done
log "final evals finished"
