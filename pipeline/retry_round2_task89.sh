#!/usr/bin/env bash
# Second retry round for instances lost to laptop-GPU OOM; runs after final_evals_task89.sh ends.
set -u
LOG=~/winner-eval/eval_queue_task89.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
while pgrep -f "final_evals_task89.s[h]" >/dev/null; do sleep 60; done
for STEP in 4000 6000 8000 9999; do
  log "retry round 2 for step $STEP (missing instances only)"
  INST="0 1 2 3 4"; [ "$STEP" = 9999 ] && INST="0 1 2 3 4 5 6 7 8 9"
  POLICY_DEVICE=gpu python3 ~/winner-eval/run_task89_ft_eval.py --step $STEP --instances $INST >> "$LOG" 2>&1
  log "step $STEP result: $(tr -d '\n ' < ~/eval_outputs/installing_a_fax_machine_ft1_step${STEP}/summary.json 2>/dev/null)"
done
log "retry round 2 finished"
