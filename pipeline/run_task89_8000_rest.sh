#!/usr/bin/env bash
# Finish the official 20-instance public_test set for task 89, checkpoint 8000.
# Instances 301-305 (indices 0-4) are already done; this runs indices 5-19 -> ids 306-320.
set -u
LOG=~/winner-eval/task89_8000_rest.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "starting instances 306-320 (indices 5-19) for task 89 ckpt 8000"
POLICY_DEVICE=gpu python3 ~/winner-eval/run_task89_ft_eval.py --step 8000 --instances 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 >> "$LOG" 2>&1
log "done: $(tr -d '\n ' < ~/eval_outputs/installing_a_fax_machine_ft1_step8000/summary.json 2>/dev/null)"
# second pass for anything lost to GPU OOM
log "retry pass for missing instances"
POLICY_DEVICE=gpu python3 ~/winner-eval/run_task89_ft_eval.py --step 8000 --instances 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 >> "$LOG" 2>&1
log "final: $(tr -d '\n ' < ~/eval_outputs/installing_a_fax_machine_ft1_step8000/summary.json 2>/dev/null)"
