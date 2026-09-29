#!/usr/bin/env bash
# Finish the official 20-instance public_test set for task 56 (make_rose_centerpieces), ckpt 19999.
# Waits for the task-89 job to finish first (one GPU), then runs indices 5-19 -> ids 306-320,
# followed by a retry pass over all 20 for anything lost to GPU OOM.
set -u
LOG=~/winner-eval/task56_19999_rest.log
CK=/media/rena/Elements/openpi_checkpoints/ft_3more
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
while pgrep -f "run_task89_8000_rest.s[h]" >/dev/null; do sleep 120; done
log "task-89 job finished; starting task 56 instances 306-320"
POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task make_rose_centerpieces --task-id 56 \
  --config pi_behavior_b1k_2026_3more --ckpt-root "$CK" --step 19999 \
  --instances 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 >> "$LOG" 2>&1
log "first pass done: $(tr -d '\n ' < ~/eval_outputs/make_rose_centerpieces_ft1_step19999/summary.json 2>/dev/null)"
log "retry pass for missing instances"
POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task make_rose_centerpieces --task-id 56 \
  --config pi_behavior_b1k_2026_3more --ckpt-root "$CK" --step 19999 \
  --instances 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 >> "$LOG" 2>&1
log "final: $(tr -d '\n ' < ~/eval_outputs/make_rose_centerpieces_ft1_step19999/summary.json 2>/dev/null)"
