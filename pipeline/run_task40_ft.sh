#!/usr/bin/env bash
# Fine-tuned task 40 (make_microwave_popcorn), ckpt 9999, official 20-instance public_test set.
# Zero-shot winner baseline skipped per user request. Runs after the task-56 job frees the GPU.
set -u
LOG=~/winner-eval/task40_ft.log
CK=/media/rena/Elements/openpi_checkpoints/ft_task40
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
while pgrep -f "run_task56_rest.s[h]" >/dev/null || pgrep -f "run_ft_eval\.p[y]" >/dev/null; do sleep 60; done
log "GPU free; starting task 40 fine-tune eval, step 9999, instances 301-320"
ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task make_microwave_popcorn --task-id 40 \
  --config pi_behavior_b1k_2026_task40 --ckpt-root "$CK" --step 9999 --run ft1 \
  --instances $ALL >> "$LOG" 2>&1
log "first pass: $(tr -d '\n ' < ~/eval_outputs/make_microwave_popcorn_ft1_step9999/summary.json 2>/dev/null)"
log "retry pass for any missing instances"
POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task make_microwave_popcorn --task-id 40 \
  --config pi_behavior_b1k_2026_task40 --ckpt-root "$CK" --step 9999 --run ft1 \
  --instances $ALL >> "$LOG" 2>&1
log "final: $(tr -d '\n ' < ~/eval_outputs/make_microwave_popcorn_ft1_step9999/summary.json 2>/dev/null)"
