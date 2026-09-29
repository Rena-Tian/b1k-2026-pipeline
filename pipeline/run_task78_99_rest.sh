#!/usr/bin/env bash
# Finish the official 20-instance public_test sets for the other two tasks of the joint
# 56/78/99 fine-tune, ckpt 19999. Task 78 first (cheaper), then task 99.
# Waits until the GPU is free: no task-40 queue script and no eval process running.
set -u
LOG=~/winner-eval/task78_99_rest.log
CK=/media/rena/Elements/openpi_checkpoints/ft_3more
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

quiet=0
while [ "$quiet" -lt 2 ]; do
  if pgrep -f "bash \./run_task10_eval\.sh" >/dev/null || pgrep -f "/home/rena/winner-eval/run_ft_eval\.py" >/dev/null; then
    quiet=0
  else
    quiet=$((quiet+1))
  fi
  sleep 120
done
log "GPU free; starting task 78 instances 306-320"

run(){ # task, id, indices...
  local t=$1 i=$2; shift 2
  POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task "$t" --task-id "$i" \
    --config pi_behavior_b1k_2026_3more --ckpt-root "$CK" --step 19999 \
    --instances "$@" >> "$LOG" 2>&1
}

run make_cabinet_doors 78 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
log "task 78 first pass: $(tr -d '\n ' < ~/eval_outputs/make_cabinet_doors_ft1_step19999/summary.json 2>/dev/null)"
run make_cabinet_doors 78 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
log "task 78 final: $(tr -d '\n ' < ~/eval_outputs/make_cabinet_doors_ft1_step19999/summary.json 2>/dev/null)"

log "starting task 99 instances 306-320"
run sorting_books_on_shelf 99 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
log "task 99 first pass: $(tr -d '\n ' < ~/eval_outputs/sorting_books_on_shelf_ft1_step19999/summary.json 2>/dev/null)"
run sorting_books_on_shelf 99 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
log "task 99 final: $(tr -d '\n ' < ~/eval_outputs/sorting_books_on_shelf_ft1_step19999/summary.json 2>/dev/null)"
