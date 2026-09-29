#!/usr/bin/env bash
# Zero-shot check: the 2025 winner's task-40 specialist (checkpoint_4) on make_microwave_popcorn
# in the 2026 harness, all 20 public instances. Runs after the task-56 job frees the GPU.
set -u
LOG=~/winner-eval/task40_zeroshot.log
CK=/media/rena/Elements/openpi_checkpoints/winner_larchenko
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
# wait for the task-56 job and for the checkpoint download
while pgrep -f "run_task56_rest.s[h]" >/dev/null; do sleep 120; done
until grep -q DONE "$CK/download_ckpt4.log" 2>/dev/null; do log "waiting for checkpoint_4 download"; sleep 120; done
log "starting zero-shot eval: task 40 make_microwave_popcorn, winner checkpoint_4, instances 301-320"
POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task make_microwave_popcorn --task-id 40 \
  --config pi_behavior_b1k_fast --ckpt-root "$CK" --step checkpoint_4 --run winner0shot \
  --instances 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 >> "$LOG" 2>&1
log "first pass: $(tr -d '\n ' < ~/eval_outputs/make_microwave_popcorn_winner0shot_stepcheckpoint_4/summary.json 2>/dev/null)"
log "retry pass for any missing instances"
POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task make_microwave_popcorn --task-id 40 \
  --config pi_behavior_b1k_fast --ckpt-root "$CK" --step checkpoint_4 --run winner0shot \
  --instances 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 >> "$LOG" 2>&1
log "final: $(tr -d '\n ' < ~/eval_outputs/make_microwave_popcorn_winner0shot_stepcheckpoint_4/summary.json 2>/dev/null)"
