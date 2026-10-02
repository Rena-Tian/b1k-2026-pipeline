#!/usr/bin/env bash
# Task 67 LEADERBOARD RUN ONLY (user, 2026-09-29 17:50: "go straight to the leaderboard run with 9999").
# Screen result that picked it: 6000 = 0/3, 8000 = 0/3 (skipped by the 3-zeros rule), 9999 = 0.167 on 311.
# Original sweep notes:
#   Phase A: choose the checkpoint on TEST instances 10-14 (ids 311-315): screen 6000, 8000, 9999.
#   Phase B: run the LEADERBOARD instances 0-9 (ids 301-310) once on the best; score = sum / 10.
# Ties in the screen go to the later step (task 99's retrain improved with every checkpoint).
# SCREEN RULE (user, 2026-09-29): if a checkpoint's first 3 screen rollouts (311-313) all score 0, stop
# screening it and move on to the next checkpoint. Checkpoints are ranked by MEAN q over their measured
# screen instances (3 or 5), so a checkpoint cut at 3 is compared fairly.
# Only crashed (no-result) instances are retried; a finished rollout is never re-run.
# Output tag ft67 -> ~/eval_outputs/boxing_food_after_dinner_ft67_step<N>/
set -u
LOG=~/winner-eval/task67_sweep.log
CK=/media/rena/Elements/openpi_checkpoints/ft_task67
STREAMLOG=~/winner-eval/stream_ft_task67.log
OUT=~/eval_outputs
T=boxing_food_after_dinner
TAG=ft67
STEPS="${STEPS:-6000 8000 9999}"
SCREEN="10 11 12 13 14"
FIRST3="10 11 12"
REPORT="0 1 2 3 4 5 6 7 8 9"
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# count/sum over a given set of instance ids for one step
score(){ python3 -c "
import json,glob
ids=set(int(x)+301 for x in '$2'.split())
qs=[d['q_score']['final'] for d in (json.load(open(f)) for f in glob.glob('$OUT/${T}_${TAG}_step$1/json/*.json')) if d['instance_id'] in ids]
print('%d %.4f'%(len(qs),sum(qs)))"; }
ready(){ grep -a -q "step $1: verified" "$STREAMLOG" && [ -f "$CK/$1/params/manifest.ocdbt" ]; }
eval_set(){ # step "indices" target_count
  local s=$1 idx=$2 n=$3
  for pass in 1 2 3 4 5 6 7 8; do
    [ "$(score $s "$idx" | cut -d' ' -f1)" -ge "$n" ] && break
    until [ -f "$CK/$s/params/manifest.ocdbt" ]; do log "checkpoint $s not visible (drive unmounted?), waiting"; sleep 120; done
    POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task $T --task-id 67 \
      --config pi_behavior_b1k_2026_task67 --ckpt-root "$CK" --step "$s" --run $TAG \
      --instances $idx >> "$LOG" 2>&1
    log "step $s [$idx] after pass $pass: $(score $s "$idx")"
  done
}

BEST=9999
while pgrep -f "/home/rena/winner-eval/run_ft_eval\.py" >/dev/null; do sleep 60; done
log "=== task 67 LEADERBOARD run: step $BEST on instances 301-310 (user chose to skip the rest of the screen) ==="
eval_set $BEST "$REPORT" 10
S=$(score $BEST "$REPORT"); N=${S%% *}; Q=${S##* }
if [ "$N" -ge 10 ]; then
  log "FINAL task 67 step $BEST (leaderboard 301-310): sum $Q over 10 -> q = $(python3 -c "print('%.4f'%($Q/10))")"
else
  log "INCOMPLETE task 67 step $BEST: only $N/10 leaderboard instances measured (sum $Q) — not a score"
fi
