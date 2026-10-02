#!/usr/bin/env bash
# Task 12 LEADERBOARD RUN ONLY (user, 2026-09-30 13:00: "stop 315, start eval with 9999").
# Screen on test 311-315: 8000 = 0,.333,.333,.333,0 (1.0/5); 9999 = .167,.333,.167,.333 (1.0/4, 315 skipped).
# Task 12 (preparing_lunch_box) eval — RICHARD's task-12 checkpoints (copied from his pod to Elements ft_task12) — 2026 leaderboard method (runbook, 2026-09-29):
#   Phase A: choose the checkpoint on TEST instances 10-14 (ids 311-315): screen 8000, 9999 (plan B, 2026-09-29).
#   Phase B: run the LEADERBOARD instances 0-9 (ids 301-310) once on the best; score = sum / 10.
# Ties in the screen go to the later step (task 99's retrain improved with every checkpoint).
# SCREEN RULE (user, 2026-09-29): if a checkpoint's first 3 screen rollouts (311-313) all score 0, stop
# screening it and move on to the next checkpoint. Checkpoints are ranked by MEAN q over their measured
# screen instances (3 or 5), so a checkpoint cut at 3 is compared fairly.
# Only crashed (no-result) instances are retried; a finished rollout is never re-run.
# Output tag ft12 -> ~/eval_outputs/preparing_lunch_box_ft12_step<N>/
set -u
LOG=~/winner-eval/task12_sweep.log
CK=/media/rena/Elements/openpi_checkpoints/ft_task12
STREAMLOG=~/winner-eval/fetch_task12.log
OUT=~/eval_outputs
T=preparing_lunch_box
TAG=ft12
STEPS="${STEPS:-8000 9999}"
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
    POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task $T --task-id 12 \
      --config pi_behavior_b1k_2026_task12 --ckpt-root "$CK" --step "$s" --run $TAG \
      --instances $idx >> "$LOG" 2>&1
    log "step $s [$idx] after pass $pass: $(score $s "$idx")"
  done
}

BEST=9999
while pgrep -f "/home/rena/winner-eval/run_ft_eval\.py" >/dev/null; do sleep 30; done
log "=== task 12 LEADERBOARD run: step $BEST on instances 301-310 ==="
eval_set $BEST "$REPORT" 10
S=$(score $BEST "$REPORT"); N=${S%% *}; Q=${S##* }
if [ "$N" -ge 10 ]; then
  log "FINAL task 12 step $BEST (leaderboard 301-310): sum $Q over 10 -> q = $(python3 -c "print('%.4f'%($Q/10))")"
else
  log "INCOMPLETE task 12 step $BEST: only $N/10 leaderboard instances measured (sum $Q) — not a score"
fi
