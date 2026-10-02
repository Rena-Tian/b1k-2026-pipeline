#!/usr/bin/env bash
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

# queued behind task 67: wait for its leaderboard script AND any eval process to be gone
while pgrep -f "bash \./run_task67_leaderboard\.sh" >/dev/null || pgrep -f "/home/rena/winner-eval/run_ft_eval\.py" >/dev/null; do sleep 60; done
log "task 67 done; GPU free"
log "=== task 12: screening $STEPS on test instances 311-315, then leaderboard 301-310 on the best ==="
for s in $STEPS; do
  until ready $s; do log "waiting for verified checkpoint $s"; sleep 300; done
  log "--- screening step $s (test 311-315) ---"
  eval_set $s "$FIRST3" 3
  F=$(score $s "$FIRST3")
  if [ "${F%% *}" -ge 3 ] && python3 -c "import sys; sys.exit(0 if float('${F##* }') == 0 else 1)"; then
    log "SCREEN RULE: step $s scored 0 on all of 311-313 -> skipping 314-315, moving to the next checkpoint"
  else
    eval_set $s "$SCREEN" 5
  fi
  log "SCREEN step $s: $(score $s "$SCREEN")"
done

# rank by mean q over the measured screen instances; ties -> later step
BEST=$(for s in $STEPS; do set -- $(score $s "$SCREEN"); python3 -c "print('$s', ($2/$1) if $1 else -1)"; done | sort -k2,2g -k1,1n | tail -1 | cut -d' ' -f1)
log "=== screen (test 311-315): $(for s in $STEPS; do echo -n "$s=[$(score $s "$SCREEN")] "; done) -> best step $BEST ==="
log "--- leaderboard instances 301-310 on step $BEST ---"
eval_set $BEST "$REPORT" 10
S=$(score $BEST "$REPORT"); N=${S%% *}; Q=${S##* }
if [ "$N" -ge 10 ]; then
  log "FINAL task 12 step $BEST (leaderboard 301-310): sum $Q over 10 -> q = $(python3 -c "print('%.4f'%($Q/10))")"
else
  log "INCOMPLETE task 12 step $BEST: only $N/10 leaderboard instances measured (sum $Q) — not a score"
fi
