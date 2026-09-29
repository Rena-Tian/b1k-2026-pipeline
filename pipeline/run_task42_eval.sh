#!/usr/bin/env bash
# Task 42 (chop_an_onion) — final checkpoint 9999, all 20 public instances.
#
# Single checkpoint, no Phase A screen: task 42 is the same recipe as tasks 40 and 10 (10k steps,
# batch 16, specialist init from a winner checkpoint), and on task 40 the last three checkpoints
# tied exactly at 2.00/5. Task 10 ran single-checkpoint on 9999 and returned 0.225. If 9999 comes
# back flat 0.000 across all 20, steps 8000/6000 are the fallback to screen.
#
# Goal structure (why this task was trained): 4 ground predicates, all starting FALSE —
#   real(diced onion) · inside(parer, sink) · inside(chopping_board, sink) · contains(bowl, onion)
# Two of them are plain placements that pay 0.25 each even if the dicing never succeeds, so unlike
# task 78's single `attached` there is partial credit to catch. Horizon 6400 steps.
set -u
LOG=~/winner-eval/task42_eval.log
CK=/media/rena/Elements/openpi_checkpoints/ft_task42
OUT=~/eval_outputs
STEP=9999
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

while pgrep -f "/home/rena/winner-eval/run_ft_eval\.py" >/dev/null; do sleep 60; done
until [ -f "$CK/$STEP/params/manifest.ocdbt" ]; do log "waiting for ft_task42/$STEP"; sleep 120; done

eval_step(){ POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task chop_an_onion --task-id 42 \
    --config pi_behavior_b1k_2026_task42 --ckpt-root "$CK" --step "$STEP" --run ft1 \
    --instances "$@" >> "$LOG" 2>&1; }
score(){ python3 -c "
import json,glob
fs=glob.glob('$OUT/chop_an_onion_ft1_step$STEP/json/*.json')
qs=[json.load(open(f))['q_score']['final'] for f in fs]
print('%d %.3f'%(len(qs),sum(qs)))"; }

ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
log "=== task 42 chop_an_onion, step $STEP, instances 301-320 ==="
eval_step $ALL
log "pass 1: $(score)"
# GPU OOM kills instances before they act; a lost instance counts as 0 in the official sum/20,
# so retry until every instance has a real rollout. Failures are cheap (minutes), successes are not.
for pass in 2 3 4 5 6 7 8; do
  n=$(score | cut -d' ' -f1)
  [ "$n" -ge 20 ] && break
  log "have $n/20 instances, retry pass $pass"
  eval_step $ALL
done
S=$(score)
log "FINAL task 42 step $STEP: $S -> official q = $(python3 -c "print('%.4f'%($(echo $S | cut -d' ' -f2)/20))")"
