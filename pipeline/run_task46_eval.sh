#!/usr/bin/env bash
# Task 46 (cook_bacon) — final checkpoint 9999, all 20 public instances.
#
# No checkpoint screening: task 46 is the same recipe as task 40 (10k steps, batch 16, specialist
# init from a winner checkpoint), and on task 40 the last three checkpoints tied exactly (9999,
# 8000, 6000 all 2.00/5) while only step 2000 was dead. So the final checkpoint is the sensible
# single choice, and screening would cost ~25 rollouts for a distinction inside the noise band.
# If 9999 comes back at 0.000 across all 20, steps 8000/6000 are the fallback.
#
# Runs after the task-40 Phase B job frees the GPU.
set -u
LOG=~/winner-eval/task46_eval.log
CK=/media/rena/Elements/openpi_checkpoints/ft_task46
OUT=~/eval_outputs
STEP=9999
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

while pgrep -f "bash \./run_task40_sweep\.sh" >/dev/null || pgrep -f "/home/rena/winner-eval/run_ft_eval\.py" >/dev/null; do sleep 60; done
# the checkpoint must be on Elements and complete
until [ -f "$CK/$STEP/params/manifest.ocdbt" ]; do log "waiting for ft_task46/$STEP to finish downloading"; sleep 120; done

eval_step(){ POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task cook_bacon --task-id 46 \
    --config pi_behavior_b1k_2026_task46 --ckpt-root "$CK" --step "$STEP" --run ft1 \
    --instances "$@" >> "$LOG" 2>&1; }
score(){ python3 -c "
import json,glob
fs=glob.glob('$OUT/cook_bacon_ft1_step$STEP/json/*.json')
qs=[json.load(open(f))['q_score']['final'] for f in fs]
print('%d %.3f'%(len(qs),sum(qs)))"; }

ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
log "=== task 46 cook_bacon, step $STEP, instances 301-320 ==="
eval_step $ALL
log "first pass: $(score)"
for pass in 1 2; do
  n=$(score | cut -d' ' -f1)
  [ "$n" -ge 20 ] && break
  log "have $n/20 instances, retry pass $pass"
  eval_step $ALL
done
S=$(score)
log "FINAL task 46 step $STEP: $S -> official q = $(python3 -c "print('%.4f'%($(echo $S | cut -d' ' -f2)/20))")"
