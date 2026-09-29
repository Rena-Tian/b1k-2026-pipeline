#!/usr/bin/env bash
# Task 10 (set_up_a_coffee_station_in_your_kitchen) — final checkpoint 9999, all 20 instances.
# Same single-checkpoint reasoning as task 46. Runs after the task-46 eval frees the GPU, and
# waits for stream_ckpts.sh to have pulled step 9999 onto Elements (training ends ~08:00 09-24).
set -u
LOG=~/winner-eval/task10_eval.log
CK=/media/rena/Elements/openpi_checkpoints/ft_task10
OUT=~/eval_outputs
STEP=9999
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

while pgrep -f "bash \./run_task46_sweep\.sh" >/dev/null || pgrep -f "/home/rena/winner-eval/run_ft_eval\.py" >/dev/null; do sleep 60; done
until [ -f "$CK/$STEP/params/manifest.ocdbt" ]; do log "waiting for ft_task10/$STEP (training + stream_ckpts)"; sleep 300; done

eval_step(){ POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task set_up_a_coffee_station_in_your_kitchen --task-id 10 \
    --config pi_behavior_b1k_2026_task10 --ckpt-root "$CK" --step "$STEP" --run ft1 \
    --instances "$@" >> "$LOG" 2>&1; }
score(){ python3 -c "
import json,glob
fs=glob.glob('$OUT/set_up_a_coffee_station_in_your_kitchen_ft1_step$STEP/json/*.json')
qs=[json.load(open(f))['q_score']['final'] for f in fs]
print('%d %.3f'%(len(qs),sum(qs)))"; }

ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
log "=== task 10 set_up_a_coffee_station_in_your_kitchen, step $STEP, instances 301-320 ==="
eval_step $ALL
log "first pass: $(score)"
for pass in 1 2; do
  n=$(score | cut -d' ' -f1)
  [ "$n" -ge 20 ] && break
  log "have $n/20 instances, retry pass $pass"
  eval_step $ALL
done
S=$(score)
log "FINAL task 10 step $STEP: $S -> official q = $(python3 -c "print('%.4f'%($(echo $S | cut -d' ' -f2)/20))")"
