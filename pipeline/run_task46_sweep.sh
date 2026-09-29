#!/usr/bin/env bash
# Task 46 (cook_bacon) — stop at 5 MEASURED instances, then hand the GPU to task 10.
#
# Why we are not running all 20 (user decision 2026-09-24): the first three measured rollouts all
# scored exactly 0.00 and all ran the full 11,520-step clock, i.e. timeouts rather than crashes.
# The goal is (forall ?bacon (cooked ?bacon)) AND (not (open ?fridge)) — a forall over all six
# slices is effectively one all-or-nothing condition, so there is almost no partial credit to
# catch, unlike task 56 which scored 0.263 purely on partials. And in the official metric an
# unrun instance and a measured-zero instance both contribute 0, so the remaining 17 rollouts
# (~20 h) could not raise the score unless one of them scored above zero.
#
# A 5-instance sample is enough to record task 46 as 0.000 and move on.
# Name kept as run_task46_sweep.sh so run_task10_eval.sh's guard still holds task 10 behind it.
set -u
LOG=~/winner-eval/task46_sweep.log
CK=/media/rena/Elements/openpi_checkpoints/ft_task46
OUT=~/eval_outputs
STEP=9999
TARGET=5
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# let the in-flight rollout from the previous controller finish — it counts toward the target
while pgrep -f "/home/rena/winner-eval/run_ft_eval\.py" >/dev/null; do sleep 60; done

count(){ ls "$OUT/cook_bacon_ft1_step$STEP"/json/*.json 2>/dev/null | wc -l; }
score(){ python3 -c "
import json,glob
fs=glob.glob('$OUT/cook_bacon_ft1_step$STEP/json/*.json')
qs=[json.load(open(f))['q_score']['final'] for f in fs]
print('%d %.3f'%(len(qs),sum(qs)))"; }
eval_one(){ POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task cook_bacon --task-id 46 \
    --config pi_behavior_b1k_2026_task46 --ckpt-root "$CK" --step "$STEP" --run ft1 \
    --instances "$1" >> "$LOG" 2>&1; }

log "=== task 46: filling up to $TARGET measured instances, currently $(count) ==="
# one index at a time so we can stop the moment the target is reached; several passes because
# ~29% of attempts survive the GPU (weights 6.3 GB + scratch ~1.5 GB + sim ~4.1 GB vs 12.2 GB)
for pass in 1 2 3 4 5 6; do
  [ "$(count)" -ge "$TARGET" ] && break
  for i in 0 1 2 3 4 5 6 7 8 9; do
    [ "$(count)" -ge "$TARGET" ] && break
    eval_one "$i"
  done
  log "after pass $pass: $(score)"
done

S=$(score)
N=$(echo $S | cut -d' ' -f1); Q=$(echo $S | cut -d' ' -f2)
log "STOPPING task 46 at $N measured instances (sum q $Q)"
log "task 46 reported q over the measured sample = $(python3 -c "print('%.4f'%($Q/$N))" 2>/dev/null); official (sum/20) = $(python3 -c "print('%.4f'%($Q/20))")"
log "handing the GPU to task 10"
