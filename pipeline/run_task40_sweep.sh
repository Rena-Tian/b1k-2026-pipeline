#!/usr/bin/env bash
# Task 40 (make_microwave_popcorn) — Phase B.
#
# Phase A screening is done. Steps 9999, 8000 and 6000 each scored 2.00/5 (a three-way tie, each
# solving a different pair of instances => one ~40% policy, not three qualities). Step 2000 scored
# 0.00. Step 4000 is SKIPPED at the user's direction: it failed 5/5 with GPU fragmentation errors
# and cannot change the tie-break.
#
# Tie-break goes to the highest step, 9999. This runs instances 306-320 on it, then a retry pass
# over all 20 to fill anything lost to GPU OOM.
#
# Keeps the name run_task40_sweep.sh so run_task78_99_rest.sh's guard still holds 78/99 behind it.
set -u
LOG=~/winner-eval/task40_sweep.log
CK=/media/rena/Elements/openpi_checkpoints/ft_task40
OUT=~/eval_outputs
BEST=9999
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

eval_step(){ # step, indices...
  local s=$1; shift
  POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task make_microwave_popcorn --task-id 40 \
    --config pi_behavior_b1k_2026_task40 --ckpt-root "$CK" --step "$s" --run ft1 \
    --instances "$@" >> "$LOG" 2>&1
}
score(){ python3 -c "
import json,glob
fs=glob.glob('$OUT/make_microwave_popcorn_ft1_step$1/json/*.json')
qs=[json.load(open(f))['q_score']['final'] for f in fs]
print('%d %.3f'%(len(qs),sum(qs)))"; }

log "=== Phase B: step $BEST, instances 306-320 (step 4000 skipped per user) ==="
eval_step "$BEST" 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
log "step $BEST first pass: $(score $BEST)"

# up to two retry passes for instances lost to GPU RESOURCE_EXHAUSTED
for pass in 1 2; do
  n=$(score "$BEST" | cut -d' ' -f1)
  [ "$n" -ge 20 ] && break
  log "step $BEST has $n/20 instances, retry pass $pass"
  eval_step "$BEST" 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
done

S=$(score $BEST)
log "FINAL task 40 step $BEST: $S -> official q = $(python3 -c "print('%.4f'%($(echo $S | cut -d' ' -f2)/20))")"
