#!/usr/bin/env bash
# Task 99 (sorting_books_on_shelf) DEDICATED retrain, W&B thsvulxu — checkpoint sweep, then full 20.
#
# Phase A: screen 5 instances (301-305) per checkpoint as each one lands on Elements, in the order
#   4000, 6000, 8000, 9999. Step 2000 is skipped: tasks 40 and 89 both scored 0 at 2000.
# Phase B: all 20 instances on the best screen (highest sum; ties go to the later step).
# Baseline to beat: the old joint 56/78/99 checkpoint ft_3more/19999 = 0.0182 official, and 0.000 on
#   instances 301-305 specifically.
#
# Outputs go to ~/eval_outputs/sorting_books_on_shelf_ft99_step<N>/ (run tag ft99) so they can never
# mix with the old ft_3more results in sorting_books_on_shelf_ft1_step8000/19999.
set -u
LOG=~/winner-eval/task99_retrain_sweep.log
CK=/media/rena/Elements/openpi_checkpoints/ft_task99
STREAMLOG=~/winner-eval/stream_ft_task99.log
OUT=~/eval_outputs
T=sorting_books_on_shelf
TAG=ft99
# STEPS can be overridden. 2026-09-28 18:25 the user stopped the sweep mid-way through step 4000
# (301-303 measured, all 0.00; 304 lost to a WeChat GPU OOM, 305 interrupted) and asked to resume at
# the NEXT checkpoint, so the restart runs STEPS="6000 8000 9999"; 4000 is left at 3/5.
STEPS="${STEPS:-4000 6000 8000 9999}"
SCREEN="0 1 2 3 4"
ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

score(){ python3 -c "
import json,glob
qs=[json.load(open(f))['q_score']['final'] for f in glob.glob('$OUT/${T}_${TAG}_step$1/json/*.json')]
print('%d %.4f'%(len(qs),sum(qs)))"; }
# a checkpoint is usable only once the streamer has verified its byte count against the pod
ready(){ grep -a -q "step $1: verified" "$STREAMLOG" && [ -f "$CK/$1/params/manifest.ocdbt" ]; }
eval_to(){ # step target_count indices...
  local s=$1 n=$2; shift 2
  for pass in 1 2 3 4 5 6 7 8; do
    [ "$(score $s | cut -d' ' -f1)" -ge "$n" ] && break
    until [ -f "$CK/$s/params/manifest.ocdbt" ]; do log "checkpoint $s not visible (drive unmounted?), waiting"; sleep 120; done
    POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task $T --task-id 99 \
      --config pi_behavior_b1k_2026_task99 --ckpt-root "$CK" --step "$s" --run $TAG \
      --instances "$@" >> "$LOG" 2>&1
    log "step $s after pass $pass: $(score $s)"
  done
}

while pgrep -f "/home/rena/winner-eval/run_ft_eval\.py" >/dev/null; do sleep 60; done
log "=== task 99 retrain sweep: screening $STEPS on instances 301-305 ==="
for s in $STEPS; do
  until ready $s; do sleep 300; done
  log "--- screening step $s ---"
  eval_to $s 5 $SCREEN
  log "SCREEN step $s: $(score $s)"
done

BEST=$(for s in $STEPS; do echo "$s $(score $s)"; done | sort -k3,3g -k1,1n | tail -1 | cut -d' ' -f1)
log "=== screen results: $(for s in $STEPS; do echo -n "$s=[$(score $s)] "; done) -> best step $BEST ==="
log "--- full 20 on step $BEST ---"
eval_to $BEST 20 $ALL
S=$(score $BEST); N=${S%% *}; Q=${S##* }
if [ "$N" -ge 20 ]; then
  log "FINAL task 99 retrain step $BEST: sum $Q over 20 -> official q = $(python3 -c "print('%.4f'%($Q/20))") (old joint ckpt: 0.0182)"
else
  log "INCOMPLETE task 99 retrain step $BEST: only $N/20 measured (sum $Q) — not an official score"
fi
