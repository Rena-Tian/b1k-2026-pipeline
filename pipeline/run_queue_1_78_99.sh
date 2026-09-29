#!/usr/bin/env bash
# Eval queue requested 2026-09-26: task 42 (already running) -> task 1 -> task 78 -> task 99.
#
#   task 1  picking_up_trash   ft_task1/9999   all 20 (fresh; same single-checkpoint plan as 42 and 10)
#   task 78 make_cabinet_doors ft_3more/19999  finish 20 (13 done, all 0.00)
#   task 99 sorting_books      ft_3more/19999  finish 20 (5 done, all 0.00) — the OLD joint run; the
#                                              dedicated task-99 retrain was never trained
#
# Every task is retried until all 20 instances have a real rollout (GPU OOM kills some before they
# act). If the Elements drive drops out, wait for it instead of burning retry passes — that is what
# produced the bogus "14/20" task-42 FINAL on 2026-09-26.
set -u
LOG=~/winner-eval/queue_1_78_99.log
OUT=~/eval_outputs
EL=/media/rena/Elements/openpi_checkpoints
ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

log "waiting for the task-42 eval to finish"
while pgrep -f "bash \./run_task42_eval\.sh" >/dev/null || pgrep -f "/home/rena/winner-eval/run_ft_eval\.py" >/dev/null; do sleep 60; done

score(){ python3 -c "
import json,glob
qs=[json.load(open(f))['q_score']['final'] for f in glob.glob('$OUT/$1_ft1_step$2/json/*.json')]
print('%d %.3f'%(len(qs),sum(qs)))"; }

eval_task(){ # name id config ckpt_dir step
  local t=$1 id=$2 cfg=$3 ck=$4 step=$5
  log "=== task $id $t, $ck/$step: starting at $(score $t $step) ==="
  for pass in 1 2 3 4 5 6 7 8; do
    [ "$(score $t $step | cut -d' ' -f1)" -ge 20 ] && break
    until [ -f "$ck/$step/params/manifest.ocdbt" ]; do log "checkpoint $ck/$step not visible (drive unmounted?), waiting"; sleep 120; done
    POLICY_DEVICE=gpu python3 ~/winner-eval/run_ft_eval.py --task "$t" --task-id "$id" --config "$cfg" \
      --ckpt-root "$ck" --step "$step" --run ft1 --instances $ALL >> "$LOG" 2>&1
    log "task $id after pass $pass: $(score $t $step)"
  done
  local S=$(score $t $step) N Q
  N=$(echo $S | cut -d' ' -f1); Q=$(echo $S | cut -d' ' -f2)
  if [ "$N" -ge 20 ]; then
    log "FINAL task $id step $step: sum $Q over 20 -> official q = $(python3 -c "print('%.4f'%($Q/20))")"
  else
    log "INCOMPLETE task $id step $step: only $N/20 measured (sum $Q) — not an official score"
  fi
}

eval_task picking_up_trash       1  pi_behavior_b1k_2026_task1  $EL/ft_task1  9999
eval_task make_cabinet_doors     78 pi_behavior_b1k_2026_3more  $EL/ft_3more  19999
eval_task sorting_books_on_shelf 99 pi_behavior_b1k_2026_3more  $EL/ft_3more  19999
log "queue done"
