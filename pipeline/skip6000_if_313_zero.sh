#!/usr/bin/env bash
# User rule (2026-09-29 13:10): "if 313 still 0, then move on to 8000, skip 6000".
# Waits for task-67 step-6000 instance 313's result. If q == 0: stop the running sweep (and the 314 rollout
# it will have started) and restart it with STEPS="8000 9999". If q > 0: leave everything as is.
set -u
J=~/eval_outputs/boxing_food_after_dinner_ft67_step6000/json/boxing_food_after_dinner_313_0.json
LOG=~/winner-eval/task67_sweep.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
until [ -f "$J" ]; do sleep 10; done
sleep 3
q=$(python3 -c "import json; print(json.load(open('$J'))['q_score']['final'])")
if python3 -c "import sys; sys.exit(0 if float('$q') > 0 else 1)"; then
  log "RULE: 313 scored $q > 0 -> keep screening step 6000 (314, 315)"
  exit 0
fi
log "RULE: 313 scored 0 (6000 = 0/3 on 311-313) -> skipping rest of step 6000, moving to 8000 (user rule)"
for p in $(pgrep -f "bash \./run_task67_sweep\.sh") $(pgrep -f "/home/rena/winner-eval/run_ft_eval\.py --task boxing_food_after_dinner"); do kill "$p" 2>/dev/null; done
sleep 3
for p in $(pgrep -f "scripts/serve_b1k\.py .*--task-id 67") $(pgrep -f "omnigibson\.eval\.eval --task-name boxing_food_after_dinner") $(pgrep -f "conda run .*boxing_food_after_dinner"); do kill "$p" 2>/dev/null; done
sleep 8
left=$(pgrep -f "run_ft_eval\.py|serve_b1k\.py|omnigibson\.eval\.eval" | wc -l)
log "RULE: step-6000 processes stopped (left: $left); relaunching sweep with STEPS=\"8000 9999\""
cd ~/winner-eval && STEPS="8000 9999" setsid nohup bash ./run_task67_sweep.sh >/dev/null 2>&1 < /dev/null &
sleep 5
log "RULE: sweep relaunched: $(pgrep -f 'bash \./run_task67_sweep\.sh' | tr '\n' ' ')"
