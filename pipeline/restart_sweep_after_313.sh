#!/usr/bin/env bash
# Hand-off to the updated run_task67_sweep.sh (with the 3-zeros SCREEN RULE) at the next safe point:
# as soon as step-6000 instance 313 has a result, stop the old sweep (and the 314 rollout it starts)
# and relaunch the new script over all three checkpoints. The new script sees 311-313 already measured
# and applies the rule itself: all 0 -> skip 6000; otherwise it finishes 314-315.
set -u
J=~/eval_outputs/boxing_food_after_dinner_ft67_step6000/json/boxing_food_after_dinner_313_0.json
LOG=~/winner-eval/task67_sweep.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
until [ -f "$J" ]; do sleep 10; done
sleep 3
log "HAND-OFF: 313 done (q=$(python3 -c "import json; print(json.load(open('$J'))['q_score']['final'])")); restarting sweep on the updated script (3-zeros screen rule)"
for p in $(pgrep -f "bash \./run_task67_sweep\.sh") $(pgrep -f "/home/rena/winner-eval/run_ft_eval\.py --task boxing_food_after_dinner"); do kill "$p" 2>/dev/null; done
sleep 3
for p in $(pgrep -f "scripts/serve_b1k\.py .*--task-id 67") $(pgrep -f "omnigibson\.eval\.eval --task-name boxing_food_after_dinner") $(pgrep -f "conda run .*boxing_food_after_dinner"); do kill "$p" 2>/dev/null; done
sleep 8
log "HAND-OFF: old processes left: $(pgrep -f 'run_ft_eval\.py|serve_b1k\.py|omnigibson\.eval\.eval' | wc -l)"
cd ~/winner-eval && STEPS="6000 8000 9999" setsid nohup bash ./run_task67_sweep.sh >/dev/null 2>&1 < /dev/null &
sleep 5
log "HAND-OFF: new sweep running: $(pgrep -f 'bash \./run_task67_sweep\.sh' | tr '\n' ' ')"
