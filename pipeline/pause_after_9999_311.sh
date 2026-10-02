#!/usr/bin/env bash
# User (2026-09-29 16:30): "after this rollout, please check the video first".
# When task-67 step-9999 instance 311 has a result, stop the sweep before it starts 312 (and kill the 312
# rollout if it already started). Nothing is lost: finished instances are kept; restart with
#   cd ~/winner-eval && STEPS="6000 8000 9999" setsid nohup bash ./run_task67_sweep.sh &
set -u
J=~/eval_outputs/boxing_food_after_dinner_ft67_step9999/json/boxing_food_after_dinner_311_0.json
LOG=~/winner-eval/task67_sweep.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
until [ -f "$J" ]; do sleep 5; done
sleep 2
for p in $(pgrep -f "bash \./run_task67_sweep\.sh") $(pgrep -f "/home/rena/winner-eval/run_ft_eval\.py --task boxing_food_after_dinner"); do kill "$p" 2>/dev/null; done
sleep 3
for p in $(pgrep -f "scripts/serve_b1k\.py .*--task-id 67") $(pgrep -f "omnigibson\.eval\.eval --task-name boxing_food_after_dinner") $(pgrep -f "conda run .*boxing_food_after_dinner"); do kill "$p" 2>/dev/null; done
sleep 5
log "PAUSED by user request after step-9999 instance 311 (q=$(python3 -c "import json; print(json.load(open('$J'))['q_score']['final'])")) to review videos; processes left: $(pgrep -f 'run_ft_eval\.py|serve_b1k\.py|omnigibson\.eval\.eval' | wc -l)"
