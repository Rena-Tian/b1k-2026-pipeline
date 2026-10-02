#!/usr/bin/env bash
# Task 20: bootstrap (download data, sanity check, start training + streamer), then arm the auto-stop.
# Usage: POD=root@<ip> PORT=<port> ./train_task20_chain.sh
set -u
: "${POD:?set POD}"; : "${PORT:?set PORT}"
cd ~/winner-eval
bash ./bootstrap_task20_training.sh
grep -q "bootstrap complete" ~/winner-eval/bootstrap_task20.log || { echo "bootstrap did not complete; auto-stop NOT armed"; exit 1; }
sleep 120   # let tmux train20 appear before the stop script starts watching it
bash ./stop_pod_after_task20.sh
