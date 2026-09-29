#!/usr/bin/env bash
# Wait for all five task-40 checkpoints to be verified locally, then stop the pod.
set -u
POD="root@185.216.21.214"; PORT=35233; POD_ID=f9o2yy3xq43l9a
LOCAL=/media/rena/Elements/openpi_checkpoints/ft_task40
LOG=~/winner-eval/stop_pod.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
while pgrep -f "fetch_task40_ckpts.s[h]" >/dev/null; do sleep 120; done
missing=""
for s in 2000 4000 6000 8000 9999; do [ -f "$LOCAL/$s/params/manifest.ocdbt" ] || missing="$missing $s"; done
if [ -n "$missing" ]; then log "NOT stopping: checkpoints missing locally:$missing"; exit 1; fi
log "all 5 checkpoints present locally ($(du -sh $LOCAL | cut -f1)); stopping pod $POD_ID"
out=$(ssh -o ConnectTimeout=20 -p $PORT $POD "runpodctl stop pod $POD_ID" </dev/null 2>&1); log "runpodctl: $out"
sleep 45
if ssh -o ConnectTimeout=15 -p $PORT $POD true </dev/null 2>/dev/null; then
  log "pod still reachable — stop it manually at console.runpod.io (pod $POD_ID)"
else
  log "pod is no longer reachable: stopped"
fi
