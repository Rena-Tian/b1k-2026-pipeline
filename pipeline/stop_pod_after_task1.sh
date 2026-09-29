#!/usr/bin/env bash
# Stop the pod once task-1 training has finished AND all five of its checkpoints are verified on
# Elements. Refuses to stop if anything is missing — the container-local disk (/root/ckpt) is wiped
# on stop, and that is exactly where task 46's final checkpoint was nearly lost.
set -u
POD="root@185.216.21.214"; PORT=35233; POD_ID=f9o2yy3xq43l9a
LOCAL=/media/rena/Elements/openpi_checkpoints/ft_task1
STEPS="2000 4000 6000 8000 9999"
LOG=~/winner-eval/stop_pod_task1.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
sshp(){ ssh -o ConnectTimeout=25 -o BatchMode=yes -p $PORT $POD "$@" </dev/null 2>/dev/null; }

log "waiting for task-1 training to finish"
while [ -n "$(sshp 'tmux ls 2>/dev/null | grep train1')" ]; do sleep 300; done
log "training finished: $(sshp 'grep -aoE "Step [0-9]+:" /root/logs/train_task1_ft1.log | tail -1')"

log "waiting for stream_ckpts.sh to fetch and verify every checkpoint"
while pgrep -f "bash \./stream_ckpts\.sh pi_behavior_b1k_2026_task1" >/dev/null; do sleep 120; done

missing=""
for s in $STEPS; do [ -f "$LOCAL/$s/params/manifest.ocdbt" ] || missing="$missing $s"; done
if [ -n "$missing" ]; then
  log "NOT STOPPING — task-1 checkpoints missing on Elements:$missing"
  log "the pod stays up so they can still be fetched; re-run this script once they are local"
  exit 1
fi

# belt and braces: nothing of value should remain only on the pod
REMOTE_LEFT=$(sshp "ls /root/ckpt/pi_behavior_b1k_2026_task1/ft1 2>/dev/null | tr '\n' ' '")
log "all 5 checkpoints verified on Elements ($(du -sh $LOCAL | cut -f1)); pod-local leftovers: ${REMOTE_LEFT:-none}"

log "stopping pod $POD_ID"
out=$(sshp "runpodctl stop pod $POD_ID" 2>&1); log "runpodctl: ${out:-<no output>}"
sleep 45
if ssh -o ConnectTimeout=15 -p $PORT $POD true </dev/null 2>/dev/null; then
  log "pod still reachable — stop it manually at console.runpod.io (pod $POD_ID)"
else
  log "pod is no longer reachable: STOPPED"
fi
log "note: the network volume survives a stop; only the container disk (/root) is wiped"
