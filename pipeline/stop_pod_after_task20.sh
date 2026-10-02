#!/usr/bin/env bash
# Stop the training pod once task-20 training has finished AND all five of its checkpoints are verified
# on Elements (user request 2026-09-28: "after the last checkpoint landed on the element disk, stop the
# pod to avoid wasting money"). Refuses to stop if anything is missing.
# runpodctl over SSH needs the API key from the pod's PID-1 environment (a plain SSH session does not
# have it — that is why stop_pod_after_task1.sh failed with "API key not found").
set -u
POD="${POD:?set POD}"; PORT="${PORT:?set PORT}"; POD_ID=""
CFG=pi_behavior_b1k_2026_task20
LOCAL=/media/rena/Elements/openpi_checkpoints/ft_task20
STREAMLOG=~/winner-eval/stream_ft_task20.log
STEPS="2000 4000 6000 8000 9999"
LOG=~/winner-eval/stop_pod_task20.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
sshp(){ ssh -o ConnectTimeout=25 -o BatchMode=yes -p $PORT $POD "$@" </dev/null 2>/dev/null; }
verified_all(){ for s in $STEPS; do grep -a -q "step $s: verified" "$STREAMLOG" && [ -f "$LOCAL/$s/params/manifest.ocdbt" ] || return 1; done; }

log "waiting for task-20 training to finish (tmux train20)"
while [ -n "$(sshp 'tmux ls 2>/dev/null | grep train20')" ]; do sleep 300; done
log "training finished: $(sshp 'grep -aoE "Step [0-9]+: action_loss=[0-9.]+" /root/logs/train_task20_ft1.log | tail -1')"

log "waiting for all 5 checkpoints to be verified on Elements"
for i in $(seq 1 180); do verified_all && break; sleep 120; done
if ! verified_all; then
  missing=""; for s in $STEPS; do grep -a -q "step $s: verified" "$STREAMLOG" || missing="$missing $s"; done
  log "NOT STOPPING — checkpoints not verified on Elements after 6 h:$missing. Pod stays up."
  exit 1
fi
log "all 5 checkpoints verified on Elements ($(du -sh $LOCAL | cut -f1))"
log "pod-side leftovers (fine to lose): $(sshp "ls /root/ckpt/$CFG/ft1 /workspace/behavior-1k-solution/outputs/checkpoints/$CFG/ft1 2>/dev/null | tr '\n' ' '")"

POD_ID=$(sshp 'tr "\0" "\n" < /proc/1/environ | grep "^RUNPOD_POD_ID=" | cut -d= -f2-')
log "stopping pod $POD_ID"
out=$(sshp 'export RUNPOD_API_KEY=$(tr "\0" "\n" < /proc/1/environ | grep "^RUNPOD_API_KEY=" | cut -d= -f2-); runpodctl stop pod '"$POD_ID" 2>&1)
log "runpodctl: ${out:-<no output>}"
# verify by reachability, never by the command's message
for i in 1 2 3 4 5 6; do
  sleep 30
  if ! ssh -o ConnectTimeout=15 -o BatchMode=yes -p $PORT $POD true </dev/null 2>/dev/null; then
    log "pod is no longer reachable: STOPPED (network volume keeps its data; container disk is wiped)"
    exit 0
  fi
done
log "POD STILL REACHABLE after 3 min — stop it manually at console.runpod.io (pod $POD_ID)"
exit 2
