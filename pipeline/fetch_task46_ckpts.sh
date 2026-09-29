#!/usr/bin/env bash
# Pull the task-46 (cook_bacon) checkpoints to Elements.
# Step 9999 lives ONLY on the pod's container-local disk (/root/ckpt) — the syncer's CKPT_SWEEP
# listed 10000 but training ended at 9999, so it was never copied to the volume. Container local
# disk is wiped when the pod stops, so 9999 is fetched FIRST and straight from /root/ckpt.
# The volume is at 187G of its 200G quota, so nothing is staged through it.
set -u
POD="root@185.216.21.214"; PORT=35233
VOL=/workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task46/ft1
LOCALDISK=/root/ckpt/pi_behavior_b1k_2026_task46/ft1
LOCAL=/media/rena/Elements/openpi_checkpoints/ft_task46
LOG=~/winner-eval/fetch_task46.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

fetch(){ # step, remote_root, delete_after(yes/no)
  local STEP=$1 ROOT=$2 DEL=$3
  [ -f "$LOCAL/$STEP/params/manifest.ocdbt" ] && { log "step $STEP already local"; return; }
  log "fetching step $STEP from $ROOT"
  until rsync -rlt --partial --info=progress2 -e "ssh -p $PORT" "$POD:$ROOT/$STEP/" "$LOCAL/$STEP/" >> "$LOG" 2>&1; do
    log "rsync failed, retry in 60s"; sleep 60
  done
  local R L
  R=$(ssh -o ConnectTimeout=20 -p $PORT $POD "du -sb $ROOT/$STEP | cut -f1" </dev/null 2>/dev/null)
  L=$(du -sb "$LOCAL/$STEP" | cut -f1)
  if [ "$R" = "$L" ]; then
    log "step $STEP verified ($L bytes)"
    [ "$DEL" = yes ] && { ssh -o ConnectTimeout=20 -p $PORT $POD "rm -rf $ROOT/$STEP" </dev/null 2>/dev/null; log "step $STEP removed from volume"; }
  else
    log "step $STEP SIZE MISMATCH local=$L remote=$R (kept remote)"
  fi
}

fetch 9999 "$LOCALDISK" no          # at-risk copy first, keep it on the pod until all is safe
for STEP in 8000 6000 4000 2000; do fetch "$STEP" "$VOL" yes; done
log "done: $(ls $LOCAL | tr '\n' ' ')"
log "volume now: $(ssh -o ConnectTimeout=20 -p $PORT $POD 'du -sh /workspace | cut -f1' </dev/null 2>/dev/null)"
