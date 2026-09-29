#!/usr/bin/env bash
# Pull the task-40 fine-tune checkpoints to the Elements drive, freeing the pod volume as we go.
set -u
POD="root@185.216.21.214"; PORT=35233
VOL=/workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task40/ft1
LOCAL=/media/rena/Elements/openpi_checkpoints/ft_task40
LOG=~/winner-eval/fetch_task40.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
for STEP in 9999 8000 6000 4000 2000; do
  [ -f "$LOCAL/$STEP/params/manifest.ocdbt" ] && { log "step $STEP already local"; continue; }
  log "fetching step $STEP"
  until rsync -rlt --partial --info=progress2 -e "ssh -p $PORT" "$POD:$VOL/$STEP/" "$LOCAL/$STEP/" >> "$LOG" 2>&1; do log "rsync failed, retry in 60s"; sleep 60; done
  R=$(ssh -o ConnectTimeout=20 -p $PORT $POD "du -sb $VOL/$STEP | cut -f1" </dev/null 2>/dev/null); L=$(du -sb "$LOCAL/$STEP" | cut -f1)
  if [ "$R" = "$L" ]; then log "step $STEP verified ($L bytes); removing from volume"; ssh -o ConnectTimeout=20 -p $PORT $POD "rm -rf $VOL/$STEP" </dev/null 2>/dev/null; else log "step $STEP SIZE MISMATCH local=$L remote=$R (kept on volume)"; fi
done
log "done: $(ls $LOCAL | tr '\n' ' ')"
