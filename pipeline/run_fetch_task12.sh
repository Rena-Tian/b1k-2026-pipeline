#!/usr/bin/env bash
# Copy Richard's task-12 checkpoints (pod at 185.216.21.214:14097) to Elements ft_task12.
# COPY ONLY — never deletes anything on the pod (these are Richard's files).
set -u
POD=root@185.216.21.214; PORT=14097
SRC=/workspace/behavior-1k-solution/outputs/checkpoints/pi_behavior_b1k_2026_task12/ft1
DST=/media/rena/Elements/openpi_checkpoints/ft_task12
LOG=~/winner-eval/fetch_task12.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
mkdir -p "$DST"
for s in 9999 8000 6000 4000 2000; do
  for try in 1 2 3 4 5; do
    log "step $s: rsync try $try"
    rsync -rlt --partial -e "ssh -o BatchMode=yes -p $PORT" "$POD:$SRC/$s/" "$DST/$s/" >> "$LOG" 2>&1 && break
    sleep 60
  done
  R=$(ssh -o BatchMode=yes -p $PORT $POD "du -sb $SRC/$s | cut -f1"); L=$(du -sb "$DST/$s" | cut -f1)
  if [ -n "$R" ] && [ "$R" = "$L" ]; then log "step $s: verified ($L bytes)"; else log "step $s: SIZE MISMATCH local=$L remote=$R"; fi
done
log "done: $(ls $DST | tr '\n' ' ')"
