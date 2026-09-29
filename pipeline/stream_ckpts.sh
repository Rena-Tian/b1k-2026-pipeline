#!/usr/bin/env bash
# Stream fine-tune checkpoints off the pod as soon as each one lands: fetch to Elements, verify
# byte-for-byte, then delete the pod copy. Keeps the 200 GB volume near-empty during training and
# means the run survives a pod stop at any moment.
#
# Usage: stream_ckpts.sh <config_name> <local_dir_name> [steps...]
#   e.g. stream_ckpts.sh pi_behavior_b1k_2026_task10 ft_task10 2000 4000 6000 8000 9999
#
# Every run uses --exp_name=ft1, so the checkpoint dir is always <config>/ft1. Keep it that way:
# exp_name is the last path component of the checkpoint directory, so changing it moves where
# checkpoints land on both the container disk and the volume.
#
# ORDERING RULE (matters): ckpt_sync.py on the pod re-copies a volume checkpoint whenever the
# volume copy is missing AND the container-local copy still exists. Deleting the volume copy of the
# newest step would therefore start an endless copy loop. So a volume copy is only deleted once its
# /root/ckpt counterpart is gone — which the syncer does for every step except the newest.
# The final step is handled at the end, after the syncer is stopped.
set -u
POD="${POD:-root@185.216.21.214}"; PORT="${PORT:-35233}"
CFG=${1:?config name}; LOCALNAME=${2:?local dir name}; shift 2
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(2000 4000 6000 8000 9999)
VOL=/workspace/behavior-1k-solution/outputs/checkpoints/$CFG/ft1
LOCALDISK=/root/ckpt/$CFG/ft1
LOCAL=/media/rena/Elements/openpi_checkpoints/$LOCALNAME
LOG=~/winner-eval/stream_${LOCALNAME}.log
mkdir -p "$LOCAL"
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
sshp(){ ssh -o ConnectTimeout=25 -o BatchMode=yes -p $PORT $POD "$@" </dev/null 2>/dev/null; }

have(){ [ -f "$LOCAL/$1/params/manifest.ocdbt" ]; }

fetch_one(){ # step, remote_root -> 0 if now verified locally
  local S=$1 ROOT=$2
  log "step $S: fetching from $ROOT"
  local tries=0
  until rsync -rlt --partial -e "ssh -p $PORT" "$POD:$ROOT/$S/" "$LOCAL/$S/" >> "$LOG" 2>&1; do
    tries=$((tries+1)); [ $tries -ge 5 ] && { log "step $S: rsync failed 5x, will retry next round"; return 1; }
    log "step $S: rsync failed, retry in 60s"; sleep 60
  done
  local R L
  R=$(sshp "du -sb $ROOT/$S | cut -f1"); L=$(du -sb "$LOCAL/$S" | cut -f1)
  if [ -n "$R" ] && [ "$R" = "$L" ]; then log "step $S: verified ($L bytes)"; return 0; fi
  log "step $S: SIZE MISMATCH local=$L remote=$R (keeping pod copy, will retry)"; return 1
}

log "=== streaming $CFG -> $LOCAL (steps: ${STEPS[*]}) ==="
ALL_DONE=0
while [ "$ALL_DONE" = 0 ]; do
  TRAINING=$(sshp 'tmux ls 2>/dev/null | grep -c train' || echo 0)
  ALL_DONE=1
  for S in "${STEPS[@]}"; do
    if have "$S"; then
      # already on Elements: drop the pod copy once the syncer has released the local one
      if [ -n "$(sshp "test -d $VOL/$S && echo y")" ] && [ -z "$(sshp "test -d $LOCALDISK/$S && echo y")" ]; then
        sshp "rm -rf $VOL/$S" && log "step $S: removed from volume (already safe on Elements)"
      fi
      continue
    fi
    ALL_DONE=0
    # a step is ready when it is on the volume, fully copied (no .copying / orbax tmp dirs)
    READY=$(sshp "test -d $VOL/$S && test -f $VOL/$S/params/manifest.ocdbt && ! ls -d $VOL/$S.copying $VOL/$S/*orbax-checkpoint-tmp* >/dev/null 2>&1 && echo y")
    if [ -n "$READY" ]; then
      if fetch_one "$S" "$VOL"; then
        if [ -z "$(sshp "test -d $LOCALDISK/$S && echo y")" ]; then
          sshp "rm -rf $VOL/$S" && log "step $S: removed from volume"
        else
          log "step $S: on Elements; volume copy kept until the syncer releases /root/ckpt/$S"
        fi
      fi
    fi
  done
  [ "$ALL_DONE" = 1 ] && break
  # training finished and nothing left to wait for? check whether the final step only exists locally
  if [ "$TRAINING" = 0 ]; then
    LAST=${STEPS[${#STEPS[@]}-1]}
    if ! have "$LAST" && [ -n "$(sshp "test -d $LOCALDISK/$LAST && echo y")" ]; then
      log "training over; final step $LAST is only on the container-local disk — fetching from there"
      fetch_one "$LAST" "$LOCALDISK" || true
    fi
  fi
  sleep 120
done

log "all steps on Elements: $(ls "$LOCAL" | tr '\n' ' ')"
# everything is verified; if training has ended, stop the syncer and clear the last pod copies
if [ -z "$(sshp 'tmux ls 2>/dev/null | grep train')" ]; then
  sshp "pkill -f 'ckpt_sync.py .*$CFG' || true"
  sleep 3
  sshp "rm -rf $LOCALDISK" && log "removed container-local copies"
  sshp "rm -rf $VOL" && log "removed volume copies"
else
  log "training still running; leaving the newest pod copy in place (the syncer would recreate it)"
fi
log "volume now: $(sshp 'du -sh /workspace | cut -f1')  local disk: $(sshp 'du -sh /root/ckpt 2>/dev/null | cut -f1')"
log "done"
