#!/usr/bin/env bash
# User (2026-09-29 23:50): stop pod 4p2ln4o04a7u6k after all task-12 checkpoints are downloaded.
# Stops only when: our 5 checkpoints are verified on Elements AND no other transfer (rsync/scp sender)
# is still reading from the pod AND nothing is training/evaluating there. Never deletes anything.
set -u
POD=root@185.216.21.214; PORT=14097; POD_ID=4p2ln4o04a7u6k
FLOG=~/winner-eval/fetch_task12.log
LOG=~/winner-eval/stop_pod_task12.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
sshp(){ ssh -o ConnectTimeout=25 -o BatchMode=yes -p $PORT $POD "$@" </dev/null 2>/dev/null; }
log "waiting for all 5 task-12 checkpoints to be verified on Elements"
until grep -q "done:" "$FLOG"; do sleep 60; done
n=$(grep -c -E "step (9999|8000|6000|4000|2000): verified" "$FLOG")
if [ "$n" -lt 5 ] || grep -q MISMATCH "$FLOG"; then log "NOT STOPPING: only $n/5 verified or a size mismatch — see $FLOG"; exit 1; fi
log "all 5 checkpoints verified on Elements ($(du -sh /media/rena/Elements/openpi_checkpoints/ft_task12 | cut -f1))"
for i in $(seq 1 72); do   # wait up to 6 h for other people's transfers / jobs on the pod
  busy=$(sshp 'pgrep -af "rsync --server|scp -f|sftp-server|train\.py|serve_b1k|omnigibson" | grep -v pgrep | wc -l')
  [ "${busy:-1}" = 0 ] && break
  [ "$i" = 1 ] && log "pod still busy with $busy other transfer/job process(es); waiting for them to finish"
  sleep 300
done
if [ "${busy:-1}" != 0 ]; then log "NOT STOPPING: pod still busy after 6 h ($busy processes)"; exit 2; fi
log "pod idle; stopping $POD_ID"
out=$(sshp 'export RUNPOD_API_KEY=$(tr "\0" "\n" < /proc/1/environ | grep "^RUNPOD_API_KEY=" | cut -d= -f2-); runpodctl stop pod '"$POD_ID" 2>&1); log "runpodctl: ${out:-<no output>}"
for i in 1 2 3 4 5 6; do
  sleep 30
  if ! ssh -o ConnectTimeout=15 -o BatchMode=yes -p $PORT $POD true </dev/null 2>/dev/null; then log "pod unreachable: STOPPED"; exit 0; fi
done
log "POD STILL REACHABLE after 3 min — stop it at console.runpod.io (pod $POD_ID)"; exit 3
