#!/usr/bin/env bash
# Migrated pod (2026-09-20). Usage: pod4_run.sh "cmd1" "cmd2" ...
{
  for cmd in "$@"; do printf '%s\n' "$cmd"; done
  printf 'exit\n'
} | ssh -tt -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=20 \
      m10dlqsjw3op81-64410cb0@ssh.runpod.io 2>&1 \
  | tr -cd '[:print:]\n'
