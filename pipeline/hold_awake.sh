#!/usr/bin/env bash
# Keep the laptop awake with the lid closed while any eval runner, eval queue or checkpoint streamer is
# alive; releases itself after. (2026-09-28: the streamer was missing here, so the laptop suspended
# the moment the eval queue ended and checkpoint streaming paused for ~2.5 h.)
while pgrep -f "bash \./(run_[a-z0-9_]+|stream_ckpts)\.sh|/home/rena/winner-eval/run_ft_eval\.py" >/dev/null; do sleep 60; done
