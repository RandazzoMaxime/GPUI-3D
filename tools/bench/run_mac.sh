#!/bin/zsh
# run_mac.sh <exe> <dump.bmp> <seconds> [VAR=value ...]
# Runs a single process, waits, stops it and waits for it to exit; checks none is left.
exe=$1; dump=$2; secs=$3; shift 3
rm -f "$dump" "$dump.partial"
env "$@" GPUI_FRAME_DUMP="$dump" "$exe" > "$dump.log" 2>&1 &
pid=$!
sleep "$secs"
if kill -0 $pid 2>/dev/null; then
  kill $pid; wait $pid 2>/dev/null
  echo "stopped after ${secs}s"
else
  wait $pid; echo "EXITED EARLY (code $?)"
fi
pgrep -f "^$exe" >/dev/null && echo "STILL RUNNING" || echo "no process left"
[ -f "$dump" ] && echo "dump $(stat -f %z "$dump") bytes" || echo "NO DUMP"
grep -v "^\s*$" "$dump.log" | tail -8
