#!/bin/zsh
# run_mac.sh <exe> <dump.bmp> <secondes> [VAR=valeur ...]
# Lance un seul process, attend, l'arrête et attend sa fin ; vérifie qu'il n'en reste aucun.
exe=$1; dump=$2; secs=$3; shift 3
rm -f "$dump" "$dump.partial"
env "$@" GPUI_FRAME_DUMP="$dump" "$exe" > "$dump.log" 2>&1 &
pid=$!
sleep "$secs"
if kill -0 $pid 2>/dev/null; then
  kill $pid; wait $pid 2>/dev/null
  echo "arrêté après ${secs}s"
else
  wait $pid; echo "SORTI TOT (code $?)"
fi
pgrep -f "^$exe" >/dev/null && echo "ENCORE VIVANT" || echo "aucun process restant"
[ -f "$dump" ] && echo "dump $(stat -f %z "$dump") octets" || echo "PAS DE DUMP"
grep -v "^\s*$" "$dump.log" | tail -8
