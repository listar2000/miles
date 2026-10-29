#!/usr/bin/env bash
# Keeps review_daemon.py running on the reviewer box, which has no systemd: restarts it after any exit
# (including its own exit when main brings a new version), and re-reads this script each time.
# Safe to start repeatedly, e.g. from a cron watchdog every few minutes: a second copy exits at once.
#   setsid nohup .claude/skills/miles-review/supervisor.sh [daemon args] >/dev/null 2>&1 &
set -u
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")"
OUT="${MILES_REVIEW_OUT:-$HOME/miles-review}"
mkdir -p "$OUT"
exec 9>"$OUT/supervisor.lock"
flock -n 9 || exit 0
echo "$(date '+%F %T') starting review_daemon.py $*" >>"$OUT/supervisor.log"
python3 review_daemon.py --out "$OUT" "$@" >>"$OUT/supervisor.log" 2>&1
rc=$?
echo "$(date '+%F %T') exited $rc" >>"$OUT/supervisor.log"
sleep 30
exec "$SELF" "$@"
