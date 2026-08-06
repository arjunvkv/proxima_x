#!/usr/bin/env bash
# Tokyo_H0 daily live firing wrapper for the Hermes scheduler.
# Launches the validated-curve runner attach-only at the FTMO demo in EXECUTE
# + MANAGE mode. It sits alive across 00:00 UTC, buys the top-5 at the 00:10
# fill bar, manages the 12-bar hold, then exits.
set -u
cd /c/Trading/Proxima_X || exit 1
unset PYTHONPATH
PROJ_VENV="./.venv/Scripts/python.exe"
if [ ! -x "$PROJ_VENV" ]; then
  echo "TOKYO_H0_ERROR: project venv python not found at $PROJ_VENV" >&2
  exit 1
fi
"$PROJ_VENV" scripts/run_tokyo_h0_live.py --execute --manage
rc=$?
echo "TOKYO_H0_EXIT rc=$rc"
exit $rc