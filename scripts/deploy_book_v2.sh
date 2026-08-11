#!/usr/bin/env bash
# deploy_book_v2.sh — book v2 live deploy (tokyo removed, gold_s3 added).
# Run from ANY machine with IPv4 reachability to the VPS (this dev box's
# IPv4 is currently down; run from another connection if needed).
#
#   bash scripts/deploy_book_v2.sh
#
set -euo pipefail

K="${1:-$HOME/Downloads/id_rsa_proxima.key}"
HOST="ubuntu@140.245.234.92"
REPO="/home/ubuntu/proxima_x"

echo "== 1/4 pull book v2 =="
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 22 -i "$K" "$HOST" \
  "cd $REPO && git pull --ff-only && echo '--- STRATS check (expect gold_s3, NO tokyo):' && grep -E 'gold_s3|tokyo' scripts/run_core_book_live.py | head -4"

echo "== 2/4 confirm no tokyo entry remains =="
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 22 -i "$K" "$HOST" \
  "cd $REPO && if grep -q '\"tokyo\"' scripts/run_core_book_live.py; then echo 'ABORT: tokyo still present'; exit 1; else echo 'tokyo absent — good'; fi"

echo "== 3/4 restart daemon =="
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 22 -i "$K" "$HOST" \
  "tmux kill-session -t corebook 2>/dev/null; sleep 1; \
   tmux new-session -d -s corebook \
     -e PROXIMA_ROOT=$REPO \
     -e MT5_PATH='/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/terminal64.exe' \
     -e WINEPREFIX=/home/ubuntu/.wine \
     'wine /home/ubuntu/.wine/drive_c/Python310/python.exe -u $REPO/scripts/run_core_book_live.py --execute --manage --daemon --init-balance 25000 2>&1 | tee $REPO/logs/core_book_daemon.log'; \
   sleep 12; echo '--- tmux:'; tmux ls; echo '--- proc:'; ps aux | grep 'python.exe -u' | grep -v grep | awk '{print \$2, \$9}'; \
   echo '--- log head:'; head -8 $REPO/logs/core_book_daemon.log"

echo "== 4/4 verify clean start =="
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 22 -i "$K" "$HOST" \
  "tail -15 $REPO/logs/core_book_daemon.log; echo '--- traceback check:'; if grep -q Traceback $REPO/logs/core_book_daemon.log; then echo 'TRACEBACK PRESENT — CHECK POSITIONS'; else echo 'no traceback — clean'; fi"

echo "DEPLOY DONE — verify a gold fire at the next exhaustion window (2-3 or 16-18 server) and journal rows in core_book_trades.jsonl"
