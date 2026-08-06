#!/bin/bash
# Compile Ultra_Monster_MT5_v107.mq5 via MetaEditor64 on Wine
# Uses Windows-style path as required by /compile flag

META_EXE='C:\Program Files\FTMO Global Markets MT5 Terminal\MetaEditor64.exe'
MQ5_WIN='C:\Program Files\FTMO Global Markets MT5 Terminal\MQL5\Experts\Ultra_Monster_MT5_v107.mq5'
EX5_PATH='/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/Ultra_Monster_MT5_v107.ex5'
LOG_PATH='/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/Ultra_Monster_MT5_v107.log'

echo "=== Compiling Ultra Monster v107 ==="
echo "MQ5 (Win path): $MQ5_WIN"

export DISPLAY=:0
export WINEPREFIX=/home/ubuntu/.wine

# Kill any stale MetaEditor instances
pkill -f MetaEditor64 2>/dev/null || true
sleep 2

# Launch MetaEditor with Windows-style /compile path
wine "$META_EXE" "/compile:$MQ5_WIN" /log &
MPID=$!
echo "MetaEditor PID: $MPID"

# Wait up to 30 seconds for .ex5 to appear
for i in $(seq 1 30); do
    sleep 1
    if [ -f "$EX5_PATH" ]; then
        MOD=$(stat -c %Y "$EX5_PATH")
        NOW=$(date +%s)
        AGE=$((NOW - MOD))
        if [ $AGE -lt 60 ]; then
            echo ""
            echo "✅ SUCCESS: Ultra_Monster_MT5_v107.ex5 compiled!"
            echo "   File size: $(stat -c %s "$EX5_PATH") bytes"
            echo "   Modified : $(stat -c %y "$EX5_PATH")"
            echo ""
            echo "--- Compile Log ---"
            cat "$LOG_PATH" 2>/dev/null || echo "(no log file)"
            exit 0
        fi
    fi
    printf "."
done

echo ""
echo "--- MetaEditor stdout after 30s ---"
echo "--- Log file ---"
cat "$LOG_PATH" 2>/dev/null || echo "(no log file)"

if [ -f "$EX5_PATH" ]; then
    echo "⚠️  .ex5 exists but may be stale ($(stat -c %y "$EX5_PATH"))"
else
    echo "❌ .ex5 still not found — compile may have failed"
    echo "--- Files in Experts folder ---"
    ls -la '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/' | grep -i ultra
fi
