"""Diagnose why no trades are being placed — check MT5 state, bars, graph."""
import time, sys
import MetaTrader5 as mt5
from collections import deque, defaultdict
import numpy as np

if not mt5.initialize():
    print("MT5 init failed", file=sys.stderr)
    sys.exit(1)

# 1. Check our positions
positions = mt5.positions_get()
cd_positions = [p for p in (positions or []) if 236000 <= p.magic < 236200]
print(f"CD positions: {len(cd_positions)}", file=sys.stderr)
for p in cd_positions:
    print(f"  {p.symbol} {'BUY' if p.type==0 else 'SELL'} pnl={p.profit:.2f}", file=sys.stderr)

# 2. Check M1 bars per symbol — simulate what TickStore sees
from config.settings import SYMBOLS
bar_counts = {}
for sym in SYMBOLS:
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, 2)
    if rates is not None and len(rates) > 0:
        bar_counts[sym] = len(rates)
    else:
        bar_counts[sym] = 0

available = {s: c for s, c in bar_counts.items() if c > 0}
missing = {s: c for s, c in bar_counts.items() if c == 0}
print(f"\nSymbols with M1 data: {len(available)}", file=sys.stderr)
print(f"Symbols missing M1: {len(missing)}", file=sys.stderr)
for s, c in sorted(available.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {s}: {c} bars", file=sys.stderr)

# 3. Check what TickStore would calculate (simulate calculate_returns)
# With window=15, need 15 bars
# But the poll_ticks uses copy_rates_from_pos(sym, M1, 0, 2) which returns complete bars
# Each call returns the latest 2 complete M1 bars
# With 5s polling, the bars may accumulate in TickStore._bars

print(f"\nTickStore._bars accumulates Ticks (not raw bars)", file=sys.stderr)
print(f"Each poll adds a Tick per symbol if bar close changed", file=sys.stderr)
print(f"With 5s polling, M1 bar changes detected ~every 60s", file=sys.stderr)
print(f"After 6 min runtime: ~6 Ticks per symbol worst case", file=sys.stderr)
print(f"window=15 requires 15 Ticks per symbol", file=sys.stderr)
print(f"-> calculate_returns returns all zeros", file=sys.stderr)
print(f"-> _active_pair_count = 0 < MIN_SOLVE_PAIRS (10)", file=sys.stderr)
print(f"-> execution_allowed = False", file=sys.stderr)
print(f"-> No trades until 15 min runtime", file=sys.stderr)

mt5.shutdown()
