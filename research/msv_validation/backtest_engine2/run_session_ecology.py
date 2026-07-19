"""
Session Ecology Expansion Test.

Theory: Tokyo H0 (00:00 UTC) proven at 80.2% WR.
London H0 (08:00 UTC) and NY H0 (13:00 UTC) may have similar
participant transition edges with different dynamics.

Each session transition:
- Prior session creates inventory imbalance
- New liquidity population arrives
- Inventory correction happens

We test this with the 120-day M5 dataset (same pipeline as Tokyo H0).

Features:
- Session H0 (first 15-30 min of each major session)
- Top-N movers from the prior session
- ATR filter (high vol bars only)
"""
import sys, os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path
import MetaTrader5 as mt5
import time

# Same import pattern as scalper backtest (Tokyo H0 engine)
project_root = str(Path(__file__).resolve().parents[3])
os.chdir(project_root)
cd_root = os.path.join(project_root, "currency_decomposition")
sys.path.insert(0, cd_root)

from config.settings import BASE_CURRENCY_MAP
ALL_PAIRS = list(BASE_CURRENCY_MAP.keys())[:15]

print("=" * 70)
print("SESSION ECOLOGY EXPANSION")
print("=" * 70)
print(f"Pairs: {ALL_PAIRS}")
print()

# Load 120 days M5 data from MT5
for attempt in range(3):
    init = mt5.initialize()
    if init:
        break
    time.sleep(1)

if not mt5.terminal_info().connected:
    print("MT5 not connected")
    mt5.shutdown()
    sys.exit(1)

end = datetime.now()
start = end - timedelta(days=120)

all_data = {}
for pair in ALL_PAIRS:
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
    if rates is not None and len(rates) > 0:
        all_data[pair] = rates
    else:
        print(f"WARN: No data for {pair}")

print(f"Loaded {len(all_data)} pairs, "
      f"{min(len(v) for v in all_data.values())} bars (M5)")
print()

# Align to common time range
min_bars = min(len(v) for v in all_data.values())
N = min_bars

# Pre-compute close prices for all pairs
closes = np.zeros((N, len(ALL_PAIRS)))
times_m5 = np.zeros(N, dtype='u8')
for pi, pair in enumerate(ALL_PAIRS):
    bars = all_data[pair]
    for i in range(N):
        closes[i, pi] = bars[i][3]  # close
        times_m5[i] = bars[i][0]

# Sessions to test
SESSIONS = {
    "TOKYO_H0": {"hour": 0, "prior_window": 3, "description": "Tokyo Open (00:00 UTC)"},
    "LONDON_H0": {"hour": 8, "prior_window": 3, "description": "London Open (08:00 UTC)"},
    "NY_H0": {"hour": 13, "prior_window": 3, "description": "NY Open (13:00 UTC)"},
    "SUNDAY_OPEN": {"hour": 22, "prior_window": 3, "description": "Sunday Open (22:00 UTC)"},
}

# For each session, test mean reversion of top-N prior-session movers
# (same mechanism as Tokyo H0)
HOLD_BARS = 3  # M5 bars hold = 15 minutes
LOOKBACK_BARS = 3  # M5 bars lookback = 15 minutes
TOP_N_VALUES = [3, 5, 7]
VOL_FILTER = True

all_session_results = []

for session_name, session_info in SESSIONS.items():
    target_hour = session_info["hour"]
    prior_window = session_info["prior_window"]
    desc = session_info["description"]
    
    print(f"{'─'*60}")
    print(f"  {session_name}: {desc}")
    print(f"{'─'*60}")
    
    for top_n in TOP_N_VALUES:
        trades = []
        
        for i in range(LOOKBACK_BARS + HOLD_BARS + 1, N):
            # Check if current bar is in the target hour
            ts = datetime.fromtimestamp(times_m5[i], tz=timezone.utc)
            h = ts.hour
            
            if h != target_hour:
                continue
            
            # Skip Sunday check for non-Sunday sessions
            if session_name == "SUNDAY_OPEN" and ts.weekday() != 6:
                continue
            if session_name != "SUNDAY_OPEN" and ts.weekday() in (5, 6):
                continue
            
            # Compute prior-session displacements (last LOOKBACK_BARS bars before H0)
            prior_close = closes[i - LOOKBACK_BARS]
            current_close = closes[i]
            
            moves = []
            total_vol = 0.0
            for pi, pair in enumerate(ALL_PAIRS):
                pct_move = (current_close[pi] - prior_close[pi]) / prior_close[pi]
                # ATR proxy
                atr_val = 0.0
                for j in range(1, LOOKBACK_BARS + 1):
                    hi = float(all_data[pair][i - j][1])
                    lo = float(all_data[pair][i - j][2])
                    pc = float(all_data[pair][i - j - 1][3])
                    tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
                    atr_val += tr / float(all_data[pair][i - j][3])
                atr_val /= LOOKBACK_BARS
                total_vol += atr_val
                moves.append((pi, pair, pct_move, atr_val))
            
            avg_vol = total_vol / len(ALL_PAIRS)
            
            # Sort by absolute move
            moves.sort(key=lambda x: abs(x[2]), reverse=True)
            top_movers = moves[:top_n]
            
            for pi, pair, pct_move, atr_val in top_movers:
                # Vol filter: skip if pair volatility is below median
                if VOL_FILTER and atr_val < avg_vol:
                    continue
                
                # Determine direction: extreme movers should revert
                # If pair moved UP in prior session, predict DOWN (mean reversion)
                direction = -1 if pct_move > 0 else 1
                
                # Hold for HOLD_BARS bars
                entry_price = float(all_data[pair][i][3])  # open of current bar
                exit_idx = min(i + HOLD_BARS, N - 1)
                exit_price = float(all_data[pair][exit_idx][3])
                
                pnl_pct = (exit_price - entry_price) / entry_price
                
                if direction == 1:  # long
                    win = pnl_pct > 0
                else:  # short
                    win = pnl_pct < 0
                
                bp_move = abs(exit_price - entry_price)
                # Convert to basis points
                if pair in ("USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CHFJPY", "NZDJPY"):
                    bp_move *= 100  # 2-decimal -> basis points
                else:
                    bp_move *= 10000  # 4-decimal -> basis points
                
                trades.append({
                    "pair": pair, "session": session_name,
                    "direction": direction, "pct_move": pct_move,
                    "pnl_pct": pnl_pct, "win": win,
                    "bp_move": bp_move, "top_n": top_n,
                    "hour": h, "weekday": ts.weekday(),
                })
        
        if not trades:
            print(f"  top_n={top_n}: 0 trades")
            continue
        
        df = pd.DataFrame(trades)
        total = len(df)
        wins = df["win"].sum()
        wr = wins / total
        avg_bp = df["bp_move"].mean()
        
        all_session_results.append({
            "session": session_name, "top_n": top_n,
            "wr": wr, "trades": total, "avg_bp": avg_bp
        })
        
        # Per-hour breakdown within the session
        hour_breakdown = df.groupby("hour")["win"].agg(["count", "mean"])
        print(f"  top_n={top_n}: WR={wr:.1%}({total}) avg_move={avg_bp:.1f}bp")
        for hr, row in hour_breakdown.iterrows():
            if row["count"] >= 20:
                print(f"    h{hr:02d}: WR={row['mean']:.1%}({int(row['count'])})")
    
    print()

print("=" * 70)
print("SESSION ECOLOGY SUMMARY")
print("=" * 70)
print()
all_session_results.sort(key=lambda r: (-r["wr"], -r["trades"]))
for r in all_session_results:
    bar = "█" * int(r["wr"] * 40)
    print(f"  {r['wr']:.1%} |{bar:<40}| {r['session']} top_n={r['top_n']} ({r['trades']} trades, {r['avg_bp']:.1f}bp avg)")

print()
print("=" * 70)
print("COMPARISON: Tokyo H0 (from Engine 1 evidence)")
print("=" * 70)
print()
print("  Tokyo H0 (00:00 UTC): 81.4% WR, 720 trades, +4.0bp avg")
print("  Mechanism: Friday close → Monday Asia capital rotation")
print("  Top 3 by 15min move, ATR filter (top 33% vol)")
print()

mt5.shutdown()
