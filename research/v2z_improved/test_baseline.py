"""Quick test: baseline V2+z on EURAUD Forward period, with fixed entry timing."""
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime

Z_THRESHOLD = 3.5
STOP_A = 3.0
TRIG_A = 1.0
GAP_A = 0.05
BASE_LOT = 0.75
COMMISSION_PER_LOT = 5.0
MAX_HOLD_BARS = 54
Z_WINDOW = 50
ATR_PERIOD = 20
SLIPPAGE_PIPS = 1.0
PAIR = "EURAUD"

def run(bars):
    trades = []
    pos = 0; entry_p = 0.0; entry_i = 0; best_p = 0.0; stop_p = 0.0; held = 0

    atr = bars['high'].sub(bars['low']).rolling(ATR_PERIOD).mean()
    rets = bars['close'].diff()
    zs = (rets - rets.rolling(Z_WINDOW).mean()) / rets.rolling(Z_WINDOW).std(ddof=1)

    slip = SLIPPAGE_PIPS * 0.0001 / 10
    n = len(bars)
    start = max(Z_WINDOW, ATR_PERIOD) + 5

    i = start
    while i < n - 1:
        hour = bars.iloc[i]['time'].hour
        if hour >= 7:
            i += 1; continue

        # Manage position
        if pos != 0:
            held += 1
            atr_v = atr.iloc[i]
            if pd.notna(atr_v) and atr_v > 0:
                tg = TRIG_A * atr_v; gp = GAP_A * atr_v
                hi = bars.iloc[i]['high']; lo = bars.iloc[i]['low']
                if pos == 1:
                    # Check stop FIRST (low), then trail update (high)
                    if lo <= stop_p:
                        raw = (stop_p - entry_p) * BASE_LOT * 100000
                        trades.append(dict(dir='LONG', entry=entry_p, exit=stop_p, raw=raw,
                                           comm=BASE_LOT*COMMISSION_PER_LOT, net=raw-BASE_LOT*COMMISSION_PER_LOT,
                                           reason='stp', held=held))
                        pos = 0; i += 1; continue
                    if hi > best_p:
                        best_p = hi
                        if best_p - entry_p > tg:
                            ns = best_p - gp
                            if ns > stop_p: stop_p = ns
                else:
                    # Check stop FIRST (high), then trail update (low)
                    if hi >= stop_p:
                        raw = (entry_p - stop_p) * BASE_LOT * 100000
                        trades.append(dict(dir='SHORT', entry=entry_p, exit=stop_p, raw=raw,
                                           comm=BASE_LOT*COMMISSION_PER_LOT, net=raw-BASE_LOT*COMMISSION_PER_LOT,
                                           reason='stp', held=held))
                        pos = 0; i += 1; continue
                    if lo < best_p:
                        best_p = lo
                        if entry_p - best_p > tg:
                            ns = best_p + gp
                            if ns < stop_p: stop_p = ns

            if held >= MAX_HOLD_BARS:
                ep = bars.iloc[i]['close']
                raw = (ep - entry_p) * BASE_LOT * 100000 if pos == 1 else (entry_p - ep) * BASE_LOT * 100000
                trades.append(dict(dir='LONG' if pos==1 else 'SHORT', entry=entry_p, exit=ep, raw=raw,
                                   comm=BASE_LOT*COMMISSION_PER_LOT, net=raw-BASE_LOT*COMMISSION_PER_LOT,
                                   reason='exp', held=held))
                pos = 0
            i += 1; continue

        # Entry check
        z_v = zs.iloc[i]; atr_v = atr.iloc[i]
        if pd.isna(z_v) or pd.isna(atr_v) or atr_v <= 0:
            i += 1; continue
        if abs(z_v) < Z_THRESHOLD:
            i += 1; continue

        direction = -1 if z_v > 0 else 1
        # Enter at NEXT bar open (FIX: removes look-ahead bias)
        entry_p = bars.iloc[i + 1]['open'] + (slip * direction * -1)
        s = STOP_A * atr_v
        stop_p = entry_p - s if direction == 1 else entry_p + s
        best_p = entry_p; held = 0; pos = direction; entry_i = i + 1
        i += 1

    return trades

if not mt5.initialize():
    print("MT5 init failed"); exit()

print(f"Downloading {PAIR} M1 Jun8-Jul10...")
r = mt5.copy_rates_range(PAIR, mt5.TIMEFRAME_M1, datetime(2026,6,8), datetime(2026,7,10))
mt5.shutdown()
if r is None: print("No data"); exit()

bars = pd.DataFrame(r)
bars['time'] = pd.to_datetime(bars['time'], unit='s')
print(f"Bars: {len(bars):,} ({len(bars)/60/24:.1f} days)")

trades = run(bars)
df = pd.DataFrame(trades)
print(f"\nTrades: {len(df)}")
wins = df[df['net'] > 0]; losses = df[df['net'] < 0]
wr = len(wins)/len(df)*100 if len(df) else 0
print(f"WR: {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
print(f"Gross PnL: ${df['raw'].sum():+.2f}")
print(f"Commission: ${df['comm'].sum():.2f}")
print(f"Net PnL: ${df['net'].sum():+.2f}")
pf = abs(wins['raw'].sum() / losses['raw'].sum()) if len(losses) and losses['raw'].sum() != 0 else float('inf')
print(f"Profit Factor: {pf:.3f}")
print(f"Avg Win: ${wins['net'].mean():+.2f}" if len(wins) else "Avg Win: N/A")
print(f"Avg Loss: ${losses['net'].mean():+.2f}" if len(losses) else "Avg Loss: N/A")
print(f"Payoff ratio: {abs(wins['net'].mean()/losses['net'].mean()):.2f}" if len(wins) and len(losses) else "N/A")
print(f"Hold: {df['held'].mean():.0f} bars avg")

# Compare to real MT5: +$313 on 215 trades for EURAUD forward
print(f"\nReal MT5 Forward (Jun8-Jul25): +$313 on 215 trades")
print(f"Our sim (Jun8-Jul10): ${df['net'].sum():+.0f} on {len(df)} trades")
