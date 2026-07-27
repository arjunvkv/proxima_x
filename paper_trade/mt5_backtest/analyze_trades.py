"""
Analyze trade details: time-of-day, direction, etc.
"""
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "EURAUD"
FROM_DATE = datetime.strptime(sys.argv[2], '%Y-%m-%d') if len(sys.argv) > 2 else datetime(2026, 6, 8)
TO_DATE = datetime.strptime(sys.argv[3], '%Y-%m-%d') if len(sys.argv) > 3 else datetime(2026, 7, 25)
Z_THRESHOLD = float(sys.argv[4]) if len(sys.argv) > 4 else 3.5
STOP_A = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
TRIG_A = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
GAP_A = float(sys.argv[7]) if len(sys.argv) > 7 else 0.05
BASE_LOT = float(sys.argv[8]) if len(sys.argv) > 8 else 0.5
COMMISSION_PER_LOT = 5.0
MAX_HOLD_BARS = 54
Z_WINDOW = 50
ATR_PERIOD = 20
SLIPPAGE_PIPS = 1.0
TRADE_START_HOUR = 0
TRADE_END_HOUR = 7

PIP_VALUE = {"EURAUD": 6.70, "EURNZD": 6.10, "GBPAUD": 6.10,
              "GBPCAD": 7.50, "GBPNZD": 5.60, "AUDNZD": 5.60}.get(SYMBOL, 10.0)

def compute_atr(bars, period=20):
    hl = bars['high'] - bars['low']
    return hl.rolling(period).mean()

def compute_zscore(returns, window=50):
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1)
    return (returns - mean) / std

def run_backtest(bars):
    trades = []
    pos = 0
    entry_price = 0.0
    entry_idx = 0
    entry_hour = 0
    best_price = 0.0
    current_stop = 0.0
    bars_held = 0

    atr = compute_atr(bars, ATR_PERIOD)
    returns = bars['close'].diff()
    z = compute_zscore(returns, Z_WINDOW)
    require_history = max(Z_WINDOW, ATR_PERIOD) + 5

    for i in range(require_history, len(bars)):
        bar = bars.iloc[i]
        hour = bar['time'].hour

        if pos != 0:
            bars_held += 1
            atr_val = atr.iloc[i]
            if pd.isna(atr_val) or atr_val <= 0:
                continue
            tg = TRIG_A * atr_val
            gp = GAP_A * atr_val

            if pos == 1:
                high, low = bar['high'], bar['low']
                if high > best_price:
                    best_price = high
                    if best_price - entry_price > tg:
                        ns = best_price - gp
                        if ns > current_stop:
                            current_stop = ns
                if low <= current_stop:
                    exit_price, reason = current_stop, 'stop'
                    raw_pnl = (exit_price - entry_price) * BASE_LOT * 100000
                    commission = BASE_LOT * COMMISSION_PER_LOT
                    pnl = raw_pnl - commission
                    trades.append({'entry_time': bars.iloc[entry_idx]['time'], 'exit_time': bar['time'],
                                   'entry_hour': entry_hour, 'exit_hour': hour,
                                   'direction': 'LONG', 'entry': entry_price, 'exit': exit_price,
                                   'raw_pnl': raw_pnl, 'commission': commission, 'pnl': pnl,
                                   'reason': reason, 'bars_held': bars_held})
                    pos = 0
                    continue
            else:
                high, low = bar['high'], bar['low']
                if low < best_price:
                    best_price = low
                    if entry_price - best_price > tg:
                        ns = best_price + gp
                        if ns < current_stop:
                            current_stop = ns
                if high >= current_stop:
                    exit_price, reason = current_stop, 'stop'
                    raw_pnl = (entry_price - exit_price) * BASE_LOT * 100000
                    commission = BASE_LOT * COMMISSION_PER_LOT
                    pnl = raw_pnl - commission
                    trades.append({'entry_time': bars.iloc[entry_idx]['time'], 'exit_time': bar['time'],
                                   'entry_hour': entry_hour, 'exit_hour': hour,
                                   'direction': 'SHORT', 'entry': entry_price, 'exit': exit_price,
                                   'raw_pnl': raw_pnl, 'commission': commission, 'pnl': pnl,
                                   'reason': reason, 'bars_held': bars_held})
                    pos = 0
                    continue

            if bars_held >= MAX_HOLD_BARS:
                exit_price = bar['close']
                raw_pnl = ((exit_price - entry_price) if pos == 1 else (entry_price - exit_price)) * BASE_LOT * 100000
                commission = BASE_LOT * COMMISSION_PER_LOT
                pnl = raw_pnl - commission
                trades.append({'entry_time': bars.iloc[entry_idx]['time'], 'exit_time': bar['time'],
                               'entry_hour': entry_hour, 'exit_hour': hour,
                               'direction': 'LONG' if pos == 1 else 'SHORT',
                               'entry': entry_price, 'exit': exit_price,
                               'raw_pnl': raw_pnl, 'commission': commission, 'pnl': pnl,
                               'reason': 'expiry', 'bars_held': bars_held})
                pos = 0
                continue

        if pos != 0:
            continue
        if hour < TRADE_START_HOUR or hour >= TRADE_END_HOUR:
            continue

        z_val = z.iloc[i]
        atr_val = atr.iloc[i]
        if pd.isna(z_val) or pd.isna(atr_val) or atr_val <= 0:
            continue

        if abs(z_val) >= Z_THRESHOLD:
            direction = -1 if z_val > 0 else 1
            entry_price = bar['close'] + (SLIPPAGE_PIPS * 0.00001 * direction * -1)
            s = STOP_A * atr_val
            current_stop = entry_price - s if direction == 1 else entry_price + s
            pos = direction
            entry_idx = i
            entry_hour = hour
            best_price = entry_price
            bars_held = 0

    return pd.DataFrame(trades) if trades else pd.DataFrame()

if __name__ == '__main__':
    if not mt5.initialize():
        print("MT5 init failed!"); sys.exit(1)
    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, FROM_DATE, TO_DATE)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        print("No data!"); sys.exit(1)

    bars = pd.DataFrame(rates)
    bars['time'] = pd.to_datetime(bars['time'], unit='s')
    trades = run_backtest(bars)

    if len(trades) == 0:
        print("No trades"); sys.exit(0)

    print(f"\n=== {SYMBOL} {FROM_DATE.date()} to {TO_DATE.date()} ===")
    print(f"Params: z={Z_THRESHOLD} stop={STOP_A} trail={TRIG_A} gap={GAP_A} lot={BASE_LOT}")
    print(f"Total trades: {len(trades)}")
    print(f"Net PnL: ${trades['pnl'].sum():+.2f}")

    print(f"\n--- By Entry Hour (0-23 UTC) ---")
    by_hour = trades.groupby('entry_hour').agg(
        trades=('pnl', 'count'), net=('pnl', 'sum'),
        wins=('pnl', lambda x: (x > 0).sum()),
        avg=('pnl', 'mean')
    )
    by_hour['wr'] = (by_hour['wins'] / by_hour['trades'] * 100)
    by_hour = by_hour.sort_values('net', ascending=False)
    for idx, row in by_hour.iterrows():
        print(f"  Hour {idx:2d} | {int(row['trades']):3d} trades | ${row['net']:+.1f} net | {row['wr']:.0f}% WR | ${row['avg']:+.1f} avg")

    print(f"\n--- Direction Bias ---")
    for d in ['LONG', 'SHORT']:
        sub = trades[trades['direction'] == d]
        if len(sub):
            wr = (sub['pnl'] > 0).sum() / len(sub) * 100
            print(f"  {d:5s}: {len(sub):3d} trades, ${sub['pnl'].sum():+7.2f} net, {wr:.0f}% WR")

    print(f"\n--- Exit Reason ---")
    for r in ['stop', 'expiry']:
        sub = trades[trades['reason'] == r]
        if len(sub):
            print(f"  {r:6s}: {len(sub):3d} trades, ${sub['pnl'].sum():+7.2f} net")

    print(f"\n--- Top 5 Winners ---")
    top = trades.nlargest(5, 'pnl')
    for _, t in top.iterrows():
        print(f"  {t['direction']:5s} | entry_h={t['entry_hour']:2d} | ${t['pnl']:+.1f}")

    print(f"\n--- Top 5 Losers ---")
    bot = trades.nsmallest(5, 'pnl')
    for _, t in bot.iterrows():
        print(f"  {t['direction']:5s} | entry_h={t['entry_hour']:2d} | ${t['pnl']:+.1f}")
