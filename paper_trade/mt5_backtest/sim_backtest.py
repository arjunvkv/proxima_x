"""
Python backtest simulation of V2z CPPF strategy.
Downloads M1 data from MT5 and replicates EA logic exactly.
"""
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "EURAUD"
FROM_DATE = datetime.strptime(sys.argv[2], '%Y-%m-%d') if len(sys.argv) > 2 else datetime(2026, 6, 8)
TO_DATE = datetime.strptime(sys.argv[3], '%Y-%m-%d') if len(sys.argv) > 3 else datetime(2026, 7, 25)

# Strategy parameters (override via command line)
Z_THRESHOLD = float(sys.argv[4]) if len(sys.argv) > 4 else 3.5
STOP_A = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
TRIG_A = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
GAP_A = float(sys.argv[7]) if len(sys.argv) > 7 else 0.05
BASE_LOT = float(sys.argv[8]) if len(sys.argv) > 8 else 0.5
COMMISSION_PER_LOT = 5.0
MAX_HOLD_BARS = 54
Z_WINDOW = 50
ATR_PERIOD = 20
MAX_SPREAD_PIPS = 5.0
SLIPPAGE_PIPS = 1.0
TRADE_START_HOUR = 0
TRADE_END_HOUR = 7

# Currency conversion for cross pairs
def pip_value_usd(symbol):
    pip_map = {
        "EURAUD": 6.70, "EURNZD": 6.10, "GBPAUD": 6.10,
        "GBPCAD": 7.50, "GBPNZD": 5.60, "AUDNZD": 5.60,
    }
    return pip_map.get(symbol, 10.0)

PIP_VALUE = pip_value_usd(SYMBOL)

def compute_atr(bars, period=20):
    """SMA of high-low ranges over period bars"""
    hl = bars['high'] - bars['low']
    return hl.rolling(period).mean()

def compute_zscore(returns, window=50):
    """Rolling z-score of last return"""
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1)
    z = (returns - mean) / std
    return z

def run_backtest(bars):
    trades = []
    pos = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
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

        # Manage existing position
        if pos != 0:
            bars_held += 1
            atr_val = atr.iloc[i]
            if pd.isna(atr_val) or atr_val <= 0:
                continue

            tg = TRIG_A * atr_val
            gp = GAP_A * atr_val

            if pos == 1:
                high = bar['high']
                low = bar['low']
                # Update best price
                if high > best_price:
                    best_price = high
                    if best_price - entry_price > tg:
                        ns = best_price - gp
                        if ns > current_stop:
                            current_stop = ns
                # Check stop hit
                if low <= current_stop:
                    exit_price = current_stop
                    raw_pnl = (exit_price - entry_price) * BASE_LOT * 100000
                    commission = BASE_LOT * COMMISSION_PER_LOT
                    pnl = raw_pnl - commission
                    trades.append({'entry_idx': entry_idx, 'exit_idx': i, 'direction': 'LONG',
                                   'entry': entry_price, 'exit': exit_price,
                                   'raw_pnl': raw_pnl, 'commission': commission, 'pnl': pnl,
                                   'reason': 'stop', 'bars_held': bars_held})
                    pos = 0
                    continue
            else:
                high = bar['high']
                low = bar['low']
                if low < best_price:
                    best_price = low
                    if entry_price - best_price > tg:
                        ns = best_price + gp
                        if ns < current_stop:
                            current_stop = ns
                if high >= current_stop:
                    exit_price = current_stop
                    raw_pnl = (entry_price - exit_price) * BASE_LOT * 100000
                    commission = BASE_LOT * COMMISSION_PER_LOT
                    pnl = raw_pnl - commission
                    trades.append({'entry_idx': entry_idx, 'exit_idx': i, 'direction': 'SHORT',
                                   'entry': entry_price, 'exit': exit_price,
                                   'raw_pnl': raw_pnl, 'commission': commission, 'pnl': pnl,
                                   'reason': 'stop', 'bars_held': bars_held})
                    pos = 0
                    continue

            # Max hold expiry
            if bars_held >= MAX_HOLD_BARS:
                exit_price = bar['close']
                if pos == 1:
                    raw_pnl = (exit_price - entry_price) * BASE_LOT * 100000
                else:
                    raw_pnl = (entry_price - exit_price) * BASE_LOT * 100000
                commission = BASE_LOT * COMMISSION_PER_LOT
                pnl = raw_pnl - commission
                trades.append({'entry_idx': entry_idx, 'exit_idx': i, 'direction': 'LONG' if pos == 1 else 'SHORT',
                               'entry': entry_price, 'exit': exit_price,
                               'raw_pnl': raw_pnl, 'commission': commission, 'pnl': pnl,
                               'reason': 'expiry', 'bars_held': bars_held})
                pos = 0
                continue

        # Check entry (only on flat)
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
            if direction == 1:
                current_stop = entry_price - s
            else:
                current_stop = entry_price + s
            pos = direction
            entry_idx = i
            best_price = entry_price
            bars_held = 0

    return trades

def summary(trades, label):
    if not trades:
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")
        print("NO TRADES")
        return {'trades': 0, 'net': 0.0}

    df = pd.DataFrame(trades)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Total trades: {len(df)}")
    print(f"Wins: {len(wins)} ({len(wins)/len(df)*100:.1f}%)")
    print(f"Losses: {len(losses)} ({len(losses)/len(df)*100:.1f}%)")
    print(f"Gross profit: ${wins['pnl'].sum():+.2f}" if len(wins) else "Gross profit: $0")
    print(f"Gross loss: ${losses['pnl'].sum():+.2f}" if len(losses) else "Gross loss: $0")
    print(f"Net PnL: ${df['pnl'].sum():+.2f}")
    print(f"Commission total: ${df['commission'].sum():.2f}")
    pf = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) and losses['pnl'].sum() != 0 else float('inf')
    print(f"Profit factor: {pf:.3f}")
    avg_win = wins['pnl'].mean() if len(wins) else 0
    avg_loss = losses['pnl'].mean() if len(losses) else 0
    print(f"Avg win: ${avg_win:+.2f}, Avg loss: ${avg_loss:+.2f}")
    print(f"Largest win: ${df['pnl'].max():+.2f}, Largest loss: ${df['pnl'].min():+.2f}")

    # Direction bias
    longs = df[df['direction'] == 'LONG']
    shorts = df[df['direction'] == 'SHORT']
    print(f"\nLong: {len(longs)} trades, ${longs['pnl'].sum():+.2f} net")
    if len(longs): print(f"  WR: {len(longs[longs['pnl']>0])/len(longs)*100:.1f}%")
    print(f"Short: {len(shorts)} trades, ${shorts['pnl'].sum():+.2f} net")
    if len(shorts): print(f"  WR: {len(shorts[shorts['pnl']>0])/len(shorts)*100:.1f}%")

    # Hold time analysis
    print(f"\nAvg hold: {df['bars_held'].mean():.0f} bars, Max hold: {df['bars_held'].max()}")
    print(f"Stop exits: {len(df[df['reason']=='stop'])}")
    print(f"Expiry exits: {len(df[df['reason']=='expiry'])}")

    return {'trades': len(df), 'net': df['pnl'].sum(), 'pf': pf,
            'wr': len(wins)/len(df)*100, 'avg_win': avg_win, 'avg_loss': avg_loss,
            'max_win': df['pnl'].max(), 'max_loss': df['pnl'].min()}

if __name__ == '__main__':
    print(f"Downloading {SYMBOL} M1 data from {FROM_DATE.date()} to {TO_DATE.date()}...")
    if not mt5.initialize():
        print("MT5 init failed!"); sys.exit(1)

    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, FROM_DATE, TO_DATE)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("No data received!"); sys.exit(1)

    bars = pd.DataFrame(rates)
    bars['time'] = pd.to_datetime(bars['time'], unit='s')
    print(f"Got {len(bars)} bars ({len(bars)/60/24:.1f} days)")

    label = (f"Iteration: z={Z_THRESHOLD} stop={STOP_A} trail={TRIG_A} "
             f"gap={GAP_A} lot={BASE_LOT} | {SYMBOL} {FROM_DATE.date()}-{TO_DATE.date()}")
    trades = run_backtest(bars)
    s = summary(trades, label)
