"""Comprehensive divergence strategy analysis — daily/weekly/monthly PnL breakdown."""
import MetaTrader5 as mt5
import numpy as np
from time import time
import sys
from collections import defaultdict
import json

ALL_CURRENCIES = ['USD','EUR','JPY','GBP','AUD','NZD','CAD','CHF']
ALL_PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD","USDCHF",
    "EURJPY","GBPJPY","EURGBP","EURAUD","EURCHF","EURCAD","EURNZD",
    "GBPAUD","GBPCAD","GBPCHF","GBPNZD","AUDJPY","AUDCAD","AUDCHF","AUDNZD",
    "NZDJPY","NZDCAD","NZDCHF","CADJPY","CADCHF","CHFJPY"]
PAIR_SPREAD_PIPS = {"AUDUSD":1.5,"EURUSD":1.5,"GBPUSD":2.0,"NZDUSD":2.0,
    "USDCAD":2.0,"USDCHF":2.0,"USDJPY":1.8,"EURJPY":2.5,"GBPJPY":4.0,
    "EURGBP":2.0,"EURAUD":2.5,"EURCHF":2.0,"EURCAD":2.0,"EURNZD":2.5,
    "GBPAUD":2.5,"GBPCAD":2.5,"GBPCHF":3.0,"GBPNZD":3.0,"AUDJPY":2.5,
    "AUDCAD":2.5,"AUDCHF":2.5,"AUDNZD":2.5,"NZDJPY":3.0,"NZDCAD":2.5,
    "NZDCHF":2.5,"CADJPY":2.5,"CADCHF":2.5,"CHFJPY":3.0}
VOL_W = 200; NC = len(ALL_CURRENCIES); NP = len(ALL_PAIRS)
SPREAD_COST = np.array([PAIR_SPREAD_PIPS.get(p, 2.0) * 5.0 for p in ALL_PAIRS])

def base_quote(p):
    for c in ALL_CURRENCIES:
        if p.startswith(c): return c, p[len(c):]
    return None, None

CM = np.zeros((NC, NP))
for ci, c in enumerate(ALL_CURRENCIES):
    for pj, p in enumerate(ALL_PAIRS):
        b, q = base_quote(p)
        if b == c: CM[ci, pj] = 1.0
        elif q == c: CM[ci, pj] = -1.0

t0 = time()
if not mt5.initialize(): exit(1)
for p in ALL_PAIRS: mt5.symbol_select(p, True)

print("Loading...", file=sys.stderr)
PC = {}
for p in ALL_PAIRS:
    r = mt5.copy_rates_from_pos(p, mt5.TIMEFRAME_M1, 0, 50000)
    if r is not None: PC[p] = np.array([float(x[4]) for x in r])
n = min(len(v) for v in PC.values())
P = np.zeros((n, NP))
timestamps = None
for pj, p in enumerate(ALL_PAIRS):
    if p in PC:
        P[:n, pj] = PC[p][:n]
        if timestamps is None:
            r = mt5.copy_rates_from_pos(p, mt5.TIMEFRAME_M1, 0, 50000)
            timestamps = np.array([int(x[0]) for x in r[:n]])

lr = np.diff(np.log(P + 1e-12), axis=0)[:50000]
n = len(lr)
# Align timestamps to returns
ts = timestamps[1:n+1] if timestamps is not None else np.arange(n)
fv = np.std(lr[:VOL_W], axis=0) + 1e-10
W_raw = 1.0 / fv
W = np.zeros((NC, NP))
for ci in range(NC):
    mask = np.abs(CM[ci]) > 0.5
    if mask.sum() > 0:
        w = W_raw.copy(); w[~mask] = 0.0; s = w.sum()
        if s > 0: w /= s
        W[ci] = w
C_MIX = CM * W
curr_rets = lr @ C_MIX.T
PIP_SZ = np.array([0.01 if "JPY" in ALL_PAIRS[pj] else 0.0001 for pj in range(NP)])

CURR_PAIR_MAP = {}
for pj, p in enumerate(ALL_PAIRS):
    b, q = base_quote(p)
    ci1 = ALL_CURRENCIES.index(b)
    ci2 = ALL_CURRENCIES.index(q)
    CURR_PAIR_MAP[(ci1, ci2)] = pj
    CURR_PAIR_MAP[(ci2, ci1)] = pj

CURR_PAIR_SIGN = {}
for ci in range(NC):
    for pj in range(NP):
        CURR_PAIR_SIGN[(ci, pj)] = CM[ci, pj]

print(f"Loaded {n} bars in {time()-t0:.1f}s", file=sys.stderr)

class RingBuf:
    __slots__ = ('size','buf','pos','filled')
    def __init__(self, size):
        self.size = size; self.buf = np.zeros(size)
        self.pos = 0; self.filled = 0
    def append(self, v):
        self.buf[self.pos] = v; self.pos = (self.pos+1) % self.size
        self.filled = min(self.filled+1, self.size)
    def last(self): return self.buf[(self.pos-1) % self.size]
    def stats(self):
        fn = self.filled
        if fn < 5: return 0, 0
        a = self.buf if fn >= self.size else self.buf[:fn]
        return float(np.mean(a)), float(np.std(a))

def run_bt_detailed(z_window, zt, spread_mode, mode="divergence"):
    buf = [RingBuf(z_window) for _ in range(NC)]
    trades = []; opens = {}
    days = n / 1440.0
    trade_id = 0
    hold = 5
    zt_actual = zt

    for i in range(1, n):
        for ci in range(NC):
            buf[ci].append(curr_rets[i-1, ci])

        for ci in list(opens.keys()):
            t = opens[ci]
            if i - t['bi'] >= hold:
                t = opens.pop(ci); pj = t['pj']
                ex = P[i, pj]
                pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
                if spread_mode == "full": pnl -= SPREAD_COST[pj]
                trades.append({
                    **t, 'pnl': round(pnl, 2),
                    'exit_bar': i, 'time': ts[i] if i < len(ts) else 0
                })

        z_scores = np.zeros(NC)
        for ci in range(NC):
            m, s = buf[ci].stats()
            if s < 1e-12: continue
            z_scores[ci] = (buf[ci].last() - m) / s

        sorted_idx = np.argsort(z_scores)

        if mode == "divergence":
            strongest_ci = sorted_idx[-1]
            weakest_ci = sorted_idx[0]
            sz = z_scores[strongest_ci]
            wz = z_scores[weakest_ci]
            if sz < zt_actual or wz > -zt_actual:
                continue
            if strongest_ci in opens or weakest_ci in opens:
                continue

            if abs(sz) >= abs(wz):
                trade_ci = strongest_ci; other_ci = weakest_ci
            else:
                trade_ci = weakest_ci; other_ci = strongest_ci

            pj = _best_pair(trade_ci)
            sg = CURR_PAIR_SIGN.get((trade_ci, pj), 0)
            if sg == 0: continue
            dir = 1 if z_scores[trade_ci] > 0 else -1
            d_star = dir * sg

            other_sg = CURR_PAIR_SIGN.get((other_ci, pj), 0)
            if other_sg == 0: continue
            other_dir = 1 if z_scores[other_ci] > 0 else -1
            other_dstar = other_dir * other_sg
            if d_star != other_dstar: continue

            opens[trade_ci] = {'id': trade_id, 'c': trade_ci, 'pj': pj, 'd': int(d_star),
                               'z': round(float(z_scores[trade_ci]), 2), 'bi': i, 'ep': P[i, pj],
                               'other_ci': other_ci, 'other_z': round(float(z_scores[other_ci]), 2),
                               'entry_bar': i}
            trade_id += 1

    for ci, t in opens.items():
        pj = t['pj']; ex = P[n-1, pj]
        pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
        if spread_mode == "full": pnl -= SPREAD_COST[pj]
        trades.append({**t, 'pnl': round(pnl, 2), 'exit_bar': n-1, 'time': ts[n-1] if n-1 < len(ts) else 0})

    return trades

def _best_pair(ci):
    BEST_PAIR = {"USD":"AUDUSD","EUR":"EURUSD","JPY":"NZDJPY","GBP":"GBPUSD",
        "AUD":"AUDUSD","NZD":"NZDUSD","CAD":"NZDCAD","CHF":"USDCHF"}
    return ALL_PAIRS.index(BEST_PAIR[ALL_CURRENCIES[ci]])

from datetime import datetime, timezone

def analyze_by_period(trades):
    """Group trades by day, week, month."""

    def ts_to_date(ts):
        if ts == 0: return "unknown"
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    def ts_to_week(ts):
        if ts == 0: return "unknown"
        d = datetime.fromtimestamp(ts, tz=timezone.utc)
        return d.strftime("%Y-W%W")

    def ts_to_month(ts):
        if ts == 0: return "unknown"
        d = datetime.fromtimestamp(ts, tz=timezone.utc)
        return d.strftime("%Y-%m")

    # Daily
    daily = defaultdict(list)
    for t in trades:
        day = ts_to_date(t['time'])
        daily[day].append(t)

    # Weekly
    weekly = defaultdict(list)
    for t in trades:
        wk = ts_to_week(t['time'])
        weekly[wk].append(t)

    # Monthly
    monthly = defaultdict(list)
    for t in trades:
        mo = ts_to_month(t['time'])
        monthly[mo].append(t)

    return daily, weekly, monthly

def print_summary(label, grouped, sort_key=None):
    """Print grouped summary sorted by time."""
    print(f"\n{'='*120}")
    print(f"  {label}")
    print('='*120)
    print(f"{'Period':<14} {'Trades':<8} {'Wins':<8} {'Losses':<8} {'WR':<8} {'Total PnL':<12} {'Avg PnL':<10} {'Max DD':<10} {'Cumul PnL':<12}")
    print('-'*120)

    if sort_key:
        keys = sorted(grouped.keys(), key=sort_key)
    else:
        keys = sorted(grouped.keys())

    cumul = 0.0
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    for k in keys:
        tlist = grouped[k]
        pnls = [t['pnl'] for t in tlist]
        wins = sum(1 for p in pnls if p > 0)
        n = len(pnls)
        total = sum(pnls)
        avg = total / n if n > 0 else 0
        cumul += total
        dd = 0.0
        running = 0.0
        peak = 0.0
        for p in pnls:
            running += p
            if running > peak: peak = running
            draw = running - peak
            if draw < dd: dd = draw

        total_trades += n
        total_wins += wins
        total_pnl += total

        label_k = k[-12:] if len(k) > 12 else k
        print(f"{k:<14} {n:<8} {wins:<8} {n-wins:<8} {wins/n*100:>5.1f}%  ${total:>+7.2f}   ${avg:>+6.2f}  ${dd:>+7.2f}  ${cumul:>+8.2f}")

    print('-'*120)
    if total_trades > 0:
        print(f"{'TOTAL':<14} {total_trades:<8} {total_wins:<8} {total_trades-total_wins:<8} {total_wins/total_trades*100:>5.1f}%  ${total_pnl:>+7.2f}")

def print_account_curve(trades, label):
    """Print account growth curve in $10 increments."""
    sorted_trades = sorted(trades, key=lambda t: t['time'])
    cumul = 0.0
    peak = 0.0
    max_dd = 0.0
    max_dd_pct = 0.0
    start_balance = 10000.0  # hypothetical $10k start
    balance = start_balance

    print(f"\n{'='*120}")
    print(f"  Account Growth — {label} (starting $10,000)")
    print('='*120)
    print(f"{'Trade#':<8} {'Date':<14} {'Pair':<8} {'Dir':<4} {'PnL':<10} {'Balance':<12} {'Drawdown':<12} {'Peak':<12}")
    print('-'*120)

    for ti, t in enumerate(sorted_trades):
        from datetime import datetime as dt, timezone
        date_str = dt.fromtimestamp(t['time'], tz=timezone.utc).strftime("%m-%d %H:%M") if t['time'] else "?"
        balance += t['pnl']
        if balance > peak: peak = balance
        dd = balance - peak
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd < max_dd: max_dd = dd
        if dd_pct < max_dd_pct: max_dd_pct = dd_pct
        dir_str = "+" if t['d'] > 0 else "-"
        print(f"{ti+1:<8} {date_str:<14} {ALL_PAIRS[t['pj']]:<8} {dir_str:<4} ${t['pnl']:>+6.2f}  ${balance:>+8.2f}  ${dd:>+7.2f}  ${peak:>+8.2f}")
        if ti < len(sorted_trades) - 1:
            current_pnl = t['pnl']
            next_pnl = sorted_trades[ti+1]['pnl']

    print('-'*120)
    end_balance = balance
    total_return = (end_balance - start_balance) / start_balance * 100
    print(f"Start: ${start_balance:.2f}  |  End: ${end_balance:.2f}  |  Return: {total_return:+.2f}%")
    print(f"Max Drawdown: ${max_dd:.2f} ({max_dd_pct:.2f}%)  |  Trades: {len(sorted_trades)}")
    print(f"Final Equity Curve (by trade): {[round(b, 2) for b in cumulative_balance(trades, start_balance)]}")

    # Sharper resolution: print every 10 trade blocks
    print(f"\n{'='*120}")
    print(f"  Equity Growth — $10 Increments (every 10 trades)")
    print('='*120)
    print(f"{'Block':<10} {'Trades':<8} {'Cumul PnL':<12} {'Balance':<12} {'Block PnL':<10}")
    print('-'*120)
    bal = start_balance
    for block_start in range(0, len(sorted_trades), 10):
        block = sorted_trades[block_start:block_start+10]
        block_pnl = sum(t['pnl'] for t in block)
        bal_before = bal
        bal += block_pnl
        print(f"{block_start//10 + 1:<10} {len(block):<8} ${bal - start_balance:>+8.2f}  ${bal:>+8.2f}  ${block_pnl:>+7.2f}")

    return max_dd, max_dd_pct, total_return

def cumulative_balance(trades, start_balance):
    balances = [start_balance]
    for t in sorted(trades, key=lambda x: x['time']):
        balances.append(balances[-1] + t['pnl'])
    return balances

# Run
print("\nRunning Divergence backtest with detailed logging...", file=sys.stderr)
trades_mid = run_bt_detailed(2000, 2.0, "mid", mode="divergence")
trades_full = run_bt_detailed(2000, 2.0, "full", mode="divergence")

print(f"\nTotal divergence trades (mid): {len(trades_mid)}", file=sys.stderr)
print(f"Total divergence trades (full): {len(trades_full)}", file=sys.stderr)

# Overall summary
mid_pnls = [t['pnl'] for t in trades_mid]
full_pnls = [t['pnl'] for t in trades_full]
mid_wins = sum(1 for p in mid_pnls if p > 0)
full_wins = sum(1 for p in full_pnls if p > 0)

print(f"\n{'='*120}")
print(f"  OVERALL DIVERGENCE (Z>2, hold5)")
print('='*120)
print(f"  Mid spread:  {len(mid_pnls)} trades, {mid_wins/len(mid_pnls)*100:.1f}% WR, ${np.mean(mid_pnls):+.2f} avg, ${np.sum(mid_pnls):+.2f} total")
print(f"  Full spread: {len(full_pnls)} trades, {full_wins/len(full_pnls)*100:.1f}% WR, ${np.mean(full_pnls):+.2f} avg, ${np.sum(full_pnls):+.2f} total")
print(f"  ECN ($1.75): {np.mean(mid_pnls)-1.75:+.2f} avg, ${np.sum(mid_pnls)-1.75*len(mid_pnls):+.2f} total")
print(f"  Trading days: {n/1440:.1f}")

# Daily analysis
daily, weekly, monthly = analyze_by_period(trades_mid)

# Date sort key
from datetime import datetime as dt, timezone
def date_sort(k):
    try:
        return dt.strptime(k, "%Y-%m-%d").timestamp()
    except:
        return 0
def week_sort(k):
    try:
        return dt.strptime(k + "-1", "%Y-W%W-%w").timestamp()
    except:
        return 0
def month_sort(k):
    try:
        return dt.strptime(k + "-01", "%Y-%m-%d").timestamp()
    except:
        return 0

print_summary("DAILY BREAKDOWN", daily, sort_key=date_sort)
print_summary("WEEKLY BREAKDOWN", weekly, sort_key=week_sort)
print_summary("MONTHLY BREAKDOWN", monthly, sort_key=month_sort)

# Account growth
print_account_curve(trades_mid, "Mid Spread")

# Full spread version too
print(f"\n\n")
max_dd_f, max_dd_pct_f, ret_f = print_account_curve(trades_full, "Full Spread")

# Per-pair breakdown
print(f"\n{'='*120}")
print(f"  PER-PAIR BREAKDOWN")
print('='*120)
pair_trades = defaultdict(list)
for t in trades_mid:
    pair_trades[ALL_PAIRS[t['pj']]].append(t)
print(f"{'Pair':<8} {'Trades':<8} {'Wins':<8} {'WR':<8} {'Total PnL':<12} {'Avg PnL':<10}")
print('-'*120)
for pair in sorted(pair_trades.keys()):
    tlist = pair_trades[pair]
    pnls = [t['pnl'] for t in tlist]
    wins = sum(1 for p in pnls if p > 0)
    n = len(pnls)
    total = sum(pnls)
    avg = total / n if n > 0 else 0
    print(f"{pair:<8} {n:<8} {wins:<8} {wins/n*100:>5.1f}%  ${total:>+7.2f}   ${avg:>+6.2f}")

mt5.shutdown()
print(f"\nTotal time: {time()-t0:.1f}s")
