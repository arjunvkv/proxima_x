"""Analyze asymmetric exits for divergence strategy — cut losses, let winners run."""
import MetaTrader5 as mt5
import numpy as np
from time import time
import sys
from collections import defaultdict

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
for pj, p in enumerate(ALL_PAIRS):
    if p in PC: P[:n, pj] = PC[p][:n]

lr = np.diff(np.log(P + 1e-12), axis=0)[:50000]
n = len(lr)
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

def _best_pair(ci):
    BEST_PAIR = {"USD":"AUDUSD","EUR":"EURUSD","JPY":"NZDJPY","GBP":"GBPUSD",
        "AUD":"AUDUSD","NZD":"NZDUSD","CAD":"NZDCAD","CHF":"USDCHF"}
    return ALL_PAIRS.index(BEST_PAIR[ALL_CURRENCIES[ci]])

def run_with_intraday(z_window, zt, spread_mode):
    """Track intra-trade PnL at every bar during hold."""
    buf = [RingBuf(z_window) for _ in range(NC)]
    trades = []
    opens = {}
    trade_id = 0
    hold = 5

    for i in range(1, n):
        for ci in range(NC):
            buf[ci].append(curr_rets[i-1, ci])

        # Update intra-trade PnL for open positions
        for ci in list(opens.keys()):
            t = opens[ci]
            pj = t['pj']
            ex = P[i, pj]
            pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
            t['intra_pnls'].append(round(pnl, 2))

        # Close positions at hold
        for ci in list(opens.keys()):
            t = opens[ci]
            if i - t['bi'] >= hold:
                t = opens.pop(ci)
                pj = t['pj']
                ex = P[i, pj]
                pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
                if spread_mode == "full": pnl -= SPREAD_COST[pj]
                t['pnl'] = round(pnl, 2)
                t['exit_bar'] = i
                trades.append(t)

        # Generate signals (same divergence logic)
        z_scores = np.zeros(NC)
        for ci in range(NC):
            m, s = buf[ci].stats()
            if s < 1e-12: continue
            z_scores[ci] = (buf[ci].last() - m) / s

        sorted_idx = np.argsort(z_scores)
        strongest_ci = sorted_idx[-1]
        weakest_ci = sorted_idx[0]
        sz = z_scores[strongest_ci]
        wz = z_scores[weakest_ci]

        if sz < zt or wz > -zt: continue
        if strongest_ci in opens or weakest_ci in opens: continue

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

        opens[trade_ci] = {
            'id': trade_id, 'c': trade_ci, 'pj': pj, 'd': int(d_star),
            'z': round(float(z_scores[trade_ci]), 2), 'bi': i, 'ep': P[i, pj],
            'other_ci': other_ci, 'other_z': round(float(z_scores[other_ci]), 2),
            'intra_pnls': []
        }
        trade_id += 1

    for ci, t in opens.items():
        pj = t['pj']; ex = P[n-1, pj]
        pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
        if spread_mode == "full": pnl -= SPREAD_COST[pj]
        t['pnl'] = round(pnl, 2)
        trades.append(t)

    return trades

print("Running divergence with intra-trade tracking...", file=sys.stderr)
trades = run_with_intraday(2000, 2.0, "mid")
print(f"Total trades: {len(trades)}", file=sys.stderr)

# --- Analyze PnL distribution ---
pnls = np.array([t['pnl'] for t in trades])
wins = pnls[pnls > 0]
losses = pnls[pnls <= 0]
n_wins = len(wins)
n_losses = len(losses)

print(f"\n{'='*100}")
print(f"  PnL DISTRIBUTION")
print('='*100)
print(f"Total trades: {len(pnls)}")
print(f"Win rate: {n_wins/len(pnls)*100:.1f}%")
print(f"Avg win: ${np.mean(wins):+.2f}  |  Avg loss: ${np.mean(losses):+.2f}")
print(f"Best trade: ${np.max(pnls):+.2f}  |  Worst trade: ${np.min(pnls):+.2f}")
print(f"Std dev: ${np.std(pnls):.2f}")
print(f"Profit factor: {abs(np.sum(wins) / np.sum(losses)):.2f}")
print(f"Expectancy: ${np.mean(pnls):+.2f}")

# Percentiles
for pct in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  {pct}th percentile: ${np.percentile(pnls, pct):+.2f}")

# Win/loss buckets
print(f"\n{'='*100}")
print(f"  PnL BUCKETS")
print('='*100)
buckets = [(-100, -50), (-50, -25), (-25, -10), (-10, -5), (-5, 0),
           (0, 5), (5, 10), (10, 25), (25, 50), (50, 100)]
for lo, hi in buckets:
    mask = (pnls > lo) & (pnls <= hi)
    cnt = mask.sum()
    if cnt > 0:
        print(f"  ${lo:+4d} to ${hi:+4d}: {cnt:4d} trades ({cnt/len(pnls)*100:5.1f}%) total=${np.sum(pnls[mask]):+.2f}")

print(f"\n{'='*100}")
print(f"  INTRA-TRADE PnL PATTERNS")
print('='*100)

# Analyze intra-trade paths: what happens at each minute
# For each trade, we have PnL at min 1, 2, 3, 4, 5
min1_pnls = []; min2_pnls = []; min3_pnls = []; min4_pnls = []; min5_pnls = []
for t in trades:
    ips = t.get('intra_pnls', [])
    for mi, p in enumerate(ips):
        if mi == 0: min1_pnls.append(p)
        elif mi == 1: min2_pnls.append(p)
        elif mi == 2: min3_pnls.append(p)
        elif mi == 3: min4_pnls.append(p)
    min5_pnls.append(t['pnl'])

for label, arr in [("Min 1", min1_pnls), ("Min 2", min2_pnls), ("Min 3", min3_pnls),
                    ("Min 4", min4_pnls), ("Min 5 (close)", min5_pnls)]:
    a = np.array(arr)
    wr = np.mean(a > 0) * 100
    print(f"  {label:<15}: avg=${np.mean(a):+6.2f}  wr={wr:5.1f}%  n={len(a)}")

# --- Test asymmetric exits on existing trade data ---

print(f"\n{'='*100}")
print(f"  ASYMMETRIC EXIT TESTS (using intra-trade paths)")
print('='*100)

def test_exit(trades, exit_fn, label):
    """Test an exit strategy using intra-trade paths."""
    results = []
    for t in trades:
        ips = t.get('intra_pnls', [])
        pnl_5min = t['pnl']
        # Check exit at each minute
        exited_early = False
        for mi, pnl in enumerate(ips):
            action = exit_fn(pnl, mi + 1, t)
            if action == "exit":
                results.append(pnl)
                exited_early = True
                break
        if not exited_early:
            results.append(pnl_5min)

    arr = np.array(results)
    n = len(arr)
    wins = np.sum(arr > 0)
    total = np.sum(arr)
    avg = np.mean(arr)

    print(f"\n  {label}:")
    print(f"    Trades: {n}  |  WR: {wins/n*100:.1f}%  |  Avg: ${avg:+.2f}  |  Total: ${total:+.2f}")

# Test 1: Cut losses early — if down >$SL at minute M, exit
for sl in [5, 10, 15, 20, 25]:
    for minute in [1, 2, 3]:
        def make_exit(sl_val=sl, min_val=minute):
            return lambda pnl, m, t: "exit" if m <= min_val and pnl < -sl_val else None
        # Need a closure-safe way
        test_exit(trades, (lambda pnl, m, t, sl=sl, min=minute: "exit" if m <= min and pnl < -sl else None),
                  f"Cut loss ${sl} if down at min {minute}")

# Test 2: Trailing stop — if peak-to-current exceeds threshold, exit
for trail in [5, 10, 15]:
    results = []
    for t in trades:
        ips = t.get('intra_pnls', [])
        pnl_5min = t['pnl']
        peak = -999
        exited = False
        for mi, pnl in enumerate(ips):
            if pnl > peak: peak = pnl
            if peak - pnl > trail:
                results.append(pnl)
                exited = True
                break
        if not exited:
            results.append(pnl_5min)
    arr = np.array(results)
    n = len(arr)
    wins = np.sum(arr > 0)
    total = np.sum(arr)
    avg = np.mean(arr)
    print(f"\n  Trailing stop ${trail}:")
    print(f"    Trades: {n}  |  WR: {wins/n*100:.1f}%  |  Avg: ${avg:+.2f}  |  Total: ${total:+.2f}")

# Test 3: Extend winners — if profitable at min 5, hold to min 10
print(f"\n{'='*100}")
print(f"  EXTEND WINNERS TEST")
print('='*100)

# For extend winners, we need the price 10 min after entry
# Re-run with hold=10 to compare against hold=5
print("Re-running with hold=10 for comparison...", file=sys.stderr)
trades_hold10 = run_with_intraday(2000, 2.0, "mid")
# Only first 5 min of intra PnL matters for comparison
pnls_5 = []
for t in trades_hold10:
    ips = t.get('intra_pnls', [])
    if len(ips) >= 5:
        pnls_5.append(ips[4])  # PnL at min 5
    else:
        pnls_5.append(t['pnl'])
pnls_10 = np.array([t['pnl'] for t in trades_hold10])
pnls_5 = np.array(pnls_5)

print(f"\n  Fixed hold 5:  avg=${np.mean(pnls_5):+.2f}  wr={np.mean(pnls_5>0)*100:.1f}%  total=${np.sum(pnls_5):+.2f}")
print(f"  Fixed hold 10: avg=${np.mean(pnls_10):+.2f}  wr={np.mean(pnls_10>0)*100:.1f}%  total=${np.sum(pnls_10):+.2f}")

# Test 4: Asymmetric — winners extend to 10, losers close at 5
asym_pnls = []
asym_wins_5 = pnls_5 > 0
for i in range(len(asym_wins_5)):
    if asym_wins_5[i]:
        asym_pnls.append(pnls_10[i])  # winner → hold 10
    else:
        asym_pnls.append(pnls_5[i])   # loser → close at 5
asym_pnls = np.array(asym_pnls)
print(f"\n  Asymmetric (winners→10, losers→5):")
print(f"    avg=${np.mean(asym_pnls):+.2f}  wr={np.mean(asym_pnls>0)*100:.1f}%  total=${np.sum(asym_pnls):+.2f}")
print(f"    Improvement vs hold5: ${np.mean(asym_pnls)-np.mean(pnls_5):+.2f}/trade")

# Test 5: Extend only strong winners (top quartile at min 5)
top25_thresh = np.percentile(pnls_5, 75)
strong_win_10 = []
for i in range(len(pnls_5)):
    if pnls_5[i] >= top25_thresh:
        strong_win_10.append(pnls_10[i])  # strong → hold 10
    else:
        strong_win_10.append(pnls_5[i])   # weak or loser → close at 5
strong_win_10 = np.array(strong_win_10)
print(f"\n  Asymmetric (top25% winners→10, rest→5):")
print(f"    avg=${np.mean(strong_win_10):+.2f}  wr={np.mean(strong_win_10>0)*100:.1f}%  total=${np.sum(strong_win_10):+.2f}")
print(f"    Improvement vs hold5: ${np.mean(strong_win_10)-np.mean(pnls_5):+.2f}/trade")

# Test 6: Cut early losers (down at min 1) + extend winners
print(f"\n{'='*100}")
print(f"  COMBINED: Cut early losers + extend winners")
print('='*100)
for early_cut_sl in [5, 10, 15]:
    results = []
    for t in trades_hold10:
        ips = t.get('intra_pnls', [])
        # Check early cut
        cut_early = False
        for mi, pnl in enumerate(ips):
            if mi < 2 and pnl < -early_cut_sl:  # cut in first 2 min if down >SL
                results.append(pnl)
                cut_early = True
                break
        if cut_early:
            continue
        # If not cut and profitable at 5 → extend to 10
        if len(ips) >= 5 and ips[4] > 0:
            results.append(t['pnl'])  # use 10-min PnL
        else:
            results.append(ips[4] if len(ips) >= 5 else t['pnl'])
    arr = np.array(results)
    n = len(arr)
    wins = np.sum(arr > 0)
    total = np.sum(arr)
    avg = np.mean(arr)
    print(f"\n  Cut loss ${early_cut_sl} in 2min, extend winners to 10:")
    print(f"    Trades: {n}  |  WR: {wins/n*100:.1f}%  |  Avg: ${avg:+.2f}  |  Total: ${total:+.2f}")

# Test 7: Optimal simple - if down >$15 at min 1, cut. Else hold 5.
print(f"\n{'='*100}")
print(f"  OPTIMAL SIMPLE EXIT SEARCH")
print('='*100)
best_avg = -999
best_params = None
for sl in [5, 10, 15, 20, 25, 30]:
    for cut_min in [1, 2, 3]:
        results = []
        for t in trades:
            ips = t.get('intra_pnls', [])
            cut = False
            for mi, pnl in enumerate(ips):
                if mi < cut_min and pnl < -sl:
                    results.append(pnl)
                    cut = True
                    break
            if not cut:
                results.append(t['pnl'])
        arr = np.array(results)
        avg = np.mean(arr)
        wr = np.mean(arr > 0) * 100
        if avg > best_avg:
            best_avg = avg
            best_params = (sl, cut_min, avg, wr, np.sum(arr))
        print(f"  Cut ${sl} at min≤{cut_min}: avg=${avg:+.2f}  wr={wr:.1f}%")

print(f"\n  BEST: Cut ${best_params[0]} at min≤{best_params[1]}: avg=${best_params[2]:+.2f} wr={best_params[3]:.1f}% total=${best_params[4]:+.2f}")

# --- MFE/MAE Analysis ---
print(f"\n{'='*100}")
print(f"  MFE/MAE ANALYSIS (Maximum Favorable/Adverse Excursion)")
print('='*100)
mfe_list = []
mae_list = []
for t in trades:
    ips = t.get('intra_pnls', [])
    if ips:
        mfe_list.append(max(ips + [t['pnl']]))
        mae_list.append(min(ips + [t['pnl']]))
mfe = np.array(mfe_list)
mae = np.array(mae_list)

print(f"  Avg Max Favorable: ${np.mean(mfe):+.2f}")
print(f"  Avg Max Adverse:   ${np.mean(mae):+.2f}")
print(f"  Trades that went positive at some point: {np.mean(mfe > 0)*100:.1f}%")
print(f"  Trades that went negative at some point: {np.mean(mae < 0)*100:.1f}%")

# MFE distribution
for pct in [10, 25, 50, 75, 90]:
    print(f"  MFE {pct}th: ${np.percentile(mfe, pct):+.2f}  |  MAE {pct}th: ${np.percentile(mae, pct):+.2f}")

# Trades that went profitable then ended negative (failed to protect)
went_profit_ended_loss = sum(1 for i in range(len(trades)) if mfe[i] > 0 and trades[i]['pnl'] <= 0)
went_loss_ended_profit = sum(1 for i in range(len(trades)) if mae[i] < 0 and trades[i]['pnl'] > 0)
print(f"\n  Went positive → ended negative: {went_profit_ended_loss} ({went_profit_ended_loss/len(trades)*100:.1f}%)")
print(f"  Went negative → ended positive: {went_loss_ended_profit} ({went_loss_ended_profit/len(trades)*100:.1f}%)")

# Trailing stop test: retracement from MFE
print(f"\n{'='*100}")
print(f"  RETRACEMENT-BASED EXITS")
print('='*100)
for retrace_pct in [25, 33, 50, 66, 75]:
    results = []
    for t in trades:
        ips = t.get('intra_pnls', []) + [t['pnl']]
        peak = -999
        exited = False
        for pnl in ips:
            if pnl > peak: peak = pnl
            if peak > 0 and peak - pnl > peak * retrace_pct / 100:
                results.append(pnl)
                exited = True
                break
        if not exited:
            results.append(t['pnl'])
    arr = np.array(results)
    n = len(arr)
    wins = np.sum(arr > 0)
    total = np.sum(arr)
    avg = np.mean(arr)
    print(f"  Trail {retrace_pct:2d}% retrace: avg=${avg:+.2f}  wr={wins/n*100:.1f}%  total=${total:+.2f}")

# Best fixed stop/target from 5-min data
print(f"\n{'='*100}")
print(f"  FIXED STOP LOSS / TAKE PROFIT ON 5-MIN HOLDS")
print('='*100)
for sl in [5, 10, 15, 20, 25, 30]:
    for tp in [10, 15, 20, 25, 30, 40, 50]:
        results = []
        for t in trades:
            ips = t.get('intra_pnls', [])
            exited = False
            for pnl in ips:
                if pnl < -sl:
                    results.append(pnl)
                    exited = True
                    break
                if pnl > tp:
                    results.append(pnl)
                    exited = True
                    break
            if not exited:
                results.append(t['pnl'])
        arr = np.array(results)
        avg = np.mean(arr)
        wr_val = np.mean(arr > 0) * 100
        if avg > best_avg - 0.5:  # print near-best
            print(f"  SL=${sl:2d}, TP=${tp:2d}: avg=${avg:+.2f}  wr={wr_val:.1f}%  total=${np.sum(arr):+.2f}")

mt5.shutdown()
print(f"\nTime: {time()-t0:.1f}s")
