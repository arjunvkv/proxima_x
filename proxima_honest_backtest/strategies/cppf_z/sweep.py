"""CPPF Z≥6.0 — Full Parameter Sweep

Cross-Pair Volatility Dislocation: LONG-only fade of extreme 15-min drops.
Uses rolling 200-bar z-score window (no lookahead).
"""
import sys, time
from pathlib import Path
BASE = Path("C:/Trading/Agentic_Trading/proxima_x")
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE / "proxima_honest_backtest"))
import numpy as np
import pandas as pd
from data.providers.mt5_provider import MT5Provider

PAIRS = ["EURAUD", "GBPAUD"]
MONTHS = [(y,m) for y,m in [(2026,1),(2026,2),(2026,3),(2026,4),(2026,5),(2026,6),(2026,7)]]
BROKERS = {"Exness":0,"FTMO":0,"FundedNext":3.0,"Fusion Markets":4.50,"Dukascopy":3.50}

def run_backtest(pairs, zt, hold, window=200, audusd_rates=None, pip_value_usd_arr=None):
    """Run CPPF backtest for given threshold and hold."""
    active = {}
    trades = []  # list of (usd_gross,)
    for i in range(window + 3, len(raw[pairs[0]])):
        for pair in pairs:
            z = z_all[pair][i]
            if np.isnan(z): continue
            close_i = raw[pair]["close"].values[i]

            if pair in active:
                if i - active[pair][0] >= hold:
                    entry_idx, ep, ez = active.pop(pair)
                    pips = (close_i - ep) / 0.0001
                    usd = pips * pip_value_usd_arr[entry_idx]
                    trades.append(usd)

            if z <= -zt and pair not in active:
                active[pair] = (i, close_i, z)

    for pair in list(active.keys()):
        entry_idx, ep, ez = active[pair]
        close_i = raw[pair]["close"].values[-1]
        pips = (close_i - ep) / 0.0001
        usd = pips * pip_value_usd_arr[entry_idx]
        trades.append(usd)

    return np.array(trades)

t0 = time.time()
print("Loading data...")
provider = MT5Provider()
raw = {}
for pair in PAIRS + ["AUDUSD"]:
    frames = [f for f in [provider.load_rates(pair,y,m,"m5") for y,m in MONTHS] if not f.empty]
    d = pd.concat(frames, ignore_index=True); d.sort_values("time", inplace=True)
    d.reset_index(drop=True, inplace=True)
    raw[pair] = d

min_len = min(len(raw[p]) for p in PAIRS)

# Precompute z-scores (rolling 200-bar window)
W = 200
z_all = {}
for pair in PAIRS:
    close = raw[pair]["close"].values[:min_len]
    ret15 = (close[3:] - close[:-3]) / close[:-3]
    n = min_len
    z = np.full(n, np.nan)
    for i in range(W + 3, n):
        idx = i - 3
        hist = ret15[idx - W: idx]
        if len(hist) < 60: continue
        mu = hist.mean(); sd = hist.std()
        if sd > 0: z[i] = (ret15[idx] - mu) / sd
    z_all[pair] = z

# Precompute per-bar pip->USD conversion (AUD-quoted crosses)
audusd_close = raw["AUDUSD"]["close"].values[:min_len]
pip_value_usd = np.full(min_len, 6.70)
for i in range(min_len):
    rate = audusd_close[i] if i < len(audusd_close) and not np.isnan(audusd_close[i]) else 0.67
    pip_value_usd[i] = 10.0 * rate

print(f"Loaded {len(raw)} pairs, {min_len} bars, z-scores done in {time.time()-t0:.1f}s")

# Sweep
Z_THRESHS = [3.0, 3.5, 4.0, 5.0, 6.0, 7.0]
HOLDS = [6, 9, 12, 18, 24]

print(f"\n{'z≥':>4s}  {'hold':>4s}  {'T':>4s}  {'WR':>5s}  {'Avg$':>7s}  {'PF':>5s}  {'Gross':>8s}", end="")
for bn in BROKERS: print(f"  {bn[:4]:>7s}", end="")
print()

best_by_pf = []
for zt in Z_THRESHS:
    for hold in HOLDS:
        usd_arr = run_backtest(PAIRS, zt, hold, W, audusd_close, pip_value_usd)
        n = len(usd_arr)
        if n < 3: continue
        wr = (usd_arr > 0).mean() * 100
        wins = usd_arr[usd_arr > 0]
        losses = usd_arr[usd_arr < 0]
        gw = wins.sum() if len(wins) > 0 else 0
        gl = abs(losses.sum()) if len(losses) > 0 else 0
        pf = gw / gl if gl > 0 else 99
        avg = usd_arr.mean()
        gross = usd_arr.sum()
        row = f"{zt:>4.1f}  {hold*5:>4d}m  {n:>4d}  {wr:>5.1f}%  {avg:>+7.2f}  {pf:>5.2f}  {gross:>+8.2f}"
        for bname, bcost in BROKERS.items():
            net = usd_arr - bcost * 2
            nt = net.sum()
            surv = "✓" if nt > 0 else "✗"
            row += f"  {nt:>+7.2f}{surv}"
        print(row)
        best_by_pf.append((pf, zt, hold, n, wr, avg, gross))

# Per-pair breakdown for best config
print("\n" + "=" * 70)
print("PER-PAIR BREAKDOWN (best configs)")
print("=" * 70)

for zt, hold in [(6.0, 18), (5.0, 18), (4.0, 18)]:
    print(f"\n  z≥{zt:.1f} hold={hold*5}m:")
    for pair in PAIRS:
        usd_arr = run_backtest([pair], zt, hold, W, audusd_close, pip_value_usd)
        n = len(usd_arr)
        if n < 3: continue
        wr = (usd_arr > 0).mean() * 100
        wins = usd_arr[usd_arr > 0]
        losses = usd_arr[usd_arr < 0]
        gw = wins.sum() if len(wins) > 0 else 0
        gl = abs(losses.sum()) if len(losses) > 0 else 0
        pf = gw / gl if gl > 0 else 99
        avg = usd_arr.mean()
        gross = usd_arr.sum()
        print(f"    {pair:>8s}: T={n:>3d}  WR={wr:>5.1f}%  Avg=${avg:>+7.2f}  PF={pf:>5.2f}  Gross=${gross:>+8.2f}")

# Best config summary
print("\n" + "=" * 70)
print("BEST CONFIGS (by PF)")
print("=" * 70)
best_by_pf.sort(reverse=True)
for pf, zt, hold, n, wr, avg, gross in best_by_pf[:10]:
    # Check Fusion survival
    usd_arr = run_backtest(PAIRS, zt, hold, W, audusd_close, pip_value_usd)
    fusion_net = (usd_arr - 4.50 * 2).sum()
    surv = "✓" if fusion_net > 0 else "✗"
    print(f"  z≥{zt:.1f} h={hold*5}m: PF={pf:>5.2f}  WR={wr:>5.1f}%  T={n:>3d}  Gross=${gross:>+8.2f}  Fusion={surv}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
