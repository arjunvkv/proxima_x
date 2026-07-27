"""Combined 26-pair V2+z backtest on Dukascopy parquet data (Apr-Jun 2026).
Shows total portfolio PnL, trade count, and risk metrics.
"""
import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

PARQUET_DIR = Path("research/phase_dislocation/dukascopy_data")
BASE_COST = {
    "eurusd": 0.15, "eurjpy": 50, "gbpjpy": 60,
    "audusd": 0.15, "nzdusd": 0.18, "usdcad": 0.20, "usdchf": 0.18,
    "audjpy": 50, "nzdjpy": 60, "cadjpy": 50, "chfjpy": 50,
    "euraud": 0.25, "eurgbp": 0.20, "eurcad": 0.25, "eurchf": 0.25,
    "gbpaud": 0.30, "gbpcad": 0.30, "gbpchf": 0.30, "gbpnzd": 0.35,
    "audcad": 0.20, "audchf": 0.20, "audnzd": 0.20,
    "nzdcad": 0.25, "nzdchf": 0.25,
    "gbpusd": 0.18,
}

def load_pair(pair):
    df = pd.read_parquet(PARQUET_DIR / f"{pair}.parquet").set_index("timestamp").astype(float)
    return df * 10000  # to MP

def hfdf_m1(b, cost, z_thresh=0.0):
    ret = b["close"].diff()
    z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
    atr = (b["high"] - b["low"]).shift(1).rolling(20).mean().clip(1e-8)
    valid = z.notna() & atr.notna()
    if z_thresh > 0:
        valid &= z.abs() >= z_thresh
    idxs = np.where(valid)[0]
    if len(idxs) < 5:
        return np.array([])
    stop_a, trig_a, gap_a = 0.15, 0.20, 0.10
    max_bars = 54
    pnls = []
    c, h, l = b["close"].values, b["high"].values, b["low"].values
    for pos in idxs:
        if pos + 2 >= len(b): continue
        direction = -1 if z.iloc[pos] > 0 else 1
        entry = c[pos]; atr_v = atr.iloc[pos]
        s = stop_a * atr_v; tg = trig_a * atr_v; gp = gap_a * atr_v
        best = entry; exited = False
        for j in range(1, max_bars + 1):
            bp = pos + j
            if bp >= len(b): break
            if direction == 1:
                if h[bp] > best: best = h[bp]
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if l[bp] <= sl: pnls.append((sl - entry) * direction); exited = True; break
            else:
                if l[bp] < best: best = l[bp]
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if h[bp] >= sl: pnls.append((sl - entry) * direction); exited = True; break
        if not exited:
            eb = min(pos + max_bars, len(b) - 1)
            pnls.append((c[eb] - entry) * direction)
    return np.array(pnls)

def scan_pair(pair, z_thresh):
    try:
        b = load_pair(pair)
    except:
        return None
    cost = BASE_COST.get(pair, 0.20)
    pnls = hfdf_m1(b, cost, z_thresh=z_thresh)
    if len(pnls) < 5:
        return None
    net_mp = np.mean(pnls) - cost
    wr = np.mean(pnls > 0)
    n_days = (b.index[-1] - b.index[0]).total_seconds() / 86400 + 1
    tpd = len(pnls) / n_days
    gross = np.sum(pnls) - cost * len(pnls)
    return {"pair": pair, "n": len(pnls), "tpd": tpd, "wr": wr, "net_mp": net_mp, "gross_mp": gross, "cost": cost, "days": n_days}

all_pairs = sorted([f.stem for f in PARQUET_DIR.glob("*.parquet")])
z_thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

print(f"\nCombined 26-pair V2+z — z>={z_thresh:.1f}")
print(f"{'Pair':>8s}  {'t/d':>5s}  {'total':>7s}  {'WR':>6s}  {'netMP':>8s}  {'grossMP':>10s}")
print("=" * 55)

results = []
for pair in all_pairs:
    r = scan_pair(pair, z_thresh)
    if r is None:
        print(f"  {pair:>8s}  {'N/A':>12s}")
        continue
    results.append(r)
    print(f"  {r['pair']:>8s}  {r['tpd']:>5.0f}  {r['n']:>7,d}  {r['wr']:>5.1%}  {r['net_mp']:>+8.2f}  {r['gross_mp']:>+10,.0f}")

if results:
    df = pd.DataFrame(results)
    total_n = df["n"].sum()
    total_gross_mp = df["gross_mp"].sum()
    avg_wr = np.average(df["wr"], weights=df["n"])
    avg_tpd = df["tpd"].sum()
    avg_cost = df["cost"].mean()
    print("=" * 55)
    print(f"  TOTAL    {avg_tpd:>5.0f}  {total_n:>7,d}  {avg_wr:>5.1%}  {'':>8s}  {total_gross_mp:>+10,.0f}")

    # $ estimate (0.01 lot per trade)
    # USD pairs (no jpy): 1 MP = 1 pip = $0.10 on 0.01 lot
    # JPY pairs: 100 MP = 1 pip ≈ $0.067 on 0.01 lot (at USDJPY=150)
    def mp_to_dollar(mp, pair):
        if "jpy" in pair:
            return mp / 100 * 0.067  # 1 pip = ¥10 on 0.01 lot, ¥10/150 ≈ $0.067
        else:
            return mp * 0.10  # 1 pip = $0.10 on 0.01 lot
    
    days = results[0]["days"]
    total_dollar = sum(mp_to_dollar(r["gross_mp"], r["pair"]) for r in results)
    daily_dollar = total_dollar / days
    
    print(f"\n  $ Estimate (0.01 lot per trade):")
    print(f"    Total PnL: ${total_dollar:,.0f} over {days:.0f} days")
    print(f"    Daily avg: ${daily_dollar:.0f}/day")
    
    # Z-sweep summary (use cached gross_mp from this run where possible)
    print(f"\n  Multi-pair summary across z-thresholds (using multi-pair scan):")
    import subprocess, os
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + str(Path.cwd())
    for zt in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        if abs(zt - z_thresh) < 0.01:
            t = sum(r["n"] for r in results)
            w = np.average([r["wr"] for r in results], weights=[r["n"] for r in results])
            tpd = sum(r["tpd"] for r in results)
            dd = sum(mp_to_dollar(r["gross_mp"], r["pair"]) for r in results) / days
            print(f"  z>={zt:.1f}: {tpd:>5.0f}/d  WR={w:.1%}  ${dd:>6.0f}/day")
        else:
            # Get from multi-pair scan output
            pass
    print(f"  (Run run_v2z_parquet_multi.py for full z-threshold table)")
