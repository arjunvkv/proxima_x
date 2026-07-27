"""Run full backtest + live flow on MT5-sourced M1 data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
import numpy as np
import datetime

DATA_DIR = "research/cppf/_mt5_data"
PAIRS = ["GBPNZD", "EURNZD", "GBPAUD", "EURAUD", "GBPCAD", "AUDNZD"]
Z_THRESH = 2.5
STOP_A, TRIG_A, GAP_A, MAX_BARS = 0.15, 0.20, 0.10, 54

result_rows = []

for pair in PAIRS:
    fpath = os.path.join(DATA_DIR, f"{pair.lower()}.parquet")
    df = pd.read_parquet(fpath)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df["range"] = df["high"] - df["low"]
    
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    ret = np.diff(c)
    
    # Z-scores
    z_arr = np.full(n, np.nan)
    for i in range(51, n):
        rw = ret[i - 51:i - 1]
        mu = rw.mean()
        sig = rw.std(ddof=1)
        z_arr[i] = (ret[i - 1] - mu) / sig if sig > 1e-10 else 0
    
    # ATR
    atr_arr = np.full(n, np.nan)
    for i in range(21, n):
        atr_arr[i] = np.mean(df["range"].values[i - 20:i])
    
    # Backtest
    valid = np.where((~np.isnan(z_arr)) & (~np.isnan(atr_arr)) & (np.abs(z_arr) >= Z_THRESH))[0]
    trades = []
    for pos in valid:
        if pos + 2 >= n:
            continue
        direction = -1 if z_arr[pos] > 0 else 1
        entry = c[pos]
        s = STOP_A * atr_arr[pos]
        tg = TRIG_A * atr_arr[pos]
        gp = GAP_A * atr_arr[pos]
        best = entry
        for j in range(1, min(MAX_BARS + 1, n - pos)):
            bp = pos + j
            if direction == 1:
                if h[bp] > best: best = h[bp]
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if l[bp] <= sl:
                    trades.append((sl - entry) * direction)
                    break
            else:
                if l[bp] < best: best = l[bp]
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if h[bp] >= sl:
                    trades.append((sl - entry) * direction)
                    break
        else:
            eb = min(pos + MAX_BARS, n - 1)
            trades.append((c[eb] - entry) * direction)
    
    pnls = [t for t in trades if t is not None]
    n_trades = len(pnls)
    wr = sum(1 for p in pnls if p > 0) / n_trades if n_trades > 0 else 0
    net = sum(pnls)
    avg = net / n_trades if n_trades > 0 else 0
    arr = np.array(pnls)
    sharpe = arr.mean() / arr.std() * np.sqrt(365) if n_trades > 0 and arr.std() > 0 else 0
    result_rows.append({
        "pair": pair, "n_bars": n, "n_trades": n_trades,
        "wr": f"{wr:.1%}", "net_pips": f"{net * 10000:.0f}",
        "avg_pips": f"{avg * 10000:.1f}", "sharpe": f"{sharpe:.2f}"
    })
    print(f"{pair}: {n_trades} trades, WR={wr:.1%}, Net={net * 10000:.0f}pips, Avg={avg * 10000:.1f}pips")

# Summary across all pairs
print(f"\n{'=' * 80}")
print("MT5 DATA BACKTEST SUMMARY (all pairs)")
print(f"{'=' * 80}")
total_trades = sum(r["n_trades"] for r in result_rows)
print(f"Pair          Bars   Trades   WR     NetPips  AvgPip  Sharpe")
for r in result_rows:
    print(f"{r['pair']:10s} {r['n_bars']:6d} {r['n_trades']:6d}  {r['wr']:5s}  {r['net_pips']:>6s}  {r['avg_pips']:>6s}  {r['sharpe']:>6s}")
print(f"{'=' * 80}")
