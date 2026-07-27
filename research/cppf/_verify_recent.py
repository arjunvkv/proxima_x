"""Verify live trades against backtest on recent MT5 data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
import datetime
import numpy as np

Z_THRESH = 2.5
STOP_A, TRIG_A, GAP_A, MAX_BARS = 0.15, 0.20, 0.10, 54
DATA_DIR = "research/cppf/_live_data"

# Live trade log from the run
LIVE_TRADES = {
    "GBPNZD": {"dir": -1, "entry": 2.30782, "exit": 2.30788, "entry_ts": 1784830140, "exit_ts": 1784830200, "pnl": -6.0, "z": 69.53},
    "EURNZD": {"dir": -1, "entry": 1.97113, "exit": 1.97123, "entry_ts": 1784830201, "exit_ts": 1784830260, "pnl": -10.0, "z": 171.90},
    "GBPAUD": {"dir": 1,  "entry": 1.91089, "exit": 1.91065, "entry_ts": 1784830261, "exit_ts": 1784830320, "pnl": -24.0, "z": -117.89},
    "EURAUD": {"dir": -1, "entry": 1.63194, "exit": 1.63202, "entry_ts": 1784830321, "exit_ts": 1784830380, "pnl": -8.0,  "z": 13.18},
    "GBPCAD": {"dir": 1,  "entry": 1.87472, "exit": 1.87459, "entry_ts": 1784830381, "exit_ts": 1784830440, "pnl": -6.93, "z": -161.32},
    "AUDNZD": {"dir": -1, "entry": 1.20770, "exit": 1.20769, "entry_ts": 1784830441, "exit_ts": 1784830500, "pnl": 1.0,   "z": 219.57},
}

def hfdf_m1_on_df(df):
    """Run hfdf_m1 backtest on a DataFrame with open/high/low/close columns."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    ret = np.diff(c)
    
    z_arr = np.full(n, np.nan)
    for i in range(51, n):
        rw = ret[i - 51:i - 1]
        mu = rw.mean()
        sig = rw.std(ddof=1)
        z_arr[i] = (ret[i - 1] - mu) / sig if sig > 1e-10 else 0
    
    atr_arr = np.full(n, np.nan)
    for i in range(21, n):
        atr_arr[i] = np.mean(df["range"].values[i - 20:i])
    
    valid = np.where((~np.isnan(z_arr)) & (~np.isnan(atr_arr)) & (np.abs(z_arr) >= Z_THRESH))[0]
    out = []
    for pos in valid:
        if pos + 2 >= n:
            continue
        direction = -1 if z_arr[pos] > 0 else 1
        entry = c[pos]
        s = STOP_A * atr_arr[pos]
        tg = TRIG_A * atr_arr[pos]
        gp = GAP_A * atr_arr[pos]
        best = entry
        exited = False
        for j in range(1, min(MAX_BARS + 1, n - pos)):
            bp = pos + j
            if direction == 1:
                if h[bp] > best: best = h[bp]
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if l[bp] <= sl:
                    out.append({"dir": direction, "entry": entry, "exit": sl,
                                "entry_bar": pos, "exit_bar": bp,
                                "entry_time": int(df.index[pos].timestamp()),
                                "exit_time": int(df.index[bp].timestamp()),
                                "raw_pnl": (sl - entry) * direction,
                                "z": z_arr[pos], "atr": atr_arr[pos], "exit_reason": "stop"})
                    exited = True; break
            else:
                if l[bp] < best: best = l[bp]
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if h[bp] >= sl:
                    out.append({"dir": direction, "entry": entry, "exit": sl,
                                "entry_bar": pos, "exit_bar": bp,
                                "entry_time": int(df.index[pos].timestamp()),
                                "exit_time": int(df.index[bp].timestamp()),
                                "raw_pnl": (sl - entry) * direction,
                                "z": z_arr[pos], "atr": atr_arr[pos], "exit_reason": "stop"})
                    exited = True; break
        if not exited:
            eb = min(pos + MAX_BARS, n - 1)
            out.append({"dir": direction, "entry": entry, "exit": c[eb],
                        "entry_bar": pos, "exit_bar": eb,
                        "entry_time": int(df.index[pos].timestamp()),
                        "exit_time": int(df.index[eb].timestamp()),
                        "raw_pnl": (c[eb] - entry) * direction,
                        "z": z_arr[pos], "atr": atr_arr[pos], "exit_reason": "expiry"})
    return out

print("=" * 80)
print("LIVE TRADE VERIFICATION against MT5-sourced M1 data")
print("=" * 80)

all_ok = True
for pair, live in sorted(LIVE_TRADES.items()):
    fpath = os.path.join(DATA_DIR, f"{pair.lower()}.parquet")
    if not os.path.exists(fpath):
        print(f"\n{pair}: NO DATA FILE")
        all_ok = False
        continue
    
    df = pd.read_parquet(fpath)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index)
    df["range"] = df["high"] - df["low"]
    
    bt_trades = hfdf_m1_on_df(df)
    
    entry_ts = live["entry_ts"]
    entry_dt = datetime.datetime.fromtimestamp(entry_ts)
    
    # Find matching backtest trade by entry_time
    match = None
    for t in bt_trades:
        if abs(t["entry_time"] - entry_ts) <= 60:  # within 1 min
            match = t
            break
    
    print(f"\n{pair} — Entry: {entry_dt} UTC")
    print(f"  Live: dir={live['dir']:+d} entry={live['entry']:.5f} exit={live['exit']:.5f} pnl={live['pnl']}")
    
    if match:
        dir_ok = match["dir"] == live["dir"]
        entry_diff = abs(match["entry"] - live["entry"])
        exit_diff = abs(match["exit"] - live["exit"])
        pnl_ok = abs(match["raw_pnl"] - live["pnl"] / 10000) < 0.0001  # pips to price diff
        z_diff = abs(match["z"] - live["z"])
        print(f"  BT:   dir={match['dir']:+d} entry={match['entry']:.5f} exit={match['exit']:.5f} pnl={match['raw_pnl']:.6f} z={match['z']:.2f} reason={match['exit_reason']}")
        print(f"  Dir match: {dir_ok}, Entry diff: {entry_diff:.8f}, Exit diff: {exit_diff:.8f}, Z diff: {z_diff:.4f}")
        if not (dir_ok and entry_diff < 1e-8 and exit_diff < 1e-8):
            print(f"  ** MISMATCH **")
            all_ok = False
        else:
            print(f"  ✓ MATCH")
    else:
        print(f"  NO BACKTEST TRADE at this time")
        # Show nearby backtest trades
        nearby = [t for t in bt_trades if abs(t["entry_time"] - entry_ts) < 600]
        if nearby:
            for t in nearby:
                tdt = datetime.datetime.fromtimestamp(t["entry_time"])
                print(f"    Nearby: {tdt} UTC dir={t['dir']:+d} z={t['z']:.2f}")
        all_ok = False

print(f"\n{'=' * 80}")
print(f"VERDICT: {'ALL MATCH' if all_ok else 'SOME MISMATCHES'}")
print(f"{'=' * 80}")
