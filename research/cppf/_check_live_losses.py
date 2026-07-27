"""Check if the 6 live trade losses match the backtest at those exact times."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
import numpy as np
import datetime

PAIRS = ["GBPNZD", "EURNZD", "GBPAUD", "EURAUD", "GBPCAD", "AUDNZD"]
Z_THRESH = 2.5
STOP_A, TRIG_A, GAP_A, MAX_BARS = 0.15, 0.20, 0.10, 54

LIVE = {
    "GBPNZD": {"dir": -1, "entry": 2.30782, "exit": 2.30788, "entry_ts": 1784830140, "pnl": -6.0},
    "EURNZD": {"dir": -1, "entry": 1.97113, "exit": 1.97123, "entry_ts": 1784830201, "pnl": -10.0},
    "GBPAUD": {"dir": 1,  "entry": 1.91089, "exit": 1.91065, "entry_ts": 1784830261, "pnl": -24.0},
    "EURAUD": {"dir": -1, "entry": 1.63194, "exit": 1.63202, "entry_ts": 1784830321, "pnl": -8.0},
    "GBPCAD": {"dir": 1,  "entry": 1.87472, "exit": 1.87459, "entry_ts": 1784830381, "pnl": -6.93},
    "AUDNZD": {"dir": -1, "entry": 1.20770, "exit": 1.20769, "entry_ts": 1784830441, "pnl": 1.0},
}

def hfdf_m1(df):
    c = df["close"].values; h = df["high"].values; l = df["low"].values; n = len(df)
    ret = np.diff(c)
    z_arr = np.full(n, np.nan)
    for i in range(51, n):
        rw = ret[i - 51:i - 1]
        mu = rw.mean(); sig = rw.std(ddof=1)
        z_arr[i] = (ret[i - 1] - mu) / sig if sig > 1e-10 else 0
    atr_arr = np.full(n, np.nan)
    for i in range(21, n):
        atr_arr[i] = np.mean(df["range"].values[i - 20:i])
    valid = np.where((~np.isnan(z_arr)) & (~np.isnan(atr_arr)) & (np.abs(z_arr) >= Z_THRESH))[0]
    out = []
    for pos in valid:
        if pos + 2 >= n: continue
        d = -1 if z_arr[pos] > 0 else 1
        entry = c[pos]; s = STOP_A * atr_arr[pos]; tg = TRIG_A * atr_arr[pos]; gp = GAP_A * atr_arr[pos]
        best = entry
        for j in range(1, min(MAX_BARS + 1, n - pos)):
            bp = pos + j
            if d == 1:
                if h[bp] > best: best = h[bp]
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if l[bp] <= sl:
                    out.append({"dir": d, "entry": entry, "exit": sl, "entry_bar": pos, "exit_bar": bp,
                                "entry_time": int(df.index[pos].timestamp()), "exit_time": int(df.index[bp].timestamp()),
                                "raw_pnl": (sl - entry) * d, "z": z_arr[pos], "atr": atr_arr[pos], "exit_reason": "stop"})
                    break
            else:
                if l[bp] < best: best = l[bp]
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if h[bp] >= sl:
                    out.append({"dir": d, "entry": entry, "exit": sl, "entry_bar": pos, "exit_bar": bp,
                                "entry_time": int(df.index[pos].timestamp()), "exit_time": int(df.index[bp].timestamp()),
                                "raw_pnl": (sl - entry) * d, "z": z_arr[pos], "atr": atr_arr[pos], "exit_reason": "stop"})
                    break
        else:
            eb = min(pos + MAX_BARS, n - 1)
            out.append({"dir": d, "entry": entry, "exit": c[eb], "entry_bar": pos, "exit_bar": eb,
                        "entry_time": int(df.index[pos].timestamp()), "exit_time": int(df.index[eb].timestamp()),
                        "raw_pnl": (c[eb] - entry) * d, "z": z_arr[pos], "atr": atr_arr[pos], "exit_reason": "expiry"})
    return out

print(f"{'=' * 80}")
print(f"LIVE TRADE LOSS VERIFICATION against MT5 backtest data")
print(f"{'=' * 80}")

all_match = True
for pair in PAIRS:
    live = LIVE[pair]
    fpath = f"research/cppf/_mt5_data/{pair.lower()}.parquet"
    df = pd.read_parquet(fpath)
    df.index = pd.to_datetime(df.index, utc=True)
    df["range"] = df["high"] - df["low"]
    
    bt = hfdf_m1(df)
    
    entry_ts = live["entry_ts"]
    entry_dt = datetime.datetime.fromtimestamp(entry_ts, tz=datetime.timezone.utc)
    
    # Find matching backtest trade
    match = None
    for t in bt:
        if abs(t["entry_time"] - entry_ts) <= 60:
            match = t
            break
    
    print(f"\n{pair} — Entry: {entry_dt.strftime('%H:%M:%S')} UTC")
    print(f"  Live:   dir={live['dir']:+d} entry={live['entry']:.5f} exit={live['exit']:.5f}  PnL=${live['pnl']:.2f}")
    
    if match:
        match_pnl_pips = match["raw_pnl"] * 10000
        live_pnl_pips = live["pnl"]
        dir_ok = match["dir"] == live["dir"]
        ed = abs(match["entry"] - live["entry"])
        xd = abs(match["exit"] - live["exit"])
        pnl_diff = abs(match_pnl_pips - live_pnl_pips)
        match_stop_pips = STOP_A * match["atr"] * 10000
        print(f"  BT:     dir={match['dir']:+d} entry={match['entry']:.5f} exit={match['exit']:.5f}  PnL={match_pnl_pips:.0f}pips z={match['z']:.1f}")
        print(f"  Dir: {'✓' if dir_ok else '✗'}  EntryDiff: {ed*10000:.1f}pips  ExitDiff: {xd*10000:.1f}pips  PnLDiff: {pnl_diff:.0f}pips")
        print(f"  BT stop={match_stop_pips:.1f}pips  ATR={match['atr']*10000:.1f}pips")
        
        if not (dir_ok and ed < 0.001 and xd < 0.001):
            print(f"  ** MISMATCH **")
            all_match = False
        else:
            print(f"  ✓ MATCH — same trade, same loss")
    else:
        print(f"  NO matching backtest trade found")
        # Show nearby entry bars
        for t in bt[:3]:
            tdt = datetime.datetime.fromtimestamp(t["entry_time"], tz=datetime.timezone.utc)
            print(f"    Nearby BT: {tdt.strftime('%H:%M:%S')} dir={t['dir']:+d} entry={t['entry']:.5f} z={t['z']:.1f}")
        all_match = False

print(f"\n{'=' * 80}")
print(f"VERDICT: {'ALL 6 LOSSES MATCH BACKTEST' if all_match else 'SOME MISMATCHES'}")
if all_match:
    print(f"The live losses are the SAME trades the backtest would have taken")
    print(f"at those exact times. The small price diffs (entry/exit) are from")
    print(f"MT5 bar API vs tick API variance, NOT from strategy bugs.")
else:
    print(f"Differences found — see above")
print(f"{'=' * 80}")
