"""cross_asset_refine.py — stop/horizon/hour grid for the USD-regime family.

The A/B found real conditional drift (z 8-10) but exact-execution with 2xATR
stops destroyed it (SL too tight for 2h gold/BTC ranges). Here: re-run the
hold-only winners with (a) stop grid 3/4 x ATR, (b) hold horizons H=12,24,36,48,
(c) US-hour-restricted entries, (d) DXY->ETH added. Research-only.
"""
import sys, os
import numpy as np
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MARKET = "audit_7_eas/market"
META = {
    "XAUUSD": {"tv": 1.0, "pt": 0.01, "spread_pts": 45.0},
    "XAGUSD": {"tv": 5.0, "pt": 0.001, "spread_pts": 59.0},
    "BTCUSD": {"tv": 0.01, "pt": 0.01, "spread_pts": 100.0},
    "ETHUSD": {"tv": 0.01, "pt": 0.01, "spread_pts": 60.0},
    "DXY.cash": {"tv": 0.1, "pt": 0.001, "spread_pts": 17.0},
}

def load(sym):
    return pl.read_parquet(f"{MARKET}/{sym}.pqt").sort("time")

def run(sig_sym, tgt_sym, W, H, sign_mult, atr_n=60, sl_x=0.0, tp_x=0.0,
        hours=None, vol=0.15):
    s = load(sig_sym); t = load(tgt_sym)
    sig = s["close"].to_numpy(); sts = s["time"].to_numpy()
    tcl = t["close"].to_numpy(); top = t["open"].to_numpy()
    thi = t["high"].to_numpy(); tlo = t["low"].to_numpy()
    tts = t["time"].to_numpy()
    tmap = {int(ts): k for k, ts in enumerate(tts)}
    meta = META[tgt_sym]
    spread_u = meta["spread_pts"] * meta["pt"]
    comm_u = (2 * 3.0 * vol) * meta["pt"] / meta["tv"]
    atr = np.full(len(tcl), np.nan)
    for i in range(atr_n, len(tcl)):
        w = tcl[i - atr_n:i]
        atr[i] = float(np.mean(np.abs(np.diff(w))))
    trades = []
    for i in range(W, len(sig) - 1):
        if hours is not None and (int(sts[i]) // 3600) % 24 not in hours:
            continue
        if int(sts[i]) not in tmap:
            continue
        j = tmap[int(sts[i])]
        if j + 1 + H >= len(tcl):
            break
        r = sig[i] / sig[i - W] - 1.0
        if r == 0:
            continue
        side = -1.0 if (sign_mult * r) > 0 else 1.0
        entry = top[j + 1]
        k_end = j + 1 + H
        ex = k_end; exit_px = tcl[k_end]; reason = "hold"
        if sl_x > 0 and not np.isnan(atr[j]):
            sl = sl_x * atr[j]; tp = tp_x * atr[j]
            for k in range(j + 1, k_end + 1):
                hi, lo = thi[k], tlo[k]
                if side < 0:
                    if hi >= entry + sl: ex, exit_px, reason = k, entry + sl, "sl"; break
                    if lo <= entry - tp: ex, exit_px, reason = k, entry - tp, "tp"; break
                else:
                    if lo <= entry - sl: ex, exit_px, reason = k, entry - sl, "sl"; break
                    if hi >= entry + tp: ex, exit_px, reason = k, entry + tp, "tp"; break
        net_u = side * (exit_px - entry) - spread_u - comm_u
        trades.append((int(sts[i]), net_u, reason))
    return trades, META[tgt_sym]

def report(trades, meta, label):
    if len(trades) < 100:
        print(f"{label}: n={len(trades)} TOO FEW"); return
    net = np.array([x[1] for x in trades])
    mult = meta["tv"] / meta["pt"]
    wins = net[net > 0].sum(); losses = -net[net < 0].sum()
    pf = wins / losses if losses > 0 else 99.0
    days = sorted({int(x[0]) // 86400 for x in trades})
    mid = days[len(days) // 2]
    n1 = sum(x[1] for x in trades if x[0] // 86400 < mid) * mult
    n2 = sum(x[1] for x in trades if x[0] // 86400 >= mid) * mult
    byday = {}
    for ts, u, _ in trades:
        d = int(ts) // 86400
        byday[d] = byday.get(d, 0.0) + u
    lodo = sum(1 for d in byday if (sum(byday.values()) - byday[d]) <= 0)
    nm = {}
    for ts, u, _ in trades:
        m = int(ts) // 86400 // 30
        nm[m] = nm.get(m, 0.0) + u
    print(f"{label}: n={len(trades)} exp=${(net*mult).mean():.2f}/lot PF={pf:.2f} "
          f"WR={100*(net>0).mean():.0f}% WF=${n1:+.0f}/${n2:+.0f} LODO={lodo}/{len(byday)} "
          f"months: " + " ".join(f"{m}:${v*mult*0.15:+.0f}" for m, v in sorted(nm.items())))

US_HOURS = [13, 14, 15, 16, 17, 18, 19, 20, 21]
cells = []
# stop grid on gold (hold base H=24)
for sx, tx in [(3.0, 4.0), (4.0, 6.0), (5.0, 8.0)]:
    cells.append((f"XAU S{sx}/T{tx}", "DXY.cash", "XAUUSD", 48, 24, 1.0, sx, tx, None))
# horizon grid, hold-only
for H in [12, 36, 48]:
    cells.append((f"XAU H{H} hold", "DXY.cash", "XAUUSD", 48, H, 1.0, 0.0, 0.0, None))
for H in [12, 36, 48]:
    cells.append((f"BTC H{H} hold", "DXY.cash", "BTCUSD", 48, H, 1.0, 0.0, 0.0, None))
# US-hour restricted holds
cells.append(("XAU H24 UShrs", "DXY.cash", "XAUUSD", 48, 24, 1.0, 0.0, 0.0, US_HOURS))
cells.append(("BTC H24 UShrs", "DXY.cash", "BTCUSD", 48, 24, 1.0, 0.0, 0.0, US_HOURS))
# DXY->ETH
cells.append(("ETH H24 hold", "DXY.cash", "ETHUSD", 48, 24, 1.0, 0.0, 0.0, None))
# best stop variant with US hours
cells.append(("XAU S3/T4 UShrs", "DXY.cash", "XAUUSD", 48, 24, 1.0, 3.0, 4.0, US_HOURS))
for label, sig, tgt, W, H, sm, sx, tx, hrs in cells:
    try:
        tr, meta = run(sig, tgt, W, H, sm, sl_x=sx, tp_x=tx, hours=hrs)
        report(tr, meta, label)
    except Exception as e:
        print(f"{label}: ERROR {e}")
