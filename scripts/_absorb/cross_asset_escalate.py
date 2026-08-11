"""cross_asset_escalate.py — cost-aware escalation of the USD-regime family.

Exact next-bar execution: signal on closed bar i (DXY/US500 momentum W bars),
entry at i+1 OPEN, exit at i+1+H close, stop-first (SL 2xATR60, TP 3xATR60)
vs hold-only. Costs: measured live spreads + $3/lot commission (units via
broker tick values), stress = 1.5x spread. Reports USD/lot, PF, WF halves,
worst day, LODO flips, monthly nets, per-server-hour nets, and an FX-proxy
(EURUSD) robustness variant. Research-only; engine/book/live untouched.
"""
import sys, os, json
import numpy as np
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MARKET = "audit_7_eas/market"
TV = json.load(open("scripts/_absorb/results/new_tick_values.json"))
# tick_value USD per POINT per lot; point_size; spread in POINTS (live probe)
META = {k: {"tv": v["tv"], "pt": v["pt"], "spread_pts": s}
        for k, v in TV.items() for s in [0]}
META["XAUUSD"] = {"tv": 1.0, "pt": 0.01, "spread_pts": 45.0}
META["XAGUSD"] = {"tv": 5.0, "pt": 0.001, "spread_pts": 59.0}
META["BTCUSD"] = {"tv": 0.01, "pt": 0.01, "spread_pts": 100.0}
META["DXY.cash"] = {"tv": 0.1, "pt": 0.001, "spread_pts": 17.0}
META["US500.cash"] = {"tv": 0.01, "pt": 0.01, "spread_pts": 60.0}
META["EURUSD"] = {"tv": 1.0, "pt": 1e-5, "spread_pts": 6.0}

def load(sym):
    return pl.read_parquet(f"{MARKET}/{sym}.pqt").sort("time")

def run(sig_sym, tgt_sym, W, H, sign_mult, atr_n=60, use_stops=True, spread_x=1.0, vol=0.15):
    s = load(sig_sym); t = load(tgt_sym)
    sig = s["close"].to_numpy(); sts = s["time"].to_numpy()
    tcl = t["close"].to_numpy(); top = t["open"].to_numpy()
    tts = t["time"].to_numpy()
    tmap = {int(ts): k for k, ts in enumerate(tts)}
    meta = META[tgt_sym]
    spread_u = meta["spread_pts"] * meta["pt"] * spread_x
    comm_u = (2 * 3.0 * vol) * meta["pt"] / meta["tv"]
    atr = np.full(len(tcl), np.nan)
    for i in range(atr_n, len(tcl)):
        w = tcl[i - atr_n:i]
        atr[i] = float(np.mean(np.abs(np.diff(w))))
    trades = []
    for i in range(W, len(sig) - 1):
        if i + 1 >= len(tts) or int(sts[i]) not in tmap:
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
        if use_stops and not np.isnan(atr[j]):
            sl = 2.0 * atr[j]; tp = 3.0 * atr[j]
            for k in range(j + 1, k_end + 1):
                hi, lo = t["high"][k], t["low"][k]
                if side < 0:
                    if hi >= entry + sl: ex, exit_px, reason = k, entry + sl, "sl"; break
                    if lo <= entry - tp: ex, exit_px, reason = k, entry - tp, "tp"; break
                else:
                    if lo <= entry - sl: ex, exit_px, reason = k, entry - sl, "sl"; break
                    if hi >= entry + tp: ex, exit_px, reason = k, entry + tp, "tp"; break
        pnl_u = side * (exit_px - entry)
        net_u = pnl_u - spread_u - comm_u
        trades.append({"ts": int(sts[i]), "day": int(sts[i]) // 86400,
                       "hour": (int(sts[i]) // 3600) % 24, "net_u": net_u,
                       "pnl_u": pnl_u, "reason": reason})
    return trades

def stats(trades, label, tgt_sym):
    if len(trades) < 50:
        print(f"{label}: n={len(trades)} TOO FEW"); return
    m = META[tgt_sym]
    mult = m["tv"] / m["pt"]  # price-unit -> USD per 1.0 lot
    net = np.array([tr["net_u"] for tr in trades])
    usd_lot = net * mult
    usd = net * 0.15 * mult
    wins = net[net > 0].sum(); losses = -net[net < 0].sum()
    pf = wins / losses if losses > 0 else 99.0
    days = sorted({tr["day"] for tr in trades})
    mid = days[len(days) // 2]
    h1 = [tr for tr in trades if tr["day"] < mid]; h2 = [tr for tr in trades if tr["day"] >= mid]
    n1 = sum(tr["net_u"] for tr in h1) * mult; n2 = sum(tr["net_u"] for tr in h2) * mult
    byday = {}
    for tr in trades:
        byday.setdefault(tr["day"], 0.0)
        byday[tr["day"]] += tr["net_u"]
    worst = min(byday.values()) * 0.15 * mult
    lodo = sum(1 for d in byday if (sum(byday.values()) - byday[d]) <= 0)
    byhour = {}
    for tr in trades:
        byhour.setdefault(tr["hour"], 0.0)
        byhour[tr["hour"]] += tr["net_u"]
    bymon = {}
    for tr in trades:
        mth = tr["day"] // 30
        bymon.setdefault(mth, 0.0)
        bymon[mth] += tr["net_u"]
    hb = sorted(byhour.items(), key=lambda x: -x[1])
    print(f"{label}: n={len(trades)} exp=${usd_lot.mean():.2f}/lot PF={pf:.2f} "
          f"WR={100*(net>0).mean():.0f}% WF(h1/h2)=${n1:+.0f}/${n2:+.0f} worst_day=${worst:+.0f} "
          f"LODO_flips={lodo}/{len(byday)}")
    print(f"   months: " + " ".join(f"{mth}:${v*0.15*mult:+.0f}" for mth, v in sorted(bymon.items())))
    print(f"   top hours: " + " ".join(f"{h}:{v:+.0f}" for h, v in hb[:3]) +
          " | bot: " + " ".join(f"{h}:{v:+.0f}" for h, v in hb[-3:]))
    print(f"   sl/tp split: " + " ".join(f"{r}:{sum(1 for x in trades if x['reason']==r)}" for r in ["hold", "sl", "tp"]))

cells = [
    ("DXY->XAU W48H24 stops", "DXY.cash", "XAUUSD", 48, 24, 1.0, True, 1.0),
    ("DXY->XAU W48H24 hold", "DXY.cash", "XAUUSD", 48, 24, 1.0, False, 1.0),
    ("DXY->XAU W48H24 STRESSx1.5", "DXY.cash", "XAUUSD", 48, 24, 1.0, True, 1.5),
    ("DXY->BTC W48H24 stops", "DXY.cash", "BTCUSD", 48, 24, 1.0, True, 1.0),
    ("DXY->BTC W48H24 hold", "DXY.cash", "BTCUSD", 48, 24, 1.0, False, 1.0),
    ("DXY->BTC W48H24 STRESSx1.5", "DXY.cash", "BTCUSD", 48, 24, 1.0, True, 1.5),
    ("US500->XAU W48H24 stops", "US500.cash", "XAUUSD", 48, 24, 1.0, True, 1.0),
    ("DXY->XAG W48H24 stops", "DXY.cash", "XAGUSD", 48, 24, 1.0, True, 1.0),
    ("EURUSD-proxy->XAU W48H24", "EURUSD", "XAUUSD", 48, 24, -1.0, True, 1.0),
    ("DXY->XAU W24H12 stops", "DXY.cash", "XAUUSD", 24, 12, 1.0, True, 1.0),
    ("DXY->XAU W12H6 stops", "DXY.cash", "XAUUSD", 12, 6, 1.0, True, 1.0),
    ("DXY->BTC W24H12 stops", "DXY.cash", "BTCUSD", 24, 12, 1.0, True, 1.0),
]
for label, sig, tgt, W, H, sm, stops, sx in cells:
    try:
        tr = run(sig, tgt, W, H, sm, use_stops=stops, spread_x=sx)
        stats(tr, label, tgt)
    except Exception as e:
        print(f"{label}: ERROR {e}")

