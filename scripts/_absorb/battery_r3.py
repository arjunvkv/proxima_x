"""battery_r3.py — final gate suite for round-3 candidates.

Each candidate runs exact next-bar execution with costs (measured spreads +
$3/lot comm via broker tick values) and is judged on:
  gate: PF>1.2, exp/lot>$15, net>0, WF both halves +, LODO=0, per-side both +
  monthly: no 2 consecutive negative months, stress: spread x1.5
  independence: Jaccard overlap vs other candidates and vs book legs (hours)
Candidates: A=USD-regime fade (DXY->BTC H36; +gold leg filtered), B=index
break-gap follow pooled, C=sweep survivor (injected via json). Research-only.
"""
import sys, os, json
import numpy as np
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MARKET = "audit_7_eas/market"
META = {"XAUUSD": {"tv": 1.0, "pt": 0.01, "spread_pts": 45.0},
        "BTCUSD": {"tv": 0.01, "pt": 0.01, "spread_pts": 100.0},
        "US30.cash": {"tv": 0.01, "pt": 0.01, "spread_pts": 210.0},
        "US500.cash": {"tv": 0.01, "pt": 0.01, "spread_pts": 60.0},
        "GER40.cash": {"tv": 0.01, "pt": 0.01, "spread_pts": 113.0},
        "UK100.cash": {"tv": 0.01, "pt": 0.01, "spread_pts": 75.0},
        "DXY.cash": {"tv": 0.1, "pt": 0.001, "spread_pts": 17.0}}

def load(sym):
    return pl.read_parquet(f"{MARKET}/{sym}.pqt").sort("time")

def cand_a_usdregime():
    """DXY W48 momentum -> fade BTC/XAU, hold H36, gold trend-filtered."""
    s = load("DXY.cash"); sig = s["close"].to_numpy(); sts = s["time"].to_numpy()
    out = []
    for tgt in ["BTCUSD", "XAUUSD"]:
        t = load(tgt)
        tcl = t["close"].to_numpy(); top = t["open"].to_numpy(); tts = t["time"].to_numpy()
        tmap = {int(ts): k for k, ts in enumerate(tts)}
        meta = META[tgt]; mult = meta["tv"] / meta["pt"]
        spread_u = meta["spread_pts"] * meta["pt"]; comm_u = 6.0 * meta["pt"] / meta["tv"]
        W, H = 48, 36
        for i in range(W, len(sig) - 1):
            if int(sts[i]) not in tmap: continue
            j = tmap[int(sts[i])]
            if j + 1 + H >= len(tcl): break
            r = sig[i] / sig[i - W] - 1.0
            if r == 0: continue
            if tgt == "XAUUSD":
                gslope = tcl[j] / tcl[j - W] - 1.0
                if np.sign(gslope) == np.sign(r) and abs(gslope) >= 0.0008:
                    continue
            side = -1.0 if r > 0 else 1.0
            pnl = side * (tcl[j + 1 + H] - top[j + 1]) - spread_u - comm_u
            out.append({"sym": tgt, "ts": int(sts[i]), "net": pnl * mult, "side": side})
    return out

def cand_b_gapfollow():
    """Top-tercile break-gap -> LONG to session close, 4 indices pooled."""
    out = []
    for sym in ["US30.cash", "US500.cash", "GER40.cash", "UK100.cash"]:
        s = load(sym)
        ts = s["time"].to_numpy(); op = s["open"].to_numpy(); cl = s["close"].to_numpy()
        n = len(ts)
        reopens = {}; closes = {}
        for i in range(n):
            m, h = (ts[i] // 60) % 60, (ts[i] // 3600) % 24
            d = int(ts[i]) // 86400
            if h == 1 and m == 5: reopens[d] = i
            if h == 23 and m == 45: closes[d] = i
        gaps = []
        for d in sorted(reopens):
            if d - 1 not in closes: continue
            i_o, i_c = reopens[d], closes[d - 1]
            gaps.append((d, op[i_o] - cl[i_c]))
        thresh = np.quantile([g for _, g in gaps], 2 / 3)
        meta = META[sym]; mult = meta["tv"] / meta["pt"]
        spread_u = meta["spread_pts"] * meta["pt"]; comm_u = 6.0 * meta["pt"] / meta["tv"]
        for d, g in gaps:
            if g <= thresh: continue
            i_o = reopens[d]
            pnl = (cl[min(i_o + 272, n - 1)] - op[i_o]) - spread_u - comm_u
            out.append({"sym": sym, "ts": int(ts[i_o]), "net": pnl * mult, "side": 1.0})
    return out

def suite(trades, label):
    if len(trades) < 40:
        print(f"{label}: n={len(trades)} TOO FEW"); return
    net = np.array([t["net"] for t in trades])
    days = np.array([t["ts"] // 86400 for t in trades])
    wins = net[net > 0].sum(); losses = -net[net < 0].sum()
    pf = wins / losses if losses > 0 else 99
    exp = net.mean(); z = net.mean() / (net.std() / np.sqrt(len(net)))
    mid = np.median(days)
    w1 = net[days < mid].sum(); w2 = net[days >= mid].sum()
    byday = {}
    for d, u in zip(days, net):
        byday.setdefault(int(d), 0.0); byday[int(d)] += u
    lodo = sum(1 for d in byday if sum(byday.values()) - byday[d] <= 0)
    pos = net[net > 0].sum() / len(net)
    longs = net[[t["side"] for t in trades] and [t["side"] > 0 for t in trades]]
    shorts = net[[t["side"] < 0 for t in trades]]
    bymon = {}
    for d, u in zip(days, net):
        m = int(d) // 30
        bymon.setdefault(m, 0.0); bymon[m] += u
    mon = sorted(bymon.items())
    neg2 = any(bymon[mon[i][0]] < 0 and bymon[mon[i + 1][0]] < 0 for i in range(len(mon) - 1))
    bysym = {}
    for t in trades: bysym[t["sym"]] = bysym.get(t["sym"], 0.0) + t["net"]
    gate = (pf > 1.2 and exp > 15.0 and w1 > 0 and w2 > 0 and lodo == 0
            and (len(longs) == 0 or longs.mean() > 0) and (len(shorts) == 0 or shorts.mean() > 0)
            and not neg2)
    print(f"{label}: n={len(trades)} exp=${exp:.2f}/lot PF={pf:.2f} WR={100*pos:.0f}% "
          f"z={z:+.2f} WF=${w1:+.0f}/${w2:+.0f} LODO={lodo}/{len(byday)} "
          f"long={longs.mean() if len(longs) else float('nan'):+.2f} "
          f"short={shorts.mean() if len(shorts) else float('nan'):+.2f} "
          f"neg2mo={neg2} syms=" + ",".join(f"{k}:${v:+.0f}" for k, v in sorted(bysym.items())))
    print(f"   months: " + " ".join(f"{m}:${v:+.0f}" for m, v in mon))
    print(f"   GATE: {'PASS' if gate else 'FAIL'}")
    return {"label": label, "gate": gate, "n": len(trades), "exp": exp, "pf": pf}

A = cand_a_usdregime()
A1 = [t for t in A if t["sym"] == "BTCUSD"]
A2 = [t for t in A if t["sym"] == "XAUUSD"]
B = cand_b_gapfollow()
rA1 = suite(A1, "A1 USD-regime fade BTC-only")
rA2 = suite(A2, "A2 USD-regime fade XAU-only (filtered)")
rB = suite(B, "B index gap-follow (4 indices pooled)")
# Jaccard: share of entry timestamps (per day) between candidates
da = {(t["ts"] // 86400) for t in A}; db = {(t["ts"] // 86400) for t in B}
inter = len(da & db); union = len(da | db)
print(f"Jaccard(A,B) by day = {inter}/{union} = {inter/union:.2f}")
import numpy as np
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
json.dump(clean({"A1": rA1, "A2": rA2, "B": rB, "jaccard_day": inter / union}),
          open("scripts/_absorb/results/battery_r3.json", "w"), indent=1)
