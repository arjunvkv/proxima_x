"""gap_follow_pooled.py — pooled break-gap follow across indices, cost-aware.

Signal: break gap (open[01:05] - close[23:45 prev]) in top tercile -> LONG at
open, hold to session end (or NY open). Costs: measured spreads + $3/lot comm
via tick values. USD/lot, PF, WF halves, LODO, monthly, per-index, stress x1.5.
"""
import sys, os
import numpy as np
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MARKET = "audit_7_eas/market"
META = {"US30.cash": {"tv": 0.01, "pt": 0.01, "spread_pts": 210.0},
        "US500.cash": {"tv": 0.01, "pt": 0.01, "spread_pts": 60.0},
        "GER40.cash": {"tv": 0.01, "pt": 0.01, "spread_pts": 113.0},
        "UK100.cash": {"tv": 0.01, "pt": 0.01, "spread_pts": 75.0}}

def load(sym):
    return pl.read_parquet(f"{MARKET}/{sym}.pqt").sort("time")

def events(sym):
    s = load(sym)
    ts = s["time"].to_numpy(); op = s["open"].to_numpy(); cl = s["close"].to_numpy()
    n = len(ts)
    reopens = {}; closes = {}
    for i in range(n):
        m, h = (ts[i] // 60) % 60, (ts[i] // 3600) % 24
        d = int(ts[i]) // 86400
        if h == 1 and m == 5: reopens[d] = i
        if h == 23 and m == 45: closes[d] = i
    rows = []
    for d in sorted(reopens):
        if d - 1 not in closes: continue
        i_o, i_c = reopens[d], closes[d - 1]
        gap = op[i_o] - cl[i_c]
        fny = cl[min(i_o + 173, n - 1)] - op[i_o]
        fday = cl[min(i_o + 272, n - 1)] - op[i_o]
        rows.append((sym, d, gap, fny, fday))
    return rows

all_rows = []
for sym in META:
    all_rows.extend(events(sym))
all_rows.sort(key=lambda r: r[1])
# tercile thresholds PER SYMBOL (avoid cross-symbol scale mismatch)
per = {}
for sym in META:
    g = np.array([r[2] for r in all_rows if r[0] == sym])
    per[sym] = np.quantile(g, 2 / 3)

def bat(sym, f, label, spread_x=1.0):
    rows = [r for r in all_rows if r[0] == sym]
    meta = META[sym]
    mult = meta["tv"] / meta["pt"]          # price units -> USD/lot
    spread_u = meta["spread_pts"] * meta["pt"] * spread_x
    comm_u = 6.0 * meta["pt"] / meta["tv"]
    sel = [r for r in rows if r[2] > per[sym]]
    if len(sel) < 15:
        return None
    usd = np.array([(f(r) - spread_u - comm_u) * mult for r in sel])
    days = np.array([r[1] for r in sel])
    wins = usd[usd > 0].sum(); losses = -usd[usd < 0].sum()
    pf = wins / losses if losses > 0 else 99
    z = usd.mean() / (usd.std() / np.sqrt(len(usd)))
    mid = np.median(days)
    n1 = usd[days < mid].sum(); n2 = usd[days >= mid].sum()
    lodo = sum(1 for d in set(days) if usd[days != d].sum() <= 0)
    bymon = {}
    for d, u in zip(days, usd):
        m = int(d) // 30
        bymon[m] = bymon.get(m, 0.0) + u
    print(f"  {sym} {label}: n={len(sel)} exp=${usd.mean():.2f}/lot PF={pf:.2f} z={z:+.2f} "
          f"WF=${n1:+.0f}/${n2:+.0f} LODO={lodo}/{len(set(days))} "
          f"months: " + " ".join(f"{m}:${v:+.0f}" for m, v in sorted(bymon.items())))
    return usd

print("=== pooled gap-follow (top tercile gap -> session) ===")
all_usd = []
for sym in META:
    meta = META[sym]; mult = meta["tv"] / meta["pt"]
    spread_u = meta["spread_pts"] * meta["pt"]; comm_u = 6.0 * meta["pt"] / meta["tv"]
    rows = [r for r in all_rows if r[0] == sym]
    sel = [r for r in rows if r[2] > per[sym]]
    usd = np.array([(r[4] - spread_u - comm_u) * mult for r in sel])
    days = np.array([r[1] for r in sel])
    wins = usd[usd > 0].sum(); losses = -usd[usd < 0].sum()
    pf = wins / losses if losses > 0 else 99
    z = usd.mean() / (usd.std() / np.sqrt(len(usd)))
    mid = np.median(days)
    n1 = usd[days < mid].sum(); n2 = usd[days >= mid].sum()
    lodo = sum(1 for d in set(days) if usd[days != d].sum() <= 0)
    print(f"  {sym}: n={len(sel)} exp=${usd.mean():.2f}/lot PF={pf:.2f} z={z:+.2f} "
          f"WF=${n1:+.0f}/${n2:+.0f} LODO={lodo}/{len(set(days))}")
    for r in sel:
        all_usd.append((r[1], (r[4] - spread_u - comm_u) * mult, sym))
usd = np.array([x[1] for x in all_usd]); days = np.array([x[0] for x in all_usd])
wins = usd[usd > 0].sum(); losses = -usd[usd < 0].sum()
pf = wins / losses if losses > 0 else 99
z = usd.mean() / (usd.std() / np.sqrt(len(usd)))
mid = np.median(days)
n1 = usd[days < mid].sum(); n2 = usd[days >= mid].sum()
lodo = sum(1 for d in set(days) if usd[days != d].sum() <= 0)
print(f"POOLED session: n={len(usd)} exp=${usd.mean():.2f}/lot PF={pf:.2f} z={z:+.2f} "
      f"WF=${n1:+.0f}/${n2:+.0f} LODO={lodo}/{len(set(days))}")
