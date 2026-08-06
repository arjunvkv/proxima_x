#!/usr/bin/env python3
"""Full today analysis with proper UTC — all trades since midnight UTC with session grouping."""

import MetaTrader5 as mt5
from datetime import datetime, timedelta
from collections import defaultdict

mt5.initialize()
mt5.login(1514168544, "$!4fwBIc", "FTMO-Demo")

# Pull ALL of today (UTC)
from_date = datetime(2026, 8, 3, 0, 0, 0)
to_date   = datetime(2026, 8, 4, 23, 59, 59)

deals = mt5.history_deals_get(from_date, to_date)
print(f"Raw deals fetched: {len(deals) if deals else 0}")

positions = {}
for d in deals:
    if not d.symbol or d.entry not in [0, 1]:
        continue
    pid = d.position_id
    if pid not in positions:
        positions[pid] = {"in": None, "out": None}
    if d.entry == 0:
        positions[pid]["in"] = d
    elif d.entry == 1:
        positions[pid]["out"] = d

trades = []
for pid, p in positions.items():
    d_in, d_out = p["in"], p["out"]
    if not d_in or not d_out:
        continue
    net = d_out.profit + d_out.swap + d_out.commission
    pip_m = 100.0 if "JPY" in d_in.symbol else 10000.0
    ttype = "BUY" if d_in.type == 0 else "SELL"
    pips = (d_out.price - d_in.price)*pip_m if ttype=="BUY" else (d_in.price - d_out.price)*pip_m
    entry_dt = datetime.utcfromtimestamp(d_in.time)
    hold_min = (d_out.time - d_in.time) / 60.0
    trades.append({
        "pid": pid,
        "symbol": d_in.symbol,
        "type": ttype,
        "lot": d_in.volume,
        "entry": d_in.price,
        "exit": d_out.price,
        "pips": round(pips, 1),
        "net": round(net, 2),
        "win": net > 0,
        "entry_dt": entry_dt,
        "hold_min": round(hold_min, 1),
        "comment": d_in.comment or "",
    })

trades.sort(key=lambda x: x["entry_dt"])

# Filter out tiny test trades (< 0.05L)
real_trades = [t for t in trades if t["lot"] >= 0.1]
test_trades = [t for t in trades if t["lot"] < 0.1]

print(f"\n{'='*110}")
print(f"ALL REAL TRADES TODAY (lot >= 0.1L) — {len(real_trades)} total, {len(test_trades)} sub-0.1L test trades excluded")
print(f"{'='*110}")
print(f"{'Time (UTC)':19} {'Symbol':8} {'T':4} {'Lot':5} {'Entry':10} {'Exit':10} {'Pips':6} {'Net':9} {'Hold':6} {'Comment':25} W/L")
print("-"*110)

wins = losses = 0
for t in real_trades:
    wl = "WIN " if t["win"] else "LOSS"
    if t["win"]: wins += 1
    else: losses += 1
    print(f"{str(t['entry_dt'])[:19]:19} {t['symbol']:8} {t['type'][:1]:4} {t['lot']:5.2f} {t['entry']:10.5f} {t['exit']:10.5f} {t['pips']:6.1f} ${t['net']:8.2f} {t['hold_min']:5.1f}m {t['comment'][:25]:25} {wl}")

total = wins + losses
wr = wins/total*100 if total else 0
net_total = sum(t["net"] for t in real_trades)
print("-"*110)
print(f"TOTAL: {total} | WINS: {wins} | LOSSES: {losses} | WR: {wr:.1f}% | NET PnL: ${net_total:.2f}")

# --- Duplicate detection ---
print(f"\n{'='*60}")
print("DUPLICATE DETECTION")
print(f"{'='*60}")
buckets = defaultdict(list)
for t in real_trades:
    minute_key = t["entry_dt"].strftime("%H:%M")
    key = f"{t['symbol']}|{t['type']}|{t['lot']:.2f}|{minute_key}"
    buckets[key].append(t)

found_dupe = False
for key, group in buckets.items():
    if len(group) > 1:
        found_dupe = True
        total_loss = sum(g["net"] for g in group if not g["win"])
        print(f"  ⚠️  DUPLICATE x{len(group)}: {key}  (extra loss: ${total_loss:.2f})")
        for g in group:
            icon = "✅" if g["win"] else "❌"
            print(f"      {icon} PID {g['pid']} | {g['pips']:+.1f}pip | ${g['net']:.2f} | hold {g['hold_min']}m | {g['comment']}")
if not found_dupe:
    print("  No duplicates found.")

# --- Session breakdown ---
print(f"\n{'='*60}")
print("SESSION BREAKDOWN (per :00 and :30 bar)")
print(f"{'='*60}")
session_trades = defaultdict(list)
for t in real_trades:
    m = t["entry_dt"].minute
    h = t["entry_dt"].hour
    snap_m = 0 if m < 30 else 30
    key = t["entry_dt"].strftime(f"%Y-%m-%d %H:{snap_m:02d}")
    session_trades[key].append(t)

for skey in sorted(session_trades.keys()):
    group = session_trades[skey]
    s_wins  = sum(1 for t in group if t["win"])
    s_pnl   = sum(t["net"] for t in group)
    lots    = sorted(set(f"{t['lot']:.2f}" for t in group))
    symbols = [t["symbol"] for t in group]
    dupe_flag = "⚠️ DUPES" if len(symbols) != len(set(symbols)) else ""
    print(f"  {skey} | {len(group):2}x | {s_wins}/{len(group)} wins | ${s_pnl:+.2f} | {dupe_flag}")
    for t in group:
        icon = "✅" if t["win"] else "❌"
        print(f"    {icon} {t['symbol']:8} {t['type']:4} {t['lot']:.2f}L | {t['pips']:+.1f}pip | ${t['net']:+.2f} | hold {t['hold_min']}m | {t['comment']}")

mt5.shutdown()
