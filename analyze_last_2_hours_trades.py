#!/usr/bin/env python3
"""Fetch and analyze all MT5 deals in the last 4 hours (queries full server time window)."""

import MetaTrader5 as mt5
from datetime import datetime, timedelta

mt5.initialize()
mt5.login(1514168544, "$!4fwBIc", "FTMO-Demo")

# Query full today history and slice by last 3 hours of activity
from_date = datetime(2026, 8, 3, 0, 0, 0)
to_date = datetime(2026, 8, 4, 23, 59, 59)

deals = mt5.history_deals_get(from_date, to_date)

positions = {}
if deals:
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
    exit_dt  = datetime.utcfromtimestamp(d_out.time)
    hold_min = (d_out.time - d_in.time) / 60.0
    cmt = (d_in.comment or "").strip()

    trades.append({
        "pid": pid,
        "symbol": d_in.symbol,
        "type": ttype,
        "lot": d_in.volume,
        "entry_price": d_in.price,
        "exit_price": d_out.price,
        "pips": round(pips, 1),
        "net_pnl": round(net, 2),
        "win": net > 0,
        "entry_time": entry_dt,
        "exit_time": exit_dt,
        "hold_min": round(hold_min, 1),
        "comment": cmt
    })

trades.sort(key=lambda x: x["entry_time"])

# Filter for trades entered in the last 4 hours of the dataset
if trades:
    latest_entry = trades[-1]["entry_time"]
    cutoff = latest_entry - timedelta(hours=3)
    recent_trades = [t for t in trades if t["entry_time"] >= cutoff]
    
    print(f"=== RECENT TRADES IN THE LAST 3 HOURS (Cutoff: {cutoff.strftime('%H:%M')} UTC) ===")
    print(f"Total recent trades found: {len(recent_trades)}")
    print(f"\n{'Entry Time (UTC)':19} {'Symbol':8} {'Type':4} {'Lot':5} {'Pips':6} {'Net PnL':9} {'Hold':6} {'Comment':25} {'Status'}")
    print("-" * 105)
    wins = losses = 0
    total_pnl = 0.0
    for t in recent_trades:
        wl = "WIN 🟢" if t["win"] else "LOSS 🔴"
        if t["win"]: wins += 1
        else: losses += 1
        total_pnl += t["net_pnl"]
        sign = "+" if t['net_pnl'] >= 0 else ""
        print(f"{str(t['entry_time'])[:19]:19} {t['symbol']:8} {t['type']:4} {t['lot']:5.2f} {t['pips']:6.1f} {sign}${t['net_pnl']:8.2f} {t['hold_min']:5.1f}m {t['comment']:25} {wl}")

    print("-" * 105)
    print(f"RECENT WINDOW SUMMARY: {len(recent_trades)} trades | Wins: {wins} | Losses: {losses} | Net PnL: ${total_pnl:.2f}")

mt5.shutdown()
