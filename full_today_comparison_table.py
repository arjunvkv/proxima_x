#!/usr/bin/env python3
"""Build complete, full-day comparison table of ALL trades executed today on MT5 VPS demo account."""

import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd

mt5.initialize()
mt5.login(1514168544, "$!4fwBIc", "FTMO-Demo")

from_date = datetime(2026, 8, 3, 0, 0, 0)
to_date   = datetime(2026, 8, 4, 23, 59, 59)

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

    # Categorize trade
    if "v106" in cmt and "v107" in cmt:
        category = "Duplicate EA Run ❌"
    elif d_in.symbol == "AUDCAD":
        category = "Invalid Symbol ❌"
    elif cmt.startswith("Test_") or d_in.volume < 0.1:
        category = "Manual Test Script ⚠️"
    elif net >= 0:
        category = "Clean Engine WIN 🟢"
    else:
        # Check if timestamp matches another trade exactly (duplicate)
        category = "Clean Engine LOSS 🔴"

    trades.append({
        "pid": pid,
        "entry_time": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": d_in.symbol,
        "type": ttype,
        "lot": d_in.volume,
        "entry_price": d_in.price,
        "exit_price": d_out.price,
        "pips": round(pips, 1),
        "net_pnl": round(net, 2),
        "hold_min": round(hold_min, 1),
        "comment": cmt,
        "category": category
    })

trades.sort(key=lambda x: x["entry_time"])

# Check for duplicates by timestamp + symbol + volume
seen = set()
for t in trades:
    t_key = f"{t['entry_time'][:16]}_{t['symbol']}_{t['lot']:.2f}"
    if t_key in seen:
        if "Clean Engine" in t["category"]:
            t["category"] = "Duplicate EA Run ❌"
    else:
        seen.add(t_key)

df = pd.DataFrame(trades)

print("=" * 135)
print("FULL TODAY MASTER COMPARISON TABLE — MT5 LIVE VPS TRADES (AUG 3, 2026)")
print("=" * 135)
print(f"{'Pos ID':9} {'Time (UTC)':19} {'Symbol':8} {'Type':4} {'Lot':5} {'Entry':9} {'Exit':9} {'Pips':6} {'Net PnL':9} {'Hold':6} {'Comment':24} {'Category'}")
print("-" * 135)

clean_wins = clean_losses = dupe_losses = test_losses = invalid_symbol_losses = 0
clean_pnl = 0.0

for t in trades:
    sign = "+" if t['net_pnl'] >= 0 else ""
    print(f"{t['pid']:<9} {t['entry_time']:19} {t['symbol']:8} {t['type']:4} {t['lot']:5.2f} {t['entry_price']:9.5f} {t['exit_price']:9.5f} {t['pips']:6.1f} {sign}${t['net_pnl']:8.2f} {t['hold_min']:5.1f}m {t['comment'][:24]:24} {t['category']}")
    
    if "Clean Engine WIN" in t["category"]:
        clean_wins += 1
        clean_pnl += t["net_pnl"]
    elif "Clean Engine LOSS" in t["category"]:
        clean_losses += 1
        clean_pnl += t["net_pnl"]
    elif "Duplicate EA" in t["category"]:
        dupe_losses += 1
    elif "Manual Test" in t["category"]:
        test_losses += 1
    elif "Invalid Symbol" in t["category"]:
        invalid_symbol_losses += 1

print("-" * 135)
total_trades = len(trades)
total_pnl = sum(t["net_pnl"] for t in trades)
print(f"TOTAL EXECUTED TRADES TODAY: {total_trades} | REALIZED PnL: ${total_pnl:.2f}")
print("=" * 135)

print("\n--- MASTER AUDIT SUMMARY BREAKDOWN ---")
print(f"  • Clean Strategy Wins          : {clean_wins} trades")
print(f"  • Clean Strategy Losses        : {clean_losses} trades")
print(f"  • Duplicate EA Extra Losses    : {dupe_losses} trades")
print(f"  • Invalid Symbol (AUDCAD)      : {invalid_symbol_losses} trades")
print(f"  • Manual Test Script Runs      : {test_losses} trades")
print(f"  • TRUE CLEAN ENGINE NET PnL    : +${clean_pnl:.2f} 🟢")
print("=" * 135)

mt5.shutdown()
