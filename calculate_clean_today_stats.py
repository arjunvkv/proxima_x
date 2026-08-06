#!/usr/bin/env python3
"""Calculate exact clean Win Rate and Net PnL today per strategy and overall (without duplicates, test trades, or invalid AUDCAD)."""

import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict

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
    hold_min = (d_out.time - d_in.time) / 60.0
    cmt = (d_in.comment or "").strip()

    # Identify strategy name
    if "UltraMonster" in cmt or "Ultra_Monster" in cmt:
        st_name = "Ultra Monster"
    elif "CPPF" in cmt:
        st_name = "CPPF Z"
    elif "CPMC" in cmt:
        st_name = "CPMC Z"
    elif "Tokyo" in cmt:
        st_name = "Tokyo H0"
    elif "NY" in cmt:
        st_name = "NY H21"
    elif "MSV" in cmt:
        st_name = "MSV Asian"
    else:
        st_name = "Manual / Test Script"

    trades.append({
        "pid": pid,
        "entry_time": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": st_name,
        "symbol": d_in.symbol,
        "type": ttype,
        "lot": d_in.volume,
        "pips": round(pips, 1),
        "net_pnl": round(net, 2),
        "win": net > 0,
        "hold_min": round(hold_min, 1),
        "comment": cmt
    })

trades.sort(key=lambda x: x["entry_time"])

# Filter out:
# 1. Manual / Test Script trades (comments starting with Test_ or lot < 0.1)
# 2. Invalid Symbol AUDCAD
# 3. Duplicate EA runs (same timestamp + symbol + lot)
clean_trades = []
seen_signatures = set()

for t in trades:
    # Filter out manual test scripts
    if t["strategy"] == "Manual / Test Script" or t["comment"].startswith("Test_") or t["lot"] < 0.1:
        continue
    # Filter out invalid AUDCAD
    if t["symbol"] == "AUDCAD":
        continue
    
    # Check for duplicate EA runs (same minute + symbol + lot)
    signature = f"{t['entry_time'][:16]}_{t['symbol']}_{t['lot']:.2f}"
    if signature in seen_signatures:
        # Ignore duplicate trade
        continue
    seen_signatures.add(signature)
    
    clean_trades.append(t)

# Calculate statistics per strategy
st_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0, "gross_win": 0.0, "gross_loss": 0.0})

for t in clean_trades:
    st = t["strategy"]
    st_stats[st]["trades"] += 1
    st_stats[st]["net_pnl"] += t["net_pnl"]
    if t["win"]:
        st_stats[st]["wins"] += 1
        st_stats[st]["gross_win"] += t["net_pnl"]
    else:
        st_stats[st]["losses"] += 1
        st_stats[st]["gross_loss"] += abs(t["net_pnl"])

print("=" * 115)
print("CLEAN TRADING SESSION PERFORMANCE TODAY (AUG 3, 2026) — WITHOUT BUGS / DUPLICATES")
print("=" * 115)
print(f"{'Strategy Name':22} {'Trades':8} {'Wins':6} {'Losses':8} {'Win Rate (%)':14} {'Net PnL ($)':14} {'Profit Factor'}")
print("-" * 115)

total_clean_trades = len(clean_trades)
total_clean_wins = sum(s["wins"] for s in st_stats.values())
total_clean_losses = sum(s["losses"] for s in st_stats.values())
total_clean_pnl = sum(s["net_pnl"] for s in st_stats.values())
total_gross_win = sum(s["gross_win"] for s in st_stats.values())
total_gross_loss = sum(s["gross_loss"] for s in st_stats.values())

for st_name, s in sorted(st_stats.items()):
    wr = (s["wins"] / s["trades"] * 100.0) if s["trades"] > 0 else 0.0
    pf = (s["gross_win"] / s["gross_loss"]) if s["gross_loss"] > 0 else (99.0 if s["gross_win"] > 0 else 0.0)
    sign = "+" if s["net_pnl"] >= 0 else ""
    print(f"{st_name:22} {s['trades']:<8} {s['wins']:<6} {s['losses']:<8} {wr:12.1f}% {sign}${s['net_pnl']:12.2f} {pf:12.2f}")

print("-" * 115)
overall_wr = (total_clean_wins / total_clean_trades * 100.0) if total_clean_trades > 0 else 0.0
overall_pf = (total_gross_win / total_gross_loss) if total_gross_loss > 0 else 0.0
overall_sign = "+" if total_clean_pnl >= 0 else ""
print(f"{'TOTAL CLEAN SESSION':22} {total_clean_trades:<8} {total_clean_wins:<6} {total_clean_losses:<8} {overall_wr:12.1f}% {overall_sign}${total_clean_pnl:12.2f} {overall_pf:12.2f}")
print("=" * 115)

mt5.shutdown()
