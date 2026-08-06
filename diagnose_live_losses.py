#!/usr/bin/env python3
"""Pull ALL trades from MT5 demo account and diagnose why losses don't match expected WR."""

import MetaTrader5 as mt5
from datetime import datetime, timedelta

mt5.initialize()
mt5.login(1514168544, "$!4fwBIc", "FTMO-Demo")

acc = mt5.account_info()
print(f"Account: {acc.login} | Balance: ${acc.balance:.2f} | Equity: ${acc.equity:.2f}")
print()

# Get ALL history
from_date = datetime(2026, 7, 28)  # last week
to_date = datetime.now() + timedelta(days=1)
deals = mt5.history_deals_get(from_date, to_date)

if not deals:
    print("No deals found")
    mt5.shutdown()
    exit()

# Build completed positions (IN+OUT pairs)
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
    entry_time = datetime.fromtimestamp(d_in.time)
    exit_time = datetime.fromtimestamp(d_out.time)
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
        "entry_time": entry_time,
        "hold_min": round(hold_min, 1),
    })

trades.sort(key=lambda x: x["entry_time"])

print(f"{'Time (UTC+3)':19} {'Symbol':8} {'Type':4} {'Lot':4} {'Entry':9} {'Exit':9} {'Pips':6} {'Net PnL':9} {'Hold':7} {'W/L'}")
print("-" * 105)
wins = losses = 0
for t in trades:
    wl = "WIN" if t["win"] else "LOSS"
    if t["win"]: wins += 1
    else: losses += 1
    print(f"{str(t['entry_time'])[:19]:19} {t['symbol']:8} {t['type']:4} {t['lot']:4} {t['entry']:9.5f} {t['exit']:9.5f} {t['pips']:6.1f} ${t['net']:8.2f} {t['hold_min']:5.0f}min {wl}")

total = wins + losses
wr = wins/total*100 if total else 0
total_pnl = sum(t["net"] for t in trades)
print("-" * 105)
print(f"TOTAL: {total} trades | WINS: {wins} | LOSSES: {losses} | WR: {wr:.1f}% | NET PnL: ${total_pnl:.2f}")

# Group by symbol
print()
print("--- BY PAIR ---")
syms = {}
for t in trades:
    s = t["symbol"]
    if s not in syms:
        syms[s] = {"trades":0,"wins":0,"pnl":0.0}
    syms[s]["trades"] += 1
    syms[s]["wins"] += int(t["win"])
    syms[s]["pnl"] += t["net"]
for s, v in sorted(syms.items()):
    wr_s = v["wins"]/v["trades"]*100
    print(f"  {s:8} | {v['trades']:3} trades | WR {wr_s:.0f}% | PnL ${v['pnl']:.2f}")

# Group by hour of day to check which hours are losing
print()
print("--- BY ENTRY HOUR (UTC+3 server time) ---")
hours = {}
for t in trades:
    h = t["entry_time"].hour
    if h not in hours:
        hours[h] = {"trades":0,"wins":0,"pnl":0.0}
    hours[h]["trades"] += 1
    hours[h]["wins"] += int(t["win"])
    hours[h]["pnl"] += t["net"]
for h in sorted(hours.keys()):
    v = hours[h]
    wr_h = v["wins"]/v["trades"]*100
    flag = " ⚠️ LOW WR" if wr_h < 50 else ""
    print(f"  {h:02d}:00 | {v['trades']:3} trades | WR {wr_h:.0f}% | PnL ${v['pnl']:.2f}{flag}")

# Check hold times — any abnormal exits?
print()
print("--- HOLD TIME DISTRIBUTION ---")
under5 = [t for t in trades if t["hold_min"] < 5]
normal = [t for t in trades if 10 <= t["hold_min"] <= 20]
over30 = [t for t in trades if t["hold_min"] > 30]
print(f"  <5 min hold  : {len(under5)} trades {' ← SUSPICIOUS (premature close?)' if under5 else ''}")
print(f"  10-20 min    : {len(normal)} trades (expected for HOLD_BARS=3)")
print(f"  >30 min hold : {len(over30)} trades {' ← SUSPICIOUS (no exit fired?)' if over30 else ''}")

if under5:
    print()
    print("  PREMATURE CLOSES (< 5 min hold):")
    for t in under5:
        print(f"    {t['entry_time']} | {t['symbol']} {t['type']} | hold {t['hold_min']}min | ${t['net']:.2f}")

mt5.shutdown()
