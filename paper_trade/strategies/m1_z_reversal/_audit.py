"""Audit MT5 trade history — isolate trades by system version (old vs new)."""
import sys, time
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

if not mt5.initialize():
    print(f"MT5 init failed: {mt5.last_error()}")
    raise SystemExit(1)

now = datetime.now()
deals = mt5.history_deals_get(now - timedelta(days=3), now)

rows = []
for d in deals:
    rows.append({
        "ticket": d.ticket, "time": d.time, "time_msc": d.time_msc,
        "symbol": d.symbol, "type": d.type, "entry": d.entry,
        "volume": d.volume, "price": d.price, "profit": d.profit,
        "position_id": d.position_id,
    })
df = pd.DataFrame(rows)
df["time_dt"] = pd.to_datetime(df["time"], unit="s")

# Reconstruct trades by position_id
trades = []
for pos_id, grp in df.groupby("position_id"):
    entries = grp[grp["entry"] == 0]
    exits = grp[grp["entry"] == 1]
    if len(entries) == 0 or len(exits) == 0:
        continue
    
    entry = entries.iloc[0]
    exit_ = exits.iloc[0]
    dur_s = exit_["time"] - entry["time"]
    dur_ms = (exit_["time_msc"] - entry["time_msc"]) / 1000.0
    
    direction = "BUY" if entry["type"] == 0 else "SELL"
    trades.append({
        "position_id": pos_id,
        "symbol": entry["symbol"], "direction": direction,
        "entry_time": entry["time_dt"], "exit_time": exit_["time_dt"],
        "entry_price": entry["price"], "exit_price": exit_["price"],
        "volume": entry["volume"],
        "profit": exit_["profit"],
        "dur_s": dur_s, "dur_ms": dur_ms,
        "entry_hour": entry["time_dt"].hour,
    })

trades_df = pd.DataFrame(trades).sort_values("entry_time")
if len(trades_df) == 0:
    print("No completed trades found")
    mt5.shutdown()
    raise SystemExit(0)

# Classify by duration pattern
def classify(row):
    d = row["dur_s"]
    if d <= 30: return "quick_stop (<=30s)"
    elif d <= 600: return "trail_stop (30s-10m)"
    elif d <= 3600: return "trail_stop (10m-1h)"
    else: return "no_stop_bug (>1h)"
trades_df["class"] = trades_df.apply(classify, axis=1)

print(f"\n{'='*80}")
print(f"TRADE AUDIT — last 3 days ({len(trades_df)} completed trades)")
print(f"{'='*80}\n")

# Version clusters by time
print("TRADE CLUSTERS (time-ordered):")
print("-"*80)
for _, r in trades_df.iterrows():
    wl = "W" if r["profit"] > 0 else "L"
    z = "Z" if r["profit"] == 0 else wl
    print(f"  {r['entry_time'].strftime('%m/%d %H:%M')} {r['symbol']:6s} "
          f"{r['direction']:4s} entry={r['entry_price']:.5f} → exit={r['exit_price']:.5f} "
          f"PnL=${r['profit']:+6.2f} [{z}] dur={r['dur_s']:>5.0f}s "
          f"class={r['class']}")

print(f"\n{'='*80}")
print("SUMMARY BY TRADE CLASS")
print(f"{'='*80}")
for cls, grp in sorted(trades_df.groupby("class")):
    w = len(grp[grp["profit"] > 0])
    l = len(grp[grp["profit"] < 0])
    z = len(grp[grp["profit"] == 0])
    total = len(grp)
    print(f"\n  {cls}:")
    print(f"    Count: {total} ({w}W/{l}L/{z}Z)")
    print(f"    WR: {w/max(total-z,1)*100:.1f}%")
    print(f"    Total PnL: ${grp['profit'].sum():+.2f}")
    print(f"    Avg PnL: ${grp['profit'].mean():+.2f}")
    if w > 0: print(f"    Avg Win: ${grp[grp['profit']>0]['profit'].mean():+.2f}")
    if l > 0: print(f"    Avg Loss: ${grp[grp['profit']<0]['profit'].mean():+.2f}")

    # Per pair
    for sym, sg in grp.groupby("symbol"):
        sw = len(sg[sg["profit"] > 0])
        sl = len(sg[sg["profit"] < 0])
        print(f"      {sym}: {len(sg)}t {sw}W/{sl}L ${sg['profit'].sum():+.2f}")

# EURUSD-only analysis (the only pair backtest says works)
print(f"\n{'='*80}")
print("EURUSD ONLY — ALL CLASSES")
print(f"{'='*80}")
eu = trades_df[trades_df["symbol"] == "EURUSD"]
if len(eu) > 0:
    w = len(eu[eu["profit"] > 0])
    l = len(eu[eu["profit"] < 0])
    print(f"  Total: {len(eu)}t {w}W/{l}L")
    print(f"  WR: {w/max(w+l,1)*100:.1f}%")
    print(f"  Total PnL: ${eu['profit'].sum():+.2f}")
    print(f"  Avg PnL: ${eu['profit'].mean():+.2f}")
    
    per_class = eu.groupby("class")
    for cls, grp in sorted(per_class):
        cw = len(grp[grp["profit"] > 0])
        cl = len(grp[grp["profit"] < 0])
        print(f"    {cls}: {len(grp)}t {cw}W/{cl}L ${grp['profit'].sum():+.2f}")
    
    # Exclude the no-stop bug trades
    clean = eu[eu["class"] != "no_stop_bug (>1h)"]
    if len(clean) > 0:
        cw = len(clean[clean["profit"] > 0])
        cl = len(clean[clean["profit"] < 0])
        print(f"\n  EURUSD (excluding no-stop bug): {len(clean)}t {cw}W/{cl}L")
        print(f"  WR: {cw/max(cw+cl,1)*100:.1f}%")
        print(f"  Total PnL: ${clean['profit'].sum():+.2f}")
        print(f"  Avg PnL: ${clean['profit'].mean():+.2f}")

# EURJPY only
print(f"\n{'='*80}")
print("EURJPY ONLY — ALL CLASSES")
print(f"{'='*80}")
ej = trades_df[trades_df["symbol"] == "EURJPY"]
if len(ej) > 0:
    w = len(ej[ej["profit"] > 0])
    l = len(ej[ej["profit"] < 0])
    print(f"  Total: {len(ej)}t {w}W/{l}L")
    print(f"  WR: {w/max(w+l,1)*100:.1f}%")
    print(f"  Total PnL: ${ej['profit'].sum():+.2f}")
    
    per_class = ej.groupby("class")
    for cls, grp in sorted(per_class):
        cw = len(grp[grp["profit"] > 0])
        cl = len(grp[grp["profit"] < 0])
        print(f"    {cls}: {len(grp)}t {cw}W/{cl}L ${grp['profit'].sum():+.2f}")

# Current open positions
print(f"\n{'='*80}")
print("CURRENT OPEN POSITIONS")
print(f"{'='*80}")
positions = mt5.positions_get()
if positions:
    for p in positions:
        age_s = time.time() - p.time
        print(f"  {p.symbol} {'BUY' if p.type==0 else 'SELL'} "
              f"open={p.price_open:.5f} vol={p.volume:.2f} "
              f"profit={p.profit:+.2f} age={age_s:.0f}s")
        if p.sl == 0 and p.tp == 0:
            print(f"    ⚠️  NO STOP LOSS — naked position")

# Risk analysis
print(f"\n{'='*80}")
print("RISK ANALYSIS")
print(f"{'='*80}")
total_loses = trades_df[trades_df["profit"] < 0]
if len(total_loses) > 0:
    largest_loss = total_loses.nsmallest(3, "profit")
    print(f"  Largest losses:")
    for _, r in largest_loss.iterrows():
        print(f"    {r['entry_time'].strftime('%m/%d %H:%M')} {r['symbol']} "
              f"{r['direction']} {r['entry_price']:.5f}→{r['exit_price']:.5f} "
              f"${r['profit']:+.2f} dur={r['dur_s']:.0f}s [{r['class']}]")

print(f"\n  Total risked: ${abs(total_loses['profit'].sum()):.2f} loss from {len(trades_df)} trades")
print(f"  Gross: ${trades_df['profit'].sum():+.2f}")
print(f"  Open risk: ${sum(p.profit for p in positions) if positions else 0:+.2f} (unrealized)")

mt5.shutdown()
