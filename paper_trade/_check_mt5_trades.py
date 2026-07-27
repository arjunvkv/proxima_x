"""Check MT5 trade history: wins, losses, execution times."""
import sys, time
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

if not mt5.initialize():
    print(f"MT5 init failed: {mt5.last_error()}")
    raise SystemExit(1)

print("Connected to MT5")

# ── Current open positions ──
positions = mt5.positions_get()
print(f"\n=== OPEN POSITIONS ({len(positions) if positions else 0}) ===")
if positions:
    for p in positions:
        age_s = time.time() - p.time
        print(f"  {p.symbol} {'BUY' if p.type==0 else 'SELL'} vol={p.volume} "
              f"open={p.price_open:.5f} profit={p.profit:.2f} "
              f"age={age_s/60:.0f}min sl={p.sl} tp={p.tp}")

# ── Closed trades ──
print(f"\n=== CLOSED TRADES (last 3 days) ===")
now = datetime.now()
deals = mt5.history_deals_get(now - timedelta(days=3), now)

if not deals:
    print("  No deals found")
    mt5.shutdown()
    raise SystemExit(0)

# Build DataFrame with position_id for pairing
rows = []
for d in deals:
    rows.append({
        "ticket": d.ticket, "time": d.time, "time_msc": d.time_msc,
        "symbol": d.symbol, "type": d.type, "entry": d.entry,
        "volume": d.volume, "price": d.price, "profit": d.profit,
        "position_id": d.position_id, "comment": d.comment,
    })
df = pd.DataFrame(rows)
df["time_dt"] = pd.to_datetime(df["time"], unit="s")

# ── Show last 30 deals raw ──
print(f"\n--- Last 30 deals raw ---")
for _, r in df.tail(30).iterrows():
    et = "in" if r["entry"] == 0 else "out"
    tp = "BUY" if r["type"] == 0 else "SELL"
    print(f"  {r['time_dt'].strftime('%m/%d %H:%M:%S')} {r['symbol']} "
          f"{tp}/{et} vol={r['volume']:.2f} price={r['price']:.5f} "
          f"pnl=${r['profit']:+.2f} pos#{r['position_id']}")

# ── Group by position_id to reconstruct trades ──
print(f"\n--- RECONSTRUCTED TRADES ---")
trades = []
for pos_id, grp in df.groupby("position_id"):
    entries = grp[grp["entry"] == 0]
    exits = grp[grp["entry"] == 1]
    if len(entries) == 0 or len(exits) == 0:
        continue  # still open
    
    entry_row = entries.iloc[0]
    exit_row = exits.iloc[0]
    
    dur_s = exit_row["time"] - entry_row["time"]
    dur_ms = (exit_row["time_msc"] - entry_row["time_msc"]) / 1000
    sym = entry_row["symbol"]
    direction = "BUY" if entry_row["type"] == 0 else "SELL"
    
    trades.append({
        "symbol": sym, "direction": direction,
        "entry_time": entry_row["time_dt"], "exit_time": exit_row["time_dt"],
        "entry_price": entry_row["price"], "exit_price": exit_row["price"],
        "volume": entry_row["volume"],
        "profit": exit_row["profit"],
        "duration_s": dur_s, "duration_ms": dur_ms,
        "position_id": pos_id,
    })

trades_df = pd.DataFrame(trades)
if len(trades_df) == 0:
    print("  No completed trades found")
else:
    trades_df = trades_df.sort_values("entry_time")
    wins = trades_df[trades_df["profit"] > 0]
    losses = trades_df[trades_df["profit"] < 0]
    
    print(f"\n  --- Summary ---")
    print(f"  Total: {len(trades_df)} trades")
    print(f"  Wins: {len(wins)} ({len(wins)/len(trades_df)*100:.1f}%)")
    print(f"  Losses: {len(losses)} ({len(losses)/len(trades_df)*100:.1f}%)")
    print(f"  Gross PnL: ${trades_df['profit'].sum():+.2f}")
    print(f"  Avg win: ${wins['profit'].mean():+.2f}" if len(wins) > 0 else "")
    print(f"  Avg loss: ${losses['profit'].mean():+.2f}" if len(losses) > 0 else "")
    print(f"  Best: ${trades_df['profit'].max():+.2f}")
    print(f"  Worst: ${trades_df['profit'].min():+.2f}")
    
    # Per pair
    print(f"\n  --- Per Pair ---")
    for sym, grp in trades_df.groupby("symbol"):
        pw = len(grp[grp["profit"] > 0])
        pl = len(grp[grp["profit"] < 0])
        print(f"  {sym}: {len(grp)} trades, {pw}W/{pl}L, "
              f"PnL=${grp['profit'].sum():+.2f}, "
              f"avg=${grp['profit'].mean():+.2f}")
    
    # Duration analysis
    print(f"\n  --- Duration Analysis ---")
    print(f"  Avg duration: {trades_df['duration_s'].mean():.0f}s ({trades_df['duration_s'].mean()/60:.1f}min)")
    print(f"  Median duration: {trades_df['duration_s'].median():.0f}s")
    print(f"  Min-Max: {trades_df['duration_s'].min():.0f}s - {trades_df['duration_s'].max():.0f}s")
    if len(wins) > 0:
        print(f"  Avg win duration: {wins['duration_s'].mean():.0f}s")
    if len(losses) > 0:
        print(f"  Avg loss duration: {losses['duration_s'].mean():.0f}s")
    
    print(f"\n  --- Duration by PnL buckets ---")
    for label, grp in [("Winners", wins), ("Losers", losses)]:
        if len(grp) == 0: continue
        print(f"  {label}:")
        print(f"    Count: {len(grp)}")
        print(f"    Duration: mean={grp['duration_s'].mean():.0f}s median={grp['duration_s'].median():.0f}s")
        print(f"    Entry prices: mean={grp['entry_price'].mean():.5f}")
        print(f"    Exit prices: mean={grp['exit_price'].mean():.5f}")
    
    # Big winners and losers
    print(f"\n  --- 5 Biggest Wins ---")
    for _, r in wins.nlargest(5, "profit").iterrows():
        print(f"  {r['entry_time'].strftime('%m/%d %H:%M')} {r['symbol']} "
              f"{r['direction']} entry={r['entry_price']:.5f} exit={r['exit_price']:.5f} "
              f"PnL=${r['profit']:+.2f} dur={r['duration_s']:.0f}s")
    
    print(f"\n  --- 5 Biggest Losses ---")
    for _, r in losses.nsmallest(5, "profit").iterrows():
        print(f"  {r['entry_time'].strftime('%m/%d %H:%M')} {r['symbol']} "
              f"{r['direction']} entry={r['entry_price']:.5f} exit={r['exit_price']:.5f} "
              f"PnL=${r['profit']:+.2f} dur={r['duration_s']:.0f}s")

mt5.shutdown()
