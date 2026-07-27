"""Fetch ticks around trade time to reconstruct true M1 bars."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
import datetime
import MetaTrader5 as mt5

PAIRS = ["GBPNZD", "EURNZD", "GBPAUD", "EURAUD", "GBPCAD", "AUDNZD"]
# Trades at 18:09-18:14 UTC July 23
# Get 10 min of ticks around each trade window
TRADE_WINDOW_START = 1784830080  # 18:08 UTC
TRADE_WINDOW_END = 1784830560    # 18:16 UTC

if not mt5.initialize():
    print("MT5 init failed")
    sys.exit(1)

print(f"Fetching ticks {datetime.datetime.fromtimestamp(TRADE_WINDOW_START)} to {datetime.datetime.fromtimestamp(TRADE_WINDOW_END)} (UTC)")

for pair in PAIRS:
    ticks = mt5.copy_ticks_range(pair, TRADE_WINDOW_START, TRADE_WINDOW_END, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        print(f"  {pair}: NO TICK DATA")
        continue
    
    df = pd.DataFrame(ticks)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    
    # Build M1 bars from ticks
    df["minute"] = df["time"].dt.floor("min")
    bars = []
    for minute, group in df.groupby("minute"):
        bars.append({
            "time": minute,
            "open": group["ask"].iloc[0] if "ask" in group.columns else group["bid"].iloc[0],
            "high": group["ask"].max() if "ask" in group.columns else group["bid"].max(),
            "low": group["bid"].min() if "bid" in group.columns else group["ask"].min(),
            "close": group["ask"].iloc[-1] if "ask" in group.columns else group["bid"].iloc[-1],
            "tick_count": len(group),
        })
    
    bar_df = pd.DataFrame(bars).set_index("time")
    out_path = f"research/cppf/_live_data/{pair.lower()}_ticks.parquet"
    os.makedirs("research/cppf/_live_data", exist_ok=True)
    bar_df.to_parquet(out_path)
    
    print(f"\n  {pair}: {len(ticks)} ticks, {len(bar_df)} M1 bars")
    print(f"  Tick range: {df['time'].min()} to {df['time'].max()}")
    
    # Show bars around trade entry window
    for idx, row in bar_df.iterrows():
        print(f"    {idx} O={row['open']:.5f} H={row['high']:.5f} L={row['low']:.5f} C={row['close']:.5f} ({row['tick_count']} ticks)")

mt5.shutdown()
