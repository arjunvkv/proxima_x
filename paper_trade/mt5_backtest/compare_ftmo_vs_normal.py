"""Compare FTMO vs normal MT5 on same sim config, trade-by-trade."""
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sim_recon import ReconSim


def load_data(path):
    data = np.load(path, allow_pickle=True)
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.columns = [c.lower() if c != 'time' else 'time' for c in df.columns]
    return df


def run_sim(df, **kwargs):
    sim = ReconSim(
        z_threshold=4.0, stop_a=3.0, trig_a=1.0, gap_a=0.05,
        max_hold=54, base_lot=0.5, max_spread=5,
        trade_dir=1, start_hour=0, end_hour=7,
        **kwargs
    )
    trades = sim.run(
        df['time'].astype(np.int64)//10**9,
        df['open'].values, df['high'].values,
        df['low'].values, df['close'].values,
        df['spread'].values, df['tick_volume'].values
    )
    return trades


def analyze_trades(name, trades):
    tp = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    neutral = [t for t in trades if abs(t.pnl) < 0.01]
    n = len(trades)
    denom = n - len(neutral)
    wr = len(wins)/denom*100 if denom > 0 else 0
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    print(f"\n=== {name} ===")
    print(f"  Trades: {n} ({len(wins)}W/{len(losses)}L/{len(neutral)}N) WR: {wr:.1f}%")
    print(f"  PnL: ${tp:.2f}  AvgWin: ${avg_win:.2f}  AvgLoss: ${avg_loss:.2f}")
    print(f"  Held bars: avg={np.mean([t.bars_held for t in trades]):.1f} "
          f"min={min(t.bars_held for t in trades)} max={max(t.bars_held for t in trades)}")
    for t in trades:
        print(f"    {t.entry_price:.5f} ATR={t.atr_entry:.6f} z={t.z_entry:.2f} "
              f"sprd={t.sprd_entry:.0f} -> {t.exit_reason} ${t.pnl:+.2f} held={t.bars_held}b")
    return trades


# Load both datasets
ftmo_df = load_data(r'C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\ftmo_audusd_m1.npy')
normal_df = load_data(r'C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\normal_audusd_m1.npy')

print(f"FTMO data:   {len(ftmo_df)} bars, {ftmo_df['time'].min()} to {ftmo_df['time'].max()}")
print(f"Normal data: {len(normal_df)} bars, {normal_df['time'].min()} to {normal_df['time'].max()}")

# Compare basic stats
print(f"\nFTMO spread range:   {ftmo_df['spread'].min()}-{ftmo_df['spread'].max()}")
print(f"Normal spread range: {normal_df['spread'].min()}-{normal_df['spread'].max()}")

# Compare spread histograms in Asian hours (0-7)
ftmo_asia = ftmo_df[ftmo_df['time'].dt.hour < 7]
normal_asia = normal_df[normal_df['time'].dt.hour < 7]
print(f"\nAsian session spread stats:")
print(f"  FTMO:   mean={ftmo_asia['spread'].mean():.1f} median={ftmo_asia['spread'].median():.1f} "
      f"std={ftmo_asia['spread'].std():.1f}")
print(f"  Normal: mean={normal_asia['spread'].mean():.1f} median={normal_asia['spread'].median():.1f} "
      f"std={normal_asia['spread'].std():.1f}")

# Compare price ranges
print(f"\nFTMO price range:   {ftmo_df['low'].min():.5f}-{ftmo_df['high'].max():.5f}")
print(f"Normal price range: {normal_df['low'].min():.5f}-{normal_df['high'].max():.5f}")

# Run configs on both
configs = [
    ("Baseline (no filters)", {}),
    ("ATR=0.00007 only", {"limit_entry_atr": 0.00007}),
    ("Gate relaxed (0.3/z0)", {"enable_stability": True, "stab_thresh": 0.3, "z_cum_min": 0.0}),
    ("ATR07+Gate(0.3/z0)", {"limit_entry_atr": 0.00007, "enable_stability": True, "stab_thresh": 0.3, "z_cum_min": 0.0}),
    ("ATR=0.00010 only", {"limit_entry_atr": 0.00010}),
    ("Gate strict (0.5/z5)", {"enable_stability": True, "stab_thresh": 0.5, "z_cum_min": 5.0}),
]

for name, kwargs in configs:
    ftmo_trades = run_sim(ftmo_df, **kwargs)
    normal_trades = run_sim(normal_df, **kwargs)
    
    analyze_trades(f"FTMO {name}", ftmo_trades)
    analyze_trades(f"Normal {name}", normal_trades)
    print()
