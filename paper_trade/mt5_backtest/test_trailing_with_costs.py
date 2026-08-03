"""Test original trailing-stop (sim_recon) with FundedNext cost model on all 3 pairs."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sim_recon import ReconSim
from datetime import datetime, timezone

DIR = os.path.dirname(__file__)
PAIRS = ["AUDUSD", "EURAUD", "GBPAUD"]
COMMISSION_PER_LOT = 3.0  # FundedNext $3/round-turn
LOT = 0.5

def pip_value_usd(pair, audusd_rate=0.70):
    """Pip value in USD per standard lot."""
    if pair == "AUDUSD":
        return 10.0
    if pair == "EURAUD":
        # EURAUD: 1 pip = 0.0001, 1 lot = 100k EUR
        # In AUD: 0.0001 * 100000 = 10 AUD
        # In USD: 10 * AUDUSD_rate
        return 10.0 * audusd_rate
    if pair == "GBPAUD":
        # GBPAUD: 1 pip = 0.0001, 1 lot = 100k GBP
        # In AUD: 0.0001 * 100000 = 10 AUD
        # In USD: 10 * AUDUSD_rate
        return 10.0 * audusd_rate
    return 10.0

for pair in PAIRS:
    fn = np.load(os.path.join(DIR, f'fundednext_{pair.lower()}_m1.npy'), allow_pickle=True)
    df = pd.DataFrame(fn)
    
    times = df['time'].values.astype(np.int64)  # seconds (no //10**9)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    spreads = df['spread'].values  # in points
    
    print(f"\n{'='*70}")
    print(f"{pair} — {len(df)} bars, spread med={np.median(spreads)}, p90={np.percentile(spreads,90)}")
    
    # Run original sim_recon with trailing stop (0.5 lot)
    sim = ReconSim(
        z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
        max_hold=54, base_lot=0.5, max_spread=10, trade_dir=0,
        start_hour=0, end_hour=24
    )
    trades = sim.run(times, opens, highs, lows, closes, spreads, df['tick_volume'].values)
    
    if len(trades) == 0:
        print("  0 trades")
        continue
    
    # Cost calculation
    # Spread cost per trade: spread_raw * point_size * contract_size * lot (round-trip)
    # Commission per trade: lot * commission_per_lot
    point_size = 0.00001
    contract_size = 100000
    
    gross = sum(t.pnl for t in trades)
    n = len(trades)
    
    total_commission = n * LOT * COMMISSION_PER_LOT
    
    # Estimate spread cost from median spread
    median_sprd = np.median(spreads)
    # Entry: pay half spread. Exit: pay half spread. Total: one full spread.
    # But the actual spread varies per trade. Use the last known spread as a rough approximation.
    total_spread_cost = sum(
        (t.sprd_entry * point_size * contract_size * LOT)  # full spread per trade
        for t in trades
    )
    
    net = gross - total_spread_cost - total_commission
    
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    neutral = [t for t in trades if abs(t.pnl) < 0.01]
    wr = len(wins) / (n - len(neutral)) * 100 if (n - len(neutral)) > 0 else 0
    
    avg = gross / n if n else 0
    avg_w = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_l = sum(t.pnl for t in losses) / len(losses) if losses else 0
    
    # Also compute net with pip value correction for cross pairs
    if pair in ["EURAUD", "GBPAUD"]:
        # Use close mid-rate for AUDUSD as conversion
        audusd_fn = np.load(os.path.join(DIR, 'fundednext_audusd_m1.npy'), allow_pickle=True)
        audusd_df = pd.DataFrame(audusd_fn)
        audusd_close = audusd_df['close'].values
        # For each trade, look up the closest AUDUSD rate
        total_pnl_usd = 0.0
        for t in trades:
            idx = t.entry_time
            rate = audusd_close[min(idx, len(audusd_close)-1)]
            total_pnl_usd += t.pnl * rate  # convert AUD→USD
        net_usd = total_pnl_usd - total_spread_cost - total_commission
        print(f"  Gross PnL: ${gross:>+7.2f} ({pair} quote, AUD)")
        print(f"  Gross PnL (USD): ${total_pnl_usd:>+7.2f} (converted)")
    else:
        print(f"  Gross PnL: ${gross:>+7.2f}")
    
    print(f"  {n} trades ({len(wins)}W/{len(losses)}L/{len(neutral)}N) WR={wr:.1f}%")
    print(f"  Avg: ${avg:+.2f}/t  Win: ${avg_w:+.2f}  Loss: ${avg_l:+.2f}")
    print(f"  Spread cost: ${total_spread_cost:.2f}  Commission: ${total_commission:.2f}")
    print(f"  Net PnL: ${net:>+7.2f}")
    
    # Max consecutive losses
    streak = 0
    max_streak = 0
    for t in trades:
        if t.pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"  Max consecutive losses: {max_streak}")
    
    # Drawdown
    cum = np.cumsum([t.pnl for t in trades])
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    print(f"  Max drawdown: ${np.max(dd):.2f}")
