"""Sweep z-threshold and lot size to find challenge-viable config on FundedNext."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sim_recon import ReconSim

DIR = os.path.dirname(__file__)
PAIRS = ["AUDUSD", "EURAUD", "GBPAUD"]
COMMISSION = 3.0
POINT_SIZE = 0.00001
CONTRACT_SIZE = 100000

SWEEPS = [
    {"z": 3.5, "sprd": 10, "label": "z3.5_sp10"},
    {"z": 3.0, "sprd": 10, "label": "z3.0_sp10"},
    {"z": 2.5, "sprd": 10, "label": "z2.5_sp10"},
    {"z": 3.5, "sprd": 15, "label": "z3.5_sp15"},
    {"z": 3.0, "sprd": 15, "label": "z3.0_sp15"},
    {"z": 3.5, "sprd": 5, "label": "z3.5_sp5"},
]

# Pre-load data
data = {}
for pair in PAIRS:
    fn = np.load(os.path.join(DIR, f'fundednext_{pair.lower()}_m1.npy'), allow_pickle=True)
    data[pair] = fn

print(f"{'Pair':>8} {'Config':<16} {'Trades':>7} {'WR':>7} {'Gross':>10} {'Sprd$':>8} {'Comm$':>8} {'Net$':>10} {'L/day':>7} {'$/t':>7}")
print("-" * 90)

for pair in PAIRS:
    fn = data[pair]
    times = fn['time'].astype(np.int64)
    opens = fn['open']
    highs = fn['high']
    lows = fn['low']
    closes = fn['close']
    spreads = fn['spread']
    volumes = fn['tick_volume']
    n_bars = len(times)
    
    for sw in SWEEPS:
        sim = ReconSim(
            z_threshold=sw['z'], stop_a=3.0, trig_a=1.0, gap_a=0.05,
            max_hold=54, base_lot=0.5, max_spread=sw['sprd'],
            trade_dir=0, start_hour=0, end_hour=24
        )
        trades = sim.run(times, opens, highs, lows, closes, spreads, volumes)
        
        if len(trades) == 0:
            continue
        
        n = len(trades)
        gross = sum(t.pnl for t in trades)
        spread_cost = sum(t.sprd_entry * POINT_SIZE * CONTRACT_SIZE * 0.5 for t in trades)
        commission = n * 0.5 * COMMISSION
        
        # For cross pairs, convert AUD to USD
        if pair in ["EURAUD", "GBPAUD"]:
            audusd_fn = data["AUDUSD"]
            audusd_c = audusd_fn['close']
            gross_usd = 0.0
            for t in trades:
                idx = t.entry_time
                rate = audusd_c[min(idx, len(audusd_c)-1)]
                gross_usd += t.pnl * rate
            gross_total = gross_usd
        else:
            gross_total = gross
        
        net = gross_total - spread_cost - commission
        
        wins = len([t for t in trades if t.pnl > 0])
        losses = len([t for t in trades if t.pnl < 0])
        neut = len([t for t in trades if abs(t.pnl) < 0.01])
        denom = n - neut
        wr = wins / denom * 100 if denom else 0
        
        days = 47.0  # approximate
        trades_per_day = n / days
        net_per_trade = net / n if n else 0
        
        label = sw['label']
        print(f"{pair:>8} {label:<16} {n:>6d}t {wr:>5.1f}% ${gross_total:>+7.2f} "
              f"${spread_cost:>6.2f} ${commission:>6.2f} ${net:>+8.2f} "
              f"{trades_per_day:>5.1f} ${net_per_trade:>+5.2f}")

print("\n" + "=" * 90)
print("CHALLENGE MATH: Which config can hit $500/5d?")
print(f"{'Pair':>8} {'Config':<16} {'$/d':>7} {'Days':>6} {'$500 in':>8}")
print("-" * 50)

for pair in PAIRS:
    fn = data[pair]
    times = fn['time'].astype(np.int64)
    opens = fn['open']
    highs = fn['high']
    lows = fn['low']
    closes = fn['close']
    spreads = fn['spread']
    volumes = fn['tick_volume']
    
    for sw in SWEEPS:
        sim = ReconSim(
            z_threshold=sw['z'], stop_a=3.0, trig_a=1.0, gap_a=0.05,
            max_hold=54, base_lot=1.0, max_spread=sw['sprd'],
            trade_dir=0, start_hour=0, end_hour=24
        )
        trades = sim.run(times, opens, highs, lows, closes, spreads, volumes)
        
        if len(trades) == 0:
            continue
        
        n = len(trades)
        gross = sum(t.pnl for t in trades)
        spread_cost = sum(t.sprd_entry * POINT_SIZE * CONTRACT_SIZE * 1.0 for t in trades)
        commission = n * 1.0 * COMMISSION
        
        if pair in ["EURAUD", "GBPAUD"]:
            audusd_fn = data["AUDUSD"]
            audusd_c = audusd_fn['close']
            gross_usd = 0.0
            for t in trades:
                idx = t.entry_time
                rate = audusd_c[min(idx, len(audusd_c)-1)]
                gross_usd += t.pnl * rate
            gross_total = gross_usd
        else:
            gross_total = gross
        
        net = gross_total - spread_cost - commission
        days = 47.0
        per_day = net / days
        days_to_500 = 500 / per_day if per_day > 0 else float('inf')
        
        label = sw['label']
        surv = "SURVIVES" if net > 0 else "DIES"
        print(f"{pair:>8} {label:<16} ${per_day:>+5.2f} {days_to_500:>5.0f}d "
              f"{days_to_500 <= 30:>8} {surv}")
