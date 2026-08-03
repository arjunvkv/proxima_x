"""Find RECON config that gives >70% WR with >$1000 PnL on FTMO data."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sim_recon import ReconSim
import numpy as np
import pandas as pd

# Load AUDUSD FTMO data
data = np.load(r'C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\ftmo_audusd_m1.npy',
               allow_pickle=True)
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'], unit='s')
times = df['time'].astype(np.int64)//10**9
opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
spreads = df['spread'].values
volumes = df['tick_volume'].values
from datetime import datetime, timezone
start_dt = datetime(2026, 6, 8, tzinfo=timezone.utc)

# Sweep: z_threshold x direction x hours x lot size
configs = []
for z in [3.5, 4.0, 4.5]:
    for direction_name, trade_dir in [("LONG", 1), ("BOTH", 0)]:
        for hours_name, (sh, eh) in [("0-7", (0,7)), ("0-12", (0,12)), ("22-10", (22,10)), ("0-24", (0,0))]:
            for lot in [0.5, 1.0, 2.0, 3.5]:
                configs.append({
                    "name": f"z={z} {direction_name} {hours_name} lot={lot}",
                    "z_threshold": z, "trade_dir": trade_dir,
                    "start_hour": sh, "end_hour": eh,
                    "base_lot": lot,
                    "enable_stability": True,
                    "stab_thresh": 0.3, "z_cum_min": 0.0,
                    "sprd_z_max": 1.5, "vol_z_max": 2.0,
                })

results = []
for cfg in configs:
    sim = ReconSim(
        z_threshold=cfg["z_threshold"], stop_a=3.0, trig_a=1.0, gap_a=0.05,
        max_hold=54, base_lot=cfg["base_lot"], max_spread=5,
        limit_entry_atr=0.0,
        enable_stability=cfg["enable_stability"],
        stab_thresh=cfg["stab_thresh"],
        z_cum_min=cfg["z_cum_min"],
        sprd_z_max=cfg["sprd_z_max"],
        vol_z_max=cfg["vol_z_max"],
        trade_dir=cfg["trade_dir"],
        start_hour=cfg["start_hour"], end_hour=cfg["end_hour"]
    )
    trades = sim.run(times, opens, highs, lows, closes, spreads, volumes, start_dt)
    total_pnl = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    neutral = [t for t in trades if abs(t.pnl) < 0.01]
    denom = len(trades) - len(neutral)
    wr = len(wins)/denom*100 if trades and denom > 0 else 0
    results.append((cfg["name"], len(trades), len(wins), len(losses), total_pnl, wr))
    print(f"{cfg['name']:45s} → {len(trades):2d} trades, {wr:5.1f}% WR, ${total_pnl:7.2f}")

print("\n\n=== CANDIDATES >70% WR AND >$1000 PnL ===")
print(f"{'Config':45s} {'Trades':>6s} {'WR':>6s} {'PnL':>8s}")
print("-"*70)
for name, n, w, l, pnl, wr in sorted(results, key=lambda r: -r[4]):
    if wr >= 70 and pnl >= 1000:
        print(f"{name:45s} {n:6d} {wr:5.1f}% ${pnl:7.2f}")
