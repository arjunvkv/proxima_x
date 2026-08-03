"""Run best RECON config and log every trade."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sim_recon import ReconSim
import numpy as np
import pandas as pd
from datetime import datetime, timezone

data = np.load(os.path.join(os.path.dirname(__file__), 'ftmo_audusd_m1.npy'),
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
start_dt = datetime(2026, 6, 8, tzinfo=timezone.utc)

# Best config from sweeps: z=4 BOTH with relaxed stability gate
config = {
    "z_threshold": 4.0, "stop_a": 3.0, "trig_a": 1.0, "gap_a": 0.05,
    "max_hold": 54, "base_lot": 0.5, "max_spread": 5,
    "limit_entry_atr": 0.0, "enable_stability": True,
    "stab_thresh": 0.3, "z_cum_min": 0.0, "sprd_z_max": 1.5, "vol_z_max": 2.0,
    "trade_dir": 0, "start_hour": 0, "end_hour": 7
}

sim = ReconSim(**config)
trades = sim.run(times, opens, highs, lows, closes, spreads, volumes, start_dt)

print("=" * 100)
print(f"{'TIME':<20} {'DIR':>4} {'ENTRY':>10} {'EXIT':>10} {'STOP':>10} {'Z':>6} "
      f"{'ATR':>8} {'SPRD':>5} {'HELD':>5} {'PnL':>8} {'REASON':<10}")
print("=" * 100)

for t in trades:
    dt_str = datetime.fromtimestamp(times[t.entry_time], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    if t.exit_reason == 'end':
        exit_px = closes[-1]
    elif t.exit_reason == 'stop':
        exit_px = t.stop_loss
    else:
        exit_px = t.exit_price if t.exit_price else 0
    print(f"{dt_str:<20} {t.direction:+3d} {t.entry_price:>10.5f} {exit_px:>10.5f} "
          f"{t.stop_loss:>10.5f} {t.z_entry:>+5.2f} {t.atr_entry:>8.6f} "
          f"{t.sprd_entry:>4.0f} {t.bars_held:>4d} ${t.pnl:>+6.2f} {t.exit_reason:<10}")

print("=" * 100)

total = sum(t.pnl for t in trades)
wins = [t for t in trades if t.pnl > 0]
losses = [t for t in trades if t.pnl < 0]
neutral = [t for t in trades if abs(t.pnl) < 0.01]
denom = len(trades) - len(neutral)
wr = len(wins)/denom*100 if denom else 0
avg_win = sum(t.pnl for t in wins)/len(wins) if wins else 0
avg_loss = sum(t.pnl for t in losses)/len(losses) if losses else 0
avg_hold = sum(t.bars_held for t in trades)/len(trades) if trades else 0
payoff = abs(avg_win/avg_loss) if avg_loss else 0
max_drawdown = 0
peak = 0
running = 0
for t in trades:
    running += t.pnl
    if running > peak:
        peak = running
    dd = peak - running
    if dd > max_drawdown:
        max_drawdown = dd

print(f"\n{'SUMMARY':^100}")
print("=" * 100)
print(f"Config: z={config['z_threshold']}, dir={'BOTH' if config['trade_dir']==0 else 'LONG'}, "
      f"hours={config['start_hour']}-{config['end_hour']}, lot={config['base_lot']}")
print(f"Stability: thresh={config['stab_thresh']}, z_cum_min={config['z_cum_min']}, "
      f"sprd_z_max={config['sprd_z_max']}, vol_z_max={config['vol_z_max']}")
print(f"Data: {df['time'].min()} to {df['time'].max()} ({len(df)} bars)")
print(f"\nTrades: {len(trades)}  W:{len(wins)} L:{len(losses)} N:{len(neutral)}")
print(f"Win Rate: {wr:.1f}%")
print(f"Gross PnL: ${total:.2f}")
print(f"Avg Win: ${avg_win:.2f}  Avg Loss: ${avg_loss:.2f}")
print(f"Payoff Ratio: {payoff:.2f}")
print(f"Avg Hold: {avg_hold:.0f} bars")
print(f"Max Drawdown: ${max_drawdown:.2f}")
print(f"Profit Factor: {sum(t.pnl for t in wins)/abs(sum(t.pnl for t in losses)):.2f}"
      if losses else "Profit Factor: N/A (no losses)")
print("=" * 100)

# Also show scaled versions
for lot in [1.0, 2.0, 3.5]:
    scaled = total * (lot / config['base_lot'])
    print(f"\nWith lot={lot:.1f}: ${scaled:.2f} PnL ({wr:.1f}% WR same)")
