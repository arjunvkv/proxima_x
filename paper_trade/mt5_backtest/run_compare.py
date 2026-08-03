"""Compare sim_recon against EA logic, checking differences."""
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

# Run sim with V2+z CPPF config
sim = ReconSim(z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
               max_hold=54, base_lot=1.0, max_spread=5,
               limit_entry_atr=0.0, enable_stability=False,
               trade_dir=0, start_hour=0, end_hour=24)
trades = sim.run(times, opens, highs, lows, closes, spreads, volumes, start_dt)

total = sum(t.pnl for t in trades)
wins = [t for t in trades if t.pnl > 0]
losses = [t for t in trades if t.pnl < 0]
neutral = [t for t in trades if abs(t.pnl) < 0.01]
denom = len(trades) - len(neutral)
wr = len(wins)/denom*100 if denom else 0
avg_win = sum(t.pnl for t in wins)/len(wins) if wins else 0
avg_loss = sum(t.pnl for t in losses)/len(losses) if losses else 0
avg_hold = sum(t.bars_held for t in trades)/len(trades) if trades else 0

print("SIM_RECON AUDUSD z=3.5 0-24 lot=1 (no stability)")
print(f"Trades: {len(trades)}  W:{len(wins)} L:{len(losses)} N:{len(neutral)}")
print(f"PnL: ${total:.2f}  WR: {wr:.1f}%")
print(f"Avg win: ${avg_win:.2f}  Avg loss: ${avg_loss:.2f}  Avg hold: {avg_hold:.0f} bars")
print(f"Payoff ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "")

# Exit reason breakdown
reasons = {}
for t in trades:
    reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
print(f"\nExit reasons: {reasons}")

# Now run same but with stability gate (relaxed) to compare
sim2 = ReconSim(z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
                max_hold=54, base_lot=1.0, max_spread=5,
                limit_entry_atr=0.0, enable_stability=True,
                stab_thresh=0.3, z_cum_min=0.0, sprd_z_max=1.5, vol_z_max=2.0,
                trade_dir=0, start_hour=0, end_hour=24)
trades2 = sim2.run(times, opens, highs, lows, closes, spreads, volumes, start_dt)
total2 = sum(t.pnl for t in trades2)
wins2 = [t for t in trades2 if t.pnl > 0]
losses2 = [t for t in trades2 if t.pnl < 0]
neutral2 = [t for t in trades2 if abs(t.pnl) < 0.01]
denom2 = len(trades2) - len(neutral2)
wr2 = len(wins2)/denom2*100 if denom2 else 0
avg_win2 = sum(t.pnl for t in wins2)/len(wins2) if wins2 else 0
avg_loss2 = sum(t.pnl for t in losses2)/len(losses2) if losses2 else 0

print(f"\nSIM_RECON AUDUSD z=3.5 0-24 lot=1 (with relaxed stability)")
print(f"Trades: {len(trades2)}  W:{len(wins2)} L:{len(losses2)} N:{len(neutral2)}")
print(f"PnL: ${total2:.2f}  WR: {wr2:.1f}%")
print(f"Avg win: ${avg_win2:.2f}  Avg loss: ${avg_loss2:.2f}")

print("\n=== DIFF ANALYSIS: sim_recon vs EA (V2z_CPPF_RECON.mq5) ===")
print("""
1. STOP CHECK: EA checks tick.bid/tick.ask on every tick.
   Sim checks M1 low/high once per bar. 
   -> EA catches intra-bar stops that sim misses (sim OVER-estimates survivorship)

2. TRAILING: EA trails on every tick (catches intra-bar highs).
   Sim trails on M1 high only.
   -> EA locks profit more aggressively (sim UNDER-estimates profit capture)

3. ENTRY PRICE: EA uses live tick.ask/tick.bid.
   Sim uses M1 open.
   -> Small difference, usually within spread

4. PnL: EA uses GetRealizedPL() = actual MT5 PnL (includes commission, swap).
   Sim uses (close_price - entry_price) * lot * contract_size.
   -> NO COMMISSION in sim. On FTMO at $5/round-turn/lot, a 10-lot=100k trade costs $7.
   With 285 trades at lot=1: 285 * $7 = $2,000 commission. Sim is MISSING $2k in costs.

5. SPREAD: EA uses live SYMBOL_SPREAD.
   Sim uses recorded spread from data.
   -> Small diff, likely similar

CONCLUSION: Sim inflates PnL 2 ways:
  A) No commission (biggest factor for high-frequency strategy)
  B) No intra-bar stop-outs (survivorship bias in M1 data)
  C) Can't reproduce tick-level bid/ask dynamics
""")
