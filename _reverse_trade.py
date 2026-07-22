"""Full reverse-engineering of EURJPY trade at t=1784653383.
Uses EXACT same seed order as live code (oldest-first)."""
import MetaTrader5 as mt5
import numpy as np
import sys, os
from datetime import datetime
sys.path.insert(0, r'C:\Trading\Agentic_Trading\proxima_x')
os.chdir(r'C:\Trading\Agentic_Trading\proxima_x')

from paper_trade.strategies.m1_z_reversal.strategy import PairState, CONFIG, TrailingStopManager

mt5.initialize()
pair = 'EURJPY'
trade_time = 1784653383

# Fetch 300 bars covering well before the trade time
rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M1, 0, 300)
if rates is None:
    print("No rates"); mt5.shutdown(); exit()

# Keep oldest-first (consistent with copy_m1_history)
bars = []
for r in rates:
    bars.append({'time': int(r[0]), 'open': float(r[1]), 'high': float(r[2]),
                 'low': float(r[3]), 'close': float(r[4])})

print(f"Fetched {len(bars)} M1 bars: t={bars[0]['time']} .. t={bars[-1]['time']}")

# Show ALL bars within 10 minutes of the trade
print(f"\n=== Bars within 10min of trade ({trade_time} / 17:03:03 UTC) ===")
for b in bars:
    if abs(b['time'] - trade_time) < 600:
        print(f"  {b['time']}  O={b['open']:.3f} H={b['high']:.3f} L={b['low']:.3f} C={b['close']:.3f}")

# Find the index of the bar that is the LAST ONE before the signal bar
# The trade was at 1784653383 (17:03:03), so the completed M1 bar that triggered
# the signal was at 1784653380 (17:03:00) or 1784653320 (17:02:00).
# The signal evaluation is triggered by a tick at ~17:03:00 (minute boundary).
# At that point, the bar for 17:02 (1784653320) completes.
# The close of that bar was determined by tick prices during 17:02.

# But I don't have tick data. Best I can do:
# Seed with bars up to 1784653260 (17:01), then check if ret from 1784653320 close
# would give z=-2.307.

def analyze_seed(seed_bar_time, label):
    """Seed PairState up to a given bar time, then check signal config."""
    seed_idx = None
    for i, b in enumerate(bars):
        if b['time'] == seed_bar_time:
            seed_idx = i
            break
    if seed_idx is None:
        return

    ps = PairState(pair, CONFIG)
    for b in bars[:seed_idx + 1]:
        ps.seed_bar(b)

    rets = ps.z_buf.returns
    n = len(rets)
    if n < 3:
        return

    mean_r = sum(rets) / n
    var_r = sum((r - mean_r)**2 for r in rets) / (n - 1) if n > 1 else 0
    std_r = var_r ** 0.5 if var_r > 1e-10 else 1e-10
    
    needed_ret = mean_r + (-2.307) * std_r
    implied_close = ps.last_close + needed_ret
    
    atr_v = ps.atr_buf.value()
    gate_v = ps.gate_buf.quantile(CONFIG['atr_pctl'])
    atr_ok = atr_v is not None and gate_v is not None and atr_v > gate_v
    
    # Also check if the implied close is within the M1 range of the signal bar
    sig_bar = None
    for b in bars:
        if b['time'] == seed_bar_time + 60:
            sig_bar = b
            break
    
    in_range = False
    if sig_bar and sig_bar['low'] <= implied_close <= sig_bar['high']:
        in_range = True
    
    print(f"  Seed up to {label}: last_close={ps.last_close:.3f} | "
          f"z_buf σ={std_r:.5f} | need ret={needed_ret:.5f} → {implied_close:.3f} | "
          f"ATR OK={atr_ok} | in M1 range={in_range}")
    return ps


print(f"\n=== Testing different seed boundaries ===")
analyze_seed(1784653260, "17:01")
analyze_seed(1784653200, "17:00")
analyze_seed(1784653140, "16:59")
analyze_seed(1784653080, "16:58")
analyze_seed(1784653020, "16:57")
analyze_seed(1784652960, "16:56")

# Also try a much deeper seed to see if z_buf changes
ps2 = PairState(pair, CONFIG)
for b in bars[:]:  # seed ALL bars
    ps2.seed_bar(b)
print(f"\n  Seed all 300: last_close={ps2.last_close:.3f} | "
      f"z_buf entries={len(ps2.z_buf.returns)} | "
      f"mean={sum(ps2.z_buf.returns)/50:.5f} std={...}")

ret_samples = ps2.z_buf.returns[-50:]
mean2 = sum(ret_samples) / len(ret_samples)
var2 = sum((r - mean2)**2 for r in ret_samples) / (len(ret_samples) - 1) if len(ret_samples) > 1 else 0
std2 = var2 ** 0.5 if var2 > 1e-10 else 1e-10
needed2 = mean2 + (-2.307) * std2
implied2 = ps2.last_close + needed2
print(f"  z_buf: mean={mean2:.5f} std={std2:.5f} | needed ret={needed2:.5f} → {implied2:.3f}")

mt5.shutdown()
