"""Analyze what filters block which entry signals."""
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sim_recon import ReconSim

data = np.load(r'C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\ftmo_audusd_m1.npy',
               allow_pickle=True)
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'], unit='s')

sim = ReconSim(z_threshold=4.0, stop_a=3.0, trig_a=1.0, gap_a=0.05,
               max_hold=54, base_lot=0.5, max_spread=5,
               limit_entry_atr=0.00007, enable_stability=True,
               stab_thresh=0.5, z_cum_min=5.0,
               trade_dir=1, start_hour=0, end_hour=7)

start_dt = datetime(2026, 6, 8, tzinfo=timezone.utc)
times = df['time'].astype(np.int64)//10**9
opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
spreads = df['spread'].values
volumes = df['tick_volume'].values

n = len(times)
all_signals = []

for i in range(n):
    dt = datetime.fromtimestamp(times[i], tz=timezone.utc)
    if dt < start_dt:
        sim.update_buffers(opens[i], highs[i], lows[i], closes[i], spreads[i], volumes[i])
        continue

    # Before updating, capture what the EA sees
    z = sim.compute_zscore()
    av = sim.get_atr()

    if sim.trade is not None:
        sim.manage_position(opens[i], highs[i], lows[i], closes[i], spreads[i])

    if abs(z) >= 4.0 and z < 0 and sim.trade is None:
        sprd = spreads[i]
        atr_block = av < 0.00007

        stab_info = {}
        if sim.z_count >= 10:
            z_cum = sim.compute_cumulative_z()
            spd_z = sim.compute_spread_z()
            vol_z = sim.compute_volume_z()
            stability = sim.compute_stability()
            stab_info['z_cum'] = z_cum
            stab_info['spd_z'] = spd_z
            stab_info['vol_z'] = vol_z
            stab_info['stability'] = stability

            if abs(z_cum) < 5.0:
                stab_info['reason'] = f'z_cum_min={abs(z_cum):.2f}<5.0'
            elif spd_z > 1.5:
                stab_info['reason'] = f'spd_z={spd_z:.2f}>1.5'
            elif vol_z > 2.0:
                stab_info['reason'] = f'vol_z={vol_z:.2f}>2.0'
            elif stability < 0.5:
                stab_info['reason'] = f'stab={stability:.3f}<0.5'
            else:
                stab_info['reason'] = 'PASS'
        else:
            stab_info['reason'] = 'no_history'

        all_signals.append({
            'time': dt,
            'atr': av,
            'z': z,
            'sprd': sprd,
            'atr_block': atr_block,
            'stab_info': stab_info
        })

    sim.update_buffers(opens[i], highs[i], lows[i], closes[i], spreads[i], volumes[i])

print(f'Total LONG signals (z<-4, hours 0-7, sprd<=5): {len(all_signals)}')

# ATR filter impact
low_atr = [s for s in all_signals if s['atr'] < 0.00007]
high_atr = [s for s in all_signals if s['atr'] >= 0.00007]
print(f'\n=== ATR < 0.00007 (would be BLOCKED): {len(low_atr)} signals ===')
for s in low_atr:
    print(f"  {s['time']} ATR={s['atr']:.6f} z={s['z']:.2f} sprd={s['sprd']}")

print(f'\n=== ATR >= 0.00007 (would PASS ATR): {len(high_atr)} signals ===')
stab_pass = [s for s in high_atr if s['stab_info'].get('reason') == 'PASS']
stab_block = [s for s in high_atr if s['stab_info'].get('reason', '') != 'PASS']
print(f'  Stability gate PASS: {len(stab_pass)}')
print(f'  Stability gate BLOCK: {len(stab_block)}')

print('\n=== Stability Gate Results for High-ATR Signals ===')
reasons = {}
for s in high_atr:
    r = s['stab_info'].get('reason', 'N/A')
    reasons[r] = reasons.get(r, 0) + 1
for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f'  {r}: {c}')

if stab_pass:
    print(f'\n=== TRADES THAT WOULD FIRE (both filters pass): {len(stab_pass)} ===')
    for s in stab_pass:
        print(f"  {s['time']} ATR={s['atr']:.6f} z={s['z']:.2f} sprd={s['sprd']} "
              f"z_cum={s['stab_info']['z_cum']:.2f} spd_z={s['stab_info']['spd_z']:.2f} "
              f"vol_z={s['stab_info']['vol_z']:.2f} stab={s['stab_info']['stability']:.3f}")
