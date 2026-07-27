"""Quick Cross-Rate Divergence thesis validation.
Loads EURUSD/USDJPY/EURJPY tick CSVs, aligns by second, computes divergence.
Hypothesis: when EURUSD+USDJPY move, EURJPY lags, then converges."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("C:/Trading/Agentic_Trading/proxima_x/strategies/tick_exhaustion_fader/tick_data")

def load_pair(name, cols={'time','bid','ask'}):
    path = DATA / name
    if not path.exists():
        # try _v2
        path = DATA / name.replace('.csv', '_v2.csv')
    print(f"  {path.name}: ", end="", flush=True)
    df = pd.read_csv(path, parse_dates=['time'], usecols=list(cols))
    df['mid'] = (df.bid + df.ask) / 2
    df = df[['time', 'mid']].dropna()
    print(f"{len(df):,} ticks, {df.time.min()} → {df.time.max()}")
    return df

print("=== Loading tick data ===")
# Use v2 files for USDJPY/EURJPY to overlap with EURUSD's Jul 23 window
eusd = load_pair("EURUSD_mt5.csv")
usdjpy = load_pair("USDJPY_mt5_v2.csv")
eurjpy = load_pair("EURJPY_mt5_v2.csv")

# Resample to 1-second bars (mid prices, last tick)
print("\n=== Resampling to 1-sec bars ===")
eusd_1s = eusd.set_index('time').resample('1s').last().dropna().rename(columns={'mid': 'eurusd'})
usdjpy_1s = usdjpy.set_index('time').resample('1s').last().dropna().rename(columns={'mid': 'usdjpy'})
eurjpy_1s = eurjpy.set_index('time').resample('1s').last().dropna().rename(columns={'mid': 'eurjpy'})

# Inner join on timestamp
aligned = eusd_1s.join(usdjpy_1s, how='inner').join(eurjpy_1s, how='inner')
print(f"Aligned 1s bars: {len(aligned):,}")
print(f"  Range: {aligned.index.min()} → {aligned.index.max()}")

# Compute divergence
aligned['implied'] = aligned.eurusd * aligned.usdjpy
aligned['div_raw'] = aligned.implied - aligned.eurjpy
aligned['div_pips'] = aligned.div_raw / 0.01  # EURJPY pip = 0.01 (standard JPY pip)
aligned['div_bps'] = (aligned.implied / aligned.eurjpy - 1) * 10000  # basis points

print("\n=== Divergence Statistics ===")
print(f"  Mean divergence: {aligned.div_pips.mean():.2f} pips")
print(f"  Std divergence:  {aligned.div_pips.std():.2f} pips")
print(f"  |div| > 3 pips:  {(aligned.div_pips.abs() > 3).sum():,} bars ({(aligned.div_pips.abs() > 3).mean()*100:.1f}%)")
print(f"  |div| > 5 pips:  {(aligned.div_pips.abs() > 5).sum():,} bars ({(aligned.div_pips.abs() > 5).mean()*100:.1f}%)")
print(f"  |div| > 10 pips: {(aligned.div_pips.abs() > 10).sum():,} bars ({(aligned.div_pips.abs() > 10).mean()*100:.1f}%)")
print(f"  Max |div|:       {aligned.div_pips.abs().max():.2f} pips")

# Find divergence events and check convergence
print("\n=== Convergence Check (|div| > 3 pips → converge within N minutes) ===")
THRESH = 3  # pips
CONVERGE_PCT = 0.3  # converge when |div| drops below 30% of initial

div_events = []
in_event = False
event_start = None
event_div = 0

for ts, row in aligned.iterrows():
    d = abs(row.div_pips)
    if not in_event:
        if d > THRESH:
            in_event = True
            event_start = ts
            event_div = d
    else:
        if d < event_div * CONVERGE_PCT:
            in_event = False
            duration = (ts - event_start).total_seconds()
            div_events.append({'start': event_start, 'duration_s': duration,
                               'peak_div': event_div, 'exit_div': d})
        elif (ts - event_start).total_seconds() > 1800:  # 30 min timeout
            in_event = False
            div_events.append({'start': event_start, 'duration_s': 1800,
                               'peak_div': event_div, 'exit_div': d})

print(f"  Total divergence events: {len(div_events)}")
if div_events:
    durations = [e['duration_s'] for e in div_events]
    print(f"  Median convergence: {np.median(durations):.0f}s")
    print(f"  Converged < 60s:   {sum(1 for d in durations if d < 60)} ({sum(1 for d in durations if d < 60)/len(durations)*100:.1f}%)")
    print(f"  Converged < 300s:  {sum(1 for d in durations if d < 300)} ({sum(1 for d in durations if d < 300)/len(durations)*100:.1f}%)")
    print(f"  Converged < 600s:  {sum(1 for d in durations if d < 600)} ({sum(1 for d in durations if d < 600)/len(durations)*100:.1f}%)")
    print(f"  Never converged:   {sum(1 for d in durations if d >= 1800)} ({sum(1 for d in durations if d >= 1800)/len(durations)*100:.1f}%)")

# Check divergence direction vs subsequent return
print("\n=== Directional Validation ===")
aligned['implied_lead'] = aligned.implied - aligned.eurjpy  # positive = implied above actual
aligned['eurjpy_fwd_60s'] = aligned.eurjpy.shift(-60)  # 60s forward
aligned['actual_return_60s'] = aligned.eurjpy_fwd_60s - aligned.eurjpy
aligned['return_pips_60s'] = aligned.actual_return_60s * 10

# When implied > actual (positive divergence), EURJPY should go UP (positive return)
aligned['signal'] = np.sign(aligned.implied_lead)
aligned['correct'] = np.sign(aligned.return_pips_60s.shift(60)) == np.sign(aligned.implied_lead)
valid = aligned.dropna(subset=['correct'])
print(f"  Direction correct (60s): {valid.correct.mean()*100:.1f}% (N={len(valid):,})")

# Only on strong signals
strong = valid[valid.div_pips.abs() > THRESH]
print(f"  Direction correct (|div|>{THRESH}pip,60s): {strong.correct.mean()*100:.1f}% (N={len(strong):,})")

print("\n=== DONE ===")
