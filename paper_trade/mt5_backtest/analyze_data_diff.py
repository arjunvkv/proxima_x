"""Analyze the FTMO vs Normal MT5 data differences."""
import numpy as np
import pandas as pd

ftmo = pd.DataFrame(np.load(r'C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\ftmo_audusd_m1.npy',
                             allow_pickle=True))
normal = pd.DataFrame(np.load(r'C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\normal_audusd_m1.npy',
                               allow_pickle=True))
ftmo['time'] = pd.to_datetime(ftmo['time'], unit='s')
normal['time'] = pd.to_datetime(normal['time'], unit='s')

print("=== ASIAN SESSION SPREAD (hours 0-6) ===")
for name, df in [('FTMO', ftmo), ('Normal', normal)]:
    asia = df[df['time'].dt.hour < 7]
    total = len(asia)
    sprd0 = (asia['spread'] == 0).sum()
    sprd5 = (asia['spread'] <= 5).sum()
    sprd_gt5 = (asia['spread'] > 5).sum()
    print(f"{name}: {total} bars | sprd=0: {sprd0} ({sprd0/total*100:.1f}%) | "
          f"sprd<=5: {sprd5} ({sprd5/total*100:.1f}%) | sprd>5: {sprd_gt5} ({sprd_gt5/total*100:.1f}%)")

print("\n=== PRICE DATA COMPARISON ===")
# Align by time
merged = ftmo.merge(normal, on='time', suffixes=('_f', '_n'), how='inner')
print(f"Common bars: {len(merged)}")
print(f"Close corr: {merged['close_f'].corr(merged['close_n']):.6f}")
print(f"Max close diff: {abs(merged['close_f'] - merged['close_n']).max():.7f}")
print(f"Mean close diff: {abs(merged['close_f'] - merged['close_n']).mean():.7f}")

# Spread comparison
print(f"\nFTMO spread mean/median:   {ftmo['spread'].mean():.2f} / {ftmo['spread'].median():.1f}")
print(f"Normal spread mean/median: {normal['spread'].mean():.2f} / {normal['spread'].median():.1f}")
print(f"FTMO sprd=0 bars: {(ftmo['spread']==0).sum()} / {len(ftmo)}")
print(f"Normal sprd=0 bars: {(normal['spread']==0).sum()} / {len(normal)}")

# What signals does normal have that FTMO blocks by spread?
print("\n=== SIGNALS LOST TO SPREAD ON FTMO ===")
# Find bars where normal has sprd<=5 and FTMO has sprd>5
diff = merged[(merged['spread_f'] > 5) & (merged['spread_n'] <= 5)]
print(f"Bars where FTMO spread>5 but Normal spread<=5: {len(diff)}")
if len(diff) > 0:
    print("Sample:")
    for _, r in diff.head(10).iterrows():
        print(f"  {r['time']} FTMO sprd={r['spread_f']} Normal sprd={r['spread_n']} close_f={r['close_f']:.5f} close_n={r['close_n']:.5f}")

# Also the reverse: FTMO low spread but normal high
diff2 = merged[(merged['spread_f'] <= 5) & (merged['spread_n'] > 5)]
print(f"\nBars where FTMO spread<=5 but Normal spread>5: {len(diff2)}")
if len(diff2) > 0:
    for _, r in diff2.head(10).iterrows():
        print(f"  {r['time']} FTMO sprd={r['spread_f']} Normal sprd={r['spread_n']}")

print("\n=== NORMAL DATA ONLY: Zero-spread bar stats ===")
zero_sprd = normal[normal['spread'] == 0]
asia_zero = zero_sprd[zero_sprd['time'].dt.hour < 7]
print(f"Total zero-spread bars: {len(zero_sprd)}")
print(f"Zero-spread in Asian session: {len(asia_zero)}")
if len(asia_zero) > 0:
    print(f"  Asian zero-spread ATR range: {asia_zero['close'].pct_change().abs().mean()*10000:.2f} pips avg")

# ATR comparison
ftmo['hl'] = ftmo['high'] - ftmo['low']
normal['hl'] = normal['high'] - normal['low']
ftmo_atr_asia = ftmo[ftmo['time'].dt.hour < 7]['hl'].rolling(20).mean().mean()
normal_atr_asia = normal[normal['time'].dt.hour < 7]['hl'].rolling(20).mean().mean()
print(f"\nAsian session avg ATR (20): FTMO={ftmo_atr_asia:.6f} Normal={normal_atr_asia:.6f}")
