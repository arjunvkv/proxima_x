"""Cross-rate laggard catch-up at tick level.
When EURJPY moves, do EURUSD and USDJPY catch up due to hedging flow?
"""
import numpy as np
import pandas as pd

# Load tick data for all 3 pairs
f = 'data/cache/ticks_EURJPY_EURUSD_USDJPY_2026-04-01_2026-04-30_80000_42.parquet'
df = pd.read_parquet(f)
print(f"Total ticks: {len(df)}")
print(f"Symbols: {df['sym'].value_counts().to_dict()}")

# Separate by pair
ej = df[df['sym'] == 'EURJPY'].reset_index(drop=True)
eu = df[df['sym'] == 'EURUSD'].reset_index(drop=True)
uj = df[df['sym'] == 'USDJPY'].reset_index(drop=True)

# We need aligned ticks. The tick_num is sequential across pairs.
# Let's find sequences: EURJPY tick at N, then what do EURUSD and USDJPY do in the next few ticks?

# Create merged timeline
df_sorted = df.sort_values('tick_num').reset_index(drop=True)

log_px = np.log(df_sorted['price'].values)
pairs_arr = df_sorted['sym'].values

# For each EURJPY tick, look at subsequent EURUSD and USDJPY ticks
print("\n" + "=" * 60)
print("LAGGARD CATCH-UP ANALYSIS")
print("=" * 60)

results = []
lookahead = 20  # Look at next 20 ticks

for i in range(len(df_sorted) - lookahead):
    if pairs_arr[i] != 'EURJPY':
        continue

    # Current EURJPY tick
    cur_ej = i
    cur_ej_price = log_px[i]

    # Next EURUSD tick after this EURJPY
    next_eu_idx = None
    next_uj_idx = None

    for j in range(i + 1, min(i + lookahead + 1, len(df_sorted))):
        if pairs_arr[j] == 'EURUSD' and next_eu_idx is None:
            next_eu_idx = j
        if pairs_arr[j] == 'USDJPY' and next_uj_idx is None:
            next_uj_idx = j
        if next_eu_idx is not None and next_uj_idx is not None:
            break

    if next_eu_idx is None or next_uj_idx is None:
        continue

    # Previous EURUSD and USDJPY ticks (before this EURJPY move)
    prev_eu_idx = None
    prev_uj_idx = None
    for j in range(i - 1, max(-1, i - 100), -1):
        if pairs_arr[j] == 'EURUSD' and prev_eu_idx is None:
            prev_eu_idx = j
        if pairs_arr[j] == 'USDJPY' and prev_uj_idx is None:
            prev_uj_idx = j
        if prev_eu_idx is not None and prev_uj_idx is not None:
            break

    if prev_eu_idx is None or prev_uj_idx is None:
        continue

    # Price changes
    ej_move = log_px[i] - (log_px[i - 1] if i > 0 and pairs_arr[i - 1] == 'EURJPY' else log_px[i])
    eu_after = log_px[next_eu_idx] - log_px[prev_eu_idx]
    uj_after = log_px[next_uj_idx] - log_px[prev_uj_idx]

    # Expected leg moves from triangle: EURJPY ≈ EURUSD × USDJPY
    # log(EURJPY) = log(EURUSD) + log(USDJPY)
    # So ej_move ≈ eu_after + uj_after
    ej_expected = eu_after + uj_after

    # How much does EURJPY move that is NOT explained by the legs so far?
    residual = ej_move - (eu_after + uj_after)

    # If EURJPY moved but legs haven't caught up, do they catch up in subsequent ticks?
    # Check a few ticks ahead
    for fwd in [3, 5, 10, 15]:
        fwd_eu = None
        fwd_uj = None
        end = min(next_eu_idx + 1 + fwd, len(df_sorted))
        for j in range(next_eu_idx + 1, end):
            if pairs_arr[j] == 'EURUSD' and fwd_eu is None:
                fwd_eu = j
            if pairs_arr[j] == 'USDJPY' and fwd_uj is None:
                fwd_uj = j
            if fwd_eu and fwd_uj:
                break

        if fwd_eu is None or fwd_uj is None:
            continue

        eu_fwd_pct = (np.exp(log_px[fwd_eu]) - np.exp(log_px[next_eu_idx])) / np.exp(log_px[next_eu_idx])
        uj_fwd_pct = (np.exp(log_px[fwd_uj]) - np.exp(log_px[next_uj_idx])) / np.exp(log_px[next_uj_idx])

        # Do the legs continue moving in the same direction as the EURJPY move?
        ej_sign = np.sign(ej_move)
        leg_continued_eu = np.sign(eu_fwd_pct) == ej_sign if abs(eu_fwd_pct) > 0 else None
        leg_continued_uj = np.sign(uj_fwd_pct) == ej_sign if abs(uj_fwd_pct) > 0 else None

        results.append({
            'ej_move': ej_move,
            'residual': residual,
            'eu_after': eu_after,
            'uj_after': uj_after,
            'eu_fwd': eu_fwd_pct,
            'uj_fwd': uj_fwd_pct,
            'ej_sign': ej_sign,
            'leg_eu_same': leg_continued_eu,
            'leg_uj_same': leg_continued_uj,
        })

rdf = pd.DataFrame(results)
print(f"Total EURJPY tick events: {len(rdf)}")
print(f"  With valid forward ticks: {len(rdf)}")

# Analysis 1: When EURJPY makes a large move, do legs follow in same direction?
for thresh_pct in [0.0, 0.001, 0.002, 0.005]:
    sub = rdf[abs(rdf['ej_move']) > thresh_pct]
    if len(sub) < 10:
        continue
    for col, name in [('leg_eu_same', 'EURUSD'), ('leg_uj_same', 'USDJPY')]:
        valid = sub[sub[col].notna()]
        if len(valid) < 5:
            continue
        same_pct = valid[col].mean() * 100
        print(f"  |ej_move|>{thresh_pct:.4f}: {name} same-direction={same_pct:.0f}% ({len(valid)} events)")

# Analysis 2: When residual is large (EURJPY moved but legs lagged), do legs catch up?
print("\n--- Residual analysis (legs started to catch up = more leg movement after EURJPY tick) ---")
for residual_thresh in [0.0005, 0.001, 0.002, 0.005]:
    sub = rdf[abs(rdf['residual']) > residual_thresh]
    if len(sub) < 5:
        continue
    for col, name in [('eu_fwd', 'EURUSD'), ('uj_fwd', 'USDJPY')]:
        vals = sub[col].values
        ej_signs = sub['ej_sign'].values
        same_dir = np.mean((np.sign(vals) == ej_signs) & (np.abs(vals) > 0)) * 100
        avg_move = np.mean(np.abs(vals)) * 10000  # bps
        print(f"  |residual|>{residual_thresh:.4f}: {name} same-dir={same_dir:.0f}% avg|move|={avg_move:.1f}bps ({len(sub)} events)")
