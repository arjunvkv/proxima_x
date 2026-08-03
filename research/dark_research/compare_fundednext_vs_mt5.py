#!/usr/bin/env python3
"""Compare FundedNext vs MT5 parquet data for same period."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os

ROOT = os.path.dirname(__file__)

# Load FundedNext data
pairs = ["eurjpy", "eurusd", "gbpjpy"]
pair_names = ["EURJPY", "EURUSD", "GBPJPY"]
fn_close = {}
for p in pairs:
    f = os.path.join(ROOT, f"fundednext_{p}_m1.npy")
    d = np.load(f, allow_pickle=True)
    df = pd.DataFrame(d)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    fn_close[p] = df.set_index("time")["close"]

fn_common = sorted(set(fn_close["eurjpy"].index) & set(fn_close["eurusd"].index) & set(fn_close["gbpjpy"].index))
print(f"FundedNext common bars: {len(fn_common)}")
print(f"FundedNext period: {fn_common[0]} - {fn_common[-1]}")

# Load MT5 parquet
mdf = pd.read_parquet(r"C:\Trading\Agentic_Trading\proxima_x\data\temp\mt5_m1_9day.parquet")
print(f"\nMT5 parquet: {len(mdf)} rows")
print(f"MT5 pairs: {mdf['pair'].unique()}")
print(f"MT5 period: {mdf['time'].min()} - {mdf['time'].max()}")

# Compare on overlapping timestamps
mdf_piv = mdf.pivot_table(index="time", columns="pair", values="close")
mdf_piv.columns = [c.lower() for c in mdf_piv.columns]
mdf_piv = mdf_piv.sort_index()
print(f"\nMT5 pivoted: {mdf_piv.shape[0]} bars, pairs: {list(mdf_piv.columns)}")

# Find overlapping timestamps
mt5_times = set(mdf_piv.index)
fn_times = set(fn_common)
overlap = sorted(mt5_times & fn_times)
print(f"Overlapping bars: {len(overlap)}")
print(f"Overlap period: {overlap[0]} - {overlap[-1]}")

if len(overlap) > 0:
    # Compare prices
    for pair in pairs:
        mt5_prices = mdf_piv.loc[overlap, pair].values if pair in mdf_piv.columns else None
        fn_prices = fn_close[pair].loc[overlap].values
        if mt5_prices is not None and len(mt5_prices) == len(fn_prices):
            diff = mt5_prices - fn_prices
            print(f"\n{pair.upper()} price comparison ({len(overlap)} bars):")
            print(f"  MT5 close: mean={np.mean(mt5_prices):.5f} std={np.std(mt5_prices):.5f}")
            print(f"  FN  close: mean={np.mean(fn_prices):.5f} std={np.std(fn_prices):.5f}")
            print(f"  Diff: mean={np.mean(diff):.5f} std={np.std(diff):.5f} max|diff|={np.max(np.abs(diff)):.5f}")
            print(f"  Identical: {np.allclose(mt5_prices, fn_prices, atol=1e-8)}")
            print(f"  Max bar time diff: {np.max(np.abs(mt5_prices - fn_prices)):.5f}")

    # Run Dark Consensus on MT5 data for the overlapping period
    print(f"\n\n=== Dark Consensus on MT5 parquet (overlapping period) ===")
    close_mt5 = np.column_stack([mdf_piv.loc[overlap, pair].values.astype(np.float64) for pair in pairs])
    times_mt5 = np.array([int(t.timestamp()) for t in overlap])
    rets_mt5 = np.diff(np.log(close_mt5), axis=0)
    up_mt5 = rets_mt5 > 0
    cons_mt5 = up_mt5.all(axis=1) | (~up_mt5).all(axis=1)
    dir_mt5 = np.where(up_mt5.all(axis=1), 1.0, -1.0)
    avg_mag_mt5 = np.mean(np.abs(rets_mt5), axis=1)
    pair_mags_mt5 = np.abs(rets_mt5)
    hour_mt5 = pd.DatetimeIndex(overlap).hour.values[1:]
    MAG95 = 0.00018741

    # Run on MT5 data
    from itertools import groupby
    gross_mt5 = []
    for t in range(1440, len(close_mt5) - 4):
        if not cons_mt5[t]: continue
        if hour_mt5[t] < 7 or hour_mt5[t] > 21: continue
        if avg_mag_mt5[t] <= MAG95: continue
        bi = int(np.argmax(pair_mags_mt5[t]))
        ep = close_mt5[t, bi]
        xp = close_mt5[t+3, bi]
        if bi == 1:
            gross_mt5.append((xp - ep) * 100000)
        else:
            usdjpy = close_mt5[t, 0] / close_mt5[t, 1]
            gross_mt5.append((xp - ep) * 100000 / usdjpy)

    gross_mt5 = np.array(gross_mt5)
    print(f"  Trades: {len(gross_mt5)}")
    print(f"  Gross WR: {np.mean(gross_mt5>0)*100:.1f}%")
    print(f"  Gross avg: ${np.mean(gross_mt5):.2f}")
    print(f"  Gross Sharpe: {np.mean(gross_mt5)/(np.std(gross_mt5)+1e-10)*np.sqrt(1440/3):.2f}")

    # Run on FundedNext data for same period
    print(f"\n=== Dark Consensus on FundedNext (overlapping period) ===")
    close_fn = np.column_stack([fn_close[p].loc[overlap].values for p in pairs])
    rets_fn = np.diff(np.log(close_fn), axis=0)
    up_fn = rets_fn > 0
    cons_fn = up_fn.all(axis=1) | (~up_fn).all(axis=1)
    dir_fn = np.where(up_fn.all(axis=1), 1.0, -1.0)
    avg_mag_fn = np.mean(np.abs(rets_fn), axis=1)
    pair_mags_fn = np.abs(rets_fn)

    gross_fn = []
    for t in range(1440, len(close_fn) - 4):
        if not cons_fn[t]: continue
        if hour_mt5[t] < 7 or hour_mt5[t] > 21: continue
        if avg_mag_fn[t] <= MAG95: continue
        bi = int(np.argmax(pair_mags_fn[t]))
        ep = close_fn[t, bi]
        xp = close_fn[t+3, bi]
        if bi == 1:
            gross_fn.append((xp - ep) * 100000)
        else:
            usdjpy = close_fn[t, 0] / close_fn[t, 1]
            gross_fn.append((xp - ep) * 100000 / usdjpy)

    gross_fn = np.array(gross_fn)
    print(f"  Trades: {len(gross_fn)}")
    print(f"  Gross WR: {np.mean(gross_fn>0)*100:.1f}%")
    print(f"  Gross avg: ${np.mean(gross_fn):.2f}")
    print(f"  Gross Sharpe: {np.mean(gross_fn)/(np.std(gross_fn)+1e-10)*np.sqrt(1440/3):.2f}")

    # Compare signal-by-signal
    print(f"\n=== Overlap analysis ===")
    # Check what % of trades fire on same bars
    mt5_trades = set()
    fn_trades = set()
    for t in range(1440, len(close_mt5) - 4):
        if cons_mt5[t] and hour_mt5[t] >= 7 and hour_mt5[t] <= 21 and avg_mag_mt5[t] > MAG95:
            mt5_trades.add(t)
        if cons_fn[t] and hour_mt5[t] >= 7 and hour_mt5[t] <= 21 and avg_mag_fn[t] > MAG95:
            fn_trades.add(t)
    common_trades = mt5_trades & fn_trades
    print(f"  MT5 trades: {len(mt5_trades)}")
    print(f"  FN trades: {len(fn_trades)}")
    print(f"  Common trade bars: {len(common_trades)} ({len(common_trades)/max(len(mt5_trades),1)*100:.1f}% of MT5)")

    # Check consensus alignment
    cons_mt5_set = set(np.where(cons_mt5)[0] + 1440)
    cons_fn_set = set(np.where(cons_fn)[0] + 1440)
    common_cons = cons_mt5_set & cons_fn_set
    print(f"  MT5 consensus events: {len(cons_mt5_set)}")
    print(f"  FN consensus events: {len(cons_fn_set)}")
    print(f"  Common consensus: {len(common_cons)} ({len(common_cons)/max(len(cons_mt5_set),1)*100:.1f}% of MT5)")

    # Check price correlation
    print(f"\n=== Price correlation ===")
    for i, pn in enumerate(pairs):
        mt5_p = close_mt5[:, i]
        fn_p = close_fn[:, i]
        corr = np.corrcoef(mt5_p, fn_p)[0, 1]
        print(f"  {pn.upper()}: corr={corr:.6f}")
        # Check return correlation
        mt5_r = np.diff(np.log(mt5_p))
        fn_r = np.diff(np.log(fn_p))
        r_corr = np.corrcoef(mt5_r, fn_r)[0, 1]
        print(f"  {pn.upper()} returns: corr={r_corr:.6f}")
