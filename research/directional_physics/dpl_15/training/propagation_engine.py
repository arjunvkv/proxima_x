import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import polars as pl
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
import MetaTrader5 as mt5

DATA_INTRA = "C:/Trading/Agentic_Trading/data/intraday"
M5_SOURCES = {"USDX", "EURUSD", "GBPUSD", "AUDUSD", "USDCNH"}
ALREADY_FETCHED = {"XAUUSD", "EURJPY", "USDJPY", "GBPJPY", "USTEC"}

TARGETS = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD"]
ALL_SOURCES = ["USDX", "EURUSD", "GBPUSD", "AUDUSD", "USDCNH", "USTEC", "XAUUSD"]

EXPECTED_SIGNS = {
    ("USDX", "USDJPY"): 1.0,
    ("USDX", "EURJPY"): -1.0,
    ("USDX", "GBPJPY"): -1.0,
    ("USDX", "XAUUSD"): -1.0,
    ("EURUSD", "EURJPY"): 1.0,
    ("USTEC", "GBPJPY"): 1.0,
    ("AUDUSD", "GBPJPY"): 1.0,
    ("AUDUSD", "EURJPY"): 1.0,
    ("USDCNH", "XAUUSD"): 1.0,
    ("EURUSD", "USDJPY"): 1.0,
    ("USDCNH", "USDJPY"): -1.0,
    ("USTEC", "USDJPY"): -1.0,
    ("GBPUSD", "GBPJPY"): 1.0,
    ("GBPUSD", "EURJPY"): 1.0,
}

TIER1 = [
    ("USDX", "USDJPY"),
    ("EURUSD", "EURJPY"),
    ("USTEC", "GBPJPY"),
    ("AUDUSD", "GBPJPY"),
    ("USDCNH", "XAUUSD"),
]


def fetch_source_data():
    if not mt5.initialize():
        print("MT5 init failed")
        return {}
    data = {}
    for sym in M5_SOURCES:
        print(f"  Fetching {sym}...", end=" ")
        mt5.symbol_select(sym, True)
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 50000)
        if rates is None:
            print(f"FAILED: {mt5.last_error()}")
            continue
        print(f"{len(rates)} bars")
        df = pl.DataFrame({
            "timestamp": [int(r["time"]) for r in rates],
            "close": [float(r["close"]) for r in rates],
        })
        path = os.path.join(DATA_INTRA, f"{sym}_M5.parquet")
        df.write_parquet(path)
        data[sym] = df["close"].to_numpy().astype(np.float64)
    mt5.shutdown()
    return data


def load_all():
    data = {}
    for sym in M5_SOURCES:
        path = os.path.join(DATA_INTRA, f"{sym}_M5.parquet")
        if os.path.exists(path):
            df = pl.read_parquet(path)
            data[sym] = df["close"].to_numpy().astype(np.float64)
            print(f"  {sym}: {len(data[sym])} bars")
    for sym in ["XAUUSD", "EURJPY", "USDJPY", "GBPJPY"]:
        path = os.path.join(DATA_INTRA, f"{sym}_M5.parquet")
        if os.path.exists(path):
            df = pl.read_parquet(path)
            data[sym] = df["close"].to_numpy().astype(np.float64)
            print(f"  {sym}: {len(data[sym])} bars")
    path = os.path.join(DATA_INTRA, "NAS100_M5.parquet")
    if os.path.exists(path):
        df = pl.read_parquet(path)
        data["USTEC"] = df["close"].to_numpy().astype(np.float64)
        print(f"  USTEC (NAS100): {len(data['USTEC'])} bars")
    return data


def align_all(data):
    min_len = min(len(v) for v in data.values())
    aligned = {}
    for k, v in data.items():
        aligned[k] = v[-min_len:]
    return aligned, min_len


def detect_impulses(prices, window=200):
    n = len(prices)
    rets = np.diff(np.log(prices), prepend=np.log(prices[0]))
    tr = np.abs(np.diff(prices, prepend=prices[0]))
    ret_p95 = np.zeros(n)
    for i in range(window, n):
        ret_p95[i] = float(np.percentile(np.abs(rets[i - window:i]), 95))
    impulses = np.zeros(n, dtype=bool)
    imp_dir = np.zeros(n)
    for i in range(window, n):
        if abs(rets[i]) > ret_p95[i] and ret_p95[i] > 0:
            impulses[i] = True
            imp_dir[i] = np.sign(rets[i])
    return impulses, imp_dir


def propagate(source_prices, target_prices, max_lag=12):
    s_rets = np.diff(np.log(source_prices), prepend=np.log(source_prices[0]))
    t_rets = np.diff(np.log(target_prices), prepend=np.log(target_prices[0]))
    n = len(s_rets)
    results = []
    for lag in [1, 2, 3, 6, 12]:
        if lag > max_lag:
            continue
        valid = np.zeros(n, dtype=bool)
        valid[:n - lag] = True
        scr = np.mean(s_rets[valid] * t_rets[np.arange(n)[valid] + lag]) if np.sum(valid) > 0 else 0.0
        results.append({"lag": lag, "scr": float(scr)})
    return results


def main():
    print("=" * 80)
    print("  DPL-15 CROSS-ASSET PROPAGATION ENGINE")
    print("=" * 80)

    print("\n  Fetching source data from MT5...")
    fetch_source_data()

    print("\n  Loading all data...")
    data = load_all()

    print(f"\n  Aligning {len(data)} series...")
    aligned, n = align_all(data)
    print(f"  Common length: {n} bars")

    print(f"\n{'=' * 60}")
    print("  TIER 1 PROPAGATION TESTS")
    print(f"{'=' * 60}")

    all_prop_results = {}

    for source, target in TIER1:
        print(f"\n  {source} -> {target}  (expected sign={EXPECTED_SIGNS.get((source, target), 0):+.0f})")

        s_prices = aligned[source]
        t_prices = aligned[target]

        s_imp, s_dir = detect_impulses(s_prices)

        n_imp = int(np.sum(s_imp))
        print(f"    Source impulses: {n_imp} ({n_imp/n*100:.1f}%)")

        s_rets = np.diff(np.log(s_prices), prepend=np.log(s_prices[0]))
        t_rets = np.diff(np.log(t_prices), prepend=np.log(t_prices[0]))

        pair_results = {}
        expected = EXPECTED_SIGNS.get((source, target), 0)

        for lag in [1, 2, 3, 6, 12]:
            lagged_rets = np.roll(t_rets, -lag)
            lagged_rets[-lag:] = 0

            scr_all = float(np.mean(s_rets * lagged_rets))
            scr_impulse = float(np.mean(s_rets[s_imp] * lagged_rets[s_imp])) if n_imp > 0 else 0.0

            n_consistent = 0
            for i in range(200, n):
                if s_imp[i] and expected != 0:
                    if s_rets[i] * lagged_rets[i] * expected > 0:
                        n_consistent += 1
            pct_consistent = round(n_consistent / max(n_imp, 1) * 100, 1)

            pair_results[f"lag{lag}"] = {
                "scr_all": round(scr_all, 8),
                "scr_impulse": round(scr_impulse, 8),
                "consistent_pct": pct_consistent,
                "n_imp": n_imp
            }

            print(f"      lag={lag:2d}:  SCR_all={scr_all:.8f}  SCR_imp={scr_impulse:.8f}  Consistent={pct_consistent:.1f}%")

        all_prop_results[f"{source}→{target}"] = pair_results

    print(f"\n{'=' * 60}")
    print("  YEAR-SPLIT STABILITY")
    print(f"{'=' * 60}")

    for source, target in TIER1:
        print(f"\n  {source} → {target}")
        s_prices = aligned[source]
        t_prices = aligned[target]
        s_imp, s_dir = detect_impulses(s_prices)
        s_rets = np.diff(np.log(s_prices), prepend=np.log(s_prices[0]))
        t_rets = np.diff(np.log(t_prices), prepend=np.log(t_prices[0]))
        expected = EXPECTED_SIGNS.get((source, target), 0)

        years_file = os.path.join(DATA_INTRA, f"{source}_M5.parquet")
        if os.path.exists(years_file):
            df_ts = pl.read_parquet(years_file)["timestamp"].to_numpy()
            years = np.array([datetime.fromtimestamp(t).year for t in df_ts])
            years = years[-n:]

            for yr in sorted(set(years)):
                mask = years == yr
                lag = 3
                lagged_rets = np.roll(t_rets, -lag)
                lagged_rets[-lag:] = 0
                scr_yr = float(np.mean((s_rets * lagged_rets)[mask & s_imp])) if np.sum(mask & s_imp) > 0 else 0.0
                n_yr = int(np.sum(mask & s_imp))
                n_consistent = 0
                for i in range(len(s_rets)):
                    if mask[i] and s_imp[i] and expected != 0:
                        if s_rets[i] * lagged_rets[i] * expected > 0:
                            n_consistent += 1
                pct = round(n_consistent / max(n_yr, 1) * 100, 1)
                print(f"    {yr}: n={n_yr}  SCR={scr_yr:.8f}  Consistent={pct}%")

    print(f"\n{'=' * 60}")
    print("  AGGREGATE PROPAGATION SUMMARY")
    print(f"{'=' * 60}")

    n_positive_scr = 0
    n_total = 0
    for key, pres in all_prop_results.items():
        scr3 = pres.get("lag3", {}).get("scr_all", 0)
        consistent = pres.get("lag3", {}).get("consistent_pct", 0)
        print(f"  {key:20s}: lag3 SCR={scr3:.8f}  consistent={consistent:.1f}%")
        if scr3 > 0:
            n_positive_scr += 1
        n_total += 1

    print(f"\n  SCR > 0: {n_positive_scr}/{n_total}")
    print(f"  Consistent > 50%: TODO")

    if n_positive_scr >= 3:
        print("\n  VERDICT: PROPAGATION EDGE DETECTED (multiple source-target pairs show positive SCR)")
    elif n_positive_scr >= 2:
        print("\n  VERDICT: MARGINAL (some pairs show propagation but not consistent)")
    else:
        print("\n  VERDICT: NO EDGE (cross-asset propagation not detected)")

    import json
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dpl15_results.json"), "w") as f:
        json.dump(all_prop_results, f, indent=2)
    print(f"\n  Results saved")


if __name__ == "__main__":
    main()
