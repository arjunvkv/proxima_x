"""
DPL-6 Walk-Forward Validation (P0-P4).
Tests whether regime sign inversion survives out-of-sample periods,
per-asset analysis, year-by-year stability, combined-regime expansion,
and regime-conditioned sizing simulation.

Usage: python research/directional_physics/dpl6_walkforward_validation.py
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent  # proxima_x/
sys.path.insert(0, str(SRC))

from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.temporal_topology import TemporalTopology

SYMBOLS = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD"]
DATA_DIR = Path(SRC) / "data" / "market"
HORIZONS = [20, 50, 100]
HORIZON_LABELS = {20: "H20", 50: "H50", 100: "H100"}
WARMUP = 100
REGIME_NAMES = {0: "accumulation", 1: "release", 2: "neutral"}


def load_symbol(symbol: str) -> dict:
    path = DATA_DIR / f"{symbol}.parquet"
    df = pl.read_parquet(str(path))
    price = df["close"].to_numpy().astype(np.float64)
    returns = np.diff(np.log(price), prepend=np.log(price[0]))
    volume = df["volume"].to_numpy().astype(np.float64) if "volume" in df.columns else np.ones(len(price))
    high = df["high"].to_numpy().astype(np.float64) if "high" in df.columns else price.copy()
    low = df["low"].to_numpy().astype(np.float64) if "low" in df.columns else price.copy()
    timestamps = df["timestamp"].to_numpy()
    return {"symbol": symbol, "price": price, "returns": returns,
            "volume": volume, "high": high, "low": low,
            "timestamps": timestamps, "n": len(price)}


def generate_signals(symbol: str, ed: EnergyDynamics, tt: TemporalTopology,
                     max_horizon: int = 100) -> list:
    """Generate all feature+label records for one symbol."""
    data = load_symbol(symbol)
    price = data["price"]
    n = data["n"]
    records = []
    for idx in range(WARMUP, n - max_horizon):
        data_dict = {k: v[:idx + 1] for k, v in data.items()
                     if k not in ("symbol", "timestamps", "n")}
        res_ed = ed.compute(data_dict)
        res_tt = tt.compute(data_dict)
        es_arr = np.nan_to_num(res_ed.get("energy_storage", np.zeros(idx + 1)), nan=0.0)
        current_es = es_arr[-1]
        es_rank = float(np.sum(es_arr <= current_es)) / len(es_arr) if len(es_arr) > 0 else 0.5
        energy_regime = int(res_ed.get("energy_regime", np.array([2]))[-1])
        time_regime = int(res_tt.get("time_regime", np.array([2]))[-1])
        combined_regime = energy_regime * 3 + time_regime
        ts = data["timestamps"][idx]
        rec = {"symbol": symbol, "idx": idx, "timestamp": str(ts)[:10],
               "es_rank": round(es_rank, 4), "energy_regime": energy_regime,
               "time_regime": time_regime, "combined_regime": combined_regime}
        for h in HORIZONS:
            fidx = idx + h
            if fidx < n and price[idx] > 0:
                ret = (price[fidx] - price[idx]) / price[idx]
                rec[f"return_h{h}"] = round(float(ret), 6)
                rec[f"direction_h{h}"] = 1 if ret > 0 else 0
            else:
                rec[f"return_h{h}"] = None
                rec[f"direction_h{h}"] = None
        if rec.get("direction_h20") is not None:
            records.append(rec)
    return records


def year_from_ts(ts_str: str) -> int:
    return int(ts_str[:4])


# ============================================================
# P0: Walk-forward splits
# ============================================================
def walk_forward_splits(records: list, train_years: list, test_year: int):
    """Split records into train/test by year."""
    train = [r for r in records if year_from_ts(r["timestamp"]) in train_years]
    test = [r for r in records if year_from_ts(r["timestamp"]) == test_year]
    return train, test


def regime_table(records, regime_key, horizon):
    groups = {}
    for r in records:
        rv = r.get(regime_key)
        if rv is None:
            continue
        d = r.get(f"direction_h{horizon}")
        if d is None:
            continue
        groups.setdefault(rv, []).append(d)
    table = {}
    for rv, dirs in sorted(groups.items()):
        n_s = len(dirs)
        p_up = float(np.mean(dirs))
        se = float(np.std(dirs) / np.sqrt(n_s)) if n_s > 1 else 0.5
        z = (p_up - 0.5) / max(se, 1e-6)
        table[str(rv)] = {"n": n_s, "p_up": round(p_up, 4), "se": round(se, 4), "z_score": round(z, 3)}
    return table


def compute_direction_bias(train_table: dict, test_table: dict) -> dict:
    """Compare train vs test: does the regime ordering survive?"""
    results = {}
    for rv in set(list(train_table.keys()) + list(test_table.keys())):
        tr = train_table.get(rv, {"p_up": 0.5, "n": 0})
        te = test_table.get(rv, {"p_up": 0.5, "n": 0})
        results[rv] = {"train_p_up": tr["p_up"], "train_n": tr["n"],
                       "test_p_up": te["p_up"], "test_n": te["n"]}
    return results


# ============================================================
# P2: Year-by-year stability
# ============================================================
def year_by_year(records, regime_key="energy_regime", horizon=20):
    years = sorted(set(year_from_ts(r["timestamp"]) for r in records))
    stability = {}
    for yr in years:
        yr_recs = [r for r in records if year_from_ts(r["timestamp"]) == yr]
        tbl = regime_table(yr_recs, regime_key, horizon)
        stability[str(yr)] = tbl
    return stability


# ============================================================
# P4: Regime-conditioned sizing simulation
# ============================================================
def sizing_simulation(records: list, horizon=20,
                      size_map: dict = None) -> dict:
    """Simulate regime-conditioned position sizing.

    Baseline: even 1.0x sizing on all signals.
    Regime-sized: multiply by size_map[regime].
    Returns: total_return_long, total_return_short, n_signals.
    """
    if size_map is None:
        size_map = {"0": 1.5, "1": 1.0, "2": 0.5}

    total_baseline_ret = 0.0
    total_regime_ret = 0.0
    n = 0
    regime_returns = {}

    for r in records:
        regime = str(r.get("energy_regime", 2))
        ret = r.get(f"return_h{horizon}")
        if ret is None:
            continue
        n += 1
        total_baseline_ret += ret
        mult = size_map.get(regime, 1.0)
        total_regime_ret += ret * mult
        regime_returns.setdefault(regime, []).append(ret)

    # Compute per-regime stats
    per_regime = {}
    for reg, rets in regime_returns.items():
        arr = np.array(rets)
        per_regime[reg] = {
            "n": len(rets),
            "mean_return": round(float(np.mean(arr)), 6),
            "std_return": round(float(np.std(arr)), 6),
            "sharpe": round(float(np.mean(arr) / max(np.std(arr), 1e-10)), 4),
            "p_up": round(float(np.mean(np.array(arr) > 0)), 4),
            "total_return": round(float(np.sum(arr)), 6),
        }

    return {
        "n": n,
        "size_map": size_map,
        "baseline_total_return": round(total_baseline_ret, 6),
        "regime_sized_total_return": round(total_regime_ret, 6),
        "improvement": round(total_regime_ret - total_baseline_ret, 6),
        "improvement_pct": round(
            (total_regime_ret - total_baseline_ret) / max(abs(total_baseline_ret), 1e-10) * 100, 2),
        "per_regime": per_regime,
    }


def run_walkforward(records, horizon=20):
    """P0: Walk-forward DPL-6 validation across multiple train/test splits."""
    years = sorted(set(year_from_ts(r["timestamp"]) for r in records))
    results = {}
    for split_idx in range(len(years) - 3):
        train_years = years[split_idx:split_idx + 2]
        test_year = years[split_idx + 2]
        train, test = walk_forward_splits(records, train_years, test_year)

        for regime_key, label in [("energy_regime", "Energy Regime"),
                                   ("time_regime", "Time Regime"),
                                   ("combined_regime", "Combined Regime")]:
            tr_table = regime_table(train, regime_key, horizon)
            te_table = regime_table(test, regime_key, horizon)
            comparison = compute_direction_bias(tr_table, te_table)

            # Check if regime ordering survives
            ordered_train = sorted(tr_table.items(), key=lambda x: x[1]["p_up"], reverse=True)
            ordered_test = sorted(te_table.items(), key=lambda x: x[1]["p_up"], reverse=True)
            # top regime match?
            top_match = ordered_train[0][0] == ordered_test[0][0] if ordered_train and ordered_test else False
            key = f"wf_{split_idx}_{regime_key}_H{horizon}"
            results[key] = {
                "split": f"{'-'.join(str(y) for y in train_years)}->{test_year}",
                "train_years": train_years,
                "test_year": test_year,
                "regime_type": label,
                "n_train": len(train),
                "n_test": len(test),
                "top_regime_match": top_match,
                "comparison": comparison,
            }
    return results


def main():
    ed = EnergyDynamics()
    tt = TemporalTopology()

    print("=" * 80)
    print("  DPL-6 WALK-FORWARD VALIDATION: P0-P4")
    print("=" * 80)
    print(f"  Data: {DATA_DIR}")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Warmup: {WARMUP} bars, Horizons: {', '.join(HORIZON_LABELS.values())}")
    print()

    # Generate all signals
    all_records = []
    per_symbol = {}
    for sym in SYMBOLS:
        print(f"  Generating signals for {sym}...", end=" ", flush=True)
        recs = generate_signals(sym, ed, tt)
        all_records.extend(recs)
        per_symbol[sym] = recs
        print(f"{len(recs)} signals")

    n_total = len(all_records)
    print(f"\n  Total signals: {n_total}")
    print()

    # ============================================================
    # P0: Walk-forward validation
    # ============================================================
    print("=" * 80)
    print("  P0: WALK-FORWARD DPL-6 VALIDATION")
    print("=" * 80)
    wf_h20 = run_walkforward(all_records, horizon=20)
    wf_h50 = run_walkforward(all_records, horizon=50)
    wf_h100 = run_walkforward(all_records, horizon=100)
    wf_all = {**wf_h20, **wf_h50, **wf_h100}

    for horizon in [20, 50, 100]:
        print(f"\n  --- H{horizon} Walk-Forward ---")
        for rk in ["energy_regime", "time_regime", "combined_regime"]:
            key_base = f"wf_0_{rk}_H{horizon}"
            entry = wf_all.get(key_base)
            if not entry:
                continue
            print(f"\n  {entry['regime_type']}:")
            print(f"    Split: {entry['split']}  |  Train: {entry['n_train']}  |  Test: {entry['n_test']}")
            print(f"    Top regime match: {'YES' if entry['top_regime_match'] else 'NO'}")
            for rv, v in sorted(entry["comparison"].items()):
                if v["train_n"] > 0 or v["test_n"] > 0:
                    print(f"    R{rv}: Train P(up)={v['train_p_up']:.4f}(n={v['train_n']})  "
                          f"Test P(up)={v['test_p_up']:.4f}(n={v['test_n']})")

    # ============================================================
    # P1: Per-asset regime analysis
    # ============================================================
    print("\n" + "=" * 80)
    print("  P1: PER-ASSET REGIME ANALYSIS")
    print("=" * 80)
    for horizon in [20, 50, 100]:
        print(f"\n  --- H{horizon} ---")
        print(f"  {'Symbol':<10} {'R0 P(up)':<12} {'R0 n':<8} {'R1 P(up)':<12} {'R1 n':<8} {'R2 P(up)':<12} {'R2 n':<8} {'Spread':<10}")
        print(f"  {'-'*10} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*10}")
        for sym in SYMBOLS:
            tbl = regime_table(per_symbol[sym], "energy_regime", horizon)
            r0 = tbl.get("0", {"p_up": 0, "n": 0})
            r1 = tbl.get("1", {"p_up": 0, "n": 0})
            r2 = tbl.get("2", {"p_up": 0, "n": 0})
            spread = r0["p_up"] - r2["p_up"]
            print(f"  {sym:<10} {r0['p_up']:<12.4f} {r0['n']:<8} {r1['p_up']:<12.4f} {r1['n']:<8} {r2['p_up']:<12.4f} {r2['n']:<8} {spread:<10.4f}")

    # ============================================================
    # P2: Year-by-year stability
    # ============================================================
    print("\n" + "=" * 80)
    print("  P2: YEAR-BY-YEAR ENERGY REGIME STABILITY (H20)")
    print("=" * 80)
    yby = year_by_year(all_records, "energy_regime", 20)
    years = sorted(yby.keys())
    print(f"\n  {'Year':<8} {'R0 P(up)':<12} {'R0 n':<8} {'R1 P(up)':<12} {'R1 n':<8} {'R2 P(up)':<12} {'R2 n':<8}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*12} {'-'*8}")
    r0_vals, r2_vals = [], []
    for yr in years:
        tbl = yby[yr]
        r0 = tbl.get("0", {"p_up": 0, "n": 0})
        r2 = tbl.get("2", {"p_up": 0, "n": 0})
        r0_vals.append(r0["p_up"])
        r2_vals.append(r2["p_up"])
        print(f"  {yr:<8} {r0['p_up']:<12.4f} {r0['n']:<8} {'-':<12} {'-':<8} {r2['p_up']:<12.4f} {r2['n']:<8}")

    if r0_vals:
        print(f"\n  R0 stability: mean={np.mean(r0_vals):.4f} std={np.std(r0_vals):.4f} "
              f"min={min(r0_vals):.4f} max={max(r0_vals):.4f}")
    if r2_vals:
        print(f"  R2 stability: mean={np.mean(r2_vals):.4f} std={np.std(r2_vals):.4f} "
              f"min={min(r2_vals):.4f} max={max(r2_vals):.4f}")
    if r0_vals and r2_vals:
        spreads = [r0_vals[i] - r2_vals[i] for i in range(min(len(r0_vals), len(r2_vals)))]
        print(f"  R0-R2 spread: mean={np.mean(spreads):.4f} std={np.std(spreads):.4f} "
              f"min={min(spreads):.4f} max={max(spreads):.4f}")
        print(f"  Spread positive in {sum(1 for s in spreads if s > 0)}/{len(spreads)} years "
              f"({100*sum(1 for s in spreads if s > 0)/len(spreads):.0f}%)")

    # ============================================================
    # P3: Combined-regime sample expansion
    # ============================================================
    print("\n" + "=" * 80)
    print("  P3: COMBINED REGIME DETAIL (H20)")
    print("=" * 80)
    cr_tbl = regime_table(all_records, "combined_regime", 20)
    print(f"\n  {'Regime':<8} {'Name':<30} {'P(up)':<10} {'N':<8} {'Z-score':<10}")
    print(f"  {'-'*8} {'-'*30} {'-'*10} {'-'*8} {'-'*10}")
    for rv in sorted(cr_tbl.keys(), key=lambda x: int(x)):
        v = cr_tbl[rv]
        er = int(rv) // 3
        tr = int(rv) % 3
        name = f"E{REGIME_NAMES[er]}|T{['low','med','high'][tr]}"
        print(f"  R{rv:<6} {name:<30} {v['p_up']:<10.4f} {v['n']:<8} {v['z_score']:<10.3f}")

    # ============================================================
    # P4: Regime-conditioned sizing simulation
    # ============================================================
    print("\n" + "=" * 80)
    print("  P4: REGIME-CONDITIONED SIZING SIMULATION (H20)")
    print("=" * 80)

    # Sizing strategies to test
    strategies = [
        {"name": "Conservative (R0=1.5x, R1=1.0x, R2=0.5x)", "map": {"0": 1.5, "1": 1.0, "2": 0.5}},
        {"name": "Aggressive (R0=2.0x, R1=1.0x, R2=0.25x)", "map": {"0": 2.0, "1": 1.0, "2": 0.25}},
        {"name": "Neutral (R0=1.0x, R1=1.0x, R2=1.0x)", "map": {"0": 1.0, "1": 1.0, "2": 1.0}},
        {"name": "R0 only (R0=1.0x, R1=0x, R2=0x)", "map": {"0": 1.0, "1": 0.0, "2": 0.0}},
        {"name": "Inverse (R0=0.5x, R1=1.0x, R2=1.5x)", "map": {"0": 0.5, "1": 1.0, "2": 1.5}},
    ]

    for strategy in strategies:
        sim = sizing_simulation(all_records, 20, strategy["map"])
        imp = sim["improvement_pct"]
        imp_str = f"+{imp}%" if imp > 0 else f"{imp}%"
        print(f"\n  {strategy['name']}:")
        print(f"    Baseline total return:     {sim['baseline_total_return']:.6f}")
        print(f"    Regime-sized total return: {sim['regime_sized_total_return']:.6f}")
        print(f"    Improvement:               {sim['improvement']:.6f} ({imp_str})")
        for reg, stats in sorted(sim["per_regime"].items()):
            name_r = REGIME_NAMES.get(int(reg), "?")
            mult = strategy["map"].get(reg, 1.0)
            print(f"    R{reg}({name_r}) mult={mult}x:  mean_ret={stats['mean_return']:.4f}  "
                  f"p_up={stats['p_up']:.3f}  sharpe={stats['sharpe']:.3f}  n={stats['n']}")

    # ============================================================
    # Aggregate findings
    # ============================================================
    print("\n" + "=" * 80)
    print("  AGGREGATE FINDINGS")
    print("=" * 80)

    # Determine if positive edge exists
    p0_survives = sum(1 for k, v in wf_all.items() if v.get("top_regime_match"))
    p0_total = sum(1 for k, v in wf_all.items() if "top_regime_match" in v)
    wf_survival_rate = p0_survives / max(p0_total, 1) * 100

    # P1: per-asset spread consistency
    per_asset_spreads = {}
    for sym in SYMBOLS:
        tbl = regime_table(per_symbol[sym], "energy_regime", 20)
        r0 = tbl.get("0", {"p_up": 0})
        r2 = tbl.get("2", {"p_up": 0})
        per_asset_spreads[sym] = r0["p_up"] - r2["p_up"]
    positive_spread_assets = sum(1 for s in per_asset_spreads.values() if s > 0.05)
    total_assets_with_data = sum(1 for s in per_asset_spreads.values() if s != 0)

    # P2: stability
    if r0_vals and r2_vals:
        r0_cv = np.std(r0_vals) / max(np.mean(r0_vals), 1e-10)
        r2_cv = np.std(r2_vals) / max(np.mean(r2_vals), 1e-10)
        spread_stability = np.std(spreads) if spreads else 1.0
    else:
        r0_cv = r2_cv = spread_stability = 1.0

    # P4: sizing improvement
    cons_sim = sizing_simulation(all_records, 20, {"0": 1.5, "1": 1.0, "2": 0.5})
    sizing_improvement = cons_sim["improvement_pct"]

    print(f"\n  P0 Walk-forward survival rate:  {wf_survival_rate:.0f}% ({p0_survives}/{p0_total} splits)")
    print(f"  P1 Assets with positive spread: {positive_spread_assets}/{total_assets_with_data}")
    print(f"  P2 R0 stability (CV):           {r0_cv:.3f}")
    print(f"  P2 R2 stability (CV):           {r2_cv:.3f}")
    print(f"  P2 Spread stability (std):      {spread_stability:.4f}")
    print(f"  P4 Sizing improvement:          {sizing_improvement:+.2f}%")

    has_edge = (
        wf_survival_rate > 60 and
        positive_spread_assets >= 3 and
        r0_cv < 0.3 and
        sizing_improvement > 5
    )

    print(f"\n  POSITIVE EDGE CONFIRMED: {'YES' if has_edge else 'NO (requires more validation)'}")

    # ============================================================
    # Save
    # ============================================================
    report_path = Path(__file__).parent / "reports" / "dpl6_walkforward_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "walkforward": wf_all,
        "per_asset": {sym: {h: regime_table(per_symbol[sym], "energy_regime", h)
                            for h in HORIZONS} for sym in SYMBOLS},
        "year_by_year": yby,
        "combined_regime": cr_tbl,
        "sizing_simulations": [{"name": s["name"],
                                "map": s["map"],
                                "results": sizing_simulation(all_records, 20, s["map"])}
                               for s in strategies],
        "aggregate": {
            "total_signals": n_total,
            "wf_survival_rate_pct": round(wf_survival_rate, 1),
            "wf_survived": p0_survives,
            "wf_total": p0_total,
            "assets_positive_spread": positive_spread_assets,
            "assets_with_data": total_assets_with_data,
            "r0_cv": round(r0_cv, 4),
            "r2_cv": round(r2_cv, 4),
            "spread_stability": round(spread_stability, 4),
            "sizing_improvement_pct": round(sizing_improvement, 2),
            "has_positive_edge": has_edge,
        }
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Full results saved to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
