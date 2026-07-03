"""
DPL-6 Replay Replication: Historical DPL-6 (Regime Sign Inversion) using
proper forward labels. Uses EnergyDynamics + TemporalTopology on parquet
data to compute per-bar regime tags and forward returns exactly as the
DelayedOutcomeEngine would resolve them.

Outputs P(up | Regime=R) tables for each symbol and aggregated across all.
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
from research.adaptive_alpha_engine.aae_validator import HORIZONS, _future_returns

SYMBOLS = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD"]
DATA_DIR = Path(SRC) / "data" / "market"
HORIZON_LABELS = {1: "H1", 5: "H5", 20: "H20", 50: "H50", 100: "H100", 500: "H500"}
WARMUP_BARS = 100  # minimum bars needed for stable feature computation


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


def compute_features(data: dict, ed: EnergyDynamics, tt: TemporalTopology, idx: int):
    """Compute features using ALL data up to and including idx (no lookahead)."""
    data_dict = {k: v[:idx + 1] for k, v in data.items() if k != "symbol" and k != "timestamps" and k != "n"}
    result_ed = ed.compute(data_dict)
    result_tt = tt.compute(data_dict)
    es_arr = np.nan_to_num(result_ed.get("energy_storage", np.zeros(idx + 1)), nan=0.0)
    current_es = es_arr[-1]
    es_rank = float(np.sum(es_arr <= current_es)) / len(es_arr) if len(es_arr) > 0 else 0.5
    energy_regime_arr = result_ed.get("energy_regime", np.array([2]))
    energy_regime = int(energy_regime_arr[-1]) if len(energy_regime_arr) > 0 else None
    time_regime_arr = result_tt.get("time_regime", np.array([2]))
    time_regime = int(time_regime_arr[-1]) if len(time_regime_arr) > 0 else None
    combined_regime = energy_regime * 3 + time_regime if (energy_regime is not None and time_regime is not None) else None
    return {"es_rank": es_rank, "energy_regime": energy_regime,
            "time_regime": time_regime, "combined_regime": combined_regime,
            "energy_storage": float(current_es)}


def compute_forward_labels(price: np.ndarray, idx: int, horizons: list):
    """Compute forward returns from idx using actual future prices (no lookahead)."""
    labels = {}
    n = len(price)
    for h in horizons:
        future_idx = idx + h
        if future_idx < n and price[idx] > 0:
            ret = (price[future_idx] - price[idx]) / price[idx]
            labels[f"return_h{h}"] = float(ret)
            labels[f"direction_h{h}"] = 1 if ret > 0 else (0 if ret < 0 else None)
        else:
            labels[f"return_h{h}"] = None
            labels[f"direction_h{h}"] = None
    return labels


def run_symbol(symbol: str, ed: EnergyDynamics, tt: TemporalTopology,
               horizons: list = None) -> dict:
    if horizons is None:
        horizons = [20, 50, 100]
    data = load_symbol(symbol)
    n = data["n"]
    price = data["price"]
    max_horizon = max(horizons)

    records = []
    for idx in range(WARMUP_BARS, n - max_horizon):
        features = compute_features(data, ed, tt, idx)
        labels = compute_forward_labels(price, idx, horizons)
        combined = {**features, **labels, "idx": idx}
        if labels.get(f"direction_h{horizons[0]}") is not None:
            records.append(combined)

    # Build regime tables
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
            n_samp = len(dirs)
            p_up = float(np.mean(dirs))
            se = float(np.std(dirs) / np.sqrt(n_samp)) if n_samp > 1 else 0.5
            z = (p_up - 0.5) / max(se, 1e-6)
            table[str(rv)] = {"n": n_samp, "p_up": round(p_up, 4),
                              "se": round(se, 4), "z_score": round(z, 3)}
        return table

    result = {"symbol": symbol, "total_bars": n, "valid_signals": len(records)}
    for h in horizons:
        hl = HORIZON_LABELS.get(h, f"H{h}")
        result[f"energy_regime_{hl}"] = regime_table(records, "energy_regime", h)
        result[f"time_regime_{hl}"] = regime_table(records, "time_regime", h)
        result[f"combined_regime_{hl}"] = regime_table(records, "combined_regime", h)
    return result


def aggregate_results(symbol_results: list[dict], horizons: list = None):
    if horizons is None:
        horizons = [20, 50, 100]
    agg = {"symbols_processed": [r["symbol"] for r in symbol_results],
           "total_signals": sum(r["valid_signals"] for r in symbol_results)}
    for h in horizons:
        hl = HORIZON_LABELS.get(h, f"H{h}")
        for regime_type in ["energy_regime", "time_regime", "combined_regime"]:
            key = f"{regime_type}_{hl}"
            all_dirs = {}
            for sr in symbol_results:
                for rv, rinfo in sr.get(key, {}).items():
                    all_dirs.setdefault(rv, []).extend([1] * rinfo["n"] if rinfo["p_up"] >= 0.5 else [0] * rinfo["n"])
                    # Actually let's be precise: collect actual directions
            # Proper aggregation
            combined = {}
            for sr in symbol_results:
                tbl = sr.get(key, {})
                for rv_str, rinfo in tbl.items():
                    combined.setdefault(rv_str, {"n": 0, "sum_up": 0})
                    combined[rv_str]["n"] += rinfo["n"]
                    # We can't recover exact count of ups from just p_up, so let's redo
            # Better approach: rebuild from per-symbol per-regime p_up
            agg[key] = {"per_symbol": {}}
            for sr in symbol_results:
                sym = sr["symbol"]
                tbl = sr.get(key, {})
                if tbl:
                    agg[key]["per_symbol"][sym] = tbl
                # Cross-symbol aggregate: average p_up weighted by n
            cross = {}
            n_total = 0
            for sr in symbol_results:
                tbl = sr.get(key, {})
                for rv_str, rinfo in tbl.items():
                    cross.setdefault(rv_str, {"n": 0, "weighted_sum": 0.0})
                    cross[rv_str]["n"] += rinfo["n"]
                    cross[rv_str]["weighted_sum"] += rinfo["p_up"] * rinfo["n"]
            cross_agg = {}
            for rv_str, v in sorted(cross.items()):
                if v["n"] > 0:
                    p_up_agg = v["weighted_sum"] / v["n"]
                    cross_agg[rv_str] = {"n": v["n"], "p_up": round(p_up_agg, 4)}
            agg[key]["cross_symbol"] = cross_agg
    return agg


def main():
    ed = EnergyDynamics()
    tt = TemporalTopology()
    print("=" * 70)
    print("  DPL-6 REPLAY REPLICATION: Historical Regime Sign Inversion Test")
    print("=" * 70)
    print(f"  Data: {DATA_DIR}")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Data range: Daily 2019-2025 (~1820 bars per symbol)")
    print(f"  Warmup: {WARMUP_BARS} bars, Horizons: H20/H50/H100")
    print("=" * 70)
    print()
    results = []
    for sym in SYMBOLS:
        print(f"Processing {sym}...", end=" ", flush=True)
        sr = run_symbol(sym, ed, tt)
        results.append(sr)
        ns = sr["valid_signals"]
        print(f"done ({ns} valid signals)")
    print()
    agg = aggregate_results(results)
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"\nTotal valid signals across all symbols: {agg['total_signals']}")
    for h in [20, 50, 100]:
        hl = HORIZON_LABELS.get(h, f"H{h}")
        print(f"\n  --- {hl} ---")
        for regime_type, label in [("energy_regime", "Energy Regime"),
                                    ("time_regime", "Time Regime"),
                                    ("combined_regime", "Combined Regime")]:
            key = f"{regime_type}_{hl}"
            data = agg.get(key, {})
            cross = data.get("cross_symbol", {})
            if not cross:
                continue
            print(f"\n  {label}:")
            for rv_str, v in sorted(cross.items()):
                print(f"    R{rv_str}:  P(up)={v['p_up']:.4f}  N={v['n']}")
            max_entry = max(cross.items(), key=lambda x: x[1]["p_up"])
            min_entry = min(cross.items(), key=lambda x: x[1]["p_up"])
            spread = max_entry[1]["p_up"] - min_entry[1]["p_up"]
            print(f"    Spread: {spread:.4f}  (R{max_entry[0]}={max_entry[1]['p_up']:.4f} - R{min_entry[0]}={min_entry[1]['p_up']:.4f})")
            if spread > 0.10:
                print(f"    *** REGIME SIGN INVERSION DETECTED (spread={spread:.1%}) ***")
            else:
                print(f"    No significant regime inversion (spread={spread:.1%})")
    print()
    print("=" * 70)
    print("  PER-SYMBOL DETAIL")
    print("=" * 70)
    for sr in results:
        sym = sr["symbol"]
        print(f"\n{sym} ({sr['valid_signals']} signals):")
        for h in [20, 50, 100]:
            hl = HORIZON_LABELS.get(h, f"H{h}")
            for regime_type, label in [("energy_regime", "Energy"),
                                        ("time_regime", "Time"),
                                        ("combined_regime", "Combined")]:
                key = f"{regime_type}_{hl}"
                tbl = sr.get(key, {})
                if tbl:
                    vals = [(rv, v["p_up"], v["n"]) for rv, v in tbl.items()]
                    vals_str = ", ".join(f"R{rv}={p:.3f}(n={n})" for rv, p, n in sorted(vals))
                    print(f"  {label} {hl}: {vals_str}")
    # Save to JSON for later analysis
    report_path = Path(__file__).parent / "reports" / "dpl6_replay_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output = {"aggregated": agg, "per_symbol": results}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to: {report_path}")


if __name__ == "__main__":
    main()
