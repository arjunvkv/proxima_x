"""ROL-2: Residual Persistence Physics.

Research Questions:
1. Does Hurst exponent increase before strong directional moves?
2. Is residual persistence regime-dependent?
3. Does persistence predict magnitude of subsequent moves?
4. Is there a persistence threshold for directional release?
5. Does persistence interact with ES level?
"""

import sys, json, warnings
from pathlib import Path
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.residual_origin.rol_core import ROLCore, SYMBOLS, save_rol_report

REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [10, 20, 50, 100]
HORIZON_MAP = {"H5": 1, "H20": 2, "H50": 3}
HORIZON_NAMES = ["H5", "H20", "H50"]
FUT_RET_IDX = [1, 2, 3]  # indices into fut_ret for H5, H20, H50


def rolling_hurst(arr, window=50):
    result = np.full(len(arr), np.nan)
    for i in range(window, len(arr)):
        chunk = arr[i - window:i]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) < 20:
            continue
        max_lag = min(20, len(chunk) // 2)
        if max_lag < 3:
            continue
        lags = list(range(2, max_lag))
        tau = []
        for lag in lags:
            diff = chunk[lag:] - chunk[:-lag]
            tau.append(np.std(diff))
        tau = np.array(tau)
        if len(tau) < 3 or np.any(tau == 0):
            continue
        slope, _, _, _, _ = stats.linregress(np.log(lags), np.log(tau))
        result[i] = slope / 2 + 1
    return result


def hurst_simple(x):
    if len(x) < 20 or np.std(x) < 1e-10:
        return np.nan
    max_lag = min(20, len(x) // 2)
    if max_lag < 3:
        return np.nan
    lags = list(range(2, max_lag))
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    if any(t == 0 for t in tau):
        return np.nan
    slope, _, _, _, _ = stats.linregress(np.log(lags), np.log(tau))
    return slope / 2 + 1


def make_report():
    rol = ROLCore()
    rol.load_all()
    print("ROL-2: Residual Persistence Physics")
    print("=" * 60)

    all_results = {}
    global_h_deciles = {}

    for sym in SYMBOLS:
        print(f"\n--- {sym} ---")
        res = rol.get_residuals(sym, "linear")
        regime = rol.get_regime(sym)
        fut_ret = rol.get_future_returns(sym)
        es = rol.get_es(sym)

        n = len(res)
        print(f"  n={n}")

        symbol_data = {}

        for w in WINDOWS:
            key = f"H_w{w}"
            print(f"  Rolling Hurst (w={w})...")
            h_vals = rolling_hurst(res, window=w)
            symbol_data[key] = h_vals.tolist()

        # Use w=50 as primary Hurst for analysis
        h_primary = np.array(symbol_data["H_w50"])

        # Filter valid (non-NaN) region — also ensure fut_ret is valid
        fut_valid = np.all(~np.isnan(fut_ret), axis=1)
        valid = ~np.isnan(h_primary) & ~np.isnan(res) & fut_valid
        valid_idx = np.where(valid)[0]

        if len(valid_idx) < 100:
            print(f"  SKIP: insufficient valid data ({len(valid_idx)})")
            all_results[sym] = {"error": "insufficient_valid_data"}
            continue

        h_valid = h_primary[valid]
        res_sign = np.sign(res[valid])
        reg_valid = regime[valid]
        es_valid = es[valid]

        record = {
            "n_total": int(n),
            "n_valid": int(len(valid_idx)),
            "h_mean": float(np.nanmean(h_valid)),
            "h_std": float(np.nanstd(h_valid)),
            "h_min": float(np.nanmin(h_valid)),
            "h_max": float(np.nanmax(h_valid)),
            "h_median": float(np.nanmedian(h_valid)),
            "pct_h_above_0_5": float(np.mean(h_valid > 0.5)),
            "pct_h_above_0_7": float(np.mean(h_valid > 0.7)),
            "pct_h_below_0_5": float(np.mean(h_valid < 0.5)),
        }

        # RQ2: Persistence by regime
        print("  RQ2: persistence by regime...")
        regime_stats = {}
        for r in [0, 1, 2]:
            mask = reg_valid == r
            n_r = int(np.sum(mask))
            if n_r < 20:
                regime_stats[str(r)] = {"count": n_r, "h_mean": None, "h_std": None}
                continue
            h_r = h_valid[mask]
            regime_stats[str(r)] = {
                "count": n_r,
                "h_mean": float(np.nanmean(h_r)),
                "h_std": float(np.nanstd(h_r)),
                "h_median": float(np.nanmedian(h_r)),
                "pct_h_above_0_5": float(np.mean(h_r > 0.5)),
            }
        record["regime_stats"] = regime_stats

        # RQ1, RQ3: Decile analysis for each horizon
        print("  RQ1/RQ3: decile analysis...")
        h_bins = np.nanpercentile(h_valid, np.linspace(0, 100, 11))
        h_bins[-1] += 1e-6  # ensure max is inclusive

        decile_results = {}
        for hname, fidx in zip(HORIZON_NAMES, FUT_RET_IDX):
            fut = fut_ret[valid_idx, fidx]
            fut_abs = np.abs(fut)
            fut_up = (fut > 0).astype(float)

            deciles = []
            for d in range(10):
                lo = h_bins[d]
                hi = h_bins[d + 1]
                mask_d = (h_valid >= lo) & (h_valid < hi)
                n_d = int(np.sum(mask_d))
                if n_d < 5:
                    deciles.append({"decile": d, "count": n_d, "h_lo": float(lo), "h_hi": float(hi)})
                    continue
                deciles.append({
                    "decile": d,
                    "count": n_d,
                    "h_lo": float(lo),
                    "h_hi": float(hi),
                    "h_mean": float(np.nanmean(h_valid[mask_d])),
                    "p_up": float(np.mean(fut_up[mask_d])),
                    "mean_return": float(np.nanmean(fut[mask_d])),
                    "std_return": float(np.nanstd(fut[mask_d])),
                    "mean_abs_return": float(np.nanmean(fut_abs[mask_d])),
                    "sharpe": float(np.nanmean(fut[mask_d]) / max(np.nanstd(fut[mask_d]), 1e-12)),
                })
            decile_results[hname] = deciles
        record["deciles"] = decile_results

        # RQ4: Persistence threshold — find where P(up) jumps
        print("  RQ4: persistence threshold...")
        thresholds = {}
        for hname, fidx in zip(HORIZON_NAMES, FUT_RET_IDX):
            fut = fut_ret[valid_idx, fidx]
            fut_up = (fut > 0).astype(float)

            # Scan across H values, find where P(up) changes significantly
            h_sorted_idx = np.argsort(h_valid)
            h_sorted = h_valid[h_sorted_idx]
            up_sorted = fut_up[h_sorted_idx]

            window = max(50, len(h_sorted) // 100)
            roll_pup = np.full(len(h_sorted), np.nan)
            for i in range(window, len(h_sorted) - window):
                roll_pup[i] = np.mean(up_sorted[i - window:i + window])

            # Find biggest jump in rolling P(up)
            thr_idx = np.where(~np.isnan(roll_pup))[0]
            if len(thr_idx) > 10:
                diffs = np.diff(roll_pup[thr_idx])
                if len(diffs) > 5:
                    max_abs_idx_inner = int(np.argmax(np.abs(diffs)))
                    max_abs_idx = thr_idx[max_abs_idx_inner]
                    threshold_h = float(np.mean([
                        h_sorted[max_abs_idx],
                        h_sorted[min(max_abs_idx + 1, len(h_sorted) - 1)]
                    ]))
                    jump_size = float(diffs[max_abs_idx_inner])
                    half_w = window // 2
                    before_start = thr_idx[max(0, max_abs_idx_inner - half_w)]
                    before_end = thr_idx[max_abs_idx_inner]
                    after_start = thr_idx[min(len(thr_idx) - 1, max_abs_idx_inner + 1)]
                    after_end = thr_idx[min(len(thr_idx) - 1, max_abs_idx_inner + 1 + half_w)]
                    threshold_info = {
                        "threshold_h": round(threshold_h, 4),
                        "jump_idx": int(max_abs_idx),
                        "jump_size": round(jump_size, 4),
                        "p_up_before": float(np.nanmean(roll_pup[before_start:before_end])),
                        "p_up_after": float(np.nanmean(roll_pup[after_start:after_end])),
                    }
                else:
                    threshold_info = {"threshold_h": None, "jump_size": 0}
            else:
                threshold_info = {"threshold_h": None, "jump_size": 0}
            thresholds[hname] = threshold_info
        record["thresholds"] = thresholds

        # RQ5: H × ES interaction
        print("  RQ5: H × ES interaction...")
        es_bins = np.nanpercentile(es_valid[~np.isnan(es_valid)], [33.33, 66.67])
        h_bins_tert = np.nanpercentile(h_valid, [33.33, 66.67])

        interaction = {}
        for hname, fidx in zip(HORIZON_NAMES, FUT_RET_IDX):
            fut = fut_ret[valid_idx, fidx]
            fut_up = (fut > 0).astype(float)

            cells = []
            for h_label, h_lo, h_hi in [("low", -np.inf, h_bins_tert[0]), ("mid", h_bins_tert[0], h_bins_tert[1]), ("high", h_bins_tert[1], np.inf)]:
                for e_label, e_lo, e_hi in [("low", -np.inf, es_bins[0]), ("mid", es_bins[0], es_bins[1]), ("high", es_bins[1], np.inf)]:
                    mask = (h_valid >= h_lo) & (h_valid < h_hi) & (es_valid >= e_lo) & (es_valid < e_hi)
                    n_cell = int(np.sum(mask))
                    if n_cell < 10:
                        cells.append({"h_level": h_label, "es_level": e_label, "count": n_cell})
                        continue
                    cells.append({
                        "h_level": h_label,
                        "es_level": e_label,
                        "count": n_cell,
                        "p_up": float(np.mean(fut_up[mask])),
                        "mean_return": float(np.nanmean(fut[mask])),
                        "mean_abs_return": float(np.nanmean(np.abs(fut[mask]))),
                    })
            interaction[hname] = cells
        record["interaction"] = interaction

        # RQ1b: Does H increase in N bars before strong moves?
        print("  RQ1b: H before strong moves...")
        pre_move_stats = {}
        for hname, fidx in zip(HORIZON_NAMES, FUT_RET_IDX):
            fut = fut_ret[:, fidx]
            fut_z = (fut - np.nanmean(fut)) / max(np.nanstd(fut), 1e-12)
            strong_up = (fut_z > 2.0) & ~np.isnan(h_primary)
            strong_down = (fut_z < -2.0) & ~np.isnan(h_primary)

            # Prevent out-of-bounds for pre_window lookback
            pre_window = 10
            lookback = max(50, pre_window)  # need at least 50 bars of H

            pre_stats = {}
            for direction_label, mask in [("up", strong_up), ("down", strong_down)]:
                events = np.where(mask)[0]
                # Filter events with enough lookback
                events = events[events - pre_window >= 0]
                events = events[events - lookback >= 0]

                if len(events) < 5:
                    pre_stats[direction_label] = {"count": 0}
                    continue

                h_before = []
                baseline = []
                all_h_50 = np.array(symbol_data.get("H_w50", [np.nan] * n))
                for ev in events:
                    pre_vals = all_h_50[ev - pre_window:ev]
                    pre_vals = pre_vals[~np.isnan(pre_vals)]
                    base_vals = all_h_50[ev - lookback:ev - pre_window]
                    base_vals = base_vals[~np.isnan(base_vals)]
                    if len(pre_vals) > 0 and len(base_vals) > 0:
                        h_before.append(np.nanmean(pre_vals))
                        baseline.append(np.nanmean(base_vals))

                h_before = np.array(h_before)
                baseline = np.array(baseline)
                if len(h_before) < 5:
                    pre_stats[direction_label] = {"count": 0}
                    continue

                h_ratio = h_before / (baseline + 1e-12)
                pre_stats[direction_label] = {
                    "count": int(len(h_before)),
                    "h_before_mean": float(np.nanmean(h_before)),
                    "h_baseline_mean": float(np.nanmean(baseline)),
                    "h_ratio_mean": float(np.nanmean(h_ratio)),
                    "pct_increase": float(np.mean(h_before > baseline)),
                }
            pre_move_stats[hname] = pre_stats
        record["pre_move"] = pre_move_stats

        # RQ3b: Correlation: H vs magnitude
        print("  RQ3b: H vs magnitude correlation...")
        magnitude_corr = {}
        for hname, fidx in zip(HORIZON_NAMES, FUT_RET_IDX):
            fut = fut_ret[valid_idx, fidx]
            fut_abs = np.abs(fut)
            valid_pair = ~np.isnan(h_valid) & ~np.isnan(fut_abs)
            if np.sum(valid_pair) > 30:
                r, p = stats.pearsonr(h_valid[valid_pair], fut_abs[valid_pair])
                magnitude_corr[hname] = {
                    "pearson_r": round(float(r), 4),
                    "p_value": round(float(p), 6),
                    "n": int(np.sum(valid_pair)),
                }
            else:
                magnitude_corr[hname] = {"pearson_r": None, "p_value": None, "n": int(np.sum(valid_pair))}
        record["magnitude_corr"] = magnitude_corr

        all_results[sym] = record
        print(f"  Done. H mean={record['h_mean']:.3f}, H>0.5={record['pct_h_above_0_5']:.1%}")

    # RQ2 — cross-symbol regime summary
    print("\n=== CROSS-SYMBOL SUMMARY ===")
    cross_summary = {}
    for sym, data in all_results.items():
        if "error" in data:
            continue
        rs = data.get("regime_stats", {})
        cross_summary[sym] = {
            "h_mean": data["h_mean"],
            "pct_h_above_0_5": data["pct_h_above_0_5"],
            "pct_h_above_0_7": data["pct_h_above_0_7"],
            "regime_h_diff": {r: rs.get(r, {}).get("h_mean") for r in ["0", "1", "2"]},
        }

    report = {
        "title": "ROL-2: Residual Persistence Physics",
        "description": "Investigates whether residual persistence (Hurst exponent) predicts directional moves",
        "windows": WINDOWS,
        "horizons": HORIZON_NAMES,
        "symbols": all_results,
        "cross_symbol_summary": cross_summary,
    }

    return report


def format_md(report):
    lines = []
    lines.append("# ROL-2: Residual Persistence Physics")
    lines.append("")
    lines.append(f"**Windows:** {report['windows']}  ")
    lines.append(f"**Horizons:** {report['horizons']}  ")
    lines.append("")

    # Cross-symbol table
    lines.append("## Cross-Symbol Summary")
    lines.append("")
    lines.append("| Symbol | H_mean | H>0.5 | H>0.7 | Regime0 H | Regime1 H | Regime2 H |")
    lines.append("|--------|--------|-------|-------|-----------|-----------|-----------|")
    css = report.get("cross_symbol_summary", {})
    for sym in SYMBOLS:
        if sym not in css:
            continue
        d = css[sym]
        rh = d.get("regime_h_diff", {})
        lines.append(
            f"| {sym} | {d['h_mean']:.3f} | {d['pct_h_above_0_5']:.1%} | "
            f"{d['pct_h_above_0_7']:.1%} | {_f(rh.get('0'))} | {_f(rh.get('1'))} | {_f(rh.get('2'))} |"
        )
    lines.append("")

    for sym in SYMBOLS:
        data = report["symbols"].get(sym)
        if data is None or "error" in data:
            continue
        lines.append(f"## {sym}")
        lines.append("")

        # RQ1: Deciles
        lines.append("### RQ1/RQ3: Hurst Decile → Return Profile")
        lines.append("")
        for hname in HORIZON_NAMES:
            deciles = data["deciles"].get(hname, [])
            lines.append(f"**{hname}**")
            lines.append("| Decile | H range | Count | P(up) | Mean Ret | Std Ret | Sharpe |")
            lines.append("|--------|---------|-------|-------|----------|---------|--------|")
            for d in deciles:
                if d["count"] < 5:
                    lines.append(f"| {d['decile']} | {d['h_lo']:.3f}-{d['h_hi']:.3f} | {d['count']} | - | - | - | - |")
                else:
                    lines.append(
                        f"| {d['decile']} | {d['h_lo']:.3f}-{d['h_hi']:.3f} | {d['count']} | "
                        f"{d['p_up']:.1%} | {d['mean_return']:.4f} | {d['std_return']:.4f} | "
                        f"{d['sharpe']:.3f} |"
                    )
            lines.append("")

        # RQ2: Regime
        lines.append("### RQ2: Regime-Dependent Persistence")
        lines.append("")
        rs = data.get("regime_stats", {})
        lines.append("| Regime | Count | H_mean | H_std | H_median | H>0.5 |")
        lines.append("|--------|-------|--------|-------|----------|-------|")
        for r in ["0", "1", "2"]:
            d = rs.get(r, {})
            if d.get("h_mean") is None:
                lines.append(f"| S{r} | {d.get('count', 0)} | - | - | - | - |")
            else:
                lines.append(
                    f"| S{r} | {d['count']} | {d['h_mean']:.3f} | {d['h_std']:.3f} | "
                    f"{d['h_median']:.3f} | {d['pct_h_above_0_5']:.1%} |"
                )
        lines.append("")

        # RQ4: Threshold
        lines.append("### RQ4: Persistence Threshold")
        lines.append("")
        for hname in HORIZON_NAMES:
            t = data["thresholds"].get(hname, {})
            if t.get("threshold_h") is not None:
                lines.append(
                    f"- **{hname}**: threshold H={t['threshold_h']:.3f}, "
                    f"jump={t['jump_size']:.3f} (before: {t['p_up_before']:.1%}, after: {t['p_up_after']:.1%})"
                )
            else:
                lines.append(f"- **{hname}**: no clear threshold detected")
        lines.append("")

        # RQ5: H × ES
        lines.append("### RQ5: H × ES Interaction")
        lines.append("")
        for hname in HORIZON_NAMES:
            cells = data["interaction"].get(hname, [])
            lines.append(f"**{hname}** — P(up) by H × ES tertile")
            lines.append("| ES \\ H | Low | Mid | High |")
            lines.append("|--------|-----|-----|------|")
            for e_label in ["low", "mid", "high"]:
                row = f"| {e_label} "
                for h_label in ["low", "mid", "high"]:
                    cell = next(
                        (c for c in cells if c.get("h_level") == h_label and c.get("es_level") == e_label),
                        None
                    )
                    if cell and cell["count"] >= 10:
                        row += f"| {cell['p_up']:.1%} (n={cell['count']}) "
                    else:
                        row += "| - "
                lines.append(row + "|")
            lines.append("")

        # RQ1b: Pre-move H
        lines.append("### RQ1b: H Before Strong Directional Moves (>2σ)")
        lines.append("")
        for hname in HORIZON_NAMES:
            pm = data["pre_move"].get(hname, {})
            lines.append(f"**{hname}**")
            lines.append("| Direction | Count | H_before_mean | H_baseline | Ratio | % Increase |")
            lines.append("|-----------|-------|---------------|------------|-------|------------|")
            for direction_label in ["up", "down"]:
                d = pm.get(direction_label, {})
                if d.get("count", 0) < 5:
                    lines.append(f"| {direction_label} | {d.get('count', 0)} | - | - | - | - |")
                else:
                    lines.append(
                        f"| {direction_label} | {d['count']} | {d['h_before_mean']:.3f} | "
                        f"{d['h_baseline_mean']:.3f} | {d['h_ratio_mean']:.3f} | "
                        f"{d['pct_increase']:.1%} |"
                    )
            lines.append("")

        # RQ3b: Correlation
        lines.append("### RQ3b: H vs Magnitude Correlation")
        lines.append("")
        for hname in HORIZON_NAMES:
            mc = data["magnitude_corr"].get(hname, {})
            if mc.get("pearson_r") is not None:
                sig = " (significant)" if mc["p_value"] < 0.05 else ""
                lines.append(f"- **{hname}**: r={mc['pearson_r']:.4f}, p={mc['p_value']:.4f}, n={mc['n']}{sig}")
            else:
                lines.append(f"- **{hname}**: insufficient data")
        lines.append("")

    return "\n".join(lines)


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.3f}"


if __name__ == "__main__":
    report = make_report()
    json_path = REPORT_DIR / "rol2_persistence_physics.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {json_path}")

    md_content = format_md(report)
    md_path = REPORT_DIR / "ROL2_PERSISTENCE_PHYSICS.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved {md_path}")

    # Print key findings
    print("\n=== KEY FINDINGS ===")
    css = report.get("cross_symbol_summary", {})
    for sym in SYMBOLS:
        d = css.get(sym)
        if d:
            print(f"{sym}: H={d['h_mean']:.3f}, persistent={d['pct_h_above_0_5']:.1%}")

    # Check if H increases before strong moves across symbols
    inc_count = 0
    total = 0
    for sym in SYMBOLS:
        data = report["symbols"].get(sym)
        if not data or "error" in data:
            continue
        for hname in HORIZON_NAMES:
            for direction in ["up", "down"]:
                pm = data.get("pre_move", {}).get(hname, {}).get(direction, {})
                if pm.get("count", 0) >= 5:
                    total += 1
                    if pm.get("pct_increase", 0) > 0.5:
                        inc_count += 1
    if total > 0:
        print(f"\nH increases before strong moves: {inc_count}/{total} cases ({inc_count/total:.1%})")
