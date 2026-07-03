"""LSV-2: Residual Generator Sensitivity.

Tests whether residual sign edge survives changes to residual construction method.
If the phenomenon is genuinely market-linked, it should survive changes to generator.
If it's model-linked, changing the generator will destroy it.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.directional_physics.dpl_core import DPLData, SYMBOLS
from research.directional_state.dsr_core import WalkForwardValidator

REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_KEYS = ["H5", "H20", "H50"]
FUT_RET_IDX = [1, 2, 3]

RESIDUAL_NAMES = [
    "linear",          # R0: baseline --- vol-model linear regression
    "return_based",    # R1: log return - rolling mean
    "vol_normalized",  # R2: ES / rolling_vol(ES)
    "es_percentile",   # R3: ES percentile - 0.5
    "mae",             # R4: ES - rolling_mean(ES, 7)
    "random_forest",   # R5: RF model residual
    "xgboost",         # R6: XGB model residual
]

WF_SPLITS = [
    ("2018-2019", "2020"),
    ("2020-2022", "2023"),
    ("2022-2024", "2025"),
]


def rolling_mean(x, window):
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(window, n):
        r[i] = np.nanmean(x[i - window:i])
    return r


def rolling_std(x, window):
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(window, n):
        r[i] = np.nanstd(x[i - window:i])
    return r


def rolling_percentile_rank(x, window):
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(window, n):
        chunk = x[i - window:i]
        r[i] = float(np.sum(chunk <= x[i])) / window
    return r


def compute_residual_sign(res):
    s = np.sign(res)
    s[np.isnan(s)] = 0
    return s.astype(np.int64)


def p_up_given_sign(sign, fut_ret, fidx):
    fut = fut_ret[:, fidx]
    p_ups = {}
    for sval in [1, -1]:
        mask = sign == sval
        n = int(np.sum(mask))
        if n < 5:
            p_ups[str(sval)] = {"n": n, "p_up": None}
        else:
            p_ups[str(sval)] = {"n": n, "p_up": float(np.nanmean(fut[mask] > 0))}
    baseline = float(np.nanmean(fut > 0))
    return p_ups, baseline


def build_alternative_residuals(d):
    n = len(d.es)
    residuals = {}

    # R0: linear (from ML model)
    residuals["linear"] = d.residuals.get("linear", np.full(n, np.nan))

    # R1: return-based residual
    ret = d.returns.copy()
    mean_ret = rolling_mean(ret, 20)
    residuals["return_based"] = ret - mean_ret

    # R2: vol-normalized residual
    es = d.es.copy()
    vol = d.vol_metrics.get("realized_vol", np.full(n, np.nan))
    vol_safe = np.maximum(vol, 1e-12)
    residuals["vol_normalized"] = es / vol_safe

    # R3: ES percentile residual
    pct = rolling_percentile_rank(es, 252)
    residuals["es_percentile"] = pct - 0.5

    # R4: MAE (moving average error, window=7)
    ma7 = rolling_mean(es, 7)
    residuals["mae"] = es - ma7

    # R5, R6: ML residuals
    residuals["random_forest"] = d.residuals.get("random_forest", np.full(n, np.nan))
    residuals["xgboost"] = d.residuals.get("xgboost", np.full(n, np.nan))

    return residuals


def extract_years(d):
    raw = d.data.get("raw")
    if raw is not None and hasattr(raw, "columns"):
        col = "timestamp" if "timestamp" in raw.columns else "time"
        times = raw[col].to_list() if hasattr(raw[col], "to_list") else list(raw[col])
        return np.array([t.year for t in times], dtype=np.int32)
    return None


def make_split_masks(years, train_desc, test_year_str):
    test_year = int(test_year_str)
    train_start = int(train_desc[:4])
    train_end = test_year - 1
    train_mask = (years >= train_start) & (years <= train_end)
    test_mask = years == test_year
    return train_mask, test_mask


def compute_accuracy(sign, fut_ret, fidx):
    fut = fut_ret[:, fidx]
    valid = ~np.isnan(fut) & (sign != 0)
    if np.sum(valid) < 5:
        return {"n": 0, "accuracy": None}
    pred_up = (sign[valid] == 1)
    actual_up = (fut[valid] > 0)
    correct = int(np.sum(pred_up == actual_up))
    total = int(np.sum(valid))
    return {"n": total, "accuracy": correct / total}


def make_report():
    print("=" * 72)
    print("  LSV-2: Residual Generator Sensitivity")
    print("  Tests whether residual sign edge survives changes to generator")
    print("=" * 72)

    all_results = {}
    cross_residual = {}  # sign correlation across residuals

    for sym in SYMBOLS:
        print(f"\n{'=' * 60}")
        print(f"  {sym}")
        print(f"{'=' * 60}")

        d = DPLData(sym)
        n = len(d.es)
        residuals = build_alternative_residuals(d)
        fut_ret = d.fut_ret
        years = extract_years(d)

        print(f"  n={n}, years={int(years[0]) if years is not None else '?'}-{int(years[-1]) if years is not None else '?'}")

        symbol_data = {}

        for rname in RESIDUAL_NAMES:
            res = residuals[rname]
            n_valid = int(np.sum(~np.isnan(res)))
            sign = compute_residual_sign(res)
            n_nonzero = int(np.sum(sign != 0))

            print(f"\n  --- {rname} (valid={n_valid}, nonzero_sign={n_nonzero}) ---")

            # Full-sample: P(up|sign) for each horizon
            horizon_results = {}
            for hk, fidx in zip(HORIZON_KEYS, FUT_RET_IDX):
                p_ups, baseline = p_up_given_sign(sign, fut_ret, fidx)
                p_pos = p_ups.get("1", {}).get("p_up")
                p_neg = p_ups.get("-1", {}).get("p_up")
                acc = compute_accuracy(sign, fut_ret, fidx)
                horizon_results[hk] = {
                    "p_up_given_pos": p_pos,
                    "p_up_given_neg": p_neg,
                    "baseline_p_up": baseline,
                    "accuracy": acc,
                    "edge": (p_pos - baseline) if p_pos is not None else None,
                }
                if p_pos is not None and p_neg is not None:
                    print(f"    {hk}: P(up|+)={p_pos:.1%}, P(up|-)={p_neg:.1%}, base={baseline:.1%}, acc={acc['accuracy']:.1%}" if acc['accuracy'] else f"    {hk}: P(up|+)={p_pos:.1%}, P(up|-)={p_neg:.1%}, base={baseline:.1%}")

            # Walk-forward on 3 splits
            wf_results = {}
            if years is not None:
                for train_desc, test_year in WF_SPLITS:
                    train_mask, test_mask = make_split_masks(years, train_desc, test_year)
                    n_train = int(np.sum(train_mask))
                    n_test = int(np.sum(test_mask))
                    if n_test < 20:
                        wf_results[f"{train_desc}_to_{test_year}"] = {"n_train": n_train, "n_test": n_test, "error": "insufficient_test_data"}
                        continue
                    wf_horizons = {}
                    for hk, fidx in zip(HORIZON_KEYS, FUT_RET_IDX):
                        test_sign = sign[test_mask]
                        test_fut = fut_ret[test_mask, fidx]
                        train_p_ups, _ = p_up_given_sign(sign[train_mask], fut_ret[train_mask, :], fidx)
                        train_pos = train_p_ups.get("1", {}).get("p_up")
                        acc = compute_accuracy(test_sign, fut_ret[test_mask, :], fidx)
                        test_p_ups, test_base = p_up_given_sign(test_sign, fut_ret[test_mask, :], fidx)
                        test_pos = test_p_ups.get("1", {}).get("p_up")
                        wf_horizons[hk] = {
                            "train_p_up_given_pos": train_pos,
                            "test_p_up_given_pos": test_pos,
                            "test_baseline": test_base,
                            "test_accuracy": acc,
                        }
                    wf_results[f"{train_desc}_to_{test_year}"] = {
                        "n_train": n_train,
                        "n_test": n_test,
                        "horizons": wf_horizons,
                    }
                    n_wf_test = wf_results[f"{train_desc}_to_{test_year}"]["n_test"]
                    print(f"    WF {train_desc}->{test_year}: n_test={n_wf_test}")
            else:
                print("    WF: no timestamps available")

            symbol_data[rname] = {
                "n": int(n),
                "n_valid": n_valid,
                "n_nonzero_sign": n_nonzero,
                "horizons": horizon_results,
                "walk_forward": wf_results,
            }

        # Cross-residual sign correlation
        print(f"\n  --- Cross-residual sign correlation matrix ---")
        n_res = len(RESIDUAL_NAMES)
        corr_matrix = np.full((n_res, n_res), np.nan)
        for i, rn1 in enumerate(RESIDUAL_NAMES):
            s1 = compute_residual_sign(residuals[rn1])
            for j, rn2 in enumerate(RESIDUAL_NAMES):
                if i > j:
                    continue
                s2 = compute_residual_sign(residuals[rn2])
                valid = (s1 != 0) & (s2 != 0)
                if np.sum(valid) < 10:
                    continue
                corr_matrix[i, j] = float(np.corrcoef(s1[valid], s2[valid])[0, 1])
                corr_matrix[j, i] = corr_matrix[i, j]

        # Print correlation matrix
        header = f"{'':>16}" + "".join(f"{rn:<16}" for rn in RESIDUAL_NAMES)
        print(f"  {header}")
        for i, rn1 in enumerate(RESIDUAL_NAMES):
            row = f"  {rn1:>16}"
            for j in range(n_res):
                v = corr_matrix[i, j]
                if np.isnan(v):
                    row += f"{'NaN':>16}"
                else:
                    row += f"{v:>16.3f}"
            print(row)

        # Determine if each residual produces sign correlated with linear baseline
        linear_idx = RESIDUAL_NAMES.index("linear")
        sign_correlations = {}
        for i, rn in enumerate(RESIDUAL_NAMES):
            if i == linear_idx:
                sign_correlations[rn] = 1.0
            else:
                sign_correlations[rn] = None if np.isnan(corr_matrix[linear_idx, i]) else float(corr_matrix[linear_idx, i])

        all_results[sym] = {
            "n": int(n),
            "residuals": symbol_data,
            "sign_correlation_with_linear": sign_correlations,
            "sign_correlation_matrix": {RESIDUAL_NAMES[i]: {RESIDUAL_NAMES[j]: (None if np.isnan(corr_matrix[i, j]) else float(corr_matrix[i, j])) for j in range(n_res)} for i in range(n_res)},
        }

    # --- VERDICT ---
    print(f"\n{'=' * 72}")
    print(f"  VERDICT: Is residual sign generator-dependent?")
    print(f"{'=' * 72}")

    ML_RESIDUALS = ["linear", "random_forest", "xgboost"]
    STAT_RESIDUALS = ["return_based", "mae", "es_percentile"]
    DEGENERATE = ["vol_normalized"]

    verdict = {}
    for sym in SYMBOLS:
        sd = all_results[sym]
        linear_edge = {}
        for hk in HORIZON_KEYS:
            le = sd["residuals"]["linear"]["horizons"][hk].get("edge")
            linear_edge[hk] = le

        residual_edges = {}
        for rn in RESIDUAL_NAMES:
            re = {}
            for hk in HORIZON_KEYS:
                e = sd["residuals"][rn]["horizons"][hk].get("edge")
                re[hk] = e
            residual_edges[rn] = re

        non_degenerate = [rn for rn in RESIDUAL_NAMES if rn not in DEGENERATE]

        pos_edge_count = sum(1 for rn in non_degenerate if residual_edges[rn].get("H20") is not None and residual_edges[rn]["H20"] > 0)
        neg_edge_count = sum(1 for rn in non_degenerate if residual_edges[rn].get("H20") is not None and residual_edges[rn]["H20"] < 0)

        ml_pos = sum(1 for rn in ML_RESIDUALS if residual_edges[rn].get("H20") is not None and residual_edges[rn]["H20"] > 0)
        ml_neg = sum(1 for rn in ML_RESIDUALS if residual_edges[rn].get("H20") is not None and residual_edges[rn]["H20"] < 0)
        stat_pos = sum(1 for rn in STAT_RESIDUALS if residual_edges[rn].get("H20") is not None and residual_edges[rn]["H20"] > 0)
        stat_neg = sum(1 for rn in STAT_RESIDUALS if residual_edges[rn].get("H20") is not None and residual_edges[rn]["H20"] < 0)

        ml_direction = "positive" if ml_pos > ml_neg else "negative" if ml_neg > ml_pos else "mixed"
        stat_direction = "positive" if stat_pos > stat_neg else "negative" if stat_neg > stat_pos else "mixed"

        if ml_direction == stat_direction:
            sym_verdict = f"MARKET-LINKED: ML ({ml_direction}) and statistical ({stat_direction}) residuals agree"
        elif ml_direction == "mixed" or stat_direction == "mixed":
            sym_verdict = f"PARTIALLY GENERATOR-DEPENDENT: ML={ml_direction}, statistical={stat_direction} within-class mixed"
        else:
            sym_verdict = f"GENERATOR-DEPENDENT but STRUCTURED: ML residuals {ml_direction}, statistical residuals {stat_direction} --- CLASS POLARITY FLIP"

        verdict[sym] = {
            "linear_edge_h20": linear_edge.get("H20"),
            "residual_edges_h20": {rn: residual_edges[rn].get("H20") for rn in RESIDUAL_NAMES},
            "ml_edge_h20": {rn: residual_edges[rn].get("H20") for rn in ML_RESIDUALS},
            "stat_edge_h20": {rn: residual_edges[rn].get("H20") for rn in STAT_RESIDUALS},
            "ml_direction": ml_direction,
            "stat_direction": stat_direction,
            "pos_edge_count": pos_edge_count,
            "neg_edge_count": neg_edge_count,
            "verdict": sym_verdict,
        }
        print(f"  {sym}: {sym_verdict}")
        print(f"    ML-based (lin/RF/XGB): {ml_pos}/{len(ML_RESIDUALS)} positive")
        print(f"    Statistical (ret/MAE/pct): {stat_pos}/{len(STAT_RESIDUALS)} positive")
        for rn in RESIDUAL_NAMES:
            e = residual_edges[rn].get("H20")
            if e is not None:
                print(f"    {rn:>20}: edge(H20) = {e:+.4f}")
            else:
                print(f"    {rn:>20}: edge(H20) = DEGENERATE/NO SIGN SPLIT")

    # Cross-symbol summary
    print(f"\n{'=' * 72}")
    print(f"  CROSS-SYMBOL CONSISTENCY")
    print(f"{'=' * 72}")

    for rn in RESIDUAL_NAMES:
        edges = []
        for sym in SYMBOLS:
            e = all_results[sym]["residuals"][rn]["horizons"]["H20"].get("edge")
            if e is not None:
                edges.append(e)
        if edges:
            mean_e = np.mean(edges)
            pos = sum(1 for e in edges if e > 0)
            neg = sum(1 for e in edges if e < 0)
            print(f"  {rn:>20}: mean_edge={mean_e:+.4f}, pos={pos}/{len(SYMBOLS)}, neg={neg}/{len(SYMBOLS)}")

    # Within-class consistency
    print(f"\n  --- WITHIN-CLASS CONSISTENCY ---")
    for cls_name, cls_list in [("ML-based (lin/RF/XGB)", ML_RESIDUALS), ("Statistical (ret/MAE/pct)", STAT_RESIDUALS)]:
        cls_edges = {rn: [] for rn in cls_list}
        for sym in SYMBOLS:
            for rn in cls_list:
                e = all_results[sym]["residuals"][rn]["horizons"]["H20"].get("edge")
                if e is not None:
                    cls_edges[rn].append(e)
        print(f"\n  {cls_name}:")
        if all(len(v) > 0 for v in cls_edges.values()):
            for rn in cls_list:
                mean_e = np.mean(cls_edges[rn])
                pos = sum(1 for e in cls_edges[rn] if e > 0)
                print(f"    {rn:>20}: mean={mean_e:+.4f}, pos={pos}/{len(SYMBOLS)}")
            # Check if all residuals in class agree on sign
            all_pos = all(np.mean(cls_edges[rn]) > 0 for rn in cls_list)
            all_neg = all(np.mean(cls_edges[rn]) < 0 for rn in cls_list)
            if all_pos:
                print(f"    -> CLASS-AGREED: All {cls_name} residuals positive across symbols")
            elif all_neg:
                print(f"    -> CLASS-AGREED: All {cls_name} residuals negative across symbols")
            else:
                print(f"    -> CLASS-MIXED: Residuals disagree within class")

    # Overall verdict — based on cross-symbol mean edge direction within each class
    print(f"\n  --- OVERALL VERDICT ---")

    # Check within-class agreement using the per-symbol per-residual means
    ml_means = {}
    for rn in ML_RESIDUALS:
        edges = [all_results[sym]["residuals"][rn]["horizons"]["H20"].get("edge") for sym in SYMBOLS]
        edges = [e for e in edges if e is not None]
        ml_means[rn] = np.mean(edges) if edges else 0
    stat_means = {}
    for rn in STAT_RESIDUALS:
        edges = [all_results[sym]["residuals"][rn]["horizons"]["H20"].get("edge") for sym in SYMBOLS]
        edges = [e for e in edges if e is not None]
        stat_means[rn] = np.mean(edges) if edges else 0

    ml_class_pos = all(v > 0 for v in ml_means.values())
    ml_class_neg = all(v < 0 for v in ml_means.values())
    stat_class_pos = all(v > 0 for v in stat_means.values())
    stat_class_neg = all(v < 0 for v in stat_means.values())

    ml_agree = ml_class_pos or ml_class_neg
    stat_agree = stat_class_pos or stat_class_neg

    # Per-symbol class agreement
    ml_sym_agreed = 0
    stat_sym_agreed = 0
    polarity_flip = 0
    for sym in SYMBOLS:
        v = verdict[sym]
        if v["ml_direction"] in ("positive", "negative"):
            ml_sym_agreed += 1
        if v["stat_direction"] in ("positive", "negative"):
            stat_sym_agreed += 1
        if v["ml_direction"] == "positive" and v["stat_direction"] == "negative":
            polarity_flip += 1
        if v["ml_direction"] == "negative" and v["stat_direction"] == "positive":
            polarity_flip += 1

    parts = []

    # Within-class description
    if ml_agree and stat_agree:
        ml_dir = "POSITIVE" if ml_class_pos else "NEGATIVE"
        stat_dir = "POSITIVE" if stat_class_pos else "NEGATIVE"
        if ml_dir == stat_dir:
            parts.append(f"Both classes agree (ML={ml_dir}, Statistical={stat_dir})")
            overall = "MARKET-LINKED: Residual sign edge survives generator changes --- consistent direction across all definitions and symbols"
        else:
            parts.append(f"CLASS POLARITY FLIP: ML={ml_dir}, Statistical={stat_dir}")
            parts.append(f"Occurs in {polarity_flip}/{len(SYMBOLS)} symbols")
            overall = "GENERATOR-DEPENDENT but STRUCTURED: Residual sign flips polarity depending on residual class (ML vs statistical), but within each class the direction is consistent across residuals. The sign captures real market structure but with definition-dependent polarity."
    elif ml_agree and not stat_agree:
        ml_dir = "POSITIVE" if ml_class_pos else "NEGATIVE"
        parts.append(f"ML class agreed ({ml_dir}), Statistical class mixed")
        overall = f"PARTIALLY STRUCTURED: ML residuals consistently {ml_dir} edge, but statistical residuals inconsistent"
    elif stat_agree and not ml_agree:
        stat_dir = "POSITIVE" if stat_class_pos else "NEGATIVE"
        parts.append(f"Statistical class agreed ({stat_dir}), ML class mixed")
        overall = f"PARTIALLY STRUCTURED: Statistical residuals consistently {stat_dir} edge, but ML residuals inconsistent"
    else:
        parts.append("Neither class agrees internally")
        overall = "GENERATOR-DEPENDENT: No consistent structure across residual classes"

    print(f"\n  {'; '.join(parts)}")
    print(f"\n  OVERALL: {overall}")

    report = {
        "title": "LSV-2: Residual Generator Sensitivity",
        "description": "Tests whether residual sign edge survives changes to residual construction method",
        "residual_definitions": RESIDUAL_NAMES,
        "horizons": HORIZON_KEYS,
        "wf_splits": WF_SPLITS,
        "symbols": all_results,
        "verdict": verdict,
        "overall_verdict": overall,
    }

    return report, overall


def format_md(report, overall):
    lines = []
    lines.append("# LSV-2: Residual Generator Sensitivity")
    lines.append("")
    lines.append("**Question:** Does residual sign edge survive CHANGES to the residual construction method?")
    lines.append("")
    lines.append("| Residual | Definition |")
    lines.append("|----------|-----------|")
    for rn in RESIDUAL_NAMES:
        desc = {
            "linear": "Linear regression: ES ~ vol_metrics (baseline)",
            "return_based": "Log return - rolling mean (window=20)",
            "vol_normalized": "ES / rolling_vol(ES, window=20)",
            "es_percentile": "Rolling percentile(ES, 252) - 0.5",
            "mae": "ES - rolling_mean(ES, 7)",
            "random_forest": "Random Forest regression residual (ES ~ vol_metrics)",
            "xgboost": "XGBoost regression residual (ES ~ vol_metrics)",
        }
        lines.append(f"| {rn} | {desc.get(rn, '')} |")
    lines.append("")

    lines.append("## Per-Symbol Per-Residual Accuracy (H20)")
    lines.append("")
    header = "| Symbol | " + " | ".join(f"{rn}" for rn in RESIDUAL_NAMES) + " |"
    sep = "|--------|" + "|".join("--------|" for _ in RESIDUAL_NAMES)
    lines.append(header)
    lines.append(sep)
    for sym in SYMBOLS:
        row = f"| {sym} "
        for rn in RESIDUAL_NAMES:
            sd = report["symbols"].get(sym, {})
            acc = sd.get("residuals", {}).get(rn, {}).get("horizons", {}).get("H20", {}).get("accuracy", {})
            acc_val = acc.get("accuracy")
            if acc_val is not None:
                row += f"| {acc_val:.1%} "
            else:
                row += "| N/A "
        lines.append(row + "|")
    lines.append("")

    lines.append("## Cross-Residual Sign Correlation Matrix")
    lines.append("")
    for sym in SYMBOLS:
        lines.append(f"### {sym}")
        lines.append("")
        cm = report["symbols"].get(sym, {}).get("sign_correlation_matrix", {})
        header = "| " + " | ".join([""] + RESIDUAL_NAMES) + " |"
        sep = "|---|" + "|".join("---|" for _ in RESIDUAL_NAMES)
        lines.append(header)
        lines.append(sep)
        for rn1 in RESIDUAL_NAMES:
            row = f"| {rn1} "
            for rn2 in RESIDUAL_NAMES:
                v = cm.get(rn1, {}).get(rn2)
                if v is not None:
                    row += f"| {v:.3f} "
                else:
                    row += "| - "
            lines.append(row + "|")
        lines.append("")

    lines.append("## Walk-Forward Consistency")
    lines.append("")
    lines.append("| Symbol | Residual | Split | Train N | Test N | H5 Test Edge | H20 Test Edge | H50 Test Edge |")
    lines.append("|--------|----------|-------|---------|--------|-------------|--------------|-------------|")
    for sym in SYMBOLS:
        sd = report["symbols"].get(sym, {})
        for rn in RESIDUAL_NAMES:
            wf = sd.get("residuals", {}).get(rn, {}).get("walk_forward", {})
            for split_name, split_data in wf.items():
                if "error" in split_data:
                    continue
                horizons = split_data.get("horizons", {})
                h5 = horizons.get("H5", {}).get("test_accuracy", {}).get("accuracy")
                h20 = horizons.get("H20", {}).get("test_accuracy", {}).get("accuracy")
                h50 = horizons.get("H50", {}).get("test_accuracy", {}).get("accuracy")
                h5s = f"{h5:.1%}" if h5 is not None else "-"
                h20s = f"{h20:.1%}" if h20 is not None else "-"
                h50s = f"{h50:.1%}" if h50 is not None else "-"
                lines.append(f"| {sym} | {rn} | {split_name} | {split_data['n_train']} | {split_data['n_test']} | {h5s} | {h20s} | {h50s} |")
    lines.append("")

    lines.append("## Edge Direction by Residual (H20)")
    lines.append("")
    header = "| Symbol | " + " | ".join(f"{rn}" for rn in RESIDUAL_NAMES) + " |"
    sep = "|--------|" + "|".join("--------|" for _ in RESIDUAL_NAMES)
    lines.append(header)
    lines.append(sep)
    for sym in SYMBOLS:
        row = f"| {sym} "
        for rn in RESIDUAL_NAMES:
            sd = report["symbols"].get(sym, {})
            e = sd.get("residuals", {}).get(rn, {}).get("horizons", {}).get("H20", {}).get("edge")
            if e is not None:
                marker = "+" if e > 0 else ""
                row += f"| {marker}{e:.4f} "
            else:
                row += "| N/A "
        lines.append(row + "|")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- **vol_normalized** is degenerate for sign analysis: ES is always non-negative, so `ES/vol` produces no negative signs. This residual is excluded from class polarity analysis.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    for sym in SYMBOLS:
        v = report.get("verdict", {}).get(sym, {})
        lines.append(f"- **{sym}**: {v.get('verdict', 'N/A')}")
        ml_e = v.get("ml_direction", "?")
        st_e = v.get("stat_direction", "?")
        lines.append(f"  - ML residuals (lin/RF/XGB): {ml_e}")
        lines.append(f"  - Statistical residuals (ret/MAE/pct): {st_e}")
    lines.append("")
    lines.append(f"**Overall:** {overall}")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    report, overall = make_report()

    json_path = REPORT_DIR / "lsv2_generator_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {json_path}")

    md_content = format_md(report, overall)
    md_path = REPORT_DIR / "LSV2_GENERATOR_SENSITIVITY.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved {md_path}")

    print(f"\n{'=' * 72}")
    print(f"  {report['title']}")
    print(f"  Overall Verdict: {overall}")
    print(f"{'=' * 72}")
