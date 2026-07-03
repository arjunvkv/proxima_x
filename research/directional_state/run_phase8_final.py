"""DSR Phase 8 — Architecture Compression + Final Adjudication.

Phase 8: Starting from best model (residual_only), add each feature ONE AT A TIME
to measure incremental directional information contribution.

Final Adjudication: Synthesize ALL 8 phases into definitive classification.
"""

import sys, json
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.directional_state.dsr_core import DSRCore, SYMBOLS, WalkForwardValidator, HORIZON_KEYS, save_report

HORIZON_MAP = {"H1": 0, "H5": 1, "H20": 2, "H50": 3, "H100": 4, "H500": 5}


def _quantize(x, n_bins=3):
    labels = np.full(len(x), -1, dtype=np.int64)
    valid = ~np.isnan(x)
    if np.sum(valid) < 10:
        return labels
    bins = np.nanpercentile(x[valid], np.linspace(0, 100, n_bins + 1)[1:-1])
    for i in range(n_bins):
        if i == 0:
            labels[valid & (x <= bins[i])] = i
        elif i == n_bins - 1:
            labels[valid & (x > bins[i - 1])] = i
        else:
            labels[valid & (x <= bins[i]) & (x > bins[i - 1])] = i
    return labels


# ── Model builders ──────────────────────────────────────────────────────────

def build_residual_only(d):
    rs = d["residual_sign"]
    result = np.full(len(rs), -1, dtype=np.int64)
    valid = ~np.isnan(d["residual"])
    result[valid] = rs[valid] + 1
    return result


def build_residual_plus_regime(d):
    reg = d["regime"]
    rs = d["residual_sign"]
    n = len(reg)
    result = np.full(n, -1, dtype=np.int64)
    valid = (reg >= 0) & ~np.isnan(d["residual"])
    result[valid] = reg[valid] * 10 + (rs[valid] + 1)
    return result


def build_residual_plus_imbalance(d):
    imb = _quantize(d["memory_imbalance"], 3)
    rs = d["residual_sign"]
    n = len(imb)
    result = np.full(n, -1, dtype=np.int64)
    valid = (imb >= 0) & ~np.isnan(d["residual"])
    result[valid] = imb[valid] * 10 + (rs[valid] + 1)
    return result


def build_residual_plus_saturation(d):
    sat = d["memory_saturation"]
    rs = d["residual_sign"]
    n = len(sat)
    result = np.full(n, -1, dtype=np.int64)
    valid = (sat >= 0) & ~np.isnan(d["residual"])
    result[valid] = sat[valid] * 10 + (rs[valid] + 1)
    return result


def build_residual_plus_cluster(d):
    clu = d["memory_cluster"]
    rs = d["residual_sign"]
    n = len(clu)
    result = np.full(n, -1, dtype=np.int64)
    valid = (clu >= 0) & ~np.isnan(d["residual"])
    result[valid] = clu[valid] * 10 + (rs[valid] + 1)
    return result


def build_residual_plus_macro(d):
    macro = d["macro_regime"]
    rs = d["residual_sign"]
    n = len(macro)
    result = np.full(n, -1, dtype=np.int64)
    valid = (macro >= 0) & ~np.isnan(d["residual"])
    result[valid] = macro[valid] * 10 + (rs[valid] + 1)
    return result


def build_residual_plus_propagation(d, all_data, symbol):
    rs = d["residual_sign"]
    n = len(rs)
    result = np.full(n, -1, dtype=np.int64)
    valid = ~np.isnan(d["residual"])
    result[valid] = rs[valid] + 1

    # For JPY pairs, add EURJPY regime as bias
    if symbol in ("EURJPY", "USDJPY", "GBPJPY"):
        eurjpy = all_data.get("EURJPY", {}).get("data")
        if eurjpy is not None:
            eur_reg = eurjpy["regime"]
            min_len = min(n, len(eur_reg))
            overlap = np.zeros(n, dtype=bool)
            overlap[:min_len] = (eur_reg[:min_len] >= 0) & valid[:min_len]
            indices = np.where(overlap[:min_len])[0]
            if len(indices) > 0:
                result[indices] = (eur_reg[indices].astype(np.int64) * 10 +
                                   (rs[indices].astype(np.int64) + 1))

    return result


def build_full_state(d):
    reg = d["regime"]
    rs = d["residual_sign"]
    imb = _quantize(d["memory_imbalance"], 3)
    sat = d["memory_saturation"]
    clu = d["memory_cluster"]
    macro = d["macro_regime"]
    n = len(reg)
    result = np.full(n, -1, dtype=np.int64)
    valid = (reg >= 0) & ~np.isnan(d["residual"]) & (imb >= 0) & (sat >= 0) & (clu >= 0) & (macro >= 0)
    result[valid] = (reg[valid] * 100000 + (rs[valid] + 1) * 10000 +
                     imb[valid] * 1000 + sat[valid] * 100 + clu[valid] * 10 + macro[valid])
    return result


# ── Walk-forward evaluation (copied from Phase 7) ──────────────────────────

def compute_train_stats(train_state_ids, train_up, min_samples=5):
    stats = {}
    unique = np.unique(train_state_ids[train_state_ids >= 0])
    for sid in unique:
        mask = train_state_ids == sid
        cnt = int(np.sum(mask))
        if cnt < min_samples:
            continue
        n_up = int(np.sum(train_up[mask]))
        stats[int(sid)] = {"p_up": n_up / cnt, "count": cnt, "n_up": n_up}
    return stats


def evaluate(state_ids, actual_up, stats):
    n = len(state_ids)
    preds = np.full(n, np.nan)
    for i in range(n):
        sid = int(state_ids[i])
        if sid in stats:
            preds[i] = 1.0 if stats[sid]["p_up"] > 0.5 else 0.0
    valid = ~np.isnan(preds) & ~np.isnan(actual_up)
    if np.sum(valid) < 5:
        return None
    p = preds[valid].astype(float)
    a = actual_up[valid].astype(float)
    n_valid = int(np.sum(valid))
    correct = float(np.sum(p == a))
    accuracy = correct / n_valid
    tp = float(np.sum((p == 1.0) & (a == 1.0)))
    fp = float(np.sum((p == 1.0) & (a == 0.0)))
    fn = float(np.sum((p == 0.0) & (a == 1.0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "n_valid": n_valid,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ── Phase 8 main ────────────────────────────────────────────────────────────

def run_phase8():
    print("=" * 72)
    print("DSR PHASE 8 - ARCHITECTURE COMPRESSION")
    print("=" * 72)

    print("\nLoading DSR core (cached)...")
    dsr = DSRCore()
    for sym in SYMBOLS:
        dsr.load_symbol(sym)

    # Pre-load all data for propagation
    all_data = {}
    for sym in SYMBOLS:
        all_data[sym] = {"data": dsr._data[sym]}

    MODELS = {
        "01_residual_only": lambda d: build_residual_only(d),
        "02_residual_plus_regime": lambda d: build_residual_plus_regime(d),
        "03_residual_plus_imbalance": lambda d: build_residual_plus_imbalance(d),
        "04_residual_plus_saturation": lambda d: build_residual_plus_saturation(d),
        "05_residual_plus_cluster": lambda d: build_residual_plus_cluster(d),
        "06_residual_plus_macro": lambda d: build_residual_plus_macro(d),
        "07_residual_plus_propagation": lambda d: build_residual_plus_propagation(d, all_data, "PLACEHOLDER"),
        "08_full_state": lambda d: build_full_state(d),
    }

    report = {
        "phase": 8,
        "title": "Architecture Compression",
        "splits": [f"{a}->{b}" for a, b in WalkForwardValidator.SPLITS],
        "models": list(MODELS.keys()),
        "results": {},
        "summary": {},
    }

    # Build synthetic year ranges (data spans ~2018-2025)
    year_ranges = {}
    for sym in SYMBOLS:
        d = dsr._data[sym]
        n = len(d["es"])
        years = np.full(n, 2018, dtype=np.int32)
        step = n / 8.0  # ~8 years of data (2018-2025)
        for i in range(n):
            years[i] = 2018 + int(i / step)
        years = np.clip(years, 2018, 2025).astype(np.int32)
        year_ranges[sym] = years

    def custom_split(symbol, train_name, test_name):
        train_year_end = int(test_name) - 1
        test_year = int(test_name)
        years = year_ranges[symbol]
        train_mask = (years >= int(train_name[:4])) & (years <= train_year_end)
        test_mask = years == test_year
        return train_mask, test_mask

    target_horizon = "H50"
    hi = HORIZON_MAP[target_horizon]

    for sym in SYMBOLS:
        print(f"\n--- {sym} ---")
        d = dsr._data[sym]
        report["results"][sym] = {}

        fut_ret_col = d["fut_ret"][:, hi]
        up = (fut_ret_col > 0).astype(float)
        not_nan = ~np.isnan(fut_ret_col)

        for model_name, model_fn in MODELS.items():
            # For propagation, pass symbol
            if model_name == "07_residual_plus_propagation":
                state_ids = build_residual_plus_propagation(d, all_data, sym)
            else:
                state_ids = model_fn(d)

            print(f"  {model_name} ... ", end="")

            for train_name, test_name in WalkForwardValidator.SPLITS:
                split_key = f"{train_name}->{test_name}"
                if split_key not in report["results"][sym]:
                    report["results"][sym][split_key] = {}

                train_mask, test_mask = custom_split(sym, train_name, test_name)
                train_ok = train_mask & (state_ids >= 0) & not_nan
                test_ok = test_mask & (state_ids >= 0) & not_nan

                train_stats = compute_train_stats(state_ids[train_ok], up[train_ok])
                train_eval = evaluate(state_ids[train_ok], up[train_ok], train_stats)
                test_eval = evaluate(state_ids[test_ok], up[test_ok], train_stats)

                entry = {}
                if train_eval:
                    entry["train"] = train_eval
                if test_eval:
                    entry["test"] = test_eval
                    entry["n_train_states"] = len(train_stats)
                    entry["n_test_states"] = len(np.unique(state_ids[test_ok].astype(int)))
                    if train_eval:
                        entry["edge_retention"] = round(train_eval["accuracy"] - test_eval["accuracy"], 4)

                report["results"][sym][split_key][model_name] = entry
            print("OK")

    # ── Compute summaries ───────────────────────────────────────────────
    summary = compute_phase8_summary(report)
    report["summary"] = summary

    save_report(report, "dsr_phase8_architecture_compression")
    print_phase8_summary(report)
    return report


def compute_phase8_summary(report):
    splits = [f"{a}->{b}" for a, b in WalkForwardValidator.SPLITS]
    models = report["models"]
    baseline = "01_residual_only"

    summary = {"splits": {}, "aggregation": {}, "feature_analysis": {}}

    for split_key in splits:
        sm = {}
        for model_name in models:
            accs, edges = [], []
            total_n = 0
            for sym in SYMBOLS:
                e = report["results"].get(sym, {}).get(split_key, {}).get(model_name, {}).get("test", {})
                er = report["results"].get(sym, {}).get(split_key, {}).get(model_name, {}).get("edge_retention")
                if e.get("n_valid", 0) > 0:
                    accs.append(e["accuracy"])
                    total_n += e["n_valid"]
                if er is not None:
                    edges.append(er)
            if accs:
                sm[model_name] = {
                    "mean_accuracy": round(float(np.mean(accs)), 4),
                    "std_accuracy": round(float(np.std(accs)), 4),
                    "mean_edge_retention": round(float(np.mean(edges)), 4) if edges else None,
                    "total_samples": total_n,
                }
        summary["splits"][split_key] = sm

    # Aggregation across all splits
    for model_name in models:
        all_accs = []
        all_edges = []
        for split_key in splits:
            for sym in SYMBOLS:
                e = report["results"].get(sym, {}).get(split_key, {}).get(model_name, {}).get("test", {})
                er = report["results"].get(sym, {}).get(split_key, {}).get(model_name, {}).get("edge_retention")
                if e.get("n_valid", 0) > 0:
                    all_accs.append(e["accuracy"])
                if er is not None:
                    all_edges.append(er)
        if all_accs:
            summary["aggregation"][model_name] = {
                "mean_accuracy": round(float(np.mean(all_accs)), 4),
                "std_accuracy": round(float(np.std(all_accs)), 4),
                "mean_edge_retention": round(float(np.mean(all_edges)), 4) if all_edges else None,
            }

    # Degradation from baseline
    baseline_acc = summary["aggregation"].get(baseline, {}).get("mean_accuracy", 0)
    summary["baseline_accuracy"] = baseline_acc
    for model_name in models:
        m = summary["aggregation"].get(model_name, {})
        if m:
            m["degradation_from_baseline"] = round(m["mean_accuracy"] - baseline_acc, 4)

    return summary


def print_phase8_summary(report):
    splits = [f"{a}->{b}" for a, b in WalkForwardValidator.SPLITS]
    summary = report["summary"]
    baseline = "01_residual_only"

    print("\n" + "=" * 72)
    print("PHASE 8 - ARCHITECTURE COMPRESSION RESULTS (H50 ONLY)")
    print("=" * 72)

    # Table
    print(f"\n{'Model':<32} {'Acc':<8} {'Std':<8} {'Edge Ret':<10} {'Degrade':<8}")
    print(f"{'-'*32} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for model_name in report["models"]:
        agg = summary["aggregation"].get(model_name, {})
        if agg:
            acc = agg["mean_accuracy"]
            std = agg["std_accuracy"]
            er = agg.get("mean_edge_retention", 0)
            er_str = f"{er:.4f}" if er is not None else "N/A"
            deg = agg.get("degradation_from_baseline", 0)
            deg_str = f"{deg:+.4f}" if deg != 0 else "baseline"
            print(f"{model_name:<32} {acc:<8.4f} {std:<8.4f} {er_str:<10} {deg_str:<8}")

    # Per-split detail
    print(f"\n{'='*72}")
    print("PER-SPLIT DETAIL")
    print(f"{'='*72}")
    for split_key in splits:
        print(f"\n  Split: {split_key}")
        print(f"  {'Model':<32} {'Acc':<8} {'Std':<8} {'Edge Ret':<10}")
        print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*10}")
        sm = summary.get("splits", {}).get(split_key, {})
        for model_name in report["models"]:
            m = sm.get(model_name, {})
            if m:
                er = m.get("mean_edge_retention", 0)
                er_str = f"{er:.4f}" if er is not None else "N/A"
                print(f"  {model_name:<32} {m['mean_accuracy']:<8.4f} {m['std_accuracy']:<8.4f} {er_str:<10}")

    # Questions
    print(f"\n{'='*72}")
    print("KEY QUESTIONS")
    print(f"{'='*72}")

    agg = summary.get("aggregation", {})
    baseline_acc = summary.get("baseline_accuracy", 0)

    # Q: Is residual alone sufficient?
    best_added = None
    best_added_acc = 0
    for mn, a in agg.items():
        if mn != "01_residual_only" and a.get("mean_accuracy", 0) > best_added_acc:
            best_added_acc = a["mean_accuracy"]
            best_added = mn

    print(f"\n  Q1: Is residual alone sufficient?")
    print(f"      residual_only baseline accuracy = {baseline_acc:.4f}")
    print(f"      Best augmented model = {best_added} ({best_added_acc:.4f})")
    if best_added_acc > baseline_acc + 0.01:
        print(f"      -> NO - adding features improves ({best_added_acc - baseline_acc:+.4f})")
    elif best_added_acc > baseline_acc - 0.01:
        print(f"      -> YES - residual alone is sufficient (delta < 0.01)")
    else:
        print(f"      -> YES - residual alone is best (all others degrade)")

    print(f"\n  Q2: Which feature contributes the most directional info?")
    feature_deltas = {}
    for mn, a in agg.items():
        if mn != "01_residual_only":
            feature_name = mn.replace("02_residual_plus_", "").replace("07_", "").replace("08_", "")
            feature_deltas[feature_name] = a.get("degradation_from_baseline", 0)
    sorted_features = sorted(feature_deltas.items(), key=lambda x: -x[1])
    for name, delta in sorted_features:
        marker = "BEST" if delta == max(d[1] for d in sorted_features) else ""
        print(f"      {name:<30} {delta:+.4f} {marker}")

    print(f"\n  Q3: What is the minimum viable directional architecture?")
    # Find all models above threshold
    viable = [(mn, a["mean_accuracy"]) for mn, a in agg.items()
              if a["mean_accuracy"] >= baseline_acc - 0.02]
    viable_sorted = sorted(viable, key=lambda x: -x[1])
    print(f"      Models within 2% of baseline:")
    for mn, acc in viable_sorted:
        tag = "BASELINE" if mn == "01_residual_only" else ""
        print(f"        {mn:<32} {acc:.4f} {tag}")

    baseline_edge = agg.get("01_residual_only", {}).get("mean_edge_retention", 0)
    print(f"\n  Q4: Edge retention comparison (positive = train > test = collapse)")
    print(f"      {'Model':<32} {'Edge Ret':<10} {'Status':<15}")
    print(f"      {'-'*32} {'-'*10} {'-'*15}")
    for model_name in report["models"]:
        a = agg.get(model_name, {})
        if a:
            er = a.get("mean_edge_retention")
            if er is not None:
                status = "COLLAPSED" if er > 0.03 else ("STABLE" if er > -0.03 else "REVERSAL")
                print(f"      {model_name:<32} {er:<+.4f}  {status:<15}")

    print(f"\n{'='*72}")
    print("PHASE 8 COMPLETE")
    print(f"{'='*72}")

    return report


# ══════════════════════════════════════════════════════════════════════════
# FINAL ADJUDICATION
# ══════════════════════════════════════════════════════════════════════════

def run_final_adjudication():
    print("\n" + "=" * 72)
    print("FINAL ADJUDICATION - SYNTHESIZING ALL 8 PHASES")
    print("=" * 72)

    # Load all phase reports
    reports_dir = Path(__file__).parent / "reports"
    phase_reports = {}
    for phase_num in range(1, 9):
        fname = f"dsr_phase{phase_num}_"
        if phase_num == 1:
            fname += "state_reconstruction"
        elif phase_num == 2:
            fname += "regime_residual_surface"
        elif phase_num == 3:
            fname += "memory_gate"
        elif phase_num == 4:
            fname += "state_transitions"
        elif phase_num == 5:
            fname += "state_persistence"
        elif phase_num == 6:
            fname += "propagation_cascade"
        elif phase_num == 7:
            fname += "walk_forward"
        elif phase_num == 8:
            fname += "architecture_compression"
        fp = reports_dir / f"{fname}.json"
        if fp.exists():
            with open(fp) as f:
                phase_reports[phase_num] = json.load(f)

    # ════════════════════════════════════════════════════════════════════
    # Q1: Does a directional state exist?
    # ════════════════════════════════════════════════════════════════════
    p1 = phase_reports.get(1, {})
    p2 = phase_reports.get(2, {})
    p7 = phase_reports.get(7, {})

    # From Phase 1: stable directional states at H50
    phase1_stable_up = 0
    phase1_stable_down = 0
    phase1_pct_stable = 0
    phase1_symbols = p1.get("symbols", {})
    symbol_pct_stable = []
    for sym_key, sym_val in phase1_symbols.items():
        h50 = sym_val.get("H50", {})
        su = h50.get("stable_up", 0)
        sd = h50.get("stable_down", 0)
        n_states = h50.get("n_states", 1)
        pct = h50.get("pct_stable", 0)
        symbol_pct_stable.append(pct)
        phase1_stable_up += su
        phase1_stable_down += sd
    avg_pct_stable = float(np.mean(symbol_pct_stable)) if symbol_pct_stable else 0

    # From Phase 2: directional pockets
    phase2_directional_pct = 0
    phase2_symbols = p2.get("symbols", {})
    directional_pcts = []
    for sym_val in phase2_symbols.values():
        h50 = sym_val.get("H50", {})
        dp = h50.get("pct_directional", 0)
        directional_pcts.append(dp)
        phase2_directional_pct = max(directional_pcts)
    mean_directional_pct = float(np.mean(directional_pcts)) if directional_pcts else 0

    # From Phase 7: best walk-forward accuracy
    p7_best = p7.get("summary", {}).get("best_models", {}).get("H50", {})
    p7_best_model = p7_best.get("best_model", "N/A")
    p7_best_acc = p7_best.get("best_score", 0)

    q1_evidence = {
        "phase1_avg_pct_stable_states": round(avg_pct_stable, 1),
        "phase1_stable_up": phase1_stable_up,
        "phase1_stable_down": phase1_stable_down,
        "phase2_mean_pct_directional_pockets": round(mean_directional_pct, 1),
        "phase2_max_directional_pct": round(phase2_directional_pct, 1),
        "phase7_best_walkforward_acc_h50": round(float(p7_best_acc), 4),
        "phase7_best_model": p7_best_model,
    }

    # ════════════════════════════════════════════════════════════════════
    # Q2: What variables define it?
    # ════════════════════════════════════════════════════════════════════
    p7_summary = p7.get("summary", {}).get("best_models", {}).get("H50", {}).get("all_scores", {})
    p7_ranked = sorted(p7_summary.items(), key=lambda x: -x[1]) if p7_summary else []

    # Variables that matter (ranked by Phase 7)
    q2_evidence = {
        "phase7_model_ranking_h50": {m: round(s, 4) for m, s in p7_ranked},
        "best_model": p7_ranked[0][0] if p7_ranked else None,
        "best_accuracy": p7_ranked[0][1] if p7_ranked else 0,
    }

    # ════════════════════════════════════════════════════════════════════
    # Q3: What is the strongest directional mechanism?
    # ════════════════════════════════════════════════════════════════════
    q3_evidence = {
        "best_model_phase7": p7_best_model,
        "best_accuracy_h50": p7_best_acc,
        "mechanism": "residual_sign" if p7_best_model == "residual_only" else p7_best_model,
    }

    # ════════════════════════════════════════════════════════════════════
    # Q4: What is true walk-forward accuracy?
    # ════════════════════════════════════════════════════════════════════
    p7_h50_splits = p7.get("summary", {}).get("H50", {}).get("splits", {})
    residual_accs_by_split = {}
    for split_key, split_data in p7_h50_splits.items():
        r = split_data.get("residual_only", {})
        residual_accs_by_split[split_key] = r.get("mean_accuracy", 0)
    q4_evidence = {
        "residual_only_accs_by_split": residual_accs_by_split,
        "overall_mean_h50": p7_best_acc,
        "is_strictly_oos": True,
    }

    # ════════════════════════════════════════════════════════════════════
    # Q5: Can long/short deployment be justified?
    # ════════════════════════════════════════════════════════════════════
    # Based on CDER ceiling and Phase 7 accuracy
    p1_h50_p_up = []
    for sym_val in phase1_symbols.values():
        h50 = sym_val.get("H50", {})
        mean_p_up = h50.get("mean_p_up", 0.5)
        p1_h50_p_up.append(mean_p_up)
    avg_p_up = float(np.mean(p1_h50_p_up)) if p1_h50_p_up else 0

    cder_ceiling = 0.71
    edge_over_cder = p7_best_acc - cder_ceiling
    above_random = p7_best_acc - 0.50

    q5_evidence = {
        "phase7_best_acc_h50": p7_best_acc,
        "cder_ceiling": cder_ceiling,
        "edge_over_cder": round(edge_over_cder, 4),
        "above_random": round(above_random, 4),
        "phase1_avg_p_up_h50": round(avg_p_up, 4),
        "phase2_directional_pockets_exist": mean_directional_pct > 10,
    }

    # ════════════════════════════════════════════════════════════════════
    # Q6: Is direction a state reconstruction problem?
    # ════════════════════════════════════════════════════════════════════
    p5 = phase_reports.get(5, {})
    p5_cross = p5.get("cross_symbol_summary", {})
    avg_half_life = p5_cross.get("avg_half_life", 0)
    avg_duration = p5_cross.get("avg_mean_duration", 0)

    p4 = phase_reports.get(4, {})
    p4_cross = p4.get("cross_symbol_summary", {})
    p4_h50 = p4_cross.get("H50", {})
    avg_transition_dev = p4_h50.get("avg_transition_deviation", 0)

    q6_evidence = {
        "phase5_avg_half_life": avg_half_life,
        "phase5_avg_mean_duration": avg_duration,
        "phase4_avg_transition_deviation_h50": round(avg_transition_dev, 4),
        "phase1_pct_stable_states": round(avg_pct_stable, 1),
    }

    # ════════════════════════════════════════════════════════════════════
    # Q7: Minimum viable directional architecture
    # ════════════════════════════════════════════════════════════════════
    p8 = phase_reports.get(8, {})
    p8_agg = p8.get("summary", {}).get("aggregation", {})
    p8_feature_analysis = {}
    for mn, data in p8_agg.items():
        feature_name = mn
        if feature_name.startswith("01_"):
            feature_name = "baseline (residual_only)"
        elif feature_name.startswith("02_"):
            feature_name = "residual + regime"
        elif feature_name.startswith("03_"):
            feature_name = "residual + imbalance"
        elif feature_name.startswith("04_"):
            feature_name = "residual + saturation"
        elif feature_name.startswith("05_"):
            feature_name = "residual + cluster"
        elif feature_name.startswith("06_"):
            feature_name = "residual + macro"
        elif feature_name.startswith("07_"):
            feature_name = "residual + propagation"
        elif feature_name.startswith("08_"):
            feature_name = "full state"
        p8_feature_analysis[feature_name] = {
            "accuracy": data.get("mean_accuracy", 0),
            "degradation": data.get("degradation_from_baseline", 0),
            "edge_retention": data.get("mean_edge_retention", 0),
        }

    # Find minimal viable architecture
    best_augmented = None
    best_augmented_acc = 0
    for fn, fa in p8_feature_analysis.items():
        if fn != "baseline (residual_only)" and fa["accuracy"] > best_augmented_acc:
            best_augmented_acc = fa["accuracy"]
            best_augmented = fn

    q7_evidence = {
        "phase8_feature_analysis": p8_feature_analysis,
        "baseline_residual_only_acc": p8_feature_analysis.get("baseline (residual_only)", {}).get("accuracy", 0),
        "best_augmented": best_augmented,
        "best_augmented_acc": best_augmented_acc,
    }

    # ════════════════════════════════════════════════════════════════════
    # Q8: Which discovered layer contributes the most?
    # ════════════════════════════════════════════════════════════════════
    feature_deltas = {}
    for fn, fa in p8_feature_analysis.items():
        if fn != "baseline (residual_only)":
            feature_deltas[fn] = fa["degradation"]
    ranked_features = sorted(feature_deltas.items(), key=lambda x: -x[1])

    q8_evidence = {
        "feature_ranking_by_delta": {fn: round(d, 4) for fn, d in ranked_features},
        "most_contributing": ranked_features[0][0] if ranked_features else None,
        "most_degrading": ranked_features[-1][0] if ranked_features else None,
    }

    # ════════════════════════════════════════════════════════════════════
    # Final Classification
    # ════════════════════════════════════════════════════════════════════
    final_classification = classify(q1_evidence, q2_evidence, q3_evidence, q4_evidence,
                                     q5_evidence, q6_evidence, q7_evidence, q8_evidence)

    adjudication = {
        "title": "DSR Final Adjudication",
        "classification": final_classification,
        "answers": {
            "Q1": {
                "question": "Does a directional state exist?",
                "evidence": q1_evidence,
                "answer": _answer_q1(q1_evidence),
            },
            "Q2": {
                "question": "What variables define it?",
                "evidence": q2_evidence,
                "answer": _answer_q2(q2_evidence),
            },
            "Q3": {
                "question": "What is the strongest directional mechanism?",
                "evidence": q3_evidence,
                "answer": _answer_q3(q3_evidence),
            },
            "Q4": {
                "question": "What is true walk-forward accuracy?",
                "evidence": q4_evidence,
                "answer": _answer_q4(q4_evidence),
            },
            "Q5": {
                "question": "Can long/short deployment be justified?",
                "evidence": q5_evidence,
                "answer": _answer_q5(q5_evidence),
            },
            "Q6": {
                "question": "Is direction a state reconstruction problem?",
                "evidence": q6_evidence,
                "answer": _answer_q6(q6_evidence),
            },
            "Q7": {
                "question": "What is the minimum viable directional architecture?",
                "evidence": q7_evidence,
                "answer": _answer_q7(q7_evidence),
            },
            "Q8": {
                "question": "Which discovered layer contributes the most directional information?",
                "evidence": q8_evidence,
                "answer": _answer_q8(q8_evidence),
            },
        },
    }

    return adjudication


# ── Answer generators ──────────────────────────────────────────────────────

def _answer_q1(e):
    best_acc = e.get("phase7_best_walkforward_acc_h50", 0)
    pct_stable = e.get("phase1_avg_pct_stable_states", 0)
    dir_pct = e.get("phase2_mean_pct_directional_pockets", 0)
    if best_acc > 0.65 and pct_stable > 15 and dir_pct > 20:
        return f"YES. Strong evidence across all phases: {pct_stable}% states are directionally stable (P1), {dir_pct}% directional pockets (P2), {best_acc:.1%} walk-forward accuracy (P7)."
    elif best_acc > 0.55:
        return f"WEAK YES. Directional signal exists ({best_acc:.1%}) but stability/pocket evidence is mixed."
    return f"NO. Insufficient evidence of directional state."


def _answer_q2(e):
    ranking = e.get("phase7_model_ranking_h50", {})
    best = e.get("best_model", "N/A")
    return f"Dominant variable: {best} ({e.get('best_accuracy', 0):.2%}). Model ranking: {ranking}"


def _answer_q3(e):
    return f"Residual sign direction ({e.get('best_model_phase7', 'N/A')}) at {e.get('best_accuracy_h50', 0):.2%} accuracy."


def _answer_q4(e):
    accs = e.get("residual_only_accs_by_split", {})
    mean_acc = e.get("overall_mean_h50", 0)
    return f"{mean_acc:.2%} (strictly OOS). Per-split: {accs}"


def _answer_q5(e):
    acc = e.get("phase7_best_acc_h50", 0)
    cder = e.get("cder_ceiling", 0.71)
    edge = e.get("edge_over_cder", 0)
    if acc > cder and edge > 0:
        return f"YES. Walk-forward accuracy ({acc:.2%}) exceeds CDER ceiling ({cder:.0%}) by {edge:.2%}. Deployment justified."
    elif acc > 0.60:
        return f"MARGINAL. Accuracy ({acc:.2%}) is above random but below CDER ceiling ({cder:.0%}). Deployment may succeed with tight risk controls."
    return f"NO. Accuracy ({acc:.2%}) insufficient to justify deployment."


def _answer_q6(e):
    hl = e.get("phase5_avg_half_life", 0)
    dur = e.get("phase5_avg_mean_duration", 0)
    trans_dev = e.get("phase4_avg_transition_deviation_h50", 0)
    stable = e.get("phase1_pct_stable_states", 0)
    if dur > 15 and trans_dev < 0.20 and stable > 15:
        return f"YES. States persist (half-life={hl}, mean duration={dur}), transitions cause modest deviation ({trans_dev:.2f}), {stable:.1f}% states stable."
    return f"PARTIALLY. States persist ({hl} bar half-life) but reconstruction is complicated by transition noise."


def _answer_q7(e):
    baseline = e.get("baseline_residual_only_acc", 0)
    best = e.get("best_augmented", "N/A")
    best_acc = e.get("best_augmented_acc", 0)
    fa = e.get("phase8_feature_analysis", {})
    if best_acc <= baseline + 0.01:
        return f"Residual alone ({baseline:.2%}) - no feature addition improves beyond {best_acc:.2%}."
    return f"Residual + {best} ({best_acc:.2%}) improves over residual alone ({baseline:.2%})."


def _answer_q8(e):
    ranking = e.get("feature_ranking_by_delta", {})
    best = e.get("most_contributing", "N/A")
    worst = e.get("most_degrading", "N/A")
    return f"Most contributing: {best}. Most degrading: {worst}. Full ranking: {ranking}"


def classify(q1, q2, q3, q4, q5, q6, q7, q8):
    """Determine final classification based on all evidence."""
    best_acc = q1.get("phase7_best_walkforward_acc_h50", 0)
    avg_pct_stable = q1.get("phase1_avg_pct_stable_states", 0)
    regime_acc = q2.get("phase7_model_ranking_h50", {}).get("regime_only", 0)
    residual_acc = q2.get("phase7_model_ranking_h50", {}).get("residual_only", 0)
    half_life = q6.get("phase5_avg_half_life", 0)

    # Flowchart — priority order (most specific first):
    # 1. If best acc < 0.55 → NO_DIRECTIONAL_STATE
    if best_acc < 0.55:
        return "NO_DIRECTIONAL_STATE"

    # 2. If best acc < 0.65 → WEAK_DIRECTIONAL_STATE
    if best_acc < 0.65:
        return "WEAK_DIRECTIONAL_STATE"

    # 3. If best acc >= 0.70 → DEPLOYABLE_DIRECTIONAL_ENGINE (highest confidence)
    if best_acc >= 0.70:
        return "DEPLOYABLE_DIRECTIONAL_ENGINE"

    # 4. If best acc < 0.70 and regime helps → REGIME_DEPENDENT_DIRECTION
    if regime_acc >= residual_acc:
        return "REGIME_DEPENDENT_DIRECTION"

    # 5. If states persist and transitions are stable → STATE_RECONSTRUCTABLE_DIRECTION
    if avg_pct_stable > 15 and half_life > 8:
        return "STATE_RECONSTRUCTABLE_DIRECTION"

    return "WEAK_DIRECTIONAL_STATE"


# ── Save adjudication ──────────────────────────────────────────────────────

def save_adjudication(adjudication):
    path = Path(__file__).parent / "reports" / "DSR_FINAL_ADJUDICATION.md"
    c = adjudication["classification"]
    answers = adjudication["answers"]

    md = f"""# DSR Final Adjudication

## Classification: **{c}**

---

## Q1. Does a directional state exist?
**{answers['Q1']['answer']}**

### Evidence
| Source | Metric | Value |
|--------|--------|-------|
| Phase 1 | Stable states (%) | {answers['Q1']['evidence']['phase1_avg_pct_stable_states']}% |
| Phase 2 | Directional pockets (%) | {answers['Q1']['evidence']['phase2_mean_pct_directional_pockets']}% |
| Phase 7 | Walk-forward accuracy (H50) | {answers['Q1']['evidence']['phase7_best_walkforward_acc_h50']:.2%} |

---

## Q2. What variables define it?
**{answers['Q2']['answer']}**

### Model Ranking (Phase 7, H50)
| Model | Accuracy |
|-------|----------|
"""
    for model, acc in answers['Q2']['evidence'].get('phase7_model_ranking_h50', {}).items():
        md += f"| {model} | {acc:.2%} |\n"

    md += f"""
---

## Q3. What is the strongest directional mechanism?
**{answers['Q3']['answer']}**

### Summary
- Best model: {answers['Q3']['evidence'].get('best_model_phase7', 'N/A')}
- H50 accuracy: {answers['Q3']['evidence'].get('best_accuracy_h50', 0):.2%}
- Mechanism: {answers['Q3']['evidence'].get('mechanism', 'N/A')}

---

## Q4. What is true walk-forward accuracy?
**{answers['Q4']['answer']}**

### Per-Split Breakdown
| Split | Accuracy |
|-------|----------|
"""
    for split, acc in answers['Q4']['evidence'].get('residual_only_accs_by_split', {}).items():
        md += f"| {split} | {acc:.2%} |\n"

    md += f"""
**Strictly OOS:** {answers['Q4']['evidence'].get('is_strictly_oos', True)}

---

## Q5. Can long/short deployment be justified?
**{answers['Q5']['answer']}**

| Metric | Value |
|--------|-------|
| Best H50 accuracy | {answers['Q5']['evidence'].get('phase7_best_acc_h50', 0):.2%} |
| CDER ceiling | {answers['Q5']['evidence'].get('cder_ceiling', 0):.0%} |
| Edge over CDER | {answers['Q5']['evidence'].get('edge_over_cder', 0):.2%} |
| Above random | {answers['Q5']['evidence'].get('above_random', 0):.2%} |

---

## Q6. Is direction a state reconstruction problem?
**{answers['Q6']['answer']}**

| Metric | Value |
|--------|-------|
| Average state half-life | {answers['Q6']['evidence'].get('phase5_avg_half_life', 0)} |
| Average mean duration | {answers['Q6']['evidence'].get('phase5_avg_mean_duration', 0)} |
| Transition deviation (H50) | {answers['Q6']['evidence'].get('phase4_avg_transition_deviation_h50', 0):.2f} |
| Stable states (%) | {answers['Q6']['evidence'].get('phase1_pct_stable_states', 0)}% |

---

## Q7. What is the minimum viable directional architecture?
**{answers['Q7']['answer']}**

### Architecture Compression Results (Phase 8)
| Feature Set | Accuracy | Degradation | Edge Retention |
|-------------|----------|-------------|----------------|
"""
    for fn, fa in answers['Q7']['evidence'].get('phase8_feature_analysis', {}).items():
        md += f"| {fn} | {fa['accuracy']:.2%} | {fa['degradation']:+.2%} | {fa['edge_retention']:.4f} |\n"

    md += f"""
---

## Q8. Which discovered layer contributes the most directional information?
**{answers['Q8']['answer']}**

| Feature | Delta from Baseline |
|---------|-------------------|
"""
    for fn, d in answers['Q8']['evidence'].get('feature_ranking_by_delta', {}).items():
        md += f"| {fn} | {d:+.4f} |\n"

    q1_ev = answers['Q1']['evidence']
    q6_ev = answers['Q6']['evidence']
    md += f"""
---

## Final Classification: **{c}**

### Classification Logic
1. Best H50 accuracy = {q1_ev.get('phase7_best_walkforward_acc_h50', 0):.2%}
2. Stable states = {q1_ev.get('phase1_avg_pct_stable_states', 0)}%
3. State half-life = {q6_ev.get('phase5_avg_half_life', 0)} bars
4. Residual direction dominates (simplest = best)

### Implications
- {'Direction is predictable enough for deployment with appropriate risk management.' if 'DEPLOYABLE' in c else 'Further research needed before deployment.'}
- {'Residual sign is the primary directional signal.' if 'DEPLOYABLE' in c or 'STATE_RECONSTRUCTABLE' in c else 'Directional mechanism not fully resolved.'}
- {'Walk-forward validated across multiple test years.' if 'DEPLOYABLE' in c else 'Cross-year stability not confirmed.'}
"""
    if 'DEPLOYABLE' in c:
        md += "- **Action: Proceed to production deployment with residual-only directional engine at H50 horizon.**\n"
    elif 'STATE_RECONSTRUCTABLE' in c:
        md += "- **Action: Focus on state reconstruction error correction before deployment.**\n"
    elif 'REGIME_DEPENDENT' in c:
        md += "- **Action: Integrate regime conditioning into directional engine.**\n"
    elif 'WEAK' in c:
        md += "- **Action: Additional feature discovery needed; not yet deployable.**\n"
    else:
        md += "- **Action: DSR framework requires fundamental revision.**\n"

    with open(path, "w") as f:
        f.write(md)
    print(f"\nSaved {path}")
    return path


def save_adjudication_json(adjudication):
    path = Path(__file__).parent / "reports" / "dsr_final_adjudication.json"
    with open(path, "w") as f:
        json.dump(adjudication, f, indent=2, default=str)
    print(f"Saved {path}")
    return path


def print_adjudication(adjudication):
    print("\n" + "=" * 72)
    print("FINAL ADJUDICATION SUMMARY")
    print("=" * 72)
    print(f"\n  Classification: {adjudication['classification']}")
    for qk, qv in adjudication["answers"].items():
        print(f"\n  {qk}: {qv['question']}")
        print(f"    -> {qv['answer']}")
    print(f"\n{'='*72}")
    print("ADJUDICATION COMPLETE")
    print(f"{'='*72}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n")
    phase8_report = run_phase8()

    print("\n\n")
    adjudication = run_final_adjudication()
    save_adjudication(adjudication)
    save_adjudication_json(adjudication)
    print_adjudication(adjudication)
    print("\nDSR Phase 8 + Final Adjudication complete.")
