"""DSR Phase 7 — Walk-Forward Directional Validation.

Strict out-of-sample validation across 3 train/test splits.
Tests 6 models: Regime Only, Residual Only, Regime+Residual,
Regime×Residual, Full State, Transition Model.

Key questions:
- Does directional state survive out-of-sample?
- Does accuracy remain stable across test years?
- Does edge collapse? (train accuracy - test accuracy)
- Which model is most robust?
"""

import sys, json
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.directional_state.dsr_core import DSRCore, SYMBOLS, WalkForwardValidator, STATE_HORIZON_KEYS, save_report

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


def _quintile_quantize(x):
    labels = np.full(len(x), -1, dtype=np.int64)
    valid = ~np.isnan(x)
    if np.sum(valid) < 10:
        return labels
    bins = np.nanpercentile(x[valid], [20, 40, 60, 80])
    labels[valid & (x <= bins[0])] = 0
    labels[valid & (x > bins[0]) & (x <= bins[1])] = 1
    labels[valid & (x > bins[1]) & (x <= bins[2])] = 2
    labels[valid & (x > bins[2]) & (x <= bins[3])] = 3
    labels[valid & (x > bins[3])] = 4
    return labels


# ── Model builders ──────────────────────────────────────────────────────────

def build_regime_only(d):
    return d["regime"].copy()


def build_residual_only(d):
    rs = d["residual_sign"]
    result = np.full(len(rs), -1, dtype=np.int64)
    valid = ~np.isnan(d["residual"])
    result[valid] = rs[valid] + 1
    return result


def build_regime_plus_residual(d):
    reg = d["regime"]
    rs = d["residual_sign"]
    n = len(reg)
    result = np.full(n, -1, dtype=np.int64)
    valid = (reg >= 0) & ~np.isnan(d["residual"])
    result[valid] = reg[valid] * 10 + (rs[valid] + 1)
    return result


def build_regime_x_residual(d):
    reg = d["regime"]
    q = _quintile_quantize(d["residual"])
    n = len(reg)
    result = np.full(n, -1, dtype=np.int64)
    valid = (reg >= 0) & (q >= 0)
    result[valid] = reg[valid] * 10 + q[valid]
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


def build_transition_model(d):
    tf = d["reg_transition_from"]
    tt = d["reg_transition_to"]
    n = len(tf)
    result = np.full(n, -1, dtype=np.int64)
    valid = (tf >= 0) & (tt >= 0)
    result[valid] = tf[valid] * 10 + tt[valid]
    return result


# ── Core walk-forward logic ─────────────────────────────────────────────────

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
    probs = np.full(n, np.nan)
    for i in range(n):
        sid = int(state_ids[i])
        if sid in stats:
            probs[i] = stats[sid]["p_up"]
            preds[i] = 1.0 if stats[sid]["p_up"] > 0.5 else 0.0
    valid = ~np.isnan(preds) & ~np.isnan(actual_up)
    if np.sum(valid) < 5:
        return None

    # All operations on valid-only arrays
    sid_valid = state_ids[valid].astype(int)
    p = preds[valid].astype(float)
    a = actual_up[valid].astype(float)
    pr = probs[valid]
    n_valid = int(np.sum(valid))

    correct = float(np.sum(p == a))
    accuracy = correct / n_valid

    tp = float(np.sum((p == 1.0) & (a == 1.0)))
    fp = float(np.sum((p == 1.0) & (a == 0.0)))
    fn = float(np.sum((p == 0.0) & (a == 1.0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    brier = float(np.mean((pr - a) ** 2))

    base = float(np.mean(a))
    ig = 0.0
    eps = 1e-12
    unique_sid = np.unique(sid_valid)
    for sid in unique_sid:
        if sid not in stats:
            continue
        m = sid_valid == sid
        if np.sum(m) < 2:
            continue
        p_pred = stats[sid]["p_up"]
        p_act = float(np.mean(a[m]))
        p_dn_act = 1.0 - p_act
        if p_pred > 0 and p_act > 0:
            ig += p_act * np.log2(p_pred / max(base, eps))
        if (1 - p_pred) > 0 and p_dn_act > 0:
            ig += p_dn_act * np.log2((1 - p_pred) / max(1 - base, eps))

    return {
        "n_valid": n_valid,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "information_gain": round(ig, 4),
        "base_p_up_actual": round(base, 4),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def run_phase7():
    print("=" * 72)
    print("DSR PHASE 7 - WALK-FORWARD DIRECTIONAL VALIDATION")
    print("=" * 72)

    print("\nLoading DSR core (cached)...")
    dsr = DSRCore()
    for sym in SYMBOLS:
        print(f"  {sym} ... ", end="")
        dsr.load_symbol(sym)
        print("OK")

    MODELS = {
        "regime_only": build_regime_only,
        "residual_only": build_residual_only,
        "regime_plus_residual": build_regime_plus_residual,
        "regime_x_residual": build_regime_x_residual,
        "full_state": build_full_state,
        "transition_model": build_transition_model,
    }

    report = {
        "phase": 7,
        "title": "Walk-Forward Directional Validation",
        "splits": [f"{a}->{b}" for a, b in WalkForwardValidator.SPLITS],
        "models": list(MODELS.keys()),
        "results": {},
        "summary": {},
    }

    wfv = WalkForwardValidator(dsr)

    for sym in SYMBOLS:
        print(f"\n--- {sym} ---")
        wfv.prepare(sym)
        d = dsr._data[sym]
        n_total = len(d["es"])
        report["results"][sym] = {}

        for horizon_key in STATE_HORIZON_KEYS:
            hi = HORIZON_MAP[horizon_key]
            fut_ret_col = d["fut_ret"][:, hi]
            up = (fut_ret_col > 0).astype(float)
            not_nan = ~np.isnan(fut_ret_col)
            report["results"][sym][horizon_key] = {}

            for model_name, model_fn in MODELS.items():
                print(f"  {horizon_key} / {model_name}")

                state_ids = model_fn(d)

                for train_name, test_name in WalkForwardValidator.SPLITS:
                    split_key = f"{train_name}->{test_name}"
                    train_mask, test_mask = wfv.split(sym, train_name, test_name)

                    train_ok = train_mask & (state_ids >= 0) & not_nan
                    test_ok = test_mask & (state_ids >= 0) & not_nan

                    # Train: compute P(up) per state
                    train_stats = compute_train_stats(
                        state_ids[train_ok], up[train_ok]
                    )

                    # Train: evaluate in-sample
                    train_eval = evaluate(
                        state_ids[train_ok], up[train_ok], train_stats
                    )

                    # Test: evaluate out-of-sample
                    test_eval = evaluate(
                        state_ids[test_ok], up[test_ok], train_stats
                    )

                    entry = {}
                    if train_eval:
                        entry["train"] = train_eval
                    if test_eval:
                        entry["test"] = test_eval
                        n_train_states = len(train_stats)
                        n_test_states = len(np.unique(state_ids[test_ok].astype(int)))
                        entry["n_train_states"] = n_train_states
                        entry["n_test_states"] = n_test_states
                        if train_eval:
                            entry["edge_retention"] = round(
                                train_eval["accuracy"] - test_eval["accuracy"], 4
                            )
                    report["results"][sym][horizon_key][split_key] = report["results"][sym][horizon_key].get(
                        split_key, {}
                    )
                    report["results"][sym][horizon_key][split_key][model_name] = entry

    # ── Compute summaries ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("COMPUTING CROSS-SYMBOL SUMMARIES")
    print("=" * 72)
    summary = _compute_summary(report)
    report["summary"] = summary

    save_report(report, "dsr_phase7_walk_forward")
    _print_summary(report)

    return report


def _compute_summary(report):
    splits = [f"{a}->{b}" for a, b in WalkForwardValidator.SPLITS]
    models = report["models"]
    summary = {}

    for horizon_key in STATE_HORIZON_KEYS:
        h = {"splits": {}, "stability": {}}

        for split_key in splits:
            sm = {}
            for model_name in models:
                accs, precs, recalls, briers, igs = [], [], [], [], []
                total_n = 0
                for sym in SYMBOLS:
                    e = report["results"].get(sym, {}).get(horizon_key, {}).get(
                        split_key, {}).get(model_name, {}).get("test", {})
                    if e.get("n_valid", 0) > 0:
                        accs.append(e["accuracy"])
                        precs.append(e.get("precision", 0))
                        recalls.append(e.get("recall", 0))
                        briers.append(e.get("brier_score", 0))
                        igs.append(e.get("information_gain", 0))
                        total_n += e["n_valid"]
                if accs:
                    sm[model_name] = {
                        "mean_accuracy": round(float(np.mean(accs)), 4),
                        "std_accuracy": round(float(np.std(accs)), 4),
                        "mean_precision": round(float(np.mean(precs)), 4),
                        "mean_recall": round(float(np.mean(recalls)), 4),
                        "mean_brier": round(float(np.mean(briers)), 4),
                        "mean_information_gain": round(float(np.mean(igs)), 4),
                        "total_samples": total_n,
                    }
            h["splits"][split_key] = sm

        # Stability across test years
        for model_name in models:
            accs_by_split = []
            for split_key in splits:
                xs = []
                for sym in SYMBOLS:
                    e = report["results"].get(sym, {}).get(horizon_key, {}).get(
                        split_key, {}).get(model_name, {}).get("test", {})
                    if e.get("n_valid", 0) > 0:
                        xs.append(e["accuracy"])
                if xs:
                    accs_by_split.append(float(np.mean(xs)))
            if len(accs_by_split) > 1:
                h["stability"][model_name] = {
                    "accuracy_variance": round(float(np.var(accs_by_split)), 6),
                    "accuracy_std": round(float(np.std(accs_by_split)), 4),
                    "accuracies": [round(a, 4) for a in accs_by_split],
                }

        summary[horizon_key] = h

    # Best model per horizon
    best_models = {}
    for horizon_key in STATE_HORIZON_KEYS:
        scores = {}
        for model_name in models:
            accs = []
            for split_key in splits:
                for sym in SYMBOLS:
                    e = report["results"].get(sym, {}).get(horizon_key, {}).get(
                        split_key, {}).get(model_name, {}).get("test", {})
                    if e.get("n_valid", 0) > 0:
                        accs.append(e["accuracy"])
            if accs:
                scores[model_name] = round(float(np.mean(accs)), 4)
        best_m = max(scores, key=scores.get) if scores else None
        best_models[horizon_key] = {
            "best_model": best_m,
            "best_score": scores.get(best_m, 0) if best_m else 0,
            "all_scores": scores,
        }
    summary["best_models"] = best_models

    # Edge retention summary
    edges = {}
    for horizon_key in STATE_HORIZON_KEYS:
        for split_key in splits:
            for model_name in models:
                vals = []
                for sym in SYMBOLS:
                    e = report["results"].get(sym, {}).get(horizon_key, {}).get(
                        split_key, {}).get(model_name, {})
                    er = e.get("edge_retention")
                    if er is not None:
                        vals.append(er)
                if vals:
                    key = f"{horizon_key}/{split_key}/{model_name}"
                    edges[key] = {
                        "mean_edge_retention": round(float(np.mean(vals)), 4),
                        "std_edge_retention": round(float(np.std(vals)), 4),
                        "per_symbol": {sym: report["results"].get(sym, {}).get(
                            horizon_key, {}).get(split_key, {}).get(model_name, {}).get(
                            "edge_retention") for sym in SYMBOLS},
                    }
    summary["edge_retention"] = edges

    return summary


def _print_summary(report):
    splits = [f"{a}->{b}" for a, b in WalkForwardValidator.SPLITS]
    models = report["models"]
    summary = report["summary"]

    print("\n" + "=" * 72)
    print("WALK-FORWARD VALIDATION RESULTS - DETAILED")
    print("=" * 72)

    for horizon_key in STATE_HORIZON_KEYS:
        print(f"\n{'-'*72}")
        print(f"  HORIZON: {horizon_key}")
        print(f"{'-'*72}")

        for split_key in splits:
            print(f"\n  Split: {split_key}")
            print(f"  {'Model':<24} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'Brier':<8} {'IG':<8} {'N':<8}")
            print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            sm = summary.get(horizon_key, {}).get("splits", {}).get(split_key, {})
            for model_name in models:
                m = sm.get(model_name, {})
                if m:
                    print(f"  {model_name:<24} {m['mean_accuracy']:<8.4f} {m['mean_precision']:<8.4f} "
                          f"{m['mean_recall']:<8.4f} {m['mean_brier']:<8.4f} {m['mean_information_gain']:<8.4f} "
                          f"{m['total_samples']:<8}")

        # Stability
        print(f"\n  Stability Across Test Years:")
        stab = summary.get(horizon_key, {}).get("stability", {})
        print(f"  {'Model':<24} {'2023':<8} {'2024':<8} {'2025':<8} {'Std':<8}")
        print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for model_name in models:
            s = stab.get(model_name, {})
            accs = s.get("accuracies", [])
            std = s.get("accuracy_std", 0)
            a_strs = [f"{a:<8.4f}" for a in accs]
            while len(a_strs) < 3:
                a_strs.append(f"{'N/A':<8}")
            print(f"  {model_name:<24} {''.join(a_strs)} {std:<8.4f}")

        # Best
        bm = summary.get("best_models", {}).get(horizon_key, {})
        print(f"\n  Best Model: {bm.get('best_model', 'N/A')} ({bm.get('best_score', 0):.4f})")
        for m, s in sorted(bm.get("all_scores", {}).items(), key=lambda x: -x[1]):
            print(f"    {m}: {s:.4f}")

    # Edge retention
    print(f"\n{'-'*72}")
    print("  EDGE RETENTION (Train Acc - Test Acc)")
    print(f"{'-'*72}")
    edges = summary.get("edge_retention", {})
    print(f"  {'Horizon/Split/Model':<42} {'Mean Ret':<10} {'Std':<8}")
    print(f"  {'-'*42} {'-'*10} {'-'*8}")
    sorted_e = sorted(edges.items(), key=lambda x: x[1].get("mean_edge_retention", 0))
    for key, val in sorted_e:
        mr = val.get("mean_edge_retention", 0)
        sd = val.get("std_edge_retention", 0)
        print(f"  {key:<42} {mr:<+10.4f} {sd:<8.4f}")

    # Per-symbol detail
    print(f"\n{'-'*72}")
    print("  PER-SYMBOL BREAKDOWN")
    print(f"{'-'*72}")
    for sym in SYMBOLS:
        print(f"\n  {sym}:")
        for horizon_key in STATE_HORIZON_KEYS:
            print(f"    {horizon_key}:")
            for split_key in splits:
                parts = []
                for model_name in models:
                    e = report["results"].get(sym, {}).get(horizon_key, {}).get(
                        split_key, {}).get(model_name, {}).get("test", {})
                    if e.get("n_valid", 0) > 0:
                        parts.append(f"{model_name[:12]}={e['accuracy']:.3f}")
                if parts:
                    print(f"      {split_key}: {', '.join(parts)}")

    # Key Questions
    print(f"\n{'='*72}")
    print("  KEY QUESTIONS")
    print(f"{'='*72}")

    bm_h50 = summary.get("best_models", {}).get("H50", {})
    best_h50_score = bm_h50.get("best_score", 0)
    print(f"\n  Q1: Does directional state survive out-of-sample?")
    print(f"      Best H50 test accuracy = {best_h50_score:.4f} "
          f"(random baseline = 0.5000)")
    print(f"      -> {'YES - directional signal survives OOS' if best_h50_score > 0.50 else 'NO - edge does not survive'}")

    stab_h50 = summary.get("H50", {}).get("stability", {})
    stds = [s.get("accuracy_std", 0) for s in stab_h50.values() if s]
    mean_std = float(np.mean(stds)) if stds else 0
    print(f"\n  Q2: Does accuracy remain stable across test years?")
    print(f"      Mean accuracy std across models = {mean_std:.4f}")
    if mean_std < 0.03:
        print(f"      -> STABLE (std < 0.03)")
    elif mean_std < 0.05:
        print(f"      -> MODERATELY VARIABLE (std 0.03-0.05)")
    else:
        print(f"      -> UNSTABLE (std > 0.05)")

    all_edges = [v.get("mean_edge_retention", 0) for v in edges.values()]
    avg_edge = float(np.mean(all_edges)) if all_edges else 0
    print(f"\n  Q3: Does edge collapse?")
    print(f"      Average edge retention = {avg_edge:.4f} "
          f"(positive = train > test = collapse)")
    if avg_edge > 0.03:
        print(f"      -> EDGE COLLAPSED (avg retention > 0.03)")
    elif avg_edge > 0:
        print(f"      -> EDGE STABLE")
    else:
        print(f"      -> EDGE REVERSAL (test > train)")

    print(f"\n  Q4: Which model is most robust?")
    for hk in STATE_HORIZON_KEYS:
        bm = summary.get("best_models", {}).get(hk, {})
        print(f"      {hk}: {bm.get('best_model', 'N/A')} = {bm.get('best_score', 0):.4f}")
        for m, s in sorted(bm.get("all_scores", {}).items(), key=lambda x: -x[1]):
            print(f"        {m}: {s:.4f}")

    # Additional: which model has least edge collapse?
    if all_edges:
        print(f"\n  Q5: Which model has best edge retention?")
        model_edge_map = {}
        for key, val in edges.items():
            _, _, model_name = key.split("/", 2)
            if model_name not in model_edge_map:
                model_edge_map[model_name] = []
            model_edge_map[model_name].append(val.get("mean_edge_retention", 0))
        for m, vals in sorted(model_edge_map.items(), key=lambda x: abs(float(np.mean(x[1])))):
            print(f"      {m}: avg retention = {float(np.mean(vals)):.4f}")

    print(f"\n{'='*72}")
    print("  PHASE 7 COMPLETE")
    print(f"{'='*72}")


if __name__ == "__main__":
    run_phase7()
