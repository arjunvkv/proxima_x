"""DSR Phase 6 — Cross-Asset Directional Cascade.

Leader/follower effects: for each pair (A,B), measure transfer entropy
and directional accuracy across lags to identify cascade structure.
"""
import sys, json
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.directional_state.dsr_core import DSRCore, SYMBOLS, save_report
from sklearn.metrics import mutual_info_score


def directional_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of times predicted matches actual (both -1 or 1)."""
    valid = ~np.isnan(predicted) & ~np.isnan(actual)
    if valid.sum() < 10:
        return 0.0
    return float(np.mean(predicted[valid] == actual[valid]))


def transfer_entropy(source: np.ndarray, target: np.ndarray, lag: int) -> float:
    """TE = MI(target[t], source[t-lag]) - MI(target[t], source[t-lag-1]).

    Measures how much information source at t-lag adds about target at t,
    beyond what source at t-lag-1 already provides.
    """
    n = min(len(source), len(target))
    src, tgt = source[:n], target[:n]
    if lag + 1 >= n:
        return 0.0
    # target[t] vs source[t-lag]
    t_lag = tgt[lag:]
    s_lag = src[:n - lag]
    # target[t] vs source[t-lag-1]
    s_lag2 = src[:n - lag - 1]
    t_lag2 = tgt[lag + 1:]
    n_eff = min(len(t_lag), len(s_lag), len(t_lag2), len(s_lag2))
    if n_eff < 20:
        return 0.0
    valid = (~np.isnan(t_lag[:n_eff])) & (~np.isnan(s_lag[:n_eff]))
    if valid.sum() < 20:
        return 0.0
    mi_lag1 = mutual_info_score(t_lag[:n_eff][valid], s_lag[:n_eff][valid])
    valid2 = (~np.isnan(t_lag2[:n_eff])) & (~np.isnan(s_lag2[:n_eff]))
    if valid2.sum() < 20:
        return 0.0
    mi_lag2 = mutual_info_score(t_lag2[:n_eff][valid2], s_lag2[:n_eff][valid2])
    return float(max(0, mi_lag1 - mi_lag2))


def regime_alignment_accuracy(
    regime_a: np.ndarray, regime_b: np.ndarray,
    direction_b: np.ndarray, lag: int
) -> dict:
    """Check if regime alignment (same regime in A and B) improves prediction."""
    n = min(len(regime_a), len(regime_b), len(direction_b))
    ra = regime_a[:n]
    rb = regime_b[:n]
    db = direction_b[:n]
    aligned = (ra[lag:] == rb[:-lag]) & (rb[:-lag] >= 0)
    misaligned = (ra[lag:] != rb[:-lag]) & (rb[:-lag] >= 0) & (ra[lag:] >= 0)

    dir_b_fwd = db[lag:]
    results = {}

    if aligned.sum() >= 10:
        results["aligned_accuracy"] = round(float(np.mean(dir_b_fwd[aligned] == 1)), 4)
        results["aligned_count"] = int(aligned.sum())
    else:
        results["aligned_accuracy"] = None
        results["aligned_count"] = 0

    if misaligned.sum() >= 10:
        results["misaligned_accuracy"] = round(float(np.mean(dir_b_fwd[misaligned] == 1)), 4)
        results["misaligned_count"] = int(misaligned.sum())
    else:
        results["misaligned_accuracy"] = None
        results["misaligned_count"] = 0

    return results


def run_phase6():
    dsr = DSRCore()
    for sym in SYMBOLS:
        print(f"  Loading {sym}...")
        dsr.load_symbol(sym)
    print("  All symbols loaded.\n")

    # Extract regime and direction arrays for each symbol
    symbol_data = {}
    for sym in SYMBOLS:
        d = dsr._data[sym]
        symbol_data[sym] = {
            "regime": d["regime"].copy(),
            "direction": d["residual_sign"].copy(),
            "fut_ret": d["fut_ret"].copy(),
            "n": len(d["regime"]),
        }

    LAGS = [1, 5, 10, 20]
    results = {
        "phase": 6,
        "title": "Cross-Asset Directional Cascade",
        "pairs": {},
        "leaders": {},
        "followers": {},
        "alignment_effects": {},
        "summary": {},
    }

    n_sym = len(SYMBOLS)

    for i in range(n_sym):
        for j in range(n_sym):
            if i == j:
                continue
            sym_a = SYMBOLS[i]
            sym_b = SYMBOLS[j]
            pair_key = f"{sym_a}->{sym_b}"

            reg_a = symbol_data[sym_a]["regime"]
            dir_b = symbol_data[sym_b]["direction"]
            n = min(len(reg_a), len(dir_b))

            pair_results = {}
            best_lag = None
            best_accuracy = 0.0

            for lag in LAGS:
                if lag >= n - 1:
                    continue

                # A's regime at t-lag predicts B's direction at t
                pred_regime = reg_a[lag:n]
                actual_dir = dir_b[lag:n]

                valid = (pred_regime >= 0) & (np.abs(actual_dir) == 1)
                if valid.sum() < 10:
                    continue

                # Directional accuracy: predict up (1) if regime 2 (high density),
                # down (-1) if regime 0 (low density), else guess 0 (skip)
                pred_direction = np.zeros_like(pred_regime, dtype=np.float64)
                pred_direction[:] = np.nan

                # Regime 2 (high density/pressure) -> predict up (1)
                # Regime 0 (low density/pressure) -> predict down (-1)
                pred_direction[pred_regime == 2] = 1.0
                pred_direction[pred_regime == 0] = -1.0

                acc = directional_accuracy(pred_direction, actual_dir)

                # Transfer entropy
                te = transfer_entropy(reg_a.astype(float), dir_b.astype(float), lag)

                # Up-accuracy separately
                up_mask = (pred_regime == 2) & valid
                up_acc = float(np.mean(actual_dir[up_mask] == 1)) if up_mask.sum() >= 5 else None

                # Regime alignment effect (A's regime vs B's regime at lag)
                reg_b = symbol_data[sym_b]["regime"]
                align = regime_alignment_accuracy(reg_a, reg_b, dir_b, lag)

                lag_result = {
                    "lag": lag,
                    "directional_accuracy": round(acc, 4),
                    "up_accuracy": round(up_acc, 4) if up_acc is not None else None,
                    "n_valid": int(valid.sum()),
                    "transfer_entropy": round(te, 6),
                    "regime_alignment": align,
                }
                pair_results[str(lag)] = lag_result

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_lag = lag

                # Also compute naive baseline: predict direction from B's own regime
                # (how often does B's regime predict B's direction?)

            if pair_results:
                results["pairs"][pair_key] = {
                    "best_lag": best_lag,
                    "best_accuracy": round(best_accuracy, 4),
                    "lags": pair_results,
                }

    # Determine leaders and followers
    # For each symbol B, find which A best predicts it (highest best_accuracy)
    for sym_b in SYMBOLS:
        best_leader = None
        best_acc = 0.0
        leaders = {}
        for sym_a in SYMBOLS:
            if sym_a == sym_b:
                continue
            pk = f"{sym_a}->{sym_b}"
            pair = results["pairs"].get(pk)
            if pair and pair["best_accuracy"] > best_acc:
                best_acc = pair["best_accuracy"]
                best_leader = sym_a
            if pair:
                leaders[sym_a] = pair["best_accuracy"]
        results["leaders"][sym_b] = {
            "primary_leader": best_leader,
            "leader_accuracy": round(best_acc, 4),
            "all_leaders": leaders,
        }

    # For each symbol A, determine which B it best predicts
    for sym_a in SYMBOLS:
        best_follower = None
        best_acc = 0.0
        followers = {}
        for sym_b in SYMBOLS:
            if sym_a == sym_b:
                continue
            pk = f"{sym_a}->{sym_b}"
            pair = results["pairs"].get(pk)
            if pair and pair["best_accuracy"] > best_acc:
                best_acc = pair["best_accuracy"]
                best_follower = sym_b
            if pair:
                followers[sym_b] = pair["best_accuracy"]
        results["followers"][sym_a] = {
            "primary_follower": best_follower,
            "follower_accuracy": round(best_acc, 4),
            "all_followers": followers,
        }

    # Check regime alignment effects across all pairs
    alignment_summary = {}
    for pair_key, pair_data in results["pairs"].items():
        for lag_str, lag_data in pair_data["lags"].items():
            al = lag_data.get("regime_alignment", {})
            if al.get("aligned_accuracy") is not None and al.get("misaligned_accuracy") is not None:
                diff = al["aligned_accuracy"] - al["misaligned_accuracy"]
                if pair_key not in alignment_summary:
                    alignment_summary[pair_key] = {"best_improvement": -999, "at_lag": None}
                if diff > alignment_summary[pair_key]["best_improvement"]:
                    alignment_summary[pair_key]["best_improvement"] = round(diff, 4)
                    alignment_summary[pair_key]["at_lag"] = int(lag_str)
                    alignment_summary[pair_key]["aligned_acc"] = al["aligned_accuracy"]
                    alignment_summary[pair_key]["misaligned_acc"] = al["misaligned_accuracy"]

    results["alignment_effects"] = alignment_summary

    # Build summary text
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("DSR PHASE 6 — CROSS-ASSET DIRECTIONAL CASCADE SUMMARY")
    summary_lines.append("=" * 60)

    summary_lines.append("\n--- LEADERBOARD (Which assets lead?) ---")
    for sym_b in SYMBOLS:
        ld = results["leaders"][sym_b]
        leader_acc_str = f"{ld['leader_accuracy']:.1%}" if ld["leader_accuracy"] > 0 else "N/A"
        summary_lines.append(f"  {sym_b} is led by {ld['primary_leader']} (acc={leader_acc_str})")

    summary_lines.append("\n--- FOLLOWERBOARD (Which assets follow?) ---")
    for sym_a in SYMBOLS:
        fl = results["followers"][sym_a]
        fl_acc_str = f"{fl['follower_accuracy']:.1%}" if fl["follower_accuracy"] > 0 else "N/A"
        summary_lines.append(f"  {sym_a} best predicts {fl['primary_follower']} (acc={fl_acc_str})")

    summary_lines.append("\n--- TOP-3 CROSS-ASSET PREDICTIVE PAIRS ---")
    all_pairs = [(k, v["best_accuracy"]) for k, v in results["pairs"].items()]
    all_pairs.sort(key=lambda x: -x[1])
    for pair_key, acc in all_pairs[:3]:
        lag = results["pairs"][pair_key]["best_lag"]
        te = results["pairs"][pair_key]["lags"].get(str(lag), {}).get("transfer_entropy", 0)
        summary_lines.append(f"  {pair_key}: acc={acc:.1%} @ lag={lag}, TE={te:.6f}")

    summary_lines.append("\n--- REGIME ALIGNMENT EFFECTS ---")
    improved = [(k, v["best_improvement"]) for k, v in alignment_summary.items()
                if v["best_improvement"] > -999]
    improved.sort(key=lambda x: -x[1])
    for pair_key, diff in improved[:5]:
        info = alignment_summary[pair_key]
        summary_lines.append(
            f"  {pair_key}: aligned={info['aligned_acc']:.1%}, "
            f"misaligned={info['misaligned_acc']:.1%}, "
            f"improvement={diff:+.1%} @ lag={info['at_lag']}"
        )

    summary_lines.append("\n--- CONCLUSION ---")
    summary_lines.append("Can leader-state information improve directional reconstruction?")
    if any(ld["leader_accuracy"] > 0.55 for ld in results["leaders"].values()):
        summary_lines.append("  YES — multiple pairs show >55% directional accuracy from leader regime.")
    else:
        summary_lines.append("  Limited — leader regime alone does not strongly predict follower direction.")
    if any(v["best_improvement"] > 0.02 for v in alignment_summary.values()):
        summary_lines.append("  YES — regime alignment significantly improves accuracy (aligned > misaligned).")
    else:
        summary_lines.append("  Regime alignment has marginal impact.")

    summary_text = "\n".join(summary_lines)
    results["summary"] = {"text": summary_text}

    # Save report
    save_report(results, "dsr_phase6_propagation_cascade")

    print(summary_text)

    return results


if __name__ == "__main__":
    run_phase6()
