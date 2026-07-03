"""DSR Phases 3, 4, 5 — Memory Topology Gate, State Transition Physics, State Persistence."""

import sys, json
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.directional_state.dsr_core import DSRCore, SYMBOLS, STATE_HORIZONS, STATE_HORIZON_KEYS, save_report

HORIZON_MAP = {h: i for i, h in enumerate([1, 5, 20, 50, 100, 500])}
HORIZON_IDX = [HORIZON_MAP[h] for h in STATE_HORIZONS]  # indices into fut_ret columns

###############################################################################
# Phase 3 — Memory Topology Gate
###############################################################################

def _quantize(x, n_bins=3):
    """Tertile-quantize a continuous array, handling NaN."""
    labels = np.full(len(x), -1, dtype=np.int64)
    valid = ~np.isnan(x)
    if np.sum(valid) < 10:
        return labels
    bins = np.nanpercentile(x[valid], np.linspace(0, 100, n_bins + 1)[1:-1])
    for i in range(n_bins - 1):
        if i == 0:
            labels[valid & (x <= bins[i])] = i
        else:
            labels[valid & (x <= bins[i]) & (x > bins[i - 1])] = i
    labels[valid & (x > bins[-1])] = n_bins - 1
    return labels


def _make_composite(*arrays):
    """Combine multiple categorical arrays into a single composite key."""
    result = np.zeros(len(arrays[0]), dtype=np.int64)
    for arr in arrays:
        result = result * 10 + np.where(arr >= 0, arr, 0)
    return result


def directional_accuracy(dsr, symbol, state_ids, horizon_idx):
    """Compute P(up) and related metrics for each state at a given horizon."""
    d = dsr._data[symbol]
    fut_ret = d["fut_ret"]
    up = (fut_ret[:, horizon_idx] > 0).astype(float)
    valid = (state_ids >= 0) & ~np.isnan(fut_ret[:, horizon_idx])
    state_ids = state_ids.copy()
    state_ids[~valid] = -1
    unique = np.unique(state_ids[state_ids >= 0])
    base_p_up = np.nanmean(up[valid])
    results = {}
    for sid in unique:
        mask = (state_ids == sid) & valid
        cnt = int(np.sum(mask))
        if cnt < 5:
            continue
        n_up = int(np.sum(up[mask]))
        p_up = n_up / cnt
        p_up_sem = np.sqrt(p_up * (1 - p_up) / cnt)
        p_down = 1 - p_up
        entropy = 0.0
        if p_up > 0 and p_down > 0:
            entropy = -(p_up * np.log2(p_up) + p_down * np.log2(p_down))
        ig = 0.0
        if p_up > 0 and base_p_up > 0:
            ig += p_up * np.log2(p_up / base_p_up)
        if p_down > 0 and (1 - base_p_up) > 0:
            ig += p_down * np.log2(p_down / (1 - base_p_up))
        results[int(sid)] = {
            "count": cnt, "n_up": n_up, "p_up": round(p_up, 4),
            "p_up_sem": round(p_up_sem, 4), "entropy": round(entropy, 4),
            "information_gain": round(ig, 4),
            "z_score": round((p_up - base_p_up) / max(p_up_sem, 1e-12), 4),
        }
    return results, base_p_up


def _model_separation_score(results):
    """Score how well a model separates up/down states: max|p_up - 0.5|, avg|p_up - 0.5|."""
    if not results:
        return {"max_deviation": 0, "avg_deviation": 0, "states": 0, "max_ig": 0}
    p_ups = [v["p_up"] for v in results.values()]
    deviations = [abs(p - 0.5) for p in p_ups]
    igs = [v["information_gain"] for v in results.values()]
    return {
        "max_deviation": round(max(deviations), 4),
        "avg_deviation": round(np.mean(deviations), 4),
        "states": len(results),
        "max_ig": round(max(igs), 4),
        "avg_ig": round(np.mean(igs), 4),
    }


def _contradictory_states(results):
    """Count states where P(up) is close to 0.5 (high uncertainty)."""
    if not results:
        return {"near_50pct": 0, "total": 0, "pct": 0}
    near = sum(1 for v in results.values() if abs(v["p_up"] - 0.5) < 0.05)
    return {"near_50pct": near, "total": len(results), "pct": round(near / len(results), 4)}


def phase3_memory_gate(dsr):
    """Compare Residual Only vs Residual + Memory models."""
    models = {}
    report = {}

    for sym in SYMBOLS:
        print(f"  Phase 3 - {sym}")
        d = dsr._data[sym]
        rs = d["residual_sign"]  # -1, 0, 1
        imb = d["memory_imbalance"]  # continuous
        sat = d["memory_saturation"]  # 0/1
        clu = d["memory_cluster"]  # 0, 1, 2

        # Quantize imbalance into tertiles
        imb_q = _quantize(imb, 3)

        sym_results = {}
        for h_idx, h_key in zip(HORIZON_IDX, STATE_HORIZON_KEYS):
            # Model 1: Residual only
            resid_state = _make_composite(rs + 1)  # shift -1,0,1 → 0,1,2
            r1, base_p = directional_accuracy(dsr, sym, resid_state, h_idx)
            # Model 2: Residual + imbalance
            rsi_state = _make_composite(rs + 1, imb_q + 1)
            r2, _ = directional_accuracy(dsr, sym, rsi_state, h_idx)
            # Model 3: Residual + saturation
            rss_state = _make_composite(rs + 1, sat)
            r3, _ = directional_accuracy(dsr, sym, rss_state, h_idx)
            # Model 4: Residual + cluster
            rsc_state = _make_composite(rs + 1, clu + 1)
            r4, _ = directional_accuracy(dsr, sym, rsc_state, h_idx)
            # Model 5: Residual + all memory
            rs_all_state = _make_composite(rs + 1, imb_q + 1, sat, clu + 1)
            r5, _ = directional_accuracy(dsr, sym, rs_all_state, h_idx)

            sym_results[h_key] = {
                "base_p_up": round(base_p, 4),
                "residual_only": {
                    **_model_separation_score(r1),
                    **_contradictory_states(r1),
                    "states_detail": r1,
                },
                "residual_plus_imbalance": {
                    **_model_separation_score(r2),
                    **_contradictory_states(r2),
                },
                "residual_plus_saturation": {
                    **_model_separation_score(r3),
                    **_contradictory_states(r3),
                },
                "residual_plus_cluster": {
                    **_model_separation_score(r4),
                    **_contradictory_states(r4),
                },
                "residual_plus_all_memory": {
                    **_model_separation_score(r5),
                    **_contradictory_states(r5),
                    "states_detail": r5,
                },
            }
        report[sym] = sym_results

    # Compute cross-symbol summary
    summary = {"improvement_separation": {}, "reduction_contradictions": {}, "best_memory_metric": {}}
    for h_key in STATE_HORIZON_KEYS:
        avg_sep_resid = np.mean([report[s][h_key]["residual_only"]["avg_deviation"] for s in SYMBOLS])
        avg_sep_all = np.mean([report[s][h_key]["residual_plus_all_memory"]["avg_deviation"] for s in SYMBOLS])
        avg_contra_resid = np.mean([report[s][h_key]["residual_only"]["pct"] for s in SYMBOLS])
        avg_contra_all = np.mean([report[s][h_key]["residual_plus_all_memory"]["pct"] for s in SYMBOLS])

        # Which memory metric contributes most? Compare avg deviation improvement
        metric_improvements = {}
        for metric, key in [("imbalance", "residual_plus_imbalance"),
                            ("saturation", "residual_plus_saturation"),
                            ("cluster", "residual_plus_cluster")]:
            imp = np.mean([report[s][h_key][key]["avg_deviation"] for s in SYMBOLS]) - avg_sep_resid
            metric_improvements[metric] = round(imp, 4)

        summary["improvement_separation"][h_key] = {
            "residual_only_avg_dev": round(avg_sep_resid, 4),
            "residual_plus_memory_avg_dev": round(avg_sep_all, 4),
            "improvement": round(avg_sep_all - avg_sep_resid, 4),
        }
        summary["reduction_contradictions"][h_key] = {
            "residual_only_pct": round(avg_contra_resid, 4),
            "residual_plus_memory_pct": round(avg_contra_all, 4),
            "improvement": round(avg_contra_resid - avg_contra_all, 4),
        }
        best_metric = max(metric_improvements, key=metric_improvements.get)
        summary["best_memory_metric"][h_key] = {
            "metric": best_metric,
            "improvements": metric_improvements,
        }

    result = {"per_symbol": report, "cross_symbol_summary": summary}
    print("  [Phase 3] Memory gate analysis complete.")
    return result


###############################################################################
# Phase 4 — State Transition Physics
###############################################################################

def phase4_state_transitions(dsr):
    """Analyze regime transitions: state(t-1) → state(t) and their directional power."""
    report = {}
    for sym in SYMBOLS:
        print(f"  Phase 4 - {sym}")
        d = dsr._data[sym]
        regime = d["regime"]
        trans_from = d["reg_transition_from"]
        trans_to = d["reg_transition_to"]
        fut_ret = d["fut_ret"]

        # Collect transition pairs (from, to) where a transition occurs
        transition_mask = (trans_from >= 0) & (trans_to >= 0)
        trans_pairs = np.stack([trans_from[transition_mask], trans_to[transition_mask]], axis=1)
        unique_trans = np.unique(trans_pairs, axis=0)

        sym_results = {}
        for h_idx, h_key in zip(HORIZON_IDX, STATE_HORIZON_KEYS):
            up = (fut_ret[:, h_idx] > 0).astype(float)
            base_p_up = np.nanmean(up[transition_mask])

            trans_analysis = {}
            for pair in unique_trans:
                f, t = pair
                mask = (trans_from == f) & (trans_to == t)
                cnt = int(np.sum(mask))
                if cnt < 5:
                    continue
                n_up = int(np.sum(up[mask]))
                p_up = n_up / cnt
                p_up_sem = np.sqrt(p_up * (1 - p_up) / cnt)
                p_down = 1 - p_up
                entropy = 0.0
                if p_up > 0 and p_down > 0:
                    entropy = -(p_up * np.log2(p_up) + p_down * np.log2(p_down))
                ig = 0.0
                if p_up > 0 and base_p_up > 0:
                    ig += p_up * np.log2(p_up / base_p_up)
                if p_down > 0 and (1 - base_p_up) > 0:
                    ig += p_down * np.log2(p_down / (1 - base_p_up))
                trans_analysis[f"S{int(f)}_to_S{int(t)}"] = {
                    "count": cnt, "n_up": n_up, "p_up": round(p_up, 4),
                    "p_up_sem": round(p_up_sem, 4), "entropy": round(entropy, 4),
                    "information_gain": round(ig, 4),
                }

            # Also analyze no-transition (stable) states
            stable_mask = ~transition_mask & (regime >= 0)
            stable_up = np.nanmean(up[stable_mask]) if np.sum(stable_mask) > 5 else None

            trans_results = list(trans_analysis.values())
            avg_p_up = np.mean([t["p_up"] for t in trans_results]) if trans_results else 0
            max_ig = max([t["information_gain"] for t in trans_results]) if trans_results else 0
            abs_devs = [abs(t["p_up"] - 0.5) for t in trans_results]

            sym_results[h_key] = {
                "base_p_up_transitions": round(base_p_up, 4),
                "num_transition_types": len(trans_analysis),
                "transitions": trans_analysis,
                "avg_p_up": round(avg_p_up, 4),
                "max_ig": max_ig,
                "avg_abs_deviation": round(np.mean(abs_devs), 4) if abs_devs else 0,
                "stable_p_up": round(float(stable_up), 4) if stable_up is not None else None,
            }
        report[sym] = sym_results

    # Cross-symbol summary: compare transition vs state predictive power
    summary = {}
    for h_key in STATE_HORIZON_KEYS:
        trans_devs = [report[s][h_key]["avg_abs_deviation"] for s in SYMBOLS]
        summary[h_key] = {
            "avg_transition_deviation": round(np.mean(trans_devs), 4),
            "per_symbol": {s: report[s][h_key]["avg_abs_deviation"] for s in SYMBOLS},
        }

    result = {"per_symbol": report, "cross_symbol_summary": summary}
    print("  [Phase 4] State transition analysis complete.")
    return result


###############################################################################
# Phase 5 — Directional State Persistence
###############################################################################

def phase5_state_persistence(dsr):
    """Measure half-life, entropy, survival probability, stability duration of directional states."""
    report = {}
    for sym in SYMBOLS:
        print(f"  Phase 5 - {sym}")
        d = dsr._data[sym]
        regime = d["regime"]
        fut_ret = d["fut_ret"]
        n = len(regime)

        # Find runs of consecutive identical regime states
        valid = regime >= 0
        run_lengths = []
        run_starts = []
        run_states = []
        i = 0
        while i < n:
            if not valid[i]:
                i += 1
                continue
            current_state = regime[i]
            j = i
            while j < n and valid[j] and regime[j] == current_state:
                j += 1
            run_len = j - i
            run_lengths.append(run_len)
            run_starts.append(i)
            run_states.append(int(current_state))
            i = j

        run_lengths = np.array(run_lengths)
        run_states_arr = np.array(run_states)
        run_starts_arr = np.array(run_starts)

        if len(run_lengths) == 0:
            report[sym] = {"error": "no valid regime data"}
            continue

        # Overall statistics
        half_life = int(np.median(run_lengths))
        mean_duration = float(np.mean(run_lengths))
        max_duration = int(np.max(run_lengths))
        min_duration = int(np.min(run_lengths))
        std_duration = float(np.std(run_lengths))

        # Survival probability: P(state unchanged after N bars)
        max_len = int(np.percentile(run_lengths, 95))
        survival_probs = {}
        for n_bars in range(1, min(max_len + 1, 51)):
            survived = np.sum(run_lengths > n_bars)
            total = len(run_lengths)
            survival_probs[n_bars] = round(survived / total, 4) if total > 0 else 0

        # Entropy of state distribution
        state_counts = {}
        for s in run_states_arr:
            state_counts[int(s)] = state_counts.get(int(s), 0) + 1
        total_runs = len(run_states_arr)
        state_entropy = 0.0
        for cnt in state_counts.values():
            p = cnt / total_runs
            state_entropy -= p * np.log2(p)

        # Per-state persistence
        per_state = {}
        for s in sorted(set(run_states_arr)):
            mask = run_states_arr == s
            if np.sum(mask) < 2:
                continue
            s_runs = run_lengths[mask]
            s_half_life = int(np.median(s_runs))
            s_mean = float(np.mean(s_runs))
            s_max = int(np.max(s_runs))
            s_count = int(np.sum(mask))

            # Directional strength decay: P(up) at start vs end of runs
            # For runs >= 5 bars, compare first 2 bars vs last 2 bars
            start_up = []
            end_up = []
            for idx in np.where(mask)[0]:
                si = run_starts_arr[idx]
                sl = run_lengths[idx]
                if sl >= 5:
                    # P(up) at H5 for first and last portions
                    for h_idx, h_key in zip(HORIZON_IDX, STATE_HORIZON_KEYS):
                        if si + 2 < n and si + sl - 2 < n:
                            start_slice = fut_ret[si:si + 2, h_idx]
                            end_slice = fut_ret[si + sl - 2:si + sl, h_idx]
                            start_up.append(np.nanmean(start_slice > 0))
                            end_up.append(np.nanmean(end_slice > 0))

            decay = None
            if start_up and end_up:
                avg_start = np.nanmean(start_up)
                avg_end = np.nanmean(end_up)
                decay = round(avg_end - avg_start, 4)

            per_state[int(s)] = {
                "count": s_count,
                "half_life": s_half_life,
                "mean_duration": round(s_mean, 2),
                "max_duration": s_max,
                "directional_decay": decay,
            }

        report[sym] = {
            "n_runs": len(run_lengths),
            "half_life": half_life,
            "mean_duration": round(mean_duration, 2),
            "max_duration": max_duration,
            "min_duration": min_duration,
            "std_duration": round(std_duration, 2),
            "state_entropy": round(state_entropy, 4),
            "survival_probability": survival_probs,
            "per_state": per_state,
        }

    # Cross-symbol summary
    half_lives = [report[s]["half_life"] for s in SYMBOLS if "half_life" in report[s]]
    mean_durs = [report[s]["mean_duration"] for s in SYMBOLS if "mean_duration" in report[s]]
    summary = {
        "avg_half_life": round(np.mean(half_lives), 1) if half_lives else None,
        "avg_mean_duration": round(np.mean(mean_durs), 1) if mean_durs else None,
        "per_symbol": {s: {"half_life": report[s]["half_life"],
                           "mean_duration": report[s]["mean_duration"],
                           "state_entropy": report[s]["state_entropy"]}
                       for s in SYMBOLS if "half_life" in report[s]},
    }

    result = {"per_symbol": report, "cross_symbol_summary": summary}
    print("  [Phase 5] State persistence analysis complete.")
    return result


###############################################################################
# Main
###############################################################################

def main():
    print("=" * 60)
    print("DSR Phases 3, 4, 5 - Memory Gate, Transitions, Persistence")
    print("=" * 60)

    print("\nLoading DSR core...")
    dsr = DSRCore()
    for sym in SYMBOLS:
        print(f"  Loading {sym}...")
        dsr.load_symbol(sym)
    print(f"  Loaded {len(dsr._data)} symbols.\n")

    # Phase 3
    print("-" * 50)
    print("Phase 3 - Memory Topology Gate")
    print("-" * 50)
    r3 = phase3_memory_gate(dsr)
    save_report(r3, "dsr_phase3_memory_gate")
    _print_phase3_summary(r3)

    # Phase 4
    print("\n" + "-" * 50)
    print("Phase 4 - State Transition Physics")
    print("-" * 50)
    r4 = phase4_state_transitions(dsr)
    save_report(r4, "dsr_phase4_state_transitions")
    _print_phase4_summary(r4)

    # Phase 5
    print("\n" + "-" * 50)
    print("Phase 5 - Directional State Persistence")
    print("-" * 50)
    r5 = phase5_state_persistence(dsr)
    save_report(r5, "dsr_phase5_state_persistence")
    _print_phase5_summary(r5)

    print("\n" + "=" * 60)
    print("DSR Phases 3-5 complete. Reports saved.")
    print("=" * 60)


def _print_phase3_summary(r3):
    s = r3["cross_symbol_summary"]
    print("\n  Cross-Symbol Summary:")
    for hk in STATE_HORIZON_KEYS:
        sep = s["improvement_separation"][hk]
        con = s["reduction_contradictions"][hk]
        best = s["best_memory_metric"][hk]
        print(f"  [{hk}]")
        print(f"    Separation: resid={sep['residual_only_avg_dev']} vs mem={sep['residual_plus_memory_avg_dev']} (delta={sep['improvement']:+.4f})")
        print(f"    Contradictions: resid={con['residual_only_pct']} vs mem={con['residual_plus_memory_pct']} (delta={con['improvement']:+.4f})")
        print(f"    Best metric: {best['metric']} ({best['improvements']})")


def _print_phase4_summary(r4):
    s = r4["cross_symbol_summary"]
    print("\n  Cross-Symbol Summary:")
    for hk in STATE_HORIZON_KEYS:
        print(f"  [{hk}] Avg transition deviation: {s[hk]['avg_transition_deviation']}")
        for sym in SYMBOLS:
            t = r4["per_symbol"][sym][hk]
            sym_data = r4["per_symbol"][sym][hk]
            n_trans = sym_data["num_transition_types"]
            print(f"    {sym}: {n_trans} transition types, avg_p_up={sym_data['avg_p_up']}, max_ig={sym_data['max_ig']}")


def _print_phase5_summary(r5):
    s = r5["cross_symbol_summary"]
    print(f"\n  Avg half-life across symbols: {s['avg_half_life']} bars")
    print(f"  Avg mean duration: {s['avg_mean_duration']} bars")
    for sym in SYMBOLS:
        ps = r5["per_symbol"][sym]
        if "half_life" not in ps:
            continue
        print(f"  {sym}: half_life={ps['half_life']} bars, mean={ps['mean_duration']} bars, "
              f"entropy={ps['state_entropy']}, runs={ps['n_runs']}")
        for state, sps in ps.get("per_state", {}).items():
            decay_str = f"decay={sps['directional_decay']}" if sps['directional_decay'] is not None else "N/A"
            print(f"    S{state}: half_life={sps['half_life']}, mean={sps['mean_duration']}, {decay_str}")


if __name__ == "__main__":
    main()
