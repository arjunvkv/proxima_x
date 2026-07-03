"""STL Phase 3 (Transition Topology) & Phase 4 (Cross-Asset Synchronization)."""
import sys, json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.state_topology.stl_core import STLCore, SYMBOLS, decode_state, save_stl_report

# fut_ret index mapping
H_MAP = {5: 1, 20: 2, 50: 3}
H_KEYS = {5: "h5", 20: "h20", 50: "h50"}


def run_phase3():
    """Transition Topology: build transition graph from state space."""
    stl = STLCore()
    stl.load_all()

    all_reports = {}

    for sym in SYMBOLS:
        d = stl._data[sym]
        sids = d["state_id"]
        fut_ret = d["fut_ret"]

        transitions = defaultdict(lambda: {
            "count": 0,
            "up_h5": 0, "up_h20": 0, "up_h50": 0,
            "total_h5": 0, "total_h20": 0, "total_h50": 0
        })

        for i in range(1, len(sids)):
            f = sids[i - 1]
            t = sids[i]
            if f < 0 or t < 0:
                continue
            key = (int(f), int(t))
            row = transitions[key]
            row["count"] += 1

            for h, hi in H_MAP.items():
                ret = fut_ret[i, hi]
                if np.isnan(ret):
                    continue
                hk = H_KEYS[h]
                row[f"total_{hk}"] += 1
                if ret > 0:
                    row[f"up_{hk}"] += 1

        # Build structured results
        symbol_results = {}
        for (f, t), v in sorted(transitions.items(), key=lambda x: -x[1]["count"]):
            p_ups = {}
            for h, hk in H_KEYS.items():
                total = v[f"total_{hk}"]
                p_ups[f"H{h}"] = round(v[f"up_{hk}"] / total, 4) if total > 0 else None

            try:
                f_dec = decode_state(f)
                t_dec = decode_state(t)
            except Exception:
                f_dec = None
                t_dec = None

            k = f"{f}->{t}"
            symbol_results[k] = {
                "from_state": int(f),
                "to_state": int(t),
                "from_decode": [int(x) for x in f_dec] if f_dec else None,
                "to_decode": [int(x) for x in t_dec] if t_dec else None,
                "count": int(v["count"]),
                "p_up": p_ups,
                "n_total": {f"H{h}": int(v[f"total_{hk}"]) for h, hk in H_KEYS.items()},
                "n_up": {f"H{h}": int(v[f"up_{hk}"]) for h, hk in H_KEYS.items()},
            }

        all_reports[sym] = symbol_results

    # Derive answers
    answers = derive_transition_insights(all_reports)

    report = {
        "phase": "STL Phase 3 — State Transition Topology",
        "transitions_per_symbol": all_reports,
        "insights": answers,
    }

    save_stl_report(report, "stl3_transitions")

    # Write markdown
    md = generate_transitions_md(all_reports, answers)
    md_path = Path(__file__).parent / "reports" / "STL3_TRANSITIONS.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved {md_path}")
    return report


def derive_transition_insights(all_reports):
    """Derive answers from transition data."""
    insights = {}
    for sym, trans in all_reports.items():
        # Directional asymmetry: |P(up) - 0.5| max
        asyms = []
        for k, v in trans.items():
            for h in ["H50"]:
                p = v["p_up"].get(h)
                if p is not None:
                    asym = abs(p - 0.5)
                    asyms.append((asym, k, h, p, v["count"]))
        asyms.sort(key=lambda x: -x[0])
        strongest_asym = [
            {"transition": k, "horizon": h, "p_up": p, "asymmetry": round(a, 4), "count": c}
            for a, k, h, p, c in asyms[:10]
        ]

        # Self-loop probability (from_state == to_state)
        self_loops = []
        for k, v in trans.items():
            f, t = k.split("->")
            if f == t:
                total_from = sum(
                    v2["count"] for k2, v2 in trans.items() if k2.startswith(f"{f}->")
                )
                self_prob = v["count"] / total_from if total_from > 0 else 0
                self_loops.append((self_prob, k, v["count"], total_from))
        self_loops.sort(key=lambda x: -x[0])

        # Transition matrix (compact)
        unique_states = set()
        for k in trans:
            f, t = k.split("->")
            unique_states.add(int(f))
            unique_states.add(int(t))

        insights[sym] = {
            "n_transitions": len(trans),
            "n_unique_states": len(unique_states),
            "strongest_directional_asymmetry": strongest_asym[:5],
            "strongest_asym_summary": (f"{strongest_asym[0]['transition']} "
                                       f"at {strongest_asym[0]['horizon']} "
                                       f"|P(up)-0.5|={strongest_asym[0]['asymmetry']}")
            if strongest_asym else None,
            "highest_self_loop": [
                {"transition": k, "self_loop_prob": round(p, 4), "count": c, "total_from": t}
                for p, k, c, t in self_loops[:5]
            ],
        }

    return insights


def generate_transitions_md(all_reports, insights):
    lines = ["# STL Phase 3 — State Transition Topology", ""]

    for sym in SYMBOLS:
        lines.append(f"## {sym}")
        lines.append("")
        ins = insights.get(sym, {})
        lines.append(f"- **Observed transitions**: {ins.get('n_transitions', 'N/A')}")
        lines.append(f"- **Unique states involved**: {ins.get('n_unique_states', 'N/A')}")
        lines.append("")

        if ins.get("strongest_asym_summary"):
            lines.append(f"### Strongest Directional Asymmetry")
            lines.append(f"- {ins['strongest_asym_summary']}")
            lines.append("")

        # Top transitions by count
        trans = all_reports.get(sym, {})
        top_by_count = sorted(trans.items(), key=lambda x: -x[1]["count"])[:15]
        lines.append("### Top Transitions by Frequency")
        lines.append("")
        lines.append("| Transition | From (ES,AT,Reg,Mem) | To (ES,AT,Reg,Mem) | Count | P(up) H5 | P(up) H20 | P(up) H50 |")
        lines.append("|---|---|---|---|---|---|---|")
        for k, v in top_by_count:
            fd = v["from_decode"]
            td = v["to_decode"]
            fd_str = f"({fd[0]},{fd[1]},{fd[2]},{fd[3]})" if fd else "?"
            td_str = f"({td[0]},{td[1]},{td[2]},{td[3]})" if td else "?"
            p5 = f"{v['p_up'].get('H5', 'N/A'):.4f}" if isinstance(v['p_up'].get('H5'), (int, float)) else "N/A"
            p20 = f"{v['p_up'].get('H20', 'N/A'):.4f}" if isinstance(v['p_up'].get('H20'), (int, float)) else "N/A"
            p50 = f"{v['p_up'].get('H50', 'N/A'):.4f}" if isinstance(v['p_up'].get('H50'), (int, float)) else "N/A"
            lines.append(f"| {k} | {fd_str} | {td_str} | {v['count']} | {p5} | {p20} | {p50} |")
        lines.append("")

        if ins.get("highest_self_loop"):
            lines.append("### Highest Self-Loop Probabilities")
            lines.append("")
            for sl in ins["highest_self_loop"]:
                lines.append(f"- **{sl['transition']}**: P(stay)={sl['self_loop_prob']:.4f} "
                             f"(count={sl['count']}, total_from={sl['total_from']})")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def run_phase4():
    """Cross-Asset State Synchronization."""
    stl = STLCore()
    stl.load_all()

    n_sym = len(SYMBOLS)
    alignment_matrix = np.zeros((n_sym, n_sym), dtype=np.float64)
    alignment_counts = np.zeros((n_sym, n_sym), dtype=np.int64)

    # Build state arrays for all symbols (aligned by index)
    sym_states = {}
    for sym in SYMBOLS:
        d = stl._data[sym]
        sids = d["state_id"]
        future = d["fut_ret"]
        sym_states[sym] = {"sids": sids, "fut_ret": future}

    min_len = min(v["sids"].shape[0] for v in sym_states.values())
    n = min_len

    # Phase 4a: State alignment
    for i, sym_a in enumerate(SYMBOLS):
        for j, sym_b in enumerate(SYMBOLS):
            if j <= i:
                continue
            sids_a = sym_states[sym_a]["sids"][:n]
            sids_b = sym_states[sym_b]["sids"][:n]
            valid = (sids_a >= 0) & (sids_b >= 0)
            if valid.sum() == 0:
                continue
            same = (sids_a[valid] == sids_b[valid]).sum()
            alignment_matrix[i, j] = same / valid.sum()
            alignment_counts[i, j] = int(valid.sum())

    # Phase 4b: A's state at t predicts B's direction at t+H
    cross_pred = {}
    for sym_a in SYMBOLS:
        cross_pred[sym_a] = {}
        for sym_b in SYMBOLS:
            if sym_a == sym_b:
                continue
            sids_a = sym_states[sym_a]["sids"][:n]
            fut_b = sym_states[sym_b]["fut_ret"][:n]

            # For each state in A, compute P(B up) at each horizon
            state_predictions = {}
            valid_idx = np.where((sids_a >= 0) & ~np.any(np.isnan(fut_b[:, [1, 2, 3]]), axis=1))[0]
            for sid in np.unique(sids_a[valid_idx]):
                if sid < 0:
                    continue
                mask = sids_a[valid_idx] == sid
                if mask.sum() < 5:
                    continue
                preds = {}
                for h, hi in H_MAP.items():
                    rets = fut_b[valid_idx[mask], hi]
                    p_up = float((rets > 0).mean())
                    preds[f"H{h}"] = round(p_up, 4)
                state_predictions[int(sid)] = {
                    "count": int(mask.sum()),
                    "p_up_b": preds,
                }
            cross_pred[sym_a][sym_b] = state_predictions

    # Phase 4c: Multi-asset state clusters (states co-occurring across 3+ assets)
    multi_clusters = []
    # Build combined state tuples per bar
    state_tuples = []
    valid_bars = np.ones(n, dtype=bool)
    sym_sids_list = []
    for sym in SYMBOLS:
        ss = sym_states[sym]["sids"][:n]
        sym_sids_list.append(ss)
        valid_bars &= (ss >= 0)

    valid_idx_arr = np.where(valid_bars)[0]
    from collections import Counter
    tuple_counts = Counter()
    for idx in valid_idx_arr:
        tup = tuple(int(sym_sids_list[i][idx]) for i in range(n_sym))
        tuple_counts[tup] += 1

    # Find tuples that appear with 3+ distinct states
    for tup, cnt in tuple_counts.most_common(50):
        distinct_states = len(set(tup))
        if distinct_states >= 3 and cnt >= 3:
            multi_clusters.append({
                "state_tuple": [int(x) for x in tup],
                "symbols": {sym: int(tup[i]) for i, sym in enumerate(SYMBOLS)},
                "count": int(cnt),
            })

    # Phase 4d: Directed answers
    answers_4 = derive_sync_insights(alignment_matrix, cross_pred, multi_clusters)

    report = {
        "phase": "STL Phase 4 — Cross-Asset State Synchronization",
        "alignment_matrix": {
            f"{SYMBOLS[i]}_vs_{SYMBOLS[j]}": round(float(alignment_matrix[i, j]), 4)
            for i in range(n_sym) for j in range(n_sym) if j > i and alignment_counts[i, j] > 0
        },
        "cross_prediction": cross_pred,
        "multi_asset_clusters": multi_clusters[:30],
        "insights": answers_4,
    }

    save_stl_report(report, "stl4_synchronization")

    md = generate_sync_md(report)
    md_path = Path(__file__).parent / "reports" / "STL4_SYNCHRONIZATION.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved {md_path}")
    return report


def derive_sync_insights(alignment_matrix, cross_pred, multi_clusters):
    """Derive answers for Phase 4."""
    insights = {}

    # EURJPY -> GBPJPY predictive power
    eur_gbp = {}
    if "EURJPY" in cross_pred and "GBPJPY" in cross_pred["EURJPY"]:
        states = cross_pred["EURJPY"]["GBPJPY"]
        best = []
        for sid, v in states.items():
            for h in ["H5", "H20", "H50"]:
                p = v["p_up_b"].get(h)
                if p is not None:
                    asym = abs(p - 0.5)
                    best.append((asym, sid, h, p, v["count"]))
        best.sort(key=lambda x: -x[0])
        eur_gbp = {
            "n_states_with_predictive_power": len(states),
            "strongest": [
                {"state_id": s, "horizon": h, "p_gbp_up": p, "asymmetry": round(a, 4), "count": c}
                for a, s, h, p, c in best[:5]
            ]
        }

    # USDJPY state alignment matter?
    usdjpy_align = {}
    for sym_b in SYMBOLS:
        if sym_b == "USDJPY":
            continue
        key = f"USDJPY_vs_{sym_b}"
        if sym_b in cross_pred.get("USDJPY", {}):
            states = cross_pred["USDJPY"][sym_b]
            best = []
            for sid, v in states.items():
                for h in ["H5", "H20", "H50"]:
                    p = v["p_up_b"].get(h)
                    if p is not None:
                        best.append((abs(p - 0.5), sid, h, p, v["count"], sym_b))
            best.sort(key=lambda x: -x[0])
            usdjpy_align[sym_b] = {
                "n_predictive_states": len(states) if best else 0,
                "strongest": [
                    {"state_id": s, "horizon": h, "p_up": p, "asymmetry": round(a, 4)}
                    for a, s, h, p, _, _ in best[:3]
                ] if best else [],
            }

    # XAUUSD state agreement
    xau_align = {}
    for sym_b in SYMBOLS:
        if sym_b == "XAUUSD":
            continue
        key = f"XAUUSD_vs_{sym_b}"
        if sym_b in cross_pred.get("XAUUSD", {}):
            states = cross_pred["XAUUSD"][sym_b]
            best = []
            for sid, v in states.items():
                for h in ["H5", "H20", "H50"]:
                    p = v["p_up_b"].get(h)
                    if p is not None:
                        best.append((abs(p - 0.5), sid, h, p, v["count"]))
            best.sort(key=lambda x: -x[0])
            xau_align[sym_b] = {
                "n_predictive_states": len(states) if best else 0,
                "strongest": [
                    {"state_id": s, "horizon": h, "p_up": p, "asymmetry": round(a, 4)}
                    for a, s, h, p, _ in best[:3]
                ] if best else [],
            }

    insights = {
        "eurjpy_predicts_gbpjpy": eur_gbp,
        "usdjpy_cross_predictions": usdjpy_align,
        "xauusd_cross_predictions": xau_align,
        "multi_asset_clusters_found": len(multi_clusters),
        "multi_asset_summary": [
            {
                "states": c["state_tuple"],
                "symbols": c["symbols"],
                "count": c["count"],
            }
            for c in multi_clusters[:10]
        ],
    }
    return insights


def generate_sync_md(report):
    lines = ["# STL Phase 4 — Cross-Asset State Synchronization", ""]

    # Alignment matrix
    lines.append("## State Alignment Matrix")
    lines.append("")
    lines.append("| Pair | Alignment (%) |")
    lines.append("|---|---|")
    align = report.get("alignment_matrix", {})
    for pair, pct in sorted(align.items(), key=lambda x: -x[1]):
        lines.append(f"| {pair} | {pct*100:.2f}% |")
    lines.append("")

    # EURJPY -> GBPJPY
    ins = report.get("insights", {})
    eurgbp = ins.get("eurjpy_predicts_gbpjpy", {})
    lines.append("## EURJPY State -> GBPJPY Direction")
    lines.append("")
    if eurgbp:
        lines.append(f"- **{eurgbp.get('n_states_with_predictive_power', 0)}** EURJPY states predict GBPJPY direction")
        for s in eurgbp.get("strongest", []):
            lines.append(f"- State **{s['state_id']}** -> GBPJPY at {s['horizon']}: "
                         f"P(up)={s['p_gbp_up']:.4f} (asym={s['asymmetry']:.4f}, n={s['count']})")
    else:
        lines.append("- No predictive relationship found.")
    lines.append("")

    # USDJPY
    lines.append("## USDJPY Cross-Asset Predictions")
    lines.append("")
    usd = ins.get("usdjpy_cross_predictions", {})
    for sym_b, info in usd.items():
        lines.append(f"### USDJPY -> {sym_b}")
        lines.append(f"- {info.get('n_predictive_states', 0)} predictive states")
        for s in info.get("strongest", []):
            lines.append(f"  - State **{s['state_id']}** at {s['horizon']}: "
                         f"P({sym_b} up)={s['p_up']:.4f} (asym={s['asymmetry']:.4f})")
    lines.append("")

    # XAUUSD
    lines.append("## XAUUSD Cross-Asset Predictions")
    lines.append("")
    xau = ins.get("xauusd_cross_predictions", {})
    for sym_b, info in xau.items():
        lines.append(f"### XAUUSD -> {sym_b}")
        lines.append(f"- {info.get('n_predictive_states', 0)} predictive states")
        for s in info.get("strongest", []):
            lines.append(f"  - State **{s['state_id']}** at {s['horizon']}: "
                         f"P({sym_b} up)={s['p_up']:.4f} (asym={s['asymmetry']:.4f})")
    lines.append("")

    # Multi-asset clusters
    lines.append("## Multi-Asset State Clusters")
    lines.append("")
    clusters = report.get("multi_asset_clusters", [])
    if clusters:
        lines.append(f"Found **{len(clusters)}** multi-asset state clusters (3+ distinct states, ≥3 occurrences):")
        lines.append("")
        for c in clusters[:15]:
            sym_str = "; ".join(f"{s}={c['symbols'][s]}" for s in SYMBOLS)
            lines.append(f"- Occurrences: **{c['count']}** -> {sym_str}")
    else:
        lines.append("No significant multi-asset state clusters found.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("STL Phase 3 — State Transition Topology")
    print("=" * 60)
    r3 = run_phase3()
    print()

    print("=" * 60)
    print("STL Phase 4 — Cross-Asset State Synchronization")
    print("=" * 60)
    r4 = run_phase4()
    print()

    # Print key findings
    print()
    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    for sym in SYMBOLS:
        ins = r3.get("insights", {}).get(sym, {})
        print(f"\n{sym} — Transitions: {ins.get('n_transitions', 'N/A')}, "
              f"States: {ins.get('n_unique_states', 'N/A')}")
        if ins.get("strongest_asym_summary"):
            print(f"  Strongest asymmetry: {ins['strongest_asym_summary']}")
        if ins.get("highest_self_loop"):
            print(f"  Top self-loop: {ins['highest_self_loop'][0]['transition']} "
                  f"(P={ins['highest_self_loop'][0]['self_loop_prob']:.4f})")

    ins4 = r4.get("insights", {})
    eurgbp = ins4.get("eurjpy_predicts_gbpjpy", {})
    print(f"\nEURJPY -> GBPJPY: {eurgbp.get('n_states_with_predictive_power', 0)} predictive states")
    print(f"Multi-asset clusters: {ins4.get('multi_asset_clusters_found', 0)}")
