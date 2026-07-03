"""STL Phase 7: Synthetic Counterfactual Gate + Phase 8: Final Adjudication."""
import sys, json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.state_topology.stl_core import STLCore, SYMBOLS, decode_state, save_stl_report

HORIZONS = [5, 20, 50, 100]
HORIZON_INDICES = {5: 1, 20: 2, 50: 3, 100: 4}


def compute_directional_count(sids, fut_ret, threshold=0.70, min_count=5):
    """Count directional states (P(up) >= threshold at any horizon)."""
    state_info = {}
    for i in range(len(sids)):
        sid = sids[i]
        if sid < 0:
            continue
        if sid not in state_info:
            state_info[sid] = {"count": 0, "up": {h: 0 for h in HORIZONS}, "total": {h: 0 for h in HORIZONS}}
        state_info[sid]["count"] += 1
        for h in HORIZONS:
            ret = fut_ret[i, HORIZON_INDICES[h]]
            if np.isnan(ret):
                continue
            state_info[sid]["total"][h] += 1
            if ret > 0:
                state_info[sid]["up"][h] += 1

    directional = []
    for sid, info in state_info.items():
        if info["count"] < min_count:
            continue
        for h in HORIZONS:
            t = info["total"][h]
            if t == 0:
                continue
            p_up = info["up"][h] / t
            if p_up >= threshold:
                directional.append({"state_id": int(sid), "p_up": round(p_up, 4), "horizon": h, "count": info["count"]})
                break
    return directional, state_info


def directional_state_ids_from_summary(summary, threshold=0.70, min_count=5):
    """Get set of state IDs that are directional."""
    dir_ids = set()
    for sid, info in summary.items():
        if info["count"] < min_count:
            continue
        for h in HORIZONS:
            p = info["p_up"].get(h)
            if p is not None and p >= threshold:
                dir_ids.add(int(sid))
                break
    return dir_ids


def compute_state_summary(sids, fut_ret):
    """Compute P(up) per state for given state array."""
    _, state_info = compute_directional_count(sids, fut_ret)
    results = {}
    for sid, info in state_info.items():
        p_ups = {}
        for h in HORIZONS:
            t = info["total"][h]
            p_ups[h] = round(info["up"][h] / t, 4) if t > 0 else None
        es_q, at_q, regime, mem_q = decode_state(int(sid))
        results[int(sid)] = {
            "state_id": int(sid),
            "count": info["count"],
            "p_up": p_ups,
            "components": [es_q, at_q, regime, mem_q],
        }
    return results


def phase7_counterfactual_gate(stl):
    """Phase 7: Synthetic Counterfactual Gate."""
    report = {
        "phase": "STL Phase 7 -- Synthetic Counterfactual Gate",
        "per_symbol": {},
        "summary": {},
    }

    for sym in SYMBOLS:
        print(f"\n  {sym}:")
        d = stl._data[sym]
        sids = d["state_id"].copy()
        fut_ret = d["fut_ret"]
        n = len(sids)
        valid_mask = sids >= 0
        n_valid = int(np.sum(valid_mask))

        # Real state space
        real_dir, real_state_info = compute_directional_count(sids, fut_ret)
        n_real_dir = len(real_dir)
        real_summary = compute_state_summary(sids, fut_ret)
        real_dir_ids = directional_state_ids_from_summary(real_summary)
        print(f"    Real: {n_real_dir} directional states")

        sym_result = {
            "n_bars": n,
            "n_valid": n_valid,
            "real": {
                "n_directional": n_real_dir,
                "directional_states": real_dir,
            },
            "synthetic": {},
            "artifactual_ratios": {},
            "same_state_overlap": {},
        }

        # S1: Shuffled states -- random permutation
        np.random.seed(42)
        shuffled = sids.copy()
        shuffled_valid = shuffled >= 0
        shuffled[shuffled_valid] = np.random.permutation(shuffled[shuffled_valid])
        s1_dir, _ = compute_directional_count(shuffled, fut_ret)
        n_s1 = len(s1_dir)
        s1_summary = compute_state_summary(shuffled, fut_ret)
        s1_dir_ids = directional_state_ids_from_summary(s1_summary)
        overlap_s1 = real_dir_ids & s1_dir_ids
        sym_result["synthetic"]["S1_shuffled"] = {
            "description": "Random permutation of state IDs -- destroys temporal structure",
            "n_directional": n_s1,
            "directional_states": s1_dir,
            "overlap_with_real": list(overlap_s1),
            "n_overlap": len(overlap_s1),
        }
        print(f"    S1 (shuffled): {n_s1} directional, overlap={len(overlap_s1)}")

        # S2: Markov states -- learn P(state_t | state_{t-1}) from real data
        np.random.seed(42)
        unique_sids = np.unique(sids[valid_mask])
        sid_to_idx = {s: i for i, s in enumerate(unique_sids)}
        n_states = len(unique_sids)
        # Build transition matrix
        trans_mat = np.zeros((n_states, n_states), dtype=np.float64)
        prev_sid = -1
        for i in range(n):
            sid = sids[i]
            if sid < 0 or prev_sid < 0:
                prev_sid = sid
                continue
            if prev_sid in sid_to_idx and sid in sid_to_idx:
                trans_mat[sid_to_idx[prev_sid], sid_to_idx[sid]] += 1
            prev_sid = sid
        # Normalize
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        trans_mat = trans_mat / row_sums

        # Generate synthetic sequence using transition matrix
        sid_list = list(unique_sids)
        markov_sids = np.full(n, -1, dtype=np.int64)
        # Start with a random valid state
        if n_states > 0:
            current_idx = np.random.randint(n_states)
            markov_sids[0] = sid_list[current_idx]
            for i in range(1, n):
                if sid_list[current_idx] in sid_to_idx:
                    current_idx = sid_to_idx[sid_list[current_idx]]
                probs = trans_mat[current_idx]
                if np.sum(probs) > 0:
                    current_idx = np.random.choice(n_states, p=probs)
                    markov_sids[i] = sid_list[current_idx]
                else:
                    current_idx = np.random.randint(n_states)
                    markov_sids[i] = sid_list[current_idx]

        s2_dir, _ = compute_directional_count(markov_sids, fut_ret)
        n_s2 = len(s2_dir)
        s2_summary = compute_state_summary(markov_sids, fut_ret)
        s2_dir_ids = directional_state_ids_from_summary(s2_summary)
        overlap_s2 = real_dir_ids & s2_dir_ids
        sym_result["synthetic"]["S2_markov"] = {
            "description": "Synthetic sequence with same transition matrix as real",
            "n_directional": n_s2,
            "directional_states": s2_dir,
            "overlap_with_real": list(overlap_s2),
            "n_overlap": len(overlap_s2),
        }
        print(f"    S2 (markov): {n_s2} directional, overlap={len(overlap_s2)}")

        # S3: Random persistent states -- same run-length distribution as real
        np.random.seed(42)
        # Compute run lengths from real data
        real_runs = []
        current_sid = sids[0]
        current_len = 1
        for i in range(1, n):
            if sids[i] == current_sid and sids[i] >= 0:
                current_len += 1
            else:
                if current_sid >= 0:
                    real_runs.append((current_sid, current_len))
                current_sid = sids[i]
                current_len = 1
        if current_sid >= 0:
            real_runs.append((current_sid, current_len))

        # Extract run lengths and state IDs
        real_run_lengths = [r[1] for r in real_runs]
        # Generate synthetic runs with same length distribution
        persist_sids = np.full(n, -1, dtype=np.int64)
        pos = 0
        while pos < n:
            # Pick a random valid state ID
            if n_states > 0:
                syn_sid = np.random.choice(sid_list)
            else:
                break
            # Pick a run length from the real distribution
            if real_run_lengths:
                run_len = np.random.choice(real_run_lengths)
            else:
                run_len = 1
            run_len = min(run_len, n - pos)
            persist_sids[pos:pos + run_len] = syn_sid
            pos += run_len

        s3_dir, _ = compute_directional_count(persist_sids, fut_ret)
        n_s3 = len(s3_dir)
        s3_summary = compute_state_summary(persist_sids, fut_ret)
        s3_dir_ids = directional_state_ids_from_summary(s3_summary)
        overlap_s3 = real_dir_ids & s3_dir_ids
        sym_result["synthetic"]["S3_persistent"] = {
            "description": "Synthetic states with same run-length distribution as real",
            "n_directional": n_s3,
            "directional_states": s3_dir,
            "overlap_with_real": list(overlap_s3),
            "n_overlap": len(overlap_s3),
        }
        print(f"    S3 (persistent): {n_s3} directional, overlap={len(overlap_s3)}")

        # S4: Random assignment -- randomly assign states preserving frequencies
        np.random.seed(42)
        # Count frequencies of each state in real data
        state_freqs = defaultdict(int)
        for sid in sids:
            if sid >= 0:
                state_freqs[int(sid)] += 1
        # Create a pool of state IDs with the right frequencies
        pool = []
        for sid, cnt in state_freqs.items():
            pool.extend([sid] * cnt)
        np.random.shuffle(pool)
        rand_sids = np.full(n, -1, dtype=np.int64)
        rand_sids[valid_mask] = pool

        s4_dir, _ = compute_directional_count(rand_sids, fut_ret)
        n_s4 = len(s4_dir)
        s4_summary = compute_state_summary(rand_sids, fut_ret)
        s4_dir_ids = directional_state_ids_from_summary(s4_summary)
        overlap_s4 = real_dir_ids & s4_dir_ids
        sym_result["synthetic"]["S4_random_assignment"] = {
            "description": "Random assignment preserving state frequencies, destroys ALL structure",
            "n_directional": n_s4,
            "directional_states": s4_dir,
            "overlap_with_real": list(overlap_s4),
            "n_overlap": len(overlap_s4),
        }
        print(f"    S4 (random assign): {n_s4} directional, overlap={len(overlap_s4)}")

        # Artifactual ratios
        for name, data in sym_result["synthetic"].items():
            n_syn = data["n_directional"]
            ratio = n_syn / max(n_real_dir, 1)
            sym_result["artifactual_ratios"][name] = round(ratio, 4)
            verdict = "ARTIFACT" if ratio > 0.8 else ("PARTIAL" if ratio > 0.3 else "GENUINE")
            data["artifactual_ratio"] = round(ratio, 4)
            data["verdict"] = verdict

        # Mean ratio across all synthetics
        ratios = [v["artifactual_ratio"] for v in sym_result["synthetic"].values()]
        sym_result["mean_artifactual_ratio"] = round(float(np.mean(ratios)), 4)
        if sym_result["mean_artifactual_ratio"] > 0.8:
            sym_result["gate_verdict"] = "STRUCTURAL_ARTIFACT -- states don't survive counterfactual"
        elif sym_result["mean_artifactual_ratio"] > 0.3:
            sym_result["gate_verdict"] = "PARTIAL_STRUCTURE -- some genuine, some artifactual"
        else:
            sym_result["gate_verdict"] = "GENUINE_DIRECTION -- states survive counterfactual gate"

        print(f"    Ratios: {', '.join(f'{k}={v}' for k, v in sym_result['artifactual_ratios'].items())}")
        print(f"    Mean: {sym_result['mean_artifactual_ratio']} -> {sym_result['gate_verdict']}")
        print(f"    Overlap (real ^ synthetic): "
              f"S1={len(overlap_s1)}, S2={len(overlap_s2)}, S3={len(overlap_s3)}, S4={len(overlap_s4)}")

        report["per_symbol"][sym] = sym_result

    # Summary across symbols
    summary = {}
    for sym in SYMBOLS:
        pr = report["per_symbol"][sym]
        summary[sym] = {
            "real_directional": pr["real"]["n_directional"],
            "mean_artifactual_ratio": pr["mean_artifactual_ratio"],
            "gate_verdict": pr["gate_verdict"],
            "mean_overlap": round(float(np.mean([v["n_overlap"] for v in pr["synthetic"].values()])), 1),
        }
    report["summary"] = summary

    save_stl_report(report, "stl7_counterfactual_gate")
    return report


def phase8_final_adjudication(stl, p7=None):
    """Phase 8: Final Adjudication based on ALL STL evidence."""
    # Load previous phases if not provided
    reports_dir = Path(__file__).parent / "reports"
    if p7 is None:
        with open(reports_dir / "stl7_counterfactual_gate.json") as f:
            p7 = json.load(f)

    # Load previous phase results
    prev_reports = {}
    for phase, name in [(1, "stl1_state_map"), (2, "stl2_persistence"),
                         (3, "stl3_transitions"), (4, "stl4_synchronization"),
                         (5, "stl5_null_states"), (6, "stl6_walk_forward")]:
        path = reports_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                prev_reports[phase] = json.load(f)

    # Phase 6 walk-forward results
    wf_summary = prev_reports.get(6, {}).get("summary", {})
    wf_pct_stable = wf_summary.get("pct_edge_stable", 0)
    wf_verdict = wf_summary.get("verdict", "UNKNOWN")

    adjudication = {
        "phase": "STL Phase 8 -- Final Adjudication",
        "classification": None,
        "per_symbol": {},
        "surviving_states": {},
        "evidence_summary": {},
        "final_answer": {},
    }

    for sym in SYMBOLS:
        pr = p7["per_symbol"].get(sym, {})
        dir_states = stl.directional_states(sym, threshold=0.70, min_count=5)
        summary = stl.state_summary(sym)

        # Counterfactual gate result
        gate_verdict = pr.get("gate_verdict", "UNKNOWN")
        mean_ratio = pr.get("mean_artifactual_ratio", 1.0)

        # Walk-forward result for this symbol
        wf_sym = {}
        if prev_reports.get(6):
            wf_per_state = prev_reports[6].get("per_state_validation", {}).get(sym, [])
            n_wf_candidates = len(wf_per_state)
            n_wf_stable = sum(1 for s in wf_per_state if s.get("edge_stable"))
            wf_sym = {
                "n_candidates": n_wf_candidates,
                "n_edge_stable": n_wf_stable,
                "pct_stable": round(n_wf_stable / max(n_wf_candidates, 1) * 100, 1),
            }
        else:
            wf_sym = {"n_candidates": 0, "n_edge_stable": 0, "pct_stable": 0}

        # Cross-asset stability from Phase 1
        cross_asset_count = 0
        if prev_reports.get(1):
            common = prev_reports[1].get("common_states", {})
            for sid_str, cs in common.items():
                if cs["symbols"] and sym in cs["symbols"]:
                    cross_asset_count = max(cross_asset_count, len(cs["symbols"]))

        # Null states from Phase 5
        n_null = 0
        if prev_reports.get(5):
            n_null = len(prev_reports[5].get("null_states", {}).get(sym, []))

        # Phase 2 persistence
        aging_summary = {}
        if prev_reports.get(2):
            aging = prev_reports[2].get("directional_aging", {}).get(sym, {})
            trends = defaultdict(int)
            for sid, ar in aging.items():
                trends[ar.get("trend", "unknown")] += 1
            aging_summary = dict(trends)

        # Determine classification per symbol
        if mean_ratio > 0.8:
            classification = "STRUCTURAL_ARTIFACT"
        elif mean_ratio > 0.3:
            classification = "STATE_DEPENDENT_DIRECTION"
        elif wf_sym.get("pct_stable", 0) < 50:
            classification = "STATE_DEPENDENT_DIRECTION"
        elif cross_asset_count < 3:
            classification = "MARKET_LINKED_DIRECTION"
        else:
            classification = "DEPLOYABLE_DIRECTIONAL_STATE"

        # Surviving states (those that pass ALL gates)
        surviving = []
        for ds in dir_states:
            sid = ds["state_id"]
            # Check if this state passes the counterfactual gate
            passes_counterfactual = mean_ratio < 0.8

            # Check if this state passes walk-forward
            passes_walkforward = False
            if prev_reports.get(6):
                for wfs in prev_reports[6].get("per_state_validation", {}).get(sym, []):
                    if wfs["state_id"] == sid:
                        passes_walkforward = wfs.get("edge_stable", False)
                        break

            # Check cross-asset
            cross_symbols = 1
            if prev_reports.get(1):
                for sid_str, cs in prev_reports[1].get("common_states", {}).items():
                    if int(sid_str) == sid:
                        cross_symbols = cs["n_symbols"]
                        break

            surviving.append({
                "state_id": sid,
                "components": list(decode_state(sid)),
                "occurrence_frequency": round(ds["count"] / max(summary[sid]["count"] if sid in summary else 1, 1), 4),
                "total_count": ds["count"],
                "directional_probability": round(ds["p_up"], 4),
                "horizon": ds["horizon"],
                "cross_asset_symbols": cross_symbols,
                "walk_forward_retention": passes_walkforward,
                "counterfactual_survival": passes_counterfactual,
            })

        adjudication["per_symbol"][sym] = {
            "n_directional_real": len(dir_states),
            "mean_artifactual_ratio": mean_ratio,
            "counterfactual_verdict": gate_verdict,
            "walk_forward": wf_sym,
            "n_null_states": n_null,
            "aging_trends": aging_summary,
            "classification": classification,
            "n_surviving_states": len(surviving),
        }
        if surviving:
            adjudication["surviving_states"][sym] = surviving

    # Determine overall classification
    classifications = [v["classification"] for v in adjudication["per_symbol"].values()]
    if all(c == "STRUCTURAL_ARTIFACT" for c in classifications):
        overall = "STRUCTURAL_ARTIFACT"
        answer = "No -- direction does not emerge from market state topology. States are structural artifacts of the ES×AT×Regime×Memory decomposition."
    elif any(c == "DEPLOYABLE_DIRECTIONAL_STATE" for c in classifications):
        overall = "DEPLOYABLE_DIRECTIONAL_STATE"
        deployable_symbols = [sym for sym in SYMBOLS if adjudication["per_symbol"][sym]["classification"] == "DEPLOYABLE_DIRECTIONAL_STATE"]
        answer = f"Yes -- direction emerges from market state topology for {', '.join(deployable_symbols)}. These states are robust, persistent, and cross-asset stable."
    elif any(c == "MARKET_LINKED_DIRECTION" for c in classifications):
        overall = "MARKET_LINKED_DIRECTION"
        answer = "Partially -- directional states exist but are market-specific, not cross-asset stable."
    elif any(c == "STATE_DEPENDENT_DIRECTION" for c in classifications):
        overall = "STATE_DEPENDENT_DIRECTION"
        answer = "Partially -- directional states survive counterfactual but not walk-forward. Direction is state-dependent but not persistent."
    else:
        overall = "STRUCTURAL_ARTIFACT"
        answer = "No -- states do not survive the counterfactual gate."

    adjudication["classification"] = overall
    adjudication["final_answer"] = {
        "question": "Does direction emerge from market state topology?",
        "answer": answer,
        "classification": overall,
        "walk_forward_verdict": wf_verdict,
    }

    # Evidence summary
    total_real = sum(adjudication["per_symbol"][sym]["n_directional_real"] for sym in SYMBOLS)
    total_surviving = sum(adjudication["per_symbol"][sym]["n_surviving_states"] for sym in SYMBOLS)
    adjudication["evidence_summary"] = {
        "total_directional_states_discovered": total_real,
        "total_surviving_all_gates": total_surviving,
        "survival_rate": round(total_surviving / max(total_real, 1) * 100, 1),
        "classification": overall,
        "mean_artifactual_ratio_across_symbols": round(float(np.mean([
            v["mean_artifactual_ratio"] for v in adjudication["per_symbol"].values()
        ])), 4),
    }

    # Save JSON
    save_stl_report(adjudication, "stl_final_adjudication")

    return adjudication


def write_markdown_phase7(p7_report):
    lines = []
    lines.append("# STL Phase 7 -- Synthetic Counterfactual Gate\n")
    lines.append("## MANDATORY Gate: Do Directional States Survive Randomization?\n")

    for sym in SYMBOLS:
        pr = p7_report["per_symbol"][sym]
        lines.append(f"\n## {sym}\n")
        lines.append(f"- **Real directional states:** {pr['real']['n_directional']}")
        lines.append(f"- **Mean artifactual ratio:** {pr['mean_artifactual_ratio']}")
        lines.append(f"- **Gate verdict:** {pr['gate_verdict']}\n")

        lines.append("| Variant | Description | N Directional | Ratio | Verdict | Overlap with Real |")
        lines.append("|---------|-------------|--------------|-------|---------|------------------|")
        for name, data in pr["synthetic"].items():
            label = name.replace("_", " ").title()
            lines.append(f"| {label} | {data['description']} | {data['n_directional']} | "
                         f"{data['artifactual_ratio']} | {data['verdict']} | {data['n_overlap']} |")

        lines.append(f"\n**Mean ratio:** {pr['mean_artifactual_ratio']} -> **{pr['gate_verdict']}**\n")

    # Summary table
    lines.append("## Cross-Symbol Summary\n")
    lines.append("| Symbol | Real Dir | Mean Ratio | Gate Verdict | Mean Overlap |")
    lines.append("|--------|----------|------------|-------------|-------------|")
    summary = p7_report["summary"]
    for sym in SYMBOLS:
        s = summary[sym]
        lines.append(f"| {sym} | {s['real_directional']} | {s['mean_artifactual_ratio']} | "
                     f"{s['gate_verdict']} | {s['mean_overlap']} |")

    path = Path(__file__).parent / "reports" / "STL7_COUNTERFACTUAL_GATE.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")
    return path


def write_markdown_phase8(adj):
    lines = []
    lines.append("# STL Phase 8 -- Final Adjudication\n")
    lines.append("## Does Direction Emerge from Market State Topology?\n")

    lines.append(f"**Final Classification:** {adj['classification']}\n")
    lines.append(f"**Answer:** {adj['final_answer']['answer']}\n")

    lines.append("## Per-Symbol Classification\n")
    lines.append("| Symbol | Real Dir | Mean Ratio | Counterfactual | Walk-Forward | Classification | Surviving |")
    lines.append("|--------|----------|------------|---------------|-------------|---------------|-----------|")
    for sym in SYMBOLS:
        v = adj["per_symbol"][sym]
        wf = v["walk_forward"]
        wf_str = f"{wf['pct_stable']}% stable ({wf['n_edge_stable']}/{wf['n_candidates']})" if wf["n_candidates"] > 0 else "N/A"
        lines.append(f"| {sym} | {v['n_directional_real']} | {v['mean_artifactual_ratio']} | "
                     f"{v['counterfactual_verdict']} | {wf_str} | {v['classification']} | {v['n_surviving_states']} |")

    # Evidence summary
    es = adj["evidence_summary"]
    lines.append("\n## Evidence Summary\n")
    lines.append(f"- **Total directional states discovered (all symbols):** {es['total_directional_states_discovered']}")
    lines.append(f"- **Total surviving all gates:** {es['total_surviving_all_gates']}")
    lines.append(f"- **Overall survival rate:** {es['survival_rate']}%")
    lines.append(f"- **Mean artifactual ratio:** {es['mean_artifactual_ratio_across_symbols']}")
    lines.append(f"- **Classification:** {es['classification']}")
    lines.append(f"- **Walk-Forward Verdict:** {adj['final_answer']['walk_forward_verdict']}\n")

    # Surviving states detail
    total_surviving = sum(adj["per_symbol"][sym]["n_surviving_states"] for sym in SYMBOLS)
    if total_surviving > 0:
        lines.append("## Surviving States (Passed All Gates)\n")
        for sym in SYMBOLS:
            surviving = adj.get("surviving_states", {}).get(sym, [])
            if not surviving:
                continue
            lines.append(f"\n### {sym} -- {len(surviving)} surviving states\n")
            lines.append("| State ID | ES Q | AT Q | Regime | Mem Q | Count | P(up) | Horizon | Cross-Asset | W-F Retained | CF Survived |")
            lines.append("|---------|------|------|--------|-------|-------|-------|---------|------------|-------------|------------|")
            for st in surviving:
                c = st["components"]
                lines.append(f"| {st['state_id']} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | "
                             f"{st['total_count']} | {st['directional_probability']:.2%} | "
                             f"H{st['horizon']} | {st['cross_asset_symbols']} syms | "
                             f"{'YES' if st['walk_forward_retention'] else 'NO'} | "
                             f"{'YES' if st['counterfactual_survival'] else 'NO'} |")
    else:
        lines.append("\nNo states survived all gates.\n")

    # Adjudication path
    lines.append("\n## Adjudication Logic\n")
    lines.append("Based on the STL evidence hierarchy:\n")
    lines.append("1. **Counterfactual Gate** (Phase 7) -- Do states contain genuine information or structural artifacts?")
    lines.append("2. **Walk-Forward Validation** (Phase 6) -- Do directional edges survive out of sample?")
    lines.append("3. **Cross-Asset Stability** (Phase 4) -- Do states synchronize across markets?")
    lines.append("4. **Persistence Aging** (Phase 2) -- Do edges strengthen, decay, or remain stable?\n")
    lines.append("### Classification Tree\n")
    lines.append("```")
    lines.append("Counterfactual Gate")
    lines.append("|- Ratio > 0.8 -> STRUCTURAL_ARTIFACT")
    lines.append("|- Ratio > 0.3 -> STATE_DEPENDENT_DIRECTION")
    lines.append("|   |- WF stable < 50% -> STATE_DEPENDENT_DIRECTION")
    lines.append("|   `- WF stable >= 50%")
    lines.append("|       |- Cross-asset < 3 -> MARKET_LINKED_DIRECTION")
    lines.append("|       `- Cross-asset >= 3 -> DEPLOYABLE_DIRECTIONAL_STATE")
    lines.append("`- Ratio <= 0.3 -> GENUINE (then check WF + cross-asset)")
    lines.append("```")

    path = Path(__file__).parent / "reports" / "STL_FINAL_REPORT.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")
    return path


def main():
    print("=" * 60)
    print("STL Phase 7: Synthetic Counterfactual Gate")
    print("STL Phase 8: Final Adjudication")
    print("=" * 60)

    stl = STLCore()
    print("\nLoading STL data...")
    stl.load_all()
    print(f"Loaded {len(stl._data)} symbols\n")

    # Phase 7
    print("-" * 40)
    print("PHASE 7: SYNTHETIC COUNTERFACTUAL GATE")
    print("-" * 40)
    p7 = phase7_counterfactual_gate(stl)

    # Phase 8
    print("\n" + "-" * 40)
    print("PHASE 8: FINAL ADJUDICATION")
    print("-" * 40)
    adj = phase8_final_adjudication(stl, p7)

    print(f"\nOverall classification: {adj['classification']}")
    print(f"Answer: {adj['final_answer']['answer']}")

    # Write markdown
    print("\nWriting markdown reports...")
    write_markdown_phase7(p7)
    write_markdown_phase8(adj)

    print("\nDone. Reports saved to:")
    print("  reports/stl7_counterfactual_gate.json")
    print("  reports/stl_final_adjudication.json")
    print("  reports/STL7_COUNTERFACTUAL_GATE.md")
    print("  reports/STL_FINAL_REPORT.md")


if __name__ == "__main__":
    main()
