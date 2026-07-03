"""STL Phase 1: State Map + Phase 2: Persistence Physics."""
import sys, json
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.state_topology.stl_core import STLCore, SYMBOLS, decode_state, save_stl_report, MAX_STATES
from collections import defaultdict


def get_run_lengths(arr):
    """Get run lengths of consecutive identical state IDs."""
    if len(arr) == 0:
        return []
    runs = []
    current_val = arr[0]
    current_len = 1
    for i in range(1, len(arr)):
        if arr[i] == current_val and arr[i] >= 0:
            current_len += 1
        else:
            if current_val >= 0:
                runs.append((int(current_val), current_len))
            current_val = arr[i]
            current_len = 1
    if current_val >= 0:
        runs.append((int(current_val), current_len))
    return runs


def survival_probability(run_lengths, max_n=100):
    """P(state unchanged after N bars) = fraction of runs with length >= N."""
    if not run_lengths:
        return {}
    lengths = np.array([r[1] for r in run_lengths])
    surv = {}
    for n in range(1, max_n + 1):
        surv[n] = float(np.mean(lengths >= n))
    return surv


def split_run_segments(run, arr_sids, arr_fut_ret, h):
    """Split a single run into early (first 25%), middle (25-75%), late (last 25%)."""
    sid, length = run
    # Need to find the actual positions in the array for this run
    # We'll do this differently - pass in the slice directly
    pass


def phase1_state_map(stl):
    """Phase 1: Complete state-space topology reconstruction."""
    report = {
        "summary_per_symbol": {},
        "directional_states": {},
        "strongly_directional_75": {},
        "strongly_directional_80": {},
        "common_states": {},
        "temporal_stability": {},
    }

    # Track state_ids across symbols for cross-asset analysis
    state_cross_asset = defaultdict(list)  # state_id -> [(symbol, p_up_H50, count)]

    for sym in SYMBOLS:
        summary = stl.state_summary(sym)
        sids_arr = stl._data[sym]["state_id"]
        fut_ret = stl._data[sym]["fut_ret"]
        n = len(sids_arr)
        valid_mask = sids_arr >= 0
        n_valid = int(np.sum(valid_mask))

        # Unique states
        unique_sids = np.unique(sids_arr[valid_mask])
        n_unique = len(unique_sids)

        # State frequency ranking
        state_counts = [(int(sid), summary[sid]["count"]) for sid in unique_sids]
        state_counts.sort(key=lambda x: -x[1])
        top_10_freq = [{"state_id": s, "count": c, "components": list(decode_state(s))} for s, c in state_counts[:10]]

        # Directional states
        dir_70 = stl.directional_states(sym, threshold=0.70, min_count=5)
        dir_75 = stl.directional_states(sym, threshold=0.75, min_count=5)
        dir_80 = stl.directional_states(sym, threshold=0.80, min_count=5)

        # By horizon
        dir_by_horizon = {h: {"states": [], "n": 0} for h in [5, 20, 50, 100]}
        for ds in dir_70:
            h = ds["horizon"]
            dir_by_horizon[h]["states"].append(ds)
            dir_by_horizon[h]["n"] += 1

        report["summary_per_symbol"][sym] = {
            "total_bars": n,
            "valid_bars": n_valid,
            "pct_valid": round(n_valid / n * 100, 2),
            "unique_states": n_unique,
            "max_possible_states": MAX_STATES,
            "coverage_pct": round(n_unique / MAX_STATES * 100, 2),
            "top_10_frequent_states": top_10_freq,
            "directional_70_total": len(dir_70),
            "directional_75_total": len(dir_75),
            "directional_80_total": len(dir_80),
            "directional_by_horizon": {str(h): v["n"] for h, v in dir_by_horizon.items()},
        }

        report["directional_states"][sym] = [
            {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
             for k, v in d.items()}
            for d in dir_70
        ]
        report["strongly_directional_75"][sym] = [
            {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
             for k, v in d.items()}
            for d in dir_75
        ]
        report["strongly_directional_80"][sym] = [
            {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
             for k, v in d.items()}
            for d in dir_80
        ]

        # Cross-asset tracking
        for sid in unique_sids:
            info = summary[int(sid)]
            p_up_h50 = info["p_up"].get(50, None)
            state_cross_asset[int(sid)].append({
                "symbol": sym,
                "p_up_H50": round(p_up_h50, 4) if p_up_h50 else None,
                "count": info["count"],
                "components": list(decode_state(int(sid))),
            })

        # Temporal stability: first half vs second half
        mid = n // 2
        first_half_sids = sids_arr[:mid]
        second_half_sids = sids_arr[mid:]
        first_half_fut = fut_ret[:mid]
        second_half_fut = fut_ret[mid:]

        temporal_states = {}
        for sid in unique_sids:
            sid_int = int(sid)
            info = summary[sid_int]
            if info["count"] < 20:
                continue
            p_up_h50 = info["p_up"].get(50, None)
            if p_up_h50 is None:
                continue

            # Compute P(up) for first half
            idx_f = np.where(first_half_sids == sid_int)[0]
            n_f = len(idx_f)
            up_f = 0
            for idx in idx_f:
                ret = first_half_fut[idx, 2]  # H50 index = 2 (since fut_ret is [1,5,20,50,100,500])
                if not np.isnan(ret) and ret > 0:
                    up_f += 1
            p_up_f = up_f / n_f if n_f > 0 else None

            # Second half
            idx_s = np.where(second_half_sids == sid_int)[0]
            n_s = len(idx_s)
            up_s = 0
            for idx in idx_s:
                ret = second_half_fut[idx, 2]
                if not np.isnan(ret) and ret > 0:
                    up_s += 1
            p_up_s = up_s / n_s if n_s > 0 else None

            if p_up_f is not None and p_up_s is not None:
                diff = abs(p_up_f - p_up_s)
                temporal_states[sid_int] = {
                    "state_id": sid_int,
                    "components": list(decode_state(sid_int)),
                    "count": info["count"],
                    "p_up_H50_first_half": round(p_up_f, 4),
                    "p_up_H50_second_half": round(p_up_s, 4),
                    "abs_diff": round(diff, 4),
                    "stable": diff < 0.10,
                }

        report["temporal_stability"][sym] = temporal_states

    # Cross-asset common states
    common_states = {}
    for sid, entries in state_cross_asset.items():
        if len(entries) >= 3:
            components = entries[0]["components"]
            common_states[sid] = {
                "state_id": sid,
                "components": components,
                "n_symbols": len(entries),
                "symbols": [e["symbol"] for e in entries],
                "details": entries,
            }
    report["common_states"] = common_states

    # Summary stats
    all_dir_counts = {}
    for sym in SYMBOLS:
        all_dir_counts[sym] = {
            "directional_70": report["summary_per_symbol"][sym]["directional_70_total"],
            "directional_75": report["summary_per_symbol"][sym]["directional_75_total"],
            "directional_80": report["summary_per_symbol"][sym]["directional_80_total"],
        }
    report["summary_all_symbols"] = all_dir_counts
    report["n_common_states_across_3plus"] = len(common_states)

    save_stl_report(report, "stl1_state_map")
    return report


def phase2_persistence(stl, phase1_report):
    """Phase 2: State persistence physics."""
    report = {
        "run_statistics": {},
        "survival_probabilities": {},
        "directional_aging": {},
    }

    for sym in SYMBOLS:
        sids_arr = stl._data[sym]["state_id"]
        fut_ret = stl._data[sym]["fut_ret"]

        runs = get_run_lengths(sids_arr.flatten() if sids_arr.ndim > 1 else sids_arr)

        # Per-state run statistics
        state_runs = defaultdict(list)
        for sid, length in runs:
            state_runs[sid].append(length)

        state_stats = {}
        for sid, lengths in state_runs.items():
            state_stats[sid] = {
                "state_id": sid,
                "components": list(decode_state(sid)),
                "n_runs": len(lengths),
                "avg_duration": round(float(np.mean(lengths)), 2),
                "median_duration": int(np.median(lengths)),
                "max_duration": int(np.max(lengths)),
                "min_duration": int(np.min(lengths)),
                "std_duration": round(float(np.std(lengths)), 2),
            }

        # All-run statistics (pooled)
        all_lengths = [l for _, l in runs]
        pool_stats = {}
        if all_lengths:
            pool_stats = {
                "n_runs": len(all_lengths),
                "avg_duration": round(float(np.mean(all_lengths)), 2),
                "median_duration": int(np.median(all_lengths)),
                "max_duration": int(np.max(all_lengths)),
                "min_duration": int(np.min(all_lengths)),
                "std_duration": round(float(np.std(all_lengths)), 2),
                "pct_runs_gt_1": round(np.mean(np.array(all_lengths) > 1) * 100, 2),
                "pct_runs_gt_5": round(np.mean(np.array(all_lengths) > 5) * 100, 2),
                "pct_runs_gt_10": round(np.mean(np.array(all_lengths) > 10) * 100, 2),
            }

        report["run_statistics"][sym] = {
            "pooled": pool_stats,
            "per_state": state_stats,
        }

        # Survival probability
        surv = survival_probability(runs, max_n=50)
        # Key points
        key_survival = {n: round(surv.get(n, 0), 4) for n in [1, 2, 3, 5, 10, 20, 50]}
        report["survival_probabilities"][sym] = {
            "full": {str(k): round(v, 4) for k, v in surv.items()},
            "key_points": key_survival,
        }

        # Directional aging: for strongly directional states (>0.70 at H50)
        # Find directional states from phase1
        summary = stl.state_summary(sym)
        dir_h50 = []
        for sid, info in summary.items():
            if info["count"] < 10:
                continue
            p = info["p_up"].get(50, None)
            if p is not None and p >= 0.70:
                dir_h50.append(int(sid))

        aging_results = {}
        for sid in dir_h50:
            # Find all runs for this state
            state_mask = sids_arr == sid
            # Find run boundaries
            padded = np.concatenate([[False], state_mask, [False]])
            starts = np.where(padded[1:] & ~padded[:-1])[0]
            ends = np.where(~padded[1:] & padded[:-1])[0]
            run_slices = [(s, e) for s, e in zip(starts, ends) if e - s >= 5]

            early_rets = []
            mid_rets = []
            late_rets = []

            for s, e in run_slices:
                length = e - s
                if length < 6:
                    continue
                early_end = s + max(1, int(length * 0.25))
                mid_end = s + max(1, int(length * 0.75))

                for idx in range(s, early_end):
                    r = fut_ret[idx, 2]
                    if not np.isnan(r):
                        early_rets.append(r)
                for idx in range(early_end, mid_end):
                    r = fut_ret[idx, 2]
                    if not np.isnan(r):
                        mid_rets.append(r)
                for idx in range(mid_end, e):
                    r = fut_ret[idx, 2]
                    if not np.isnan(r):
                        late_rets.append(r)

            def p_up(rets):
                return round(np.mean(np.array(rets) > 0), 4) if len(rets) > 0 else None

            p_early = p_up(early_rets)
            p_mid = p_up(mid_rets)
            p_late = p_up(late_rets)

            n_early = len(early_rets)
            n_mid = len(mid_rets)
            n_late = len(late_rets)

            # Determine trend
            if p_early is not None and p_mid is not None and p_late is not None:
                if p_late > p_early + 0.05:
                    trend = "strengthening"
                elif p_late < p_early - 0.05:
                    trend = "decaying"
                else:
                    trend = "stable"
            else:
                trend = "unknown"

            aging_results[sid] = {
                "state_id": sid,
                "components": list(decode_state(sid)),
                "n_runs_analyzed": len(run_slices),
                "n_early": n_early,
                "n_mid": n_mid,
                "n_late": n_late,
                "p_up_early": p_early,
                "p_up_mid": p_mid,
                "p_up_late": p_late,
                "trend": trend,
                "total_count": summary[sid]["count"],
            }

        report["directional_aging"][sym] = aging_results

    # Aggregated aging summary
    all_trends = defaultdict(int)
    for sym in SYMBOLS:
        for sid, ar in report["directional_aging"][sym].items():
            all_trends[ar["trend"]] += 1
    report["aging_trend_summary"] = dict(all_trends)

    save_stl_report(report, "stl2_persistence")
    return report


def write_markdown_phase1(phase1_report):
    """Write Phase 1 markdown summary."""
    lines = []
    lines.append("# STL Phase 1 â€” State Topology Reconstruction\n")
    lines.append("## State Map: ES Ã— AT Ã— Regime Ã— Memory\n")

    # Summary table
    lines.append("### Per-Symbol Summary\n")
    lines.append("| Symbol | Total Bars | Valid % | Unique States | Coverage % | Dirâ‰¥0.70 | Dirâ‰¥0.75 | Dirâ‰¥0.80 |")
    lines.append("|--------|-----------|---------|--------------|-----------|---------|---------|---------|")
    for sym in SYMBOLS:
        s = phase1_report["summary_per_symbol"][sym]
        lines.append(f"| {sym} | {s['total_bars']} | {s['pct_valid']}% | {s['unique_states']} | {s['coverage_pct']}% | "
                     f"{s['directional_70_total']} | {s['directional_75_total']} | {s['directional_80_total']} |")

    # Top frequent states
    lines.append("\n### Most Frequent States (Top 3 per Symbol)\n")
    for sym in SYMBOLS:
        lines.append(f"\n#### {sym}")
        top3 = phase1_report["summary_per_symbol"][sym]["top_10_frequent_states"][:3]
        lines.append("| State ID | ES Q | AT Q | Regime | Mem Q | Count |")
        lines.append("|---------|------|------|--------|-------|-------|")
        for t in top3:
            c = t["components"]
            lines.append(f"| {t['state_id']} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {t['count']} |")

    # Directional states
    lines.append("\n### Directional States (P(up) â‰¥ 0.70 by Horizon)\n")
    for sym in SYMBOLS:
        ds = phase1_report["directional_states"][sym]
        if ds:
            lines.append(f"\n#### {sym} â€” {len(ds)} states")
            lines.append("| State ID | ES Q | AT Q | Regime | Mem Q | Count | P(up) | Horizon |")
            lines.append("|---------|------|------|--------|-------|-------|-------|---------|")
            for d in ds[:10]:
                lines.append(f"| {d['state_id']} | {d['es_q']} | {d['at_q']} | {d['regime']} | {d['mem_q']} | "
                             f"{d['count']} | {d['p_up']:.2%} | H{d['horizon']} |")
            if len(ds) > 10:
                lines.append(f"  *... and {len(ds) - 10} more*")

    # Strongly directional
    for threshold, key in [(0.75, "strongly_directional_75"), (0.80, "strongly_directional_80")]:
        lines.append(f"\n### Strongly Directional States (P(up) â‰¥ {threshold})\n")
        for sym in SYMBOLS:
            ds = phase1_report[key][sym]
            if ds:
                lines.append(f"- {sym}: {len(ds)} states")
                for d in ds[:5]:
                    lines.append(f"  - State {d['state_id']}: P(up)={d['p_up']:.2%} @ H{d['horizon']}, "
                                 f"count={d['count']} (ES{d['es_q']} AT{d['at_q']} R{d['regime']} M{d['mem_q']})")

    # Common states
    lines.append(f"\n### Cross-Asset Common States (3+ symbols)")
    lines.append(f"Total: {phase1_report['n_common_states_across_3plus']}\n")
    for sid, cs in sorted(phase1_report["common_states"].items(), key=lambda x: -x[1]["n_symbols"]):
        comp = cs["components"]
        lines.append(f"- State {sid} (ES{comp[0]} AT{comp[1]} R{comp[2]} M{comp[3]}): "
                     f"present in {cs['n_symbols']} symbols: {', '.join(cs['symbols'])}")

    # Temporal stability
    lines.append("\n### Temporal Stability (First Half vs Second Half)\n")
    for sym in SYMBOLS:
        ts = phase1_report["temporal_stability"][sym]
        stable = sum(1 for v in ts.values() if v["stable"])
        unstable = sum(1 for v in ts.values() if not v["stable"])
        lines.append(f"- {sym}: {stable} stable, {unstable} unstable states (diff < 0.10)")
        # Show most unstable
        unstable_states = [(k, v) for k, v in ts.items() if not v["stable"]]
        unstable_states.sort(key=lambda x: -x[1]["abs_diff"])
        if unstable_states:
            lines.append("  *Most unstable:*")
            for sid, v in unstable_states[:3]:
                lines.append(f"    - State {sid}: first={v['p_up_H50_first_half']:.2%}, "
                             f"second={v['p_up_H50_second_half']:.2%}, diff={v['abs_diff']:.2%}")

    path = Path(__file__).parent / "reports" / "STL1_STATE_MAP.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")
    return path


def write_markdown_phase2(phase2_report):
    """Write Phase 2 markdown summary."""
    lines = []
    lines.append("# STL Phase 2 â€” State Persistence Physics\n")
    lines.append("## Run Lengths, Survival, and Directional Aging\n")

    # Run statistics
    lines.append("### Run Length Statistics (Pooled)\n")
    lines.append("| Symbol | N Runs | Avg | Median | Max | Std | >1 bar % | >5 bar % | >10 bar % |")
    lines.append("|--------|--------|-----|--------|-----|-----|----------|----------|-----------|")
    for sym in SYMBOLS:
        p = phase2_report["run_statistics"][sym]["pooled"]
        if p:
            lines.append(f"| {sym} | {p['n_runs']} | {p['avg_duration']} | {p['median_duration']} | "
                         f"{p['max_duration']} | {p['std_duration']} | {p['pct_runs_gt_1']}% | "
                         f"{p['pct_runs_gt_5']}% | {p['pct_runs_gt_10']}% |")
        else:
            lines.append(f"| {sym} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

    # Survival probabilities
    lines.append("\n### Survival Probability (State Unchanged After N Bars)\n")
    lines.append("| Symbol | 1 bar | 2 bars | 3 bars | 5 bars | 10 bars | 20 bars | 50 bars |")
    lines.append("|--------|-------|--------|--------|--------|---------|---------|---------|")
    for sym in SYMBOLS:
        sp = phase2_report["survival_probabilities"][sym]["key_points"]
        lines.append(f"| {sym} | {sp.get(1, 'N/A'):.1%} | {sp.get(2, 'N/A'):.1%} | "
                     f"{sp.get(3, 'N/A'):.1%} | {sp.get(5, 'N/A'):.1%} | "
                     f"{sp.get(10, 'N/A'):.1%} | {sp.get(20, 'N/A'):.1%} | "
                     f"{sp.get(50, 'N/A'):.1%} |")

    # Top longest-lived states
    lines.append("\n### Longest-Lived States (Top 5 per Symbol by Avg Duration)\n")
    for sym in SYMBOLS:
        ps = phase2_report["run_statistics"][sym]["per_state"]
        if ps:
            sorted_states = sorted(ps.values(), key=lambda x: -x["avg_duration"])[:5]
            lines.append(f"\n#### {sym}")
            lines.append("| State ID | ES Q | AT Q | Regime | Mem Q | Avg | Median | Max | N Runs |")
            lines.append("|---------|------|------|--------|-------|-----|--------|-----|--------|")
            for s in sorted_states:
                c = s["components"]
                lines.append(f"| {s['state_id']} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | "
                             f"{s['avg_duration']} | {s['median_duration']} | {s['max_duration']} | {s['n_runs']} |")

    # Directional aging
    lines.append("\n### Directional Aging\n")
    lines.append("Does directional strength increase, decrease, or stay constant with age?\n")

    trend_summary = phase2_report["aging_trend_summary"]
    total_aged = sum(trend_summary.values())
    lines.append(f"**States analyzed:** {total_aged}\n")
    lines.append("| Trend | Count | % |")
    lines.append("|-------|-------|---|")
    for trend in ["strengthening", "stable", "decaying", "unknown"]:
        cnt = trend_summary.get(trend, 0)
        pct = cnt / total_aged * 100 if total_aged > 0 else 0
        lines.append(f"| {trend} | {cnt} | {pct:.1f}% |")

    lines.append("\n\n### Per-Symbol Aging Detail\n")
    for sym in SYMBOLS:
        aging = phase2_report["directional_aging"][sym]
        if not aging:
            lines.append(f"\n#### {sym} â€” No directional states with sufficient runs\n")
            continue
        lines.append(f"\n#### {sym} â€” {len(aging)} directional states\n")
        lines.append("| State ID | ES Q | AT Q | Regime | Mem Q | P(up) Early | P(up) Mid | P(up) Late | Trend | N Runs |")
        lines.append("|---------|------|------|--------|-------|------------|----------|-----------|-------|--------|")
        sorted_age = sorted(aging.values(), key=lambda x: -x["n_runs_analyzed"])
        for ar in sorted_age[:10]:
            c = ar["components"]
            p_early = f"{ar['p_up_early']:.1%}" if ar['p_up_early'] is not None else "N/A"
            p_mid = f"{ar['p_up_mid']:.1%}" if ar['p_up_mid'] is not None else "N/A"
            p_late = f"{ar['p_up_late']:.1%}" if ar['p_up_late'] is not None else "N/A"
            lines.append(f"| {ar['state_id']} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | "
                         f"{p_early} | {p_mid} | {p_late} | {ar['trend']} | {ar['n_runs_analyzed']} |")

    path = Path(__file__).parent / "reports" / "STL2_PERSISTENCE.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")
    return path


def main():
    print("=" * 60)
    print("STL Phase 1: State Topology Reconstruction")
    print("STL Phase 2: State Persistence Physics")
    print("=" * 60)

    stl = STLCore()
    print("Loading data...")
    stl.load_all()
    print(f"Loaded {len(stl._data)} symbols\n")

    # Phase 1
    print("-" * 40)
    print("PHASE 1: STATE MAP")
    print("-" * 40)
    p1 = phase1_state_map(stl)

    for sym in SYMBOLS:
        s = p1["summary_per_symbol"][sym]
        print(f"{sym}: {s['unique_states']} unique states ({s['coverage_pct']}% coverage), "
              f"{s['directional_70_total']} directional (â‰¥0.70), {s['directional_80_total']} (â‰¥0.80)")
    print(f"Common states across 3+ symbols: {p1['n_common_states_across_3plus']}")

    # Phase 2
    print("\n" + "-" * 40)
    print("PHASE 2: PERSISTENCE PHYSICS")
    print("-" * 40)
    p2 = phase2_persistence(stl, p1)

    for sym in SYMBOLS:
        p = p2["run_statistics"][sym]["pooled"]
        if p:
            n_dir = len(p2["directional_aging"][sym])
            print(f"{sym}: avg run={p['avg_duration']} bars, median={p['median_duration']}, "
                  f"max={p['max_duration']}, directional aged={n_dir}")

    aging = p2["aging_trend_summary"]
    print(f"\nDirectional aging: {aging}")

    # Write markdown
    print("\nWriting markdown reports...")
    write_markdown_phase1(p1)
    write_markdown_phase2(p2)

    print("\nDone. Reports saved to reports/stl1_state_map.json, reports/stl2_persistence.json")
    print("Markdown: reports/STL1_STATE_MAP.md, reports/STL2_PERSISTENCE.md")


if __name__ == "__main__":
    main()
