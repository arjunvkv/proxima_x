"""STL Phase 5: Anti-Directional Null States + Phase 6: Walk-Forward Validation."""
import sys, json
from pathlib import Path
import numpy as np
from collections import defaultdict

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.state_topology.stl_core import STLCore, SYMBOLS, decode_state, save_stl_report
from research.directional_state.dsr_core import WalkForwardValidator, DSRCore


def phase5_null_states(stl):
    """Phase 5: Anti-Directional Null States analysis."""
    report = {
        "null_states": {},
        "anti_directional_states": {},
        "null_state_characterization": {},
        "cross_asset_null_consistency": {},
        "summary": {},
    }

    for sym in SYMBOLS:
        nulls = stl.null_states(sym, lower=0.45, upper=0.55, min_count=10)

        summary = stl.state_summary(sym)
        anti_dir = []
        for sid, info in summary.items():
            if info["count"] < 10:
                continue
            for h in [5, 20, 50, 100]:
                p = info["p_up"].get(h)
                if p is not None and (p < 0.30 or p > 0.70):
                    direction = "down" if p < 0.30 else "up"
                    anti_dir.append({
                        "state_id": int(sid),
                        "es_q": info["es_q"], "at_q": info["at_q"],
                        "regime": info["regime"], "mem_q": info["mem_q"],
                        "count": info["count"], "p_up": round(p, 4),
                        "horizon": h, "symbol": sym, "direction": direction,
                    })
                    break

        report["null_states"][sym] = [
            {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
             for k, v in n.items()}
            for n in nulls
        ]
        report["anti_directional_states"][sym] = [
            {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
             for k, v in a.items()}
            for a in anti_dir
        ]

        dim_counts = {"es_q": defaultdict(int), "at_q": defaultdict(int),
                      "regime": defaultdict(int), "mem_q": defaultdict(int)}
        for n in nulls:
            dim_counts["es_q"][n["es_q"]] += 1
            dim_counts["at_q"][n["at_q"]] += 1
            dim_counts["regime"][n["regime"]] += 1
            dim_counts["mem_q"][n["mem_q"]] += 1
        dim_pcts = {}
        for dim, counts in dim_counts.items():
            total = sum(counts.values())
            dim_pcts[dim] = {str(k): round(v / total * 100, 1) if total > 0 else 0
                             for k, v in sorted(counts.items())}

        null_combo_counts = defaultdict(int)
        for n in nulls:
            null_combo_counts[(n["es_q"], n["at_q"], n["regime"], n["mem_q"])] += 1
        top_null_combos = [
            {"components": list(c), "count": cnt,
             "pct_of_nulls": round(cnt / max(len(nulls), 1) * 100, 1)}
            for c, cnt in sorted(null_combo_counts.items(), key=lambda x: -x[1])[:5]
        ]

        report["null_state_characterization"][sym] = {
            "n_null_states": len(nulls),
            "dimension_pcts": dim_pcts,
            "top_null_combos": top_null_combos,
            "avg_null_p_up": round(float(np.mean([n["p_up"] for n in nulls])), 4) if nulls else None,
        }

    null_by_state_id = defaultdict(list)
    for sym in SYMBOLS:
        for n in report["null_states"][sym]:
            null_by_state_id[n["state_id"]].append({
                "symbol": sym, "p_up": n["p_up"],
                "horizon": n["horizon"], "count": n["count"],
            })
    cross_null = {}
    for sid, entries in null_by_state_id.items():
        if len(entries) >= 3:
            cross_null[sid] = {
                "state_id": sid, "components": list(decode_state(sid)),
                "n_symbols": len(entries),
                "symbols": [e["symbol"] for e in entries],
                "details": entries,
            }
    report["cross_asset_null_consistency"] = {
        "null_across_3plus_symbols": cross_null,
        "n_null_across_3plus": len(cross_null),
    }

    total_null = sum(len(v) for v in report["null_states"].values())
    total_anti = sum(len(v) for v in report["anti_directional_states"].values())
    report["summary"] = {
        "total_null_states": total_null,
        "total_anti_directional_states": total_anti,
        "null_across_3plus_symbols": len(cross_null),
        "per_symbol": {sym: {"null": len(report["null_states"][sym]),
                             "anti_directional": len(report["anti_directional_states"][sym])}
                       for sym in SYMBOLS},
    }
    save_stl_report(report, "stl5_null_states")
    return report


_H_IDX = {h: [1, 5, 20, 50, 100, 500].index(h) for h in [5, 20, 50]}
_SPLITS = WalkForwardValidator.SPLITS


def phase6_walk_forward(stl):
    """Phase 6: Walk-Forward Validation of directional states."""
    dsr = DSRCore()
    dsr.run_all_symbols()
    wfv = WalkForwardValidator(dsr)

    report = {"per_state_validation": {}, "edge_retention_summary": {}, "summary": {}}

    for sym in SYMBOLS:
        print(f"\n  {sym}:")
        summary = stl.state_summary(sym)
        sids_arr = stl._data[sym]["state_id"]
        fut_ret = stl._data[sym]["fut_ret"]
        n_total = len(sids_arr)

        candidates = []
        for sid, info in summary.items():
            if info["count"] < 15:
                continue
            for h in [20, 50]:
                p = info["p_up"].get(h)
                if p is not None and p >= 0.70:
                    candidates.append({
                        "state_id": int(sid), "es_q": info["es_q"],
                        "at_q": info["at_q"], "regime": info["regime"],
                        "mem_q": info["mem_q"], "count": info["count"],
                        "p_up_full": {str(hh): info["p_up"].get(hh, None) for hh in [5, 20, 50]},
                    })
                    break

        print(f"    {len(candidates)} candidate states")

        years = wfv.prepare(sym)
        sym_results = []

        for cand in candidates:
            sid = cand["state_id"]
            sid_result = {
                "state_id": sid, "components": list(decode_state(sid)),
                "full_sample_p_up": cand["p_up_full"],
                "total_count_full": cand["count"],
                "splits": {},
            }

            splits_tested = 0
            splits_passed = 0
            split_details = []

            for train_name, test_name in _SPLITS:
                train_mask, test_mask = wfv.split(sym, train_name, test_name)
                train_mask = train_mask[:n_total]
                test_mask = test_mask[:n_total]

                state_in_train = (sids_arr == sid) & train_mask
                state_in_test = (sids_arr == sid) & test_mask
                train_count = int(np.sum(state_in_train))
                test_count = int(np.sum(state_in_test))

                split_result = {
                    "train_name": train_name, "test_name": test_name,
                    "state_train_count": train_count,
                    "state_test_count": test_count,
                    "horizons": {},
                }

                split_passed = False

                for h in [20, 50]:
                    h_idx = _H_IDX[h]
                    hkey = f"H{h}"

                    train_ret = fut_ret[state_in_train, h_idx]
                    test_ret = fut_ret[state_in_test, h_idx]
                    n_train = int(np.sum(~np.isnan(train_ret)))
                    n_test = int(np.sum(~np.isnan(test_ret)))

                    if n_train < 5 or n_test < 3:
                        split_result["horizons"][hkey] = {
                            "train_count": n_train, "test_count": n_test,
                            "train_p_up": None, "test_p_up": None,
                            "validated": False,
                            "reason": "insufficient_train" if n_train < 5 else "insufficient_test",
                        }
                        continue

                    train_p_up = float(np.mean(train_ret[~np.isnan(train_ret)] > 0))
                    test_p_up = float(np.mean(test_ret[~np.isnan(test_ret)] > 0))

                    direction_same = (train_p_up > 0.5) == (test_p_up > 0.5)
                    edge_above_noise = test_p_up > 0.55
                    edge_stable = abs(test_p_up - train_p_up) <= 0.20
                    validated = direction_same and edge_above_noise and edge_stable

                    split_result["horizons"][hkey] = {
                        "train_count": n_train, "test_count": n_test,
                        "train_p_up": round(train_p_up, 4),
                        "test_p_up": round(test_p_up, 4),
                        "direction_same": direction_same,
                        "edge_above_noise": edge_above_noise,
                        "edge_stable": edge_stable,
                        "validated": validated,
                    }
                    if validated:
                        split_passed = True

                split_result["passes_validation"] = split_passed
                sid_result["splits"][f"{train_name}->{test_name}"] = split_result

                # Track: only count splits where state appeared in test
                if test_count >= 3:
                    splits_tested += 1
                    if split_passed:
                        splits_passed += 1
                    split_details.append({
                        "split": f"{train_name}->{test_name}",
                        "test_count": test_count,
                        "passed": split_passed,
                    })

            sid_result["splits_with_test_data"] = splits_tested
            sid_result["splits_passed"] = splits_passed
            sid_result["pass_rate"] = round(splits_passed / max(splits_tested, 1), 4)
            sid_result["state_validated"] = splits_tested >= 1 and splits_passed >= 1
            sid_result["split_details"] = split_details
            sym_results.append(sid_result)

        report["per_state_validation"][sym] = sym_results

        n_validated = sum(1 for r in sym_results if r["state_validated"])
        n_tested = sum(1 for r in sym_results if r["splits_with_test_data"] > 0)
        report["edge_retention_summary"][sym] = {
            "n_candidates": len(sym_results),
            "n_tested_oos": n_tested,
            "n_validated_oos": n_validated,
            "pct_tested_validated": round(n_validated / max(n_tested, 1) * 100, 1),
        }
        print(f"    Tested OOS: {n_tested}, Validated: {n_validated}")

    total_candidates = sum(v["n_candidates"] for v in report["edge_retention_summary"].values())
    total_tested = sum(v["n_tested_oos"] for v in report["edge_retention_summary"].values())
    total_validated = sum(v["n_validated_oos"] for v in report["edge_retention_summary"].values())
    report["summary"] = {
        "total_candidates": total_candidates,
        "total_tested_oos": total_tested,
        "total_validated_oos": total_validated,
        "pct_tested_validated": round(total_validated / max(total_tested, 1) * 100, 1),
        "per_symbol": {sym: report["edge_retention_summary"][sym] for sym in SYMBOLS},
        "verdict": (
            "DIRECTIONAL EDGES SURVIVE WALK-FORWARD"
            if total_validated / max(total_tested, 1) >= 0.5
            else "DIRECTIONAL EDGES PARTIALLY SURVIVE"
        ),
    }

    print(f"\n  Walk-Forward: {total_validated}/{total_tested} tested states validated ({report['summary']['pct_tested_validated']}%)")
    print(f"  Verdict: {report['summary']['verdict']}")

    save_stl_report(report, "stl6_walk_forward")
    return report


def write_markdown_phase5(p5):
    lines = []
    lines.append("# STL Phase 5 - Anti-Directional Null States")
    lines.append("## States That Destroy Directional Edge\n")
    lines.append("### Per-Symbol Summary\n")
    lines.append("| Symbol | Null States (45-55%) | Anti-Directional (P<30% / P>70%) |")
    lines.append("|--------|---------------------|---------------------------------|")
    for sym in SYMBOLS:
        s = p5["summary"]["per_symbol"][sym]
        lines.append(f"| {sym} | {s['null']} | {s['anti_directional']} |")

    lines.append("\n### Null States (P(up) in [0.45, 0.55])\n")
    for sym in SYMBOLS:
        nulls = p5["null_states"][sym]
        if nulls:
            lines.append(f"\n#### {sym} - {len(nulls)} null states")
            lines.append("| State ID | ES Q | AT Q | Regime | Mem Q | Count | P(up) | Horizon |")
            lines.append("|---------|------|------|--------|-------|-------|-------|---------|")
            for n in nulls[:15]:
                lines.append(f"| {n['state_id']} | {n['es_q']} | {n['at_q']} | {n['regime']} | {n['mem_q']} | "
                             f"{n['count']} | {n['p_up']:.2%} | H{n['horizon']} |")
            if len(nulls) > 15:
                lines.append(f"  *... and {len(nulls) - 15} more*")

    lines.append("\n### Null State Characterization\n")
    for sym in SYMBOLS:
        char = p5["null_state_characterization"][sym]
        if char["n_null_states"] == 0:
            continue
        lines.append(f"\n**{sym}:** {char['n_null_states']} null states | avg P(up)={char['avg_null_p_up']:.2%}")
        for dim, pcts in char["dimension_pcts"].items():
            top_val = max(pcts.items(), key=lambda x: x[1])
            lines.append(f"- {dim}: most null at value {top_val[0]} ({top_val[1]}%)")
        lines.append("- Top combos:")
        for combo in char["top_null_combos"][:3]:
            c = combo["components"]
            lines.append(f"  - ES{c[0]} AT{c[1]} R{c[2]} M{c[3]}: {combo['count']}x ({combo['pct_of_nulls']}%)")

    lines.append("\n### Anti-Directional States\n")
    for sym in SYMBOLS:
        anti = p5["anti_directional_states"][sym]
        if anti:
            lines.append(f"\n#### {sym} - {len(anti)}")
            lines.append("| State ID | ES Q | AT Q | Regime | Mem Q | Count | P(up) | Direction | Horizon |")
            lines.append("|---------|------|------|--------|-------|-------|-------|-----------|---------|")
            for a in anti[:10]:
                lines.append(f"| {a['state_id']} | {a['es_q']} | {a['at_q']} | {a['regime']} | {a['mem_q']} | "
                             f"{a['count']} | {a['p_up']:.2%} | {a['direction']} | H{a['horizon']} |")

    cn = p5["cross_asset_null_consistency"]
    lines.append(f"\n### Cross-Asset Null States (3+ symbols): {cn['n_null_across_3plus']}\n")
    for sid, entry in sorted(cn.get("null_across_3plus_symbols", {}).items(), key=lambda x: -x[1]["n_symbols"]):
        c = entry["components"]
        lines.append(f"- State {sid} (ES{c[0]} AT{c[1]} R{c[2]} M{c[3]}): {', '.join(entry['symbols'])}")

    lines.append("\n### Conclusions")
    lines.append(f"- Total null states: {p5['summary']['total_null_states']}")
    lines.append(f"- Total anti-directional states: {p5['summary']['total_anti_directional_states']}")
    lines.append("- **States that destroy edge:** All null states (P(up) approx 0.5)")
    lines.append("- **States to never trade:** Null states + unrecognized anti-directional states")

    path = Path(__file__).parent / "reports" / "STL5_NULL_STATES.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")


def write_markdown_phase6(p6):
    lines = []
    lines.append("# STL Phase 6 - Walk-Forward Validation")
    lines.append("## Do Directional States Survive Out of Sample?\n")

    s = p6["summary"]
    lines.append(f"**Candidates:** {s['total_candidates']} | "
                 f"**Tested OOS:** {s['total_tested_oos']} | "
                 f"**Validated:** {s['total_validated_oos']} ({s['pct_tested_validated']}%)")
    lines.append(f"**Verdict:** {s['verdict']}\n")

    lines.append("### Per-Symbol Summary\n")
    lines.append("| Symbol | Candidates | Tested OOS | Validated | % Validated |")
    lines.append("|--------|-----------|-----------|----------|------------|")
    for sym in SYMBOLS:
        ps = s["per_symbol"][sym]
        lines.append(f"| {sym} | {ps['n_candidates']} | {ps['n_tested_oos']} | "
                     f"{ps['n_validated_oos']} | {ps['pct_tested_validated']}% |")

    lines.append("\n### Validated States\n")
    for sym in SYMBOLS:
        states = p6["per_state_validation"][sym]
        validated = [st for st in states if st["state_validated"]]
        if validated:
            lines.append(f"\n#### {sym} - {len(validated)} validated")
            for st in validated:
                p_up_str = ", ".join(f"H{h}={st['full_sample_p_up'].get(str(h), 'N/A'):.0%}"
                                      for h in [5, 20, 50] if st['full_sample_p_up'].get(str(h)))
                lines.append(f"- State {st['state_id']} (ES{st['components'][0]} AT{st['components'][1]} "
                             f"R{st['components'][2]} M{st['components'][3]}): {p_up_str} "
                             f"| passed {st['splits_passed']}/{st['splits_with_test_data']} splits")
                for sd in st["split_details"]:
                    lines.append(f"  - {sd['split']}: {sd['test_count']} test occ, {'PASS' if sd['passed'] else 'FAIL'}")

    lines.append("\n### States Where Edge Collapsed\n")
    for sym in SYMBOLS:
        states = p6["per_state_validation"][sym]
        collapsed = [st for st in states if st["splits_with_test_data"] > 0 and not st["state_validated"]]
        if collapsed:
            lines.append(f"\n#### {sym} - {len(collapsed)} collapsed")
            for st in collapsed[:5]:
                lines.append(f"- State {st['state_id']}: passed {st['splits_passed']}/{st['splits_with_test_data']} splits")
                for sd in st["split_details"]:
                    for hkey, hdata in sorted(st["splits"].get(sd["split"], {}).get("horizons", {}).items()):
                        if hdata.get("train_p_up") is not None:
                            lines.append(f"  - {sd['split']} {hkey}: train={hdata['train_p_up']:.0%} -> "
                                         f"test={hdata['test_p_up']:.0%} {'OK' if hdata.get('validated') else 'FAIL'}")

    lines.append("\n### Conclusions")
    tested = s['total_tested_oos']
    validated = s['total_validated_oos']
    pct = s['pct_tested_validated']
    lines.append(f"1. **OOS validation rate:** {pct}% ({validated}/{tested} states weather walk-forward)")
    if pct >= 50:
        lines.append("2. Directional edges are REAL and survive OOS for most testable states")
    elif pct >= 25:
        lines.append("2. Directional edges PARTIALLY survive - some states robust, others collapse")
    else:
        lines.append("2. Directional edges WEAK - most states fail OOS validation")
    lines.append("3. Low test-set occurrence (~250 bars/year) limits statistical power")
    lines.append("4. **Use only walk-forward validated states** for live trading")

    path = Path(__file__).parent / "reports" / "STL6_WALK_FORWARD.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")


def main():
    print("=" * 60)
    print("STL Phase 5: Anti-Directional Null States")
    print("STL Phase 6: Walk-Forward Validation")
    print("=" * 60)

    stl = STLCore()
    print("\nLoading STL data...")
    stl.load_all()
    print(f"Loaded {len(stl._data)} symbols\n")

    # Phase 5
    print("-" * 40)
    print("PHASE 5: ANTI-DIRECTIONAL NULL STATES")
    print("-" * 40)
    p5 = phase5_null_states(stl)
    for sym in SYMBOLS:
        print(f"  {sym}: {len(p5['null_states'][sym])} null, {len(p5['anti_directional_states'][sym])} anti-dir")
    print(f"  Cross-asset null (3+): {p5['cross_asset_null_consistency']['n_null_across_3plus']}")

    # Phase 6
    print("\n" + "-" * 40)
    print("PHASE 6: WALK-FORWARD VALIDATION")
    print("-" * 40)
    p6 = phase6_walk_forward(stl)

    print("\nWriting markdown reports...")
    write_markdown_phase5(p5)
    write_markdown_phase6(p6)

    print("\nDone. Reports saved to reports/stl5_null_states.json, reports/stl6_walk_forward.json")
    print("Markdown: reports/STL5_NULL_STATES.md, reports/STL6_WALK_FORWARD.md")


if __name__ == "__main__":
    main()
