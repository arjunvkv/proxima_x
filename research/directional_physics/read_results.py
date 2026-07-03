"""Read and display all DPL results."""
import json, sys
from pathlib import Path

reports = Path(__file__).parent / "reports"

# DPL-1
d1 = json.load(open(reports / "dpl1_results.json"))
print("=" * 60)
print("DPL-1: ES Magnitude vs Direction")
print("=" * 60)
print(f"  Final classification: {d1['final_classification']}")
for s, v in d1["per_symbol"].items():
    print(f"  {s}: {v['classification']} (abs_wins={v['abs_wins']}/{v['total_horizons']})")
    for hl, hv in v["by_horizon"].items():
        print(f"    {hl}: n={hv['n']}, pearson_sign={hv['pearson_sign']}, pearson_abs={hv['pearson_abs']}, abs_greater={hv['abs_greater_than_sign']}")
print()

# DPL-2
d2 = json.load(open(reports / "dpl2_results.json"))
print("=" * 60)
print("DPL-2: Residual Direction Hypothesis")
print("=" * 60)
print(f"  Cross-asset mean accuracy: {d2.get('cross_asset_mean_accuracy', 'N/A')}")
print(f"  Cross-asset std accuracy: {d2.get('cross_asset_std_accuracy', 'N/A')}")
for sym, sv in d2["per_symbol"].items():
    for rt, rv in sv.items():
        if rt == "meta":
            continue
        accs = [v.get("directional_accuracy") for v in rv.values() if isinstance(v, dict) and v.get("directional_accuracy")]
        accs = [a for a in accs if a is not None]
        if accs:
            print(f"  {sym} {rt}: mean_acc={sum(accs)/len(accs):.4f} over {len(accs)} horizons")
print()

# DPL-3
d3 = json.load(open(reports / "dpl3_results.json"))
print("=" * 60)
print("DPL-3: Memory Positioning Hypothesis")
print("=" * 60)
for sym, sv in d3["per_symbol"].items():
    print(f"  {sym}:")
    for hl, hv in sv.items():
        if hv["n"] > 0:
            d_imp = hv.get("distance_improves_es", False)
            p_above = hv.get('p_up_above_memory_center', 'N/A')
            p_below = hv.get('p_up_below_memory_center', 'N/A')
            print(f"    {hl}: n={hv['n']}, p_up_above={p_above}, p_up_below={p_below}, dist_improves={d_imp}")
print()

# DPL-4
d4 = json.load(open(reports / "dpl4_results.json"))
print("=" * 60)
print("DPL-4: Energy Gradient Theory")
print("=" * 60)
for sym, sv in d4["per_symbol"].items():
    print(f"  {sym}:")
    for hl, hv in sv.items():
        if hv["n_high_es"] > 0:
            g_beats = hv.get("gradient_beats_es", False)
            p_rise = hv.get('p_up_rising_gradient', 'N/A')
            p_fall = hv.get('p_up_falling_gradient', 'N/A')
            c_grad = hv.get('corr_gradient_direction', 'N/A')
            print(f"    {hl}: n={hv['n_high_es']}, p_up_rising={p_rise}, p_up_falling={p_fall}, corr_grad_dir={c_grad}, gradient_beats_es={g_beats}")
print()

# DPL-5
d5 = json.load(open(reports / "dpl5_results.json"))
print("=" * 60)
print("DPL-5: State Transition Directionality")
print("=" * 60)
for sym, sv in d5["per_symbol"].items():
    print(f"  {sym}:")
    for hl, hv in sv.items():
        if hv["n_transitions"] > 0:
            up_prob = hv.get('best_up_probability', '?')
            print(f"    {hl}: n_states={hv['n_states']}, transitions={hv['n_transitions']}, up_prob={up_prob}, n={hv.get('best_n', 0)}")
print()

# DPL-6
d6 = json.load(open(reports / "dpl6_results.json"))
print("=" * 60)
print("DPL-6: Regime Sign Inversion")
print("=" * 60)
for sym, sv in d6["per_symbol"].items():
    flip_horizons = []
    for hl, hv in sv.items():
        if hv.get("has_sign_inversion"):
            flip_horizons.append(hl)
    print(f"  {sym}: sign_flip_horizons={flip_horizons}")
print()

# DPL-7
d7 = json.load(open(reports / "dpl7_results.json"))
print("=" * 60)
print("DPL-7: Information Flow Layer")
print("=" * 60)
if d7.get("propagation_graph"):
    edges = d7["strongest_edges"]
    for k, v in edges.items():
        k_clean = k.encode('ascii', 'replace').decode('ascii')
        print(f"  {k_clean}: {v}")
print()

# DPL-8
d8 = json.load(open(reports / "dpl8_results.json"))
print("=" * 60)
print("DPL-8: Directional Survivorship Tournament")
print("=" * 60)
print(f"  Winner: {d8['winner']}")
for r in d8["rankings"]:
    print(f"  #{r['rank']} {r['candidate']}: acc={r['directional_accuracy']}, gain={r['info_gain']}, score={r['composite_score']}")
