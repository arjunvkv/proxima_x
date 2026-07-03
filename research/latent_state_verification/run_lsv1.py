"""LSV-1: Synthetic Null Models — tests whether residual sign edge survives synthetic variants."""
import sys, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.latent_state_verification.lsv_core import LSVCore, SYMBOLS, save_lsv_report
from research.directional_state.dsr_core import WalkForwardValidator

HORIZONS = [5, 20, 50]
HORIZON_MAP = {5: "H5", 20: "H20", 50: "H50"}
N_SEEDS = 10

RANDOM_SEEDS = [42, 123, 256, 512, 1024, 2048, 4096, 8192, 9999, 1337]


def run_length_distribution(sign):
    runs = []
    current = 0
    length = 0
    for s in sign:
        if s == 0:
            if length > 0:
                runs.append((current, length))
                length = 0
            continue
        if s == current:
            length += 1
        else:
            if length > 0:
                runs.append((current, length))
            current = s
            length = 1
    if length > 0:
        runs.append((current, length))
    return runs


def generate_random_persistence_sign(original, rng):
    sign = np.sign(original)
    sign = sign.astype(np.int64)
    runs = run_length_distribution(sign)
    lengths = [r[1] for r in runs]
    signs_vals = [r[0] for r in runs]
    if len(lengths) == 0:
        return np.zeros_like(original, dtype=np.int64)
    synthetic = np.zeros(len(original), dtype=np.int64)
    i = 0
    while i < len(original):
        l = rng.choice(lengths)
        s = rng.choice(signs_vals)
        for j in range(min(l, len(original) - i)):
            synthetic[i + j] = s
        i += l
    return synthetic


def compute_conditional_probs(sign, fut_ret, horizon_idx):
    valid = ~np.isnan(fut_ret[:, horizon_idx]) & (sign != 0)
    if np.sum(valid) < 10:
        return {"p_up_given_pos": np.nan, "p_up_given_neg": np.nan, "n_pos": 0, "n_neg": 0}
    pos_mask = valid & (sign > 0)
    neg_mask = valid & (sign < 0)
    fut = fut_ret[:, horizon_idx]
    p_up_pos = np.mean(fut[pos_mask] > 0) if np.sum(pos_mask) > 0 else np.nan
    p_up_neg = np.mean(fut[neg_mask] > 0) if np.sum(neg_mask) > 0 else np.nan
    return {
        "p_up_given_pos": float(p_up_pos),
        "p_up_given_neg": float(p_up_neg),
        "n_pos": int(np.sum(pos_mask)),
        "n_neg": int(np.sum(neg_mask)),
    }


def directional_accuracy_from_rule(sign, fut_ret, horizon_idx):
    """Apply rule: go long when sign=+1 (expect up), go short when sign=-1 (expect down)."""
    valid = ~np.isnan(fut_ret[:, horizon_idx]) & (sign != 0)
    if np.sum(valid) < 10:
        return np.nan
    fut = fut_ret[:, horizon_idx]
    pred = np.where(sign[valid] > 0, 1, -1)
    correct = (pred * fut[valid]) > 0
    return float(np.mean(correct))


def accuracy_from_conditional(sign, fut_ret, horizon_idx, p_up_pos, p_up_neg):
    valid = ~np.isnan(fut_ret[:, horizon_idx]) & (sign != 0)
    if np.sum(valid) < 10:
        return np.nan
    fut = fut_ret[:, horizon_idx]
    pred = np.where(sign[valid] > 0, 1 if p_up_pos > 0.5 else -1, 1 if p_up_neg > 0.5 else -1)
    correct = (pred * fut[valid]) > 0
    return float(np.mean(correct))


def compute_run_length_stats(sign):
    """Compute mean run length and persistence probability."""
    runs = run_length_distribution(sign)
    if len(runs) == 0:
        return {"mean_run_length": np.nan, "p_flip": np.nan}
    lengths = [r[1] for r in runs]
    mean_len = float(np.mean(lengths))
    non_zero = sign[sign != 0]
    if len(non_zero) < 2:
        return {"mean_run_length": mean_len, "p_flip": np.nan}
    n_flips = np.sum(non_zero[1:] != non_zero[:-1])
    p_flip = float(n_flips) / len(non_zero)
    return {"mean_run_length": mean_len, "p_flip": p_flip}


def generate_synthetic_variants(lsv, sym, seed, rng):
    """Generate all 5 synthetic sign variants for a given symbol."""
    original_residual = lsv.residual(sym)
    original_sign = np.sign(original_residual).astype(np.int64)
    original_sign[np.isnan(original_residual)] = 0

    # Set global seed before each variant that uses global np.random
    np.random.seed(seed + 1)
    shuffled = lsv.shuffled_residual(original_residual.copy())
    s1_sign = np.sign(shuffled).astype(np.int64)
    s1_sign[np.isnan(shuffled)] = 0

    # S2: Lagged (deterministic, no randomness)
    lagged = lsv.lagged_residual(original_residual.copy(), lag=5)
    s2_sign = np.sign(lagged).astype(np.int64)
    s2_sign[np.isnan(lagged)] = 0

    # S3: Markov
    np.random.seed(seed + 2)
    s3_sign = lsv.markov_sign(original_residual.copy())

    # S4: fGn with H=0.86
    np.random.seed(seed + 3)
    s4_sign = lsv.sign_from_hurst(original_residual.copy(), H=0.86)

    # S5: Random persistence
    np.random.seed(seed + 4)
    s5_sign = generate_random_persistence_sign(original_residual.copy(), rng)

    return {
        "S1_Shuffled": s1_sign,
        "S2_Lagged": s2_sign,
        "S3_Markov": s3_sign,
        "S4_fGn_H086": s4_sign,
        "S5_RandomPersistence": s5_sign,
        "REAL_ResidualSign": original_sign,
    }


def run_single_variant(sign_train, sign_test, fut_ret_train, fut_ret_test, horizon_key, horizon_idx):
    """Train conditional probabilities on train, test accuracy on test."""
    cond = compute_conditional_probs(sign_train, fut_ret_train, horizon_idx)
    if np.isnan(cond["p_up_given_pos"]) or np.isnan(cond["p_up_given_neg"]):
        return {"accuracy": np.nan, "cond_train": cond}
    acc = accuracy_from_conditional(
        sign_test, fut_ret_test, horizon_idx,
        cond["p_up_given_pos"], cond["p_up_given_neg"]
    )
    # Also compute direct rule accuracy on test
    rule_acc = directional_accuracy_from_rule(sign_test, fut_ret_test, horizon_idx)
    return {
        "accuracy": acc,
        "rule_accuracy": rule_acc,
        "p_up_given_pos_train": cond["p_up_given_pos"],
        "p_up_given_neg_train": cond["p_up_given_neg"],
        "n_pos_train": cond["n_pos"],
        "n_neg_train": cond["n_neg"],
    }


def run_split(lsv, sym, train_name, test_name, variants, custom_split_fn):
    train_mask, test_mask = custom_split_fn(sym, train_name, test_name)
    d = lsv.rol._data.get(sym)
    if d is None:
        d = lsv.rol.dsr.load_symbol(sym)

    fut_ret = d["fut_ret"]
    results = {}
    for vname, vsign in variants.items():
        vr = {}
        for hidx in HORIZONS:
            hkey = HORIZON_MAP[hidx]
            vi = HORIZONS.index(hidx)
            vr[hkey] = run_single_variant(
                vsign[train_mask], vsign[test_mask],
                fut_ret[train_mask], fut_ret[test_mask],
                hkey, vi
            )
        results[vname] = vr
    return results


def run_lsv1():
    print("=" * 100)
    print("LSV-1: SYNTHETIC NULL MODELS")
    print("Testing whether residual sign edge survives 5 synthetic variants")
    print("=" * 100)

    lsv = LSVCore()
    lsv.load_all()

    # Build synthetic year ranges (data spans ~2018-2025)
    year_ranges = {}
    for sym in SYMBOLS:
        d = lsv.rol._data.get(sym)
        if d is None:
            d = lsv.rol.dsr.load_symbol(sym)
        n = len(d["es"])
        years = np.full(n, 2018, dtype=np.int32)
        step = n / 8.0
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

    splits = WalkForwardValidator.SPLITS

    all_results = {}
    per_seed_results = {}

    for sym in SYMBOLS:
        print(f"\n{'#' * 100}")
        print(f"SYMBOL: {sym}")
        print(f"{'#' * 100}")

        original_residual = lsv.residual(sym)
        original_sign = np.sign(original_residual).astype(np.int64)
        original_sign[np.isnan(original_residual)] = 0

        orig_rl = compute_run_length_stats(original_sign)
        print(f"\n  Original sign stats:")
        print(f"    Mean run length: {orig_rl['mean_run_length']:.2f}")
        print(f"    Flip probability: {orig_rl['p_flip']:.4f}")
        print(f"    N={len(original_sign)}, non-zero={np.sum(original_sign != 0)}")

        symbol_seeds = {}

        for seed in RANDOM_SEEDS:
            rng = np.random.RandomState(seed)
            variants = generate_synthetic_variants(lsv, sym, seed, rng)

            variant_run_lengths = {}
            for vname, vsign in variants.items():
                rl = compute_run_length_stats(vsign)
                variant_run_lengths[vname] = rl

            split_results = {}
            for train_name, test_name in splits:
                sr = run_split(lsv, sym, train_name, test_name, variants, custom_split)
                split_results[f"{train_name}->{test_name}"] = sr

            symbol_seeds[seed] = {
                "variants": {k: v.tolist() for k, v in variants.items()},
                "run_lengths": variant_run_lengths,
                "split_results": split_results,
            }

        per_seed_results[sym] = symbol_seeds

        # ---- Aggregate across seeds ----
        # For each variant x split x horizon, compute mean ± std accuracy across seeds
        print(f"\n  {'=' * 90}")
        print(f"  AGGREGATED RESULTS (N={N_SEEDS} seeds)")
        print(f"  {'=' * 90}")

        aggregate = {}
        for vname in ["S1_Shuffled", "S2_Lagged", "S3_Markov", "S4_fGn_H086", "S5_RandomPersistence"]:
            aggregate[vname] = {}
            for train_name, test_name in splits:
                split_key = f"{train_name}->{test_name}"
                aggregate[vname][split_key] = {}
                for hidx in HORIZONS:
                    hkey = HORIZON_MAP[hidx]
                    accs = []
                    rule_accs = []
                    for seed in RANDOM_SEEDS:
                        sa = symbol_seeds[seed]["split_results"][split_key][vname][hkey]
                        if not np.isnan(sa["accuracy"]) if sa["accuracy"] is not None else True:
                            accs.append(sa["accuracy"] if sa["accuracy"] is not None else np.nan)
                        else:
                            accs.append(np.nan)
                        if sa["rule_accuracy"] is not None and not np.isnan(sa["rule_accuracy"]):
                            rule_accs.append(sa["rule_accuracy"])
                    accs = [a for a in accs if not np.isnan(a)]
                    rule_accs = [a for a in rule_accs if not np.isnan(a)]
                    mean_acc = float(np.mean(accs)) if len(accs) > 0 else np.nan
                    std_acc = float(np.std(accs)) if len(accs) > 1 else np.nan
                    mean_rule = float(np.mean(rule_accs)) if len(rule_accs) > 0 else np.nan
                    aggregate[vname][split_key][hkey] = {
                        "accuracy_mean": mean_acc,
                        "accuracy_std": std_acc,
                        "rule_accuracy_mean": mean_rule,
                        "n_seeds_valid": len(accs),
                    }

        # Add REAL variant (no seed averaging needed)
        aggregate["REAL_ResidualSign"] = {}
        for train_name, test_name in splits:
            split_key = f"{train_name}->{test_name}"
            aggregate["REAL_ResidualSign"][split_key] = {}
            variants_base = generate_synthetic_variants(lsv, sym, RANDOM_SEEDS[0], np.random.RandomState(RANDOM_SEEDS[0]))
            d = lsv.rol._data.get(sym)
            if d is None:
                d = lsv.rol.dsr.load_symbol(sym)
            fut_ret = d["fut_ret"]
            train_mask, test_mask = custom_split(sym, train_name, test_name)
            for hidx in HORIZONS:
                hkey = HORIZON_MAP[hidx]
                vi = HORIZONS.index(hidx)
                vr = run_single_variant(
                    variants_base["REAL_ResidualSign"][train_mask],
                    variants_base["REAL_ResidualSign"][test_mask],
                    fut_ret[train_mask], fut_ret[test_mask],
                    hkey, vi
                )
                aggregate["REAL_ResidualSign"][split_key][hkey] = {
                    "accuracy": vr["accuracy"],
                    "rule_accuracy": vr["rule_accuracy"],
                }

        all_results[sym] = aggregate

        # ---- Print summary table ----
        for train_name, test_name in splits:
            split_key = f"{train_name}->{test_name}"
            print(f"\n  --- Split: {split_key} ---")
            header = f"  {'Variant':<25} {'H5':>10} {'H20':>10} {'H50':>10} {'H5_rule':>10} {'H20_rule':>10} {'H50_rule':>10}"
            print(header)
            print("  " + "-" * 85)
            real_accs = {}
            for hidx in HORIZONS:
                hkey = HORIZON_MAP[hidx]
                real_accs[hkey] = aggregate["REAL_ResidualSign"][split_key][hkey].get("accuracy", np.nan)
            for vname in ["REAL_ResidualSign", "S1_Shuffled", "S2_Lagged", "S3_Markov", "S4_fGn_H086", "S5_RandomPersistence"]:
                v = aggregate[vname][split_key]
                if vname == "REAL_ResidualSign":
                    line = f"  {vname:<25}"
                    for hidx in HORIZONS:
                        hkey = HORIZON_MAP[hidx]
                        a = v[hkey].get("accuracy", np.nan)
                        line += f" {a:>10.4f}" if not np.isnan(a) else f" {'N/A':>10}"
                    for hidx in HORIZONS:
                        hkey = HORIZON_MAP[hidx]
                        a = v[hkey].get("rule_accuracy", np.nan)
                        line += f" {a:>10.4f}" if not np.isnan(a) else f" {'N/A':>10}"
                    print(line)
                else:
                    line = f"  {vname:<25}"
                    for hidx in HORIZONS:
                        hkey = HORIZON_MAP[hidx]
                        if hkey in v:
                            m = v[hkey].get("accuracy_mean", np.nan)
                            s = v[hkey].get("accuracy_std", np.nan)
                            ns = v[hkey].get("n_seeds_valid", 0)
                            if not np.isnan(m) and not np.isnan(s):
                                line += f" {m:>7.4f}+-{s:<5.4f}"
                            elif not np.isnan(m):
                                line += f" {m:>10.4f}"
                            else:
                                line += f" {'N/A':>10}"
                        else:
                            line += f" {'N/A':>10}"
                    for hidx in HORIZONS:
                        hkey = HORIZON_MAP[hidx]
                        if hkey in v:
                            m = v[hkey].get("rule_accuracy_mean", np.nan)
                            if not np.isnan(m):
                                line += f" {m:>10.4f}"
                            else:
                                line += f" {'N/A':>10}"
                        else:
                            line += f" {'N/A':>10}"
                    print(line)

            print(f"\n  Edge Collapse Ratios (synthetic_acc / real_acc):")
            ecr_header = f"  {'Variant':<25} {'H5_ECR':>10} {'H20_ECR':>10} {'H50_ECR':>10} {'Verdict':>30}"
            print(ecr_header)
            print("  " + "-" * 75)
            for vname in ["S1_Shuffled", "S2_Lagged", "S3_Markov", "S4_fGn_H086", "S5_RandomPersistence"]:
                v = aggregate[vname][split_key]
                line = f"  {vname:<25}"
                verdicts = []
                for hidx in HORIZONS:
                    hkey = HORIZON_MAP[hidx]
                    real_acc = real_accs.get(hkey, np.nan)
                    if hkey in v:
                        syn_acc = v[hkey].get("accuracy_mean", np.nan)
                    else:
                        syn_acc = np.nan
                    if not np.isnan(real_acc) and not np.isnan(syn_acc) and real_acc != 0:
                        ecr = syn_acc / real_acc
                        line += f" {ecr:>10.4f}"
                        if ecr >= 0.9:
                            verdicts.append("STRUCTURAL")
                        elif ecr >= 0.7:
                            verdicts.append("PARTIAL")
                        else:
                            verdicts.append("MARKET")
                    else:
                        line += f" {'N/A':>10}"
                        verdicts.append("N/A")
                # Overall verdict
                structural_ct = sum(1 for vv in verdicts if vv == "STRUCTURAL")
                market_ct = sum(1 for vv in verdicts if vv == "MARKET")
                if structural_ct >= 2:
                    verdict_str = "ARTIFACT (structural)"
                elif market_ct >= 2:
                    verdict_str = "GENUINE (market-linked)"
                else:
                    verdict_str = "INCONCLUSIVE"
                line += f" {verdict_str:>30}"
                print(line)

            # Edge retention
            print(f"\n  Edge Retention (syn_acc - 0.5):")
            for vname in ["REAL_ResidualSign", "S1_Shuffled", "S2_Lagged", "S3_Markov", "S4_fGn_H086", "S5_RandomPersistence"]:
                v = aggregate[vname][split_key]
                line = f"  {vname:<25}"
                for hidx in HORIZONS:
                    hkey = HORIZON_MAP[hidx]
                    if vname == "REAL_ResidualSign":
                        a = v[hkey].get("accuracy", np.nan)
                    else:
                        if hkey in v:
                            a = v[hkey].get("accuracy_mean", np.nan)
                        else:
                            a = np.nan
                    if not np.isnan(a):
                        er = a - 0.5
                        sign = "+" if er > 0 else ""
                        line += f" {sign}{er:>10.4f}"
                    else:
                        line += f" {'N/A':>10}"
                print(line)

        # ---- Statistical significance ----
        print(f"\n  Statistical Significance (z-test: synthetic vs real):")
        for train_name, test_name in splits:
            split_key = f"{train_name}->{test_name}"
            for hidx in HORIZONS:
                hkey = HORIZON_MAP[hidx]
                r_acc = aggregate["REAL_ResidualSign"][split_key][hkey].get("accuracy", np.nan)
                print(f"    {split_key} {hkey}: REAL={r_acc:.4f}" if not np.isnan(r_acc) else f"    {split_key} {hkey}: REAL=N/A")
                for vname in ["S1_Shuffled", "S2_Lagged", "S3_Markov", "S4_fGn_H086", "S5_RandomPersistence"]:
                    v = aggregate[vname][split_key]
                    if hkey in v:
                        m = v[hkey].get("accuracy_mean", np.nan)
                        s = v[hkey].get("accuracy_std", np.nan)
                        ns = v[hkey].get("n_seeds_valid", 0)
                        if not np.isnan(m) and not np.isnan(s) and ns > 0 and not np.isnan(r_acc):
                            diff = r_acc - m
                            se = s / np.sqrt(ns) if ns > 1 else np.nan
                            if not np.isnan(se) and se > 0:
                                z = diff / se
                                p_val = 2 * (1 - min(abs(z) / 3.0, 1.0))
                                sig = "SIG" if p_val < 0.05 else "ns"
                                print(f"      {vname:<25}: syn={m:.4f}+-{s:.4f} diff={diff:+.4f} z={z:+.2f} p~={p_val:.3f} [{sig}]")
                            else:
                                print(f"      {vname:<25}: syn={m:.4f}+-{s:.4f} (single seed, no SE)")
                        else:
                            print(f"      {vname:<25}: N/A")

    # ---- Final Verdict ----
    print(f"\n{'#' * 100}")
    print("FINAL VERDICT: LSV-1 Synthetic Null Models")
    print(f"{'#' * 100}")

    for sym in SYMBOLS:
        aggregate = all_results[sym]
        print(f"\n  {sym}:")
        overall_structural = 0
        overall_market = 0
        total_variants = 0
        for vname in ["S1_Shuffled", "S2_Lagged", "S3_Markov", "S4_fGn_H086", "S5_RandomPersistence"]:
            counts = {"STRUCTURAL": 0, "MARKET": 0, "PARTIAL": 0}
            for train_name, test_name in splits:
                split_key = f"{train_name}->{test_name}"
                v = aggregate[vname][split_key]
                for hidx in HORIZONS:
                    hkey = HORIZON_MAP[hidx]
                    total_variants += 1
                    real_acc = aggregate["REAL_ResidualSign"][split_key][hkey].get("accuracy", np.nan)
                    if hkey in v:
                        syn_acc = v[hkey].get("accuracy_mean", np.nan)
                    else:
                        syn_acc = np.nan
                    if not np.isnan(real_acc) and not np.isnan(syn_acc) and real_acc != 0:
                        ecr = syn_acc / real_acc
                        if ecr >= 0.9:
                            counts["STRUCTURAL"] += 1
                        elif ecr >= 0.7:
                            counts["PARTIAL"] += 1
                        else:
                            counts["MARKET"] += 1
            print(f"    {vname:<25}: STRUCTURAL={counts['STRUCTURAL']} PARTIAL={counts['PARTIAL']} MARKET={counts['MARKET']}")
            overall_structural += counts["STRUCTURAL"]
            overall_market += counts["MARKET"]

        total = overall_structural + overall_market
        if total > 0:
            structural_pct = overall_structural / total * 100
            market_pct = overall_market / total * 100
            if structural_pct > 60:
                sym_verdict = "ARTIFACT — edge is structural (persistence alone generates it)"
            elif market_pct > 60:
                sym_verdict = "GENUINE — edge is market-linked"
            else:
                sym_verdict = "INCONCLUSIVE — mixed evidence"
        else:
            sym_verdict = "INCONCLUSIVE — insufficient data"
        print(f"    >>> VERDICT: {sym_verdict}")

    # ---- Save reports ----
    report = {
        "metadata": {
            "experiment": "LSV-1 Synthetic Null Models",
            "description": "Tests whether residual sign edge survives 5 synthetic variants across 5 symbols, 3 splits, 3 horizons",
            "n_seeds": N_SEEDS,
            "random_seeds": RANDOM_SEEDS,
            "splits": splits,
        },
        "results": all_results,
        "global_verdict": "See stdout for printed results",
    }

    save_lsv_report(report, "lsv1_synthetic_null_models")
    print(f"\nJSON report saved.")

    # Generate markdown
    md = generate_markdown_report(all_results, splits)
    md_path = Path(__file__).parent / "reports" / "LSV1_SYNTHETIC_NULL_MODELS.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown report saved to {md_path}")


def generate_markdown_report(all_results, splits):
    lines = []
    lines.append("# LSV-1: Synthetic Null Models")
    lines.append("")
    lines.append("## Research Questions")
    lines.append("1. Does shuffled residual sign preserve the edge?")
    lines.append("2. Does lagged residual sign preserve the edge?")
    lines.append("3. Does Markov-chain residual sign preserve the edge?")
    lines.append("4. Does fractional Brownian motion with H=0.86 preserve the edge?")
    lines.append("5. Does random sign with same run-length distribution preserve the edge?")
    lines.append("")
    lines.append("## Interpretation Rules")
    lines.append("- Edge retention = synthetic_accuracy - 0.5")
    lines.append("- Edge collapse ratio = synthetic_accuracy / real_accuracy")
    lines.append("- If collapse_ratio ~= 1.0: edge is STRUCTURAL (artifact)")
    lines.append("- If collapse_ratio ~= 0.5: edge is MARKET-LINKED (genuine)")
    lines.append("- If edge collapses in ALL variants -> genuinely tied to market evolution")
    lines.append("- If edge SURVIVES in ANY variant -> structural artifact")
    lines.append("")

    for sym in SYMBOLS:
        if sym not in all_results:
            continue
        aggregate = all_results[sym]
        lines.append(f"---")
        lines.append(f"## {sym}")
        lines.append("")

        for train_name, test_name in splits:
            split_key = f"{train_name}->{test_name}"
            lines.append(f"### Split: {split_key}")
            lines.append("")
            lines.append("| Variant | H5 Acc | H20 Acc | H50 Acc | H5 Rule | H20 Rule | H50 Rule |")
            lines.append("|---------|--------|---------|---------|---------|----------|---------|")

            for vname in ["REAL_ResidualSign", "S1_Shuffled", "S2_Lagged", "S3_Markov", "S4_fGn_H086", "S5_RandomPersistence"]:
                v = aggregate[vname][split_key]
                if vname == "REAL_ResidualSign":
                    row = f"| **{vname}**"
                    for hidx in [5, 20, 50]:
                        hkey = f"H{hidx}"
                        a = v[hkey].get("accuracy", np.nan)
                        row += f" | {a:.4f}" if not np.isnan(a) else " | N/A"
                    for hidx in [5, 20, 50]:
                        hkey = f"H{hidx}"
                        a = v[hkey].get("rule_accuracy", np.nan)
                        row += f" | {a:.4f}" if not np.isnan(a) else " | N/A"
                else:
                    row = f"| {vname}"
                    for hidx in [5, 20, 50]:
                        hkey = f"H{hidx}"
                        if hkey in v:
                            m = v[hkey].get("accuracy_mean", np.nan)
                            s = v[hkey].get("accuracy_std", np.nan)
                            if not np.isnan(m) and not np.isnan(s):
                                row += f" | {m:.4f}+-{s:.4f}"
                            elif not np.isnan(m):
                                row += f" | {m:.4f}"
                            else:
                                row += " | N/A"
                        else:
                            row += " | N/A"
                    for hidx in [5, 20, 50]:
                        hkey = f"H{hidx}"
                        if hkey in v:
                            m = v[hkey].get("rule_accuracy_mean", np.nan)
                            row += f" | {m:.4f}" if not np.isnan(m) else " | N/A"
                        else:
                            row += " | N/A"
                lines.append(row)

            lines.append("")
            lines.append("#### Edge Collapse Ratios")
            lines.append("")
            lines.append("| Variant | H5 ECR | H20 ECR | H50 ECR | Verdict |")
            lines.append("|---------|--------|---------|---------|---------|")
            for vname in ["S1_Shuffled", "S2_Lagged", "S3_Markov", "S4_fGn_H086", "S5_RandomPersistence"]:
                v = aggregate[vname][split_key]
                row = f"| {vname}"
                verdicts = []
                for hidx in [5, 20, 50]:
                    hkey = f"H{hidx}"
                    real_acc = aggregate["REAL_ResidualSign"][split_key][hkey].get("accuracy", np.nan)
                    if hkey in v:
                        syn_acc = v[hkey].get("accuracy_mean", np.nan)
                    else:
                        syn_acc = np.nan
                    if not np.isnan(real_acc) and not np.isnan(syn_acc) and real_acc != 0:
                        ecr = syn_acc / real_acc
                        row += f" | {ecr:.4f}"
                        if ecr >= 0.9:
                            verdicts.append("STRUCTURAL")
                        elif ecr >= 0.7:
                            verdicts.append("PARTIAL")
                        else:
                            verdicts.append("MARKET")
                    else:
                        row += " | N/A"
                        verdicts.append("N/A")
                structural_ct = sum(1 for vv in verdicts if vv == "STRUCTURAL")
                market_ct = sum(1 for vv in verdicts if vv == "MARKET")
                if structural_ct >= 2:
                    vv = "ARTIFACT (structural)"
                elif market_ct >= 2:
                    vv = "GENUINE (market-linked)"
                else:
                    vv = "INCONCLUSIVE"
                row += f" | {vv}"
                lines.append(row)

            lines.append("")
            lines.append("#### Edge Retention (accuracy - 0.5)")
            lines.append("")
            lines.append("| Variant | H5 | H20 | H50 |")
            lines.append("|---------|----|-----|-----|")
            for vname in ["REAL_ResidualSign", "S1_Shuffled", "S2_Lagged", "S3_Markov", "S4_fGn_H086", "S5_RandomPersistence"]:
                v = aggregate[vname][split_key]
                row = f"| {vname}"
                for hidx in [5, 20, 50]:
                    hkey = f"H{hidx}"
                    if vname == "REAL_ResidualSign":
                        a = v[hkey].get("accuracy", np.nan)
                    else:
                        if hkey in v:
                            a = v[hkey].get("accuracy_mean", np.nan)
                        else:
                            a = np.nan
                    if not np.isnan(a):
                        er = a - 0.5
                        row += f" | {er:+.4f}"
                    else:
                        row += " | N/A"
                lines.append(row)
            lines.append("")

        # Per-symbol summary
        lines.append(f"### {sym} — Final Verdict")
        lines.append("")
        lines.append("| Variant | STRUCTURAL | PARTIAL | MARKET |")
        lines.append("|---------|------------|---------|--------|")
        overall_structural = 0
        overall_market = 0
        for vname in ["S1_Shuffled", "S2_Lagged", "S3_Markov", "S4_fGn_H086", "S5_RandomPersistence"]:
            counts = {"STRUCTURAL": 0, "MARKET": 0, "PARTIAL": 0}
            for train_name, test_name in splits:
                split_key = f"{train_name}->{test_name}"
                v = aggregate[vname][split_key]
                for hidx in [5, 20, 50]:
                    hkey = f"H{hidx}"
                    real_acc = aggregate["REAL_ResidualSign"][split_key][hkey].get("accuracy", np.nan)
                    if hkey in v:
                        syn_acc = v[hkey].get("accuracy_mean", np.nan)
                    else:
                        syn_acc = np.nan
                    if not np.isnan(real_acc) and not np.isnan(syn_acc) and real_acc != 0:
                        ecr = syn_acc / real_acc
                        if ecr >= 0.9:
                            counts["STRUCTURAL"] += 1
                        elif ecr >= 0.7:
                            counts["PARTIAL"] += 1
                        else:
                            counts["MARKET"] += 1
            lines.append(f"| {vname} | {counts['STRUCTURAL']} | {counts['PARTIAL']} | {counts['MARKET']} |")
            overall_structural += counts["STRUCTURAL"]
            overall_market += counts["MARKET"]

        total = overall_structural + overall_market
        if total > 0:
            structural_pct = overall_structural / total * 100
            market_pct = overall_market / total * 100
            if structural_pct > 60:
                sym_verdict = "ARTIFACT — edge is structural (persistence alone generates it)"
            elif market_pct > 60:
                sym_verdict = "GENUINE — edge is market-linked"
            else:
                sym_verdict = "INCONCLUSIVE — mixed evidence"
        else:
            sym_verdict = "INCONCLUSIVE — insufficient data"
        lines.append(f"**Verdict: {sym_verdict}**")
        lines.append("")

    # Global summary
    lines.append("---")
    lines.append("## Global Summary")
    lines.append("")
    lines.append("| Symbol | Verdict |")
    lines.append("|--------|---------|")
    for sym in SYMBOLS:
        if sym not in all_results:
            continue
        aggregate = all_results[sym]
        overall_structural = 0
        overall_market = 0
        for vname in ["S1_Shuffled", "S2_Lagged", "S3_Markov", "S4_fGn_H086", "S5_RandomPersistence"]:
            for train_name, test_name in splits:
                split_key = f"{train_name}->{test_name}"
                v = aggregate[vname][split_key]
                for hidx in [5, 20, 50]:
                    hkey = f"H{hidx}"
                    real_acc = aggregate["REAL_ResidualSign"][split_key][hkey].get("accuracy", np.nan)
                    if hkey in v:
                        syn_acc = v[hkey].get("accuracy_mean", np.nan)
                    else:
                        syn_acc = np.nan
                    if not np.isnan(real_acc) and not np.isnan(syn_acc) and real_acc != 0:
                        ecr = syn_acc / real_acc
                        if ecr >= 0.9:
                            overall_structural += 1
                        elif ecr < 0.7:
                            overall_market += 1
        total = overall_structural + overall_market
        if total > 0:
            sp = overall_structural / total * 100
            mp = overall_market / total * 100
            if sp > 60:
                vv = "ARTIFACT"
            elif mp > 60:
                vv = "GENUINE"
            else:
                vv = "INCONCLUSIVE"
        else:
            vv = "INCONCLUSIVE"
        lines.append(f"| {sym} | {vv} |")

    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated with {N_SEEDS} random seeds per stochastic variant*")
    lines.append(f"*Seeds: {RANDOM_SEEDS}*")

    return "\n".join(lines)


if __name__ == "__main__":
    run_lsv1()
