"""LSV-4: Global Field Test — is residual sign a global market field across all 5 assets?"""
import sys, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.latent_state_verification.lsv_core import LSVCore, SYMBOLS, save_lsv_report
from research.directional_state.dsr_core import WalkForwardValidator, HORIZONS, HORIZON_KEYS

HORIZON_NAMES = ["H5", "H20", "H50"]
HORIZON_IDX = {hk: i for i, (h, hk) in enumerate(zip(HORIZONS, HORIZON_KEYS)) if hk in HORIZON_NAMES}
REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.RandomState(42)


# ─────────────────────────────────────────────
# STEP 1: Build global sign array
# ─────────────────────────────────────────────
def build_global_sign(lsv):
    signs = {}
    min_len = None
    for sym in SYMBOLS:
        s = lsv.residual_sign(sym)
        signs[sym] = s
        if min_len is None or len(s) < min_len:
            min_len = len(s)

    for sym in SYMBOLS:
        signs[sym] = signs[sym][:min_len]

    sign_stack = np.array([signs[sym] for sym in SYMBOLS])
    global_sign = np.sign(np.sum(sign_stack, axis=0))
    global_sign[global_sign == 0] = 1
    return signs, global_sign, sign_stack


# ─────────────────────────────────────────────
# STEP 2: Model comparison
# ─────────────────────────────────────────────
def model_accuracy(local_sign, global_sign, fut_ret_col):
    """Return P(up) given each model state."""
    valid = ~np.isnan(fut_ret_col) & (local_sign != 0)
    ls, gs, fr = local_sign[valid], global_sign[valid], fut_ret_col[valid]
    if len(ls) < 10:
        return {}

    up = (fr > 0).astype(float)
    n = len(ls)

    # M1: local sign only
    loc_pos = ls == 1
    loc_neg = ls == -1
    m1 = {
        "p_up_given_local_pos": float(np.mean(up[loc_pos])) if np.sum(loc_pos) > 0 else np.nan,
        "p_up_given_local_neg": float(np.mean(up[loc_neg])) if np.sum(loc_neg) > 0 else np.nan,
        "n_local_pos": int(np.sum(loc_pos)),
        "n_local_neg": int(np.sum(loc_neg)),
    }
    m1["delta"] = (m1["p_up_given_local_pos"] - m1["p_up_given_local_neg"]) if not (np.isnan(m1["p_up_given_local_pos"]) or np.isnan(m1["p_up_given_local_neg"])) else np.nan

    # M2: global sign only
    glob_pos = gs == 1
    glob_neg = gs == -1
    m2 = {
        "p_up_given_global_pos": float(np.mean(up[glob_pos])) if np.sum(glob_pos) > 0 else np.nan,
        "p_up_given_global_neg": float(np.mean(up[glob_neg])) if np.sum(glob_neg) > 0 else np.nan,
        "n_global_pos": int(np.sum(glob_pos)),
        "n_global_neg": int(np.sum(glob_neg)),
    }
    m2["delta"] = (m2["p_up_given_global_pos"] - m2["p_up_given_global_neg"]) if not (np.isnan(m2["p_up_given_global_pos"]) or np.isnan(m2["p_up_given_global_neg"])) else np.nan

    # M3: local + global combined — 4 states
    state_labels = {
        (1, 1): "local_pos_global_pos",
        (1, -1): "local_pos_global_neg",
        (-1, 1): "local_neg_global_pos",
        (-1, -1): "local_neg_global_neg",
    }
    m3 = {}
    for (l, g), label in state_labels.items():
        mask = (ls == l) & (gs == g)
        cnt = int(np.sum(mask))
        if cnt >= 5:
            m3[label] = {
                "p_up": round(float(np.mean(up[mask])), 4),
                "count": cnt,
            }

    # M4: disagreement analysis
    agree = ls == gs
    disagree = ls != gs
    m4 = {
        "p_up_when_agree": float(np.mean(up[agree])) if np.sum(agree) > 0 else np.nan,
        "p_up_when_disagree": float(np.mean(up[disagree])) if np.sum(disagree) > 0 else np.nan,
        "n_agree": int(np.sum(agree)),
        "n_disagree": int(np.sum(disagree)),
    }
    m4["delta"] = (m4["p_up_when_agree"] - m4["p_up_when_disagree"]) if not (np.isnan(m4["p_up_when_agree"]) or np.isnan(m4["p_up_when_disagree"])) else np.nan

    # Accuracy (directional prediction): predict up if model says +1
    def _accuracy(pred_sign, y):
        if len(pred_sign) == 0:
            return 0.0
        pred_up = (pred_sign > 0).astype(float)
        return float(np.mean(pred_up == y))

    m1_acc = _accuracy(ls, up) if n > 0 else 0
    m2_acc = _accuracy(gs, up) if n > 0 else 0

    # M3: combined prediction — if local=global, use that sign; else fallback to global
    m3_pred = np.where(ls == gs, ls, gs)
    m3_acc = _accuracy(m3_pred, up) if n > 0 else 0

    return {
        "M1_local_sign": {k: round(v, 4) if isinstance(v, float) else v for k, v in m1.items()},
        "M2_global_sign": {k: round(v, 4) if isinstance(v, float) else v for k, v in m2.items()},
        "M3_combined": {k: {sk: sv for sk, sv in sv.items()} for k, sv in m3.items()},
        "M3_combined_accuracy": round(m3_acc, 4),
        "M4_disagreement": {k: round(v, 4) if isinstance(v, float) else v for k, v in m4.items()},
        "accuracy_local": round(m1_acc, 4),
        "accuracy_global": round(m2_acc, 4),
        "n_total": n,
    }


# ─────────────────────────────────────────────
# STEP 3: Global field temporal properties
# ─────────────────────────────────────────────
def global_field_properties(global_sign, signs):
    n = len(global_sign)
    results = {}

    # Flip frequency
    flips = np.sum((global_sign[1:] != global_sign[:-1]) & (global_sign[1:] != 0))
    flip_freq = flips / max(n, 1)
    results["flip_frequency"] = round(float(flip_freq), 6)

    # Run length distribution
    runs = []
    if n > 0:
        current = global_sign[0]
        run_len = 1
        for i in range(1, n):
            if global_sign[i] == current:
                run_len += 1
            else:
                runs.append(run_len)
                current = global_sign[i]
                run_len = 1
        runs.append(run_len)

    results["run_length"] = {
        "mean": round(float(np.mean(runs)), 2) if runs else 0,
        "median": round(float(np.median(runs)), 2) if runs else 0,
        "max": int(np.max(runs)) if runs else 0,
        "min": int(np.min(runs)) if runs else 0,
        "n_runs": len(runs),
    }

    # Persistence: probability sign stays same next bar
    same = np.sum(global_sign[1:] == global_sign[:-1]) if n > 1 else 0
    results["persistence"] = round(same / max(n - 1, 1), 4)

    # Per-asset disagreement rate with global
    disagreement_rates = {}
    for sym in SYMBOLS:
        s = signs[sym]
        valid = (s != 0) & (global_sign != 0)
        if np.sum(valid) > 0:
            disagree_rate = float(np.mean(s[valid] != global_sign[valid]))
        else:
            disagree_rate = np.nan
        disagreement_rates[sym] = round(disagree_rate, 4)

    results["disagreement_rates"] = disagreement_rates

    # Master asset: whose sign = global sign most often
    agreement_counts = {}
    for sym in SYMBOLS:
        s = signs[sym]
        valid = (s != 0) & (global_sign != 0)
        if np.sum(valid) > 0:
            agreement_counts[sym] = float(np.mean(s[valid] == global_sign[valid]))
        else:
            agreement_counts[sym] = 0.0
    master = max(agreement_counts, key=agreement_counts.get)
    results["master_asset"] = {
        "symbol": master,
        "agreement_rate": round(agreement_counts[master], 4),
    }
    results["agreement_rates"] = {k: round(v, 4) for k, v in agreement_counts.items()}

    # Time-series structure: autocorrelation of global sign at various lags
    acf = {}
    for lag in [1, 5, 20, 50]:
        if n > lag:
            g0 = global_sign[:n-lag]
            g1 = global_sign[lag:]
            valid = (g0 != 0) & (g1 != 0)
            if np.sum(valid) >= 10:
                acf[f"lag_{lag}"] = round(float(np.mean(g0[valid] == g1[valid])), 4)
    results["autocorrelation"] = acf

    # Entropy of global sign sequence (bit per bar)
    p_pos = float(np.mean(global_sign == 1))
    p_neg = float(np.mean(global_sign == -1))
    entropy = 0.0
    if p_pos > 0 and p_neg > 0:
        entropy = -(p_pos * np.log2(p_pos) + p_neg * np.log2(p_neg))
    results["entropy"] = round(entropy, 4)

    # Compare global sign flip frequency vs each local sign flip frequency
    local_flip_comparison = {}
    for sym in SYMBOLS:
        s = signs[sym]
        local_flips = np.sum((s[1:] != s[:-1]) & (s[1:] != 0) & (s[:-1] != 0))
        local_flip_freq = local_flips / max(np.sum((s[:-1] != 0) & (s[1:] != 0)), 1)
        local_flip_comparison[sym] = round(float(local_flip_freq), 6)
    results["local_flip_frequencies"] = local_flip_comparison
    results["global_flip_frequency"] = round(float(flip_freq), 6)

    # Sync strength: how many assets agree with majority
    sync_counts = []
    for i in range(n):
        col = np.array([signs[sym][i] for sym in SYMBOLS])
        valid = col != 0
        if np.sum(valid) >= 3:
            majority = np.sign(np.sum(col[valid]))
            sync_counts.append(float(np.mean(col[valid] == majority)))
    results["mean_sync_strength"] = round(float(np.mean(sync_counts)), 4) if sync_counts else 0

    return results


# ─────────────────────────────────────────────
# STEP 4: Leading indicator test
# ─────────────────────────────────────────────
def leading_indicator_test(global_sign, signs, fut_ret_dict, lags=[5, 20, 50]):
    n = len(global_sign)
    results = {}
    for lag in lags:
        lag_results = {}
        for sym in SYMBOLS:
            sym_res = {}
            fr = fut_ret_dict[sym]
            min_n = min(n, fr.shape[0])
            gs = global_sign[:min_n]
            fr_aligned = fr[:min_n]
            for hk in HORIZON_NAMES:
                col = HORIZON_IDX[hk]
                fr_col = fr_aligned[:, col]
                gs_lagged = gs[:-lag] if lag < len(gs) else np.array([])
                fr_lagged = fr_col[lag:] if lag < len(fr_col) else np.array([])
                mlen = min(len(gs_lagged), len(fr_lagged))
                if mlen < 20:
                    continue
                gs_lagged, fr_lagged = gs_lagged[:mlen], fr_lagged[:mlen]
                valid = ~np.isnan(fr_lagged) & (gs_lagged != 0)
                gv, fv = gs_lagged[valid], fr_lagged[valid]
                if len(gv) < 10:
                    continue
                up = (fv > 0).astype(float)
                pos = gv == 1
                neg = gv == -1
                p_up_pos = float(np.mean(up[pos])) if np.sum(pos) > 0 else np.nan
                p_up_neg = float(np.mean(up[neg])) if np.sum(neg) > 0 else np.nan
                delta = (p_up_pos - p_up_neg) if not (np.isnan(p_up_pos) or np.isnan(p_up_neg)) else np.nan
                sym_res[hk] = {
                    "p_up_given_global_pos": round(p_up_pos, 4) if not np.isnan(p_up_pos) else None,
                    "p_up_given_global_neg": round(p_up_neg, 4) if not np.isnan(p_up_neg) else None,
                    "delta": round(delta, 4) if not np.isnan(delta) else None,
                    "n": int(np.sum(valid)),
                }
            lag_results[sym] = sym_res
        results[f"lag_{lag}"] = lag_results
    return results


# ─────────────────────────────────────────────
# STEP 5: Walk-forward validation
# ─────────────────────────────────────────────
def walkforward_models(lsv, signs, global_sign, fut_ret_dict):
    try:
        validator = WalkForwardValidator(lsv.rol.dsr)
    except Exception:
        return []

    splits = []
    for sym in SYMBOLS:
        try:
            validator.prepare(sym)
        except Exception:
            continue

    for train_name, test_name in WalkForwardValidator.SPLITS:
        test_masks = {}
        for sym in SYMBOLS:
            try:
                _, test_mask = validator.split(sym, train_name, test_name)
                test_masks[sym] = test_mask
            except Exception:
                test_masks[sym] = None

        split_entry = {"split": f"{train_name}_vs_{test_name}"}
        n = len(global_sign)

        for hk in HORIZON_NAMES:
            col = HORIZON_IDX[hk]
            h_results = {}
            for sym in SYMBOLS:
                tm = test_masks.get(sym)
                if tm is None:
                    continue
                fr = fut_ret_dict[sym][:n, col]
                ls = signs[sym][:n]
                gs = global_sign[:n]
                tm = tm[:n]
                test_idx = np.where(tm & ~np.isnan(fr) & (ls != 0))[0]
                if len(test_idx) < 20:
                    continue
                y_true = (fr[test_idx] > 0).astype(float)
                ls_t = ls[test_idx]
                gs_t = gs[test_idx]
                n_test = len(y_true)

                # M1: local sign
                pred_m1 = (ls_t > 0).astype(float)
                acc_m1 = float(np.mean(pred_m1 == y_true))

                # M2: global sign
                pred_m2 = (gs_t > 0).astype(float)
                acc_m2 = float(np.mean(pred_m2 == y_true))

                # M3: combined (if agree use that, else fallback to global)
                pred_m3 = np.where(ls_t == gs_t, (ls_t > 0).astype(float), (gs_t > 0).astype(float))
                acc_m3 = float(np.mean(pred_m3 == y_true))

                h_results[sym] = {
                    "M1_local_accuracy": round(acc_m1, 4),
                    "M2_global_accuracy": round(acc_m2, 4),
                    "M3_combined_accuracy": round(acc_m3, 4),
                    "n_test": n_test,
                }
            split_entry[hk] = h_results
        splits.append(split_entry)
    return splits


# ─────────────────────────────────────────────
# STEP 6: Global field strength per asset
# ─────────────────────────────────────────────
def global_field_strength(global_sign, signs, fut_ret_dict):
    """Does global sign predict each asset's direction with different strength?"""
    n = len(global_sign)
    results = {}
    for sym in SYMBOLS:
        ls = signs[sym][:n]
        fr = fut_ret_dict[sym][:n]
        sym_res = {}
        for hk in HORIZON_NAMES:
            col = HORIZON_IDX[hk]
            fr_col = fr[:, col]
            valid = ~np.isnan(fr_col) & (global_sign != 0)
            gs, fc = global_sign[valid], fr_col[valid]
            if len(gs) < 10:
                continue
            up = (fc > 0).astype(float)
            gpos = gs == 1
            gneg = gs == -1
            p_up_gpos = float(np.mean(up[gpos])) if np.sum(gpos) > 0 else np.nan
            p_up_gneg = float(np.mean(up[gneg])) if np.sum(gneg) > 0 else np.nan
            delta = p_up_gpos - p_up_gneg if not (np.isnan(p_up_gpos) or np.isnan(p_up_gneg)) else np.nan

            # also test local sign prediction for comparison
            lvalid = ~np.isnan(fr_col) & (ls != 0)
            lv, lfc = ls[lvalid], fr_col[lvalid]
            lup = (lfc > 0).astype(float)
            lpos = lv == 1
            lneg = lv == -1
            l_p_up_pos = float(np.mean(lup[lpos])) if np.sum(lpos) > 0 else np.nan
            l_p_up_neg = float(np.mean(lup[lneg])) if np.sum(lneg) > 0 else np.nan
            l_delta = l_p_up_pos - l_p_up_neg if not (np.isnan(l_p_up_pos) or np.isnan(l_p_up_neg)) else np.nan

            sym_res[hk] = {
                "global_p_up_pos": round(p_up_gpos, 4) if not np.isnan(p_up_gpos) else None,
                "global_p_up_neg": round(p_up_gneg, 4) if not np.isnan(p_up_gneg) else None,
                "global_delta": round(delta, 4) if not np.isnan(delta) else None,
                "local_p_up_pos": round(l_p_up_pos, 4) if not np.isnan(l_p_up_pos) else None,
                "local_p_up_neg": round(l_p_up_neg, 4) if not np.isnan(l_p_up_neg) else None,
                "local_delta": round(l_delta, 4) if not np.isnan(l_delta) else None,
                "n_global": int(np.sum(valid)),
                "n_local": int(np.sum(lvalid)),
            }
        results[sym] = sym_res
    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 72)
    print("LSV-4: Global Field Test")
    print("=" * 72)

    lsv = LSVCore()
    lsv.load_all()
    print("LSV core loaded.\n")

    signs, global_sign, sign_stack = build_global_sign(lsv)
    n = len(global_sign)
    print(f"Global sign built: {n} bars across {len(SYMBOLS)} assets")

    # Load future returns
    fut_ret_dict = {}
    for sym in SYMBOLS:
        fut_ret_dict[sym] = lsv.future_returns(sym)

    report = {
        "metadata": {
            "title": "LSV-4: Global Field Test",
            "symbols": SYMBOLS,
            "n_bars": n,
            "horizons": HORIZON_NAMES,
        }
    }
    md_lines = [
        "# LSV-4: Global Field Test",
        "",
        "**Hypothesis:** Residual sign is a GLOBAL MARKET FIELD rather than asset-specific.",
        f"**OMS-3 found:** mean cross-asset sync = 4.49/5 assets — residual markers appear simultaneously.",
        "",
        "## Research Questions",
        "1. Does GLOBAL sign predict direction as well as LOCAL sign?",
        "2. Is local sign useful when it DISAGREES with global sign?",
        "3. Does a model using ONLY global sign beat the local-sign baseline?",
        "4. Is the global field stronger in some assets than others?",
        "5. Does the global field have a distinct temporal structure?",
        "",
        "---",
        "",
    ]

    # ── Model Comparison ──
    print("\n" + "-" * 72)
    print("RQ1-3: MODEL COMPARISON (Local vs Global vs Combined)")
    print("-" * 72)

    model_comparison = {}
    for sym in SYMBOLS:
        print(f"\n  {sym}:")
        sym_models = {}
        for hk in HORIZON_NAMES:
            col = HORIZON_IDX[hk]
            fr = fut_ret_dict[sym]
            min_n = min(n, fr.shape[0])
            ls = signs[sym][:min_n]
            gs = global_sign[:min_n]
            fr_col = fr[:min_n, col]
            result = model_accuracy(ls, gs, fr_col)
            sym_models[hk] = result
            if result:
                m1_d = result.get("M1_local_sign", {}).get("delta", "N/A")
                m2_d = result.get("M2_global_sign", {}).get("delta", "N/A")
                m1_acc = result.get("accuracy_local", "N/A")
                m2_acc = result.get("accuracy_global", "N/A")
                print(f"    {hk}: local_delta={m1_d} global_delta={m2_d} | local_acc={m1_acc} global_acc={m2_acc}")
        model_comparison[sym] = sym_models
    report["model_comparison"] = model_comparison

    # ── Global Field Properties ──
    print("\n" + "-" * 72)
    print("RQ5: GLOBAL FIELD PROPERTIES")
    print("-" * 72)

    field_props = global_field_properties(global_sign, signs)
    report["global_field_properties"] = field_props

    print(f"  Flip frequency: {field_props['global_flip_frequency']:.6f}")
    print(f"  Mean run length: {field_props['run_length']['mean']}")
    print(f"  Persistence: {field_props['persistence']:.4f}")
    print(f"  Entropy: {field_props['entropy']:.4f}")
    print(f"  Mean sync strength: {field_props['mean_sync_strength']}")
    print(f"  Master asset: {field_props['master_asset']['symbol']} ({field_props['master_asset']['agreement_rate']:.4f})")
    print(f"  Disagreement rates:")
    for sym, dr in field_props["disagreement_rates"].items():
        print(f"    {sym}: {dr:.4f}")
    print(f"  Agreement rates with global:")
    for sym, ar in field_props["agreement_rates"].items():
        print(f"    {sym}: {ar:.4f}")
    print(f"  Local flip frequencies:")
    for sym, lf in field_props["local_flip_frequencies"].items():
        print(f"    {sym}: {lf:.6f}  (global: {field_props['global_flip_frequency']:.6f})")

    # ── Global Field Strength per Asset (RQ4) ──
    print("\n" + "-" * 72)
    print("RQ4: GLOBAL FIELD STRENGTH PER ASSET")
    print("-" * 72)

    strength = global_field_strength(global_sign, signs, fut_ret_dict)
    report["global_field_strength_per_asset"] = strength

    for sym in SYMBOLS:
        print(f"\n  {sym}:")
        for hk in HORIZON_NAMES:
            sd = strength.get(sym, {}).get(hk, {})
            if sd:
                gd = sd.get("global_delta")
                ld = sd.get("local_delta")
                print(f"    {hk}: global_delta={gd} local_delta={ld}")

    # ── Leading Indicator ──
    print("\n" + "-" * 72)
    print("LEADING INDICATOR TEST")
    print("-" * 72)

    leading = leading_indicator_test(global_sign, signs, fut_ret_dict)
    report["leading_indicator"] = leading

    for lag_key, lag_data in leading.items():
        print(f"\n  {lag_key}:")
        for sym in SYMBOLS:
            for hk in HORIZON_NAMES:
                hd = lag_data.get(sym, {}).get(hk, {})
                if hd and hd.get("delta") is not None:
                    print(f"    {sym} {hk}: delta={hd['delta']:.4f} (n={hd['n']})")

    # ── Walk-Forward Validation ──
    print("\n" + "-" * 72)
    print("WALK-FORWARD VALIDATION")
    print("-" * 72)

    wf_results = walkforward_models(lsv, signs, global_sign, fut_ret_dict)
    report["walk_forward"] = wf_results

    for wf in wf_results:
        print(f"\n  {wf['split']}:")
        for hk in HORIZON_NAMES:
            hd = wf.get(hk, {})
            if not hd:
                continue
            for sym in SYMBOLS:
                sd = hd.get(sym, {})
                if sd:
                    print(f"    {sym} {hk}: M1_local={sd['M1_local_accuracy']:.4f} M2_global={sd['M2_global_accuracy']:.4f} M3_combined={sd['M3_combined_accuracy']:.4f} n={sd['n_test']}")

    # ── Build Markdown ──
    md_lines.append("## Model Comparison (Local vs Global)")
    md_lines.append("")
    md_lines.append("| Symbol | Horizon | Local Delta | Global Delta | Local Acc | Global Acc | Combined Acc |")
    md_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for sym in SYMBOLS:
        for hk in HORIZON_NAMES:
            mc = model_comparison.get(sym, {}).get(hk, {})
            if not mc:
                continue
            m1_d = mc.get("M1_local_sign", {}).get("delta", "N/A")
            m2_d = mc.get("M2_global_sign", {}).get("delta", "N/A")
            m1_a = mc.get("accuracy_local", "N/A")
            m2_a = mc.get("accuracy_global", "N/A")
            m3_a = mc.get("M3_combined_accuracy", "N/A")
            md_lines.append(f"| {sym} | {hk} | {m1_d} | {m2_d} | {m1_a} | {m2_a} | {m3_a} |")
    md_lines.append("")

    md_lines.append("## Disagreement Analysis (M4)")
    md_lines.append("")
    md_lines.append("| Symbol | Horizon | P(up \\| agree) | P(up \\| disagree) | Delta | N(agree) | N(disagree) |")
    md_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for sym in SYMBOLS:
        for hk in HORIZON_NAMES:
            mc = model_comparison.get(sym, {}).get(hk, {})
            m4 = mc.get("M4_disagreement", {})
            if m4:
                p_a = m4.get("p_up_when_agree", "N/A")
                p_d = m4.get("p_up_when_disagree", "N/A")
                d4 = m4.get("delta", "N/A")
                n_a = m4.get("n_agree", 0)
                n_d = m4.get("n_disagree", 0)
                md_lines.append(f"| {sym} | {hk} | {p_a} | {p_d} | {d4} | {n_a} | {n_d} |")
    md_lines.append("")

    md_lines.append("## Global Field Properties")
    md_lines.append("")
    md_lines.append(f"- **Flip frequency (global):** {field_props['global_flip_frequency']}")
    md_lines.append(f"- **Mean run length:** {field_props['run_length']['mean']}")
    md_lines.append(f"- **Median run length:** {field_props['run_length']['median']}")
    md_lines.append(f"- **Max run length:** {field_props['run_length']['max']}")
    md_lines.append(f"- **Persistence (P[stay]):** {field_props['persistence']}")
    md_lines.append(f"- **Entropy (bits/bar):** {field_props['entropy']}")
    md_lines.append(f"- **Mean sync strength:** {field_props['mean_sync_strength']}")
    md_lines.append(f"- **Master asset:** {field_props['master_asset']['symbol']} (agreement={field_props['master_asset']['agreement_rate']})")
    md_lines.append("")
    md_lines.append("### Agreement Rates with Global")
    md_lines.append("")
    md_lines.append("| Symbol | Agreement Rate | Disagreement Rate | Local Flip Freq |")
    md_lines.append("|:---|:---:|:---:|:---:|")
    for sym in SYMBOLS:
        ar = field_props["agreement_rates"].get(sym, "N/A")
        dr = field_props["disagreement_rates"].get(sym, "N/A")
        lf = field_props["local_flip_frequencies"].get(sym, "N/A")
        md_lines.append(f"| {sym} | {ar} | {dr} | {lf} |")
    md_lines.append("")

    md_lines.append("### Autocorrelation (global sign)")
    md_lines.append("")
    for lag_key, val in field_props.get("autocorrelation", {}).items():
        md_lines.append(f"- **{lag_key}:** {val}")
    md_lines.append("")

    md_lines.append("## Global Field Strength Per Asset")
    md_lines.append("")
    md_lines.append("| Symbol | Horizon | Global Delta | Local Delta | N(global) |")
    md_lines.append("|:---|:---:|:---:|:---:|:---:|")
    for sym in SYMBOLS:
        for hk in HORIZON_NAMES:
            sd = strength.get(sym, {}).get(hk, {})
            if sd:
                gd = sd.get("global_delta", "N/A")
                ld = sd.get("local_delta", "N/A")
                ng = sd.get("n_global", "N/A")
                md_lines.append(f"| {sym} | {hk} | {gd} | {ld} | {ng} |")
    md_lines.append("")

    md_lines.append("## Leading Indicator Test (Global Sign at t-lag)")
    md_lines.append("")
    for lag_key, lag_data in leading.items():
        md_lines.append(f"### {lag_key}")
        md_lines.append("")
        md_lines.append("| Symbol | Horizon | P(up \\| global_pos) | P(up \\| global_neg) | Delta | N |")
        md_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        for sym in SYMBOLS:
            for hk in HORIZON_NAMES:
                hd = lag_data.get(sym, {}).get(hk, {})
                if hd:
                    pp = hd.get("p_up_given_global_pos", "N/A")
                    pn = hd.get("p_up_given_global_neg", "N/A")
                    d_ = hd.get("delta", "N/A")
                    n_ = hd.get("n", "N/A")
                    md_lines.append(f"| {sym} | {hk} | {pp} | {pn} | {d_} | {n_} |")
        md_lines.append("")

    md_lines.append("## Walk-Forward Validation")
    md_lines.append("")
    md_lines.append("| Split | Symbol | Horizon | M1 Local | M2 Global | M3 Combined | N |")
    md_lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|")
    for wf in wf_results:
        for hk in HORIZON_NAMES:
            hd = wf.get(hk, {})
            if not hd:
                continue
            for sym in SYMBOLS:
                sd = hd.get(sym, {})
                if sd:
                    md_lines.append(f"| {wf['split']} | {sym} | {hk} | {sd['M1_local_accuracy']} | {sd['M2_global_accuracy']} | {sd['M3_combined_accuracy']} | {sd['n_test']} |")
    md_lines.append("")

    md_lines.append("## Combined State Model (M3: Local x Global — 4 States)")
    md_lines.append("")
    md_lines.append("| Symbol | Horizon | L+ G+ | L+ G- | L- G+ | L- G- |")
    md_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
    for sym in SYMBOLS:
        for hk in HORIZON_NAMES:
            mc = model_comparison.get(sym, {}).get(hk, {})
            m3 = mc.get("M3_combined", {})
            if m3:
                pp = m3.get("local_pos_global_pos", {}).get("p_up", "N/A")
                pn = m3.get("local_pos_global_neg", {}).get("p_up", "N/A")
                np_ = m3.get("local_neg_global_pos", {}).get("p_up", "N/A")
                nn = m3.get("local_neg_global_neg", {}).get("p_up", "N/A")
                md_lines.append(f"| {sym} | {hk} | {pp} | {pn} | {np_} | {nn} |")
    md_lines.append("")

    md_lines.append("---")
    md_lines.append("## Verdict")
    md_lines.append("")
    md_lines.append("**Is residual sign a global field?**")
    md_lines.append("")

    # Build verdict from data
    global_wins = 0
    local_wins = 0
    total_comparisons = 0
    for sym in SYMBOLS:
        for hk in HORIZON_NAMES:
            mc = model_comparison.get(sym, {}).get(hk, {})
            if mc:
                m1_acc = mc.get("accuracy_local", 0)
                m2_acc = mc.get("accuracy_global", 0)
                if m1_acc and m2_acc:
                    total_comparisons += 1
                    if m2_acc >= m1_acc:
                        global_wins += 1
                    else:
                        local_wins += 1
    if total_comparisons > 0:
        md_lines.append(f"- Global sign beats or ties local sign in **{global_wins}/{total_comparisons}** symbol-horizon comparisons ({100*global_wins/total_comparisons:.1f}%).")
        md_lines.append(f"- Local sign wins in **{local_wins}/{total_comparisons}** ({100*local_wins/total_comparisons:.1f}%).")
    md_lines.append(f"- Mean sync strength: **{field_props['mean_sync_strength']}** (1.0 = perfect agreement).")
    md_lines.append(f"- Master asset: **{field_props['master_asset']['symbol']}** agrees with global {field_props['master_asset']['agreement_rate']*100:.1f}% of the time.")
    md_lines.append(f"- Global sign persistence: **{field_props['persistence']}** (P[stay] next bar).")
    md_lines.append(f"- Global sign flip frequency: **{field_props['global_flip_frequency']:.6f}** per bar.")
    md_lines.append("")
    md_lines.append("*Generated by LSV-4: Global Field Test*")

    # Save reports
    json_path = REPORT_DIR / "lsv4_global_field.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {json_path}")

    md_path = REPORT_DIR / "LSV4_GLOBAL_FIELD.md"
    md_content = "\n".join(md_lines)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved {md_path}")

    print("\n" + "=" * 72)
    print("LSV-4 COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
