"""ROL-4: Cross-Asset Residual Propagation.

Tests whether residual propagates across assets better than price direction.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import mutual_info_score

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.residual_origin.rol_core import ROLCore, SYMBOLS, save_rol_report

warnings.filterwarnings("ignore")

LAGS = [1, 5, 10, 20]
H50_IDX = 3


def sign_strict(x):
    s = np.sign(x)
    s[np.isnan(s)] = 0
    return s.astype(np.int64)


def directional_accuracy(pred, actual):
    mask = (pred != 0) & (actual != 0) & np.isfinite(pred) & np.isfinite(actual)
    total = int(np.sum(mask))
    if total == 0:
        return 0.0, 0
    correct = int(np.sum(pred[mask] == actual[mask]))
    return correct / total, total


def safe_mutual_info(a, b):
    valid = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    if np.sum(valid) < 20:
        return 0.0
    return float(mutual_info_score(a[valid], b[valid]))


def safe_corr(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 20:
        return 0.0
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def compute_price_return(fut_ret):
    ret = np.full(len(fut_ret), np.nan)
    ret[1:] = fut_ret[:-1, 0]
    return ret


def _align(a, b):
    """Truncate both arrays to the minimum length."""
    m = min(len(a), len(b))
    return a[:m], b[:m]


def compute_pair_metrics(rol, sym_a, sym_b, horizon_idx=H50_IDX):
    res_a, res_b = _align(rol.get_residuals(sym_a), rol.get_residuals(sym_b))
    sign_a = sign_strict(res_a)
    sign_b = sign_strict(res_b)

    fut_b = rol._data[sym_b]["fut_ret"][:len(res_b)]
    dir_b = sign_strict(fut_b[:, horizon_idx])

    fut_a = rol._data[sym_a]["fut_ret"][:len(res_a)]
    price_ret_a = compute_price_return(fut_a)
    price_dir_a = sign_strict(price_ret_a)

    reg_a = rol._data[sym_a]["regime"][:len(res_a)]
    reg_b = rol._data[sym_b]["regime"][:len(res_b)]

    n = len(res_a)
    results = {}
    for lag in LAGS:
        if lag >= n:
            continue

        resid_pred = sign_a[:n - lag]
        target = dir_b[lag:]
        m = min(len(resid_pred), len(target))
        resid_pred, target = resid_pred[:m], target[:m]

        resid_acc, resid_n = directional_accuracy(resid_pred, target)
        resid_mi = safe_mutual_info(target, resid_pred)

        price_pred = price_dir_a[:n - lag]
        m = min(len(price_pred), len(target))
        price_pred = price_pred[:m]
        price_acc, price_n = directional_accuracy(price_pred, target)
        price_mi = safe_mutual_info(target, price_pred)

        resid_b_target = sign_b[lag:][:m]
        resid_prop_acc, resid_prop_n = directional_accuracy(resid_pred, resid_b_target)

        ra = res_a[:m]
        rb = res_b[lag:][:m]
        corr = safe_corr(ra, rb)

        r_a = reg_a[:m]
        r_b = reg_b[lag:][:m]
        aligned = (r_a == r_b) & (r_a >= 0) & (r_b >= 0)
        misaligned = (r_a != r_b) & (r_a >= 0) & (r_b >= 0)
        aligned_acc = directional_accuracy(resid_pred[aligned], target[aligned])[0] if np.sum(aligned) > 10 else 0.0
        misaligned_acc = directional_accuracy(resid_pred[misaligned], target[misaligned])[0] if np.sum(misaligned) > 10 else 0.0

        results[str(lag)] = {
            "residual_directional_accuracy": round(resid_acc, 4),
            "residual_n": resid_n,
            "residual_mutual_info": round(resid_mi, 4),
            "price_directional_accuracy": round(price_acc, 4),
            "price_n": price_n,
            "price_mutual_info": round(price_mi, 4),
            "residual_propagation_accuracy": round(resid_prop_acc, 4),
            "residual_propagation_n": resid_prop_n,
            "residual_correlation": round(corr, 4),
            "regime_aligned_accuracy": round(aligned_acc, 4),
            "regime_misaligned_accuracy": round(misaligned_acc, 4),
            "residual_better_than_price": resid_acc > price_acc,
        }
    return results


def compute_nas100_impact(rol, horizon_idx=H50_IDX):
    nas_res = rol.get_residuals("NAS100")
    nas_sign = sign_strict(nas_res)
    jpy_crosses = [s for s in SYMBOLS if "JPY" in s]
    results = {}
    for sym in jpy_crosses:
        nas_res_s, res_sym = _align(nas_res, rol.get_residuals(sym))
        nas_sign_s = sign_strict(nas_res_s)
        fut = rol._data[sym]["fut_ret"][:len(res_sym)]
        dir_sym = sign_strict(fut[:, horizon_idx])
        corr = safe_corr(nas_res_s, res_sym)
        sym_res = {"residual_correlation_with_NAS100": round(corr, 4)}
        n_s = len(nas_res_s)
        for lag in LAGS:
            if lag >= n_s:
                continue
            pred = nas_sign_s[:n_s - lag]
            actual = dir_sym[lag:]
            acc, n_total = directional_accuracy(pred, actual)
            mi = safe_mutual_info(actual, pred)
            sym_res[str(lag)] = {
                "directional_accuracy": round(acc, 4),
                "n": n_total,
                "mutual_info": round(mi, 4),
            }
        results[sym] = sym_res
    return results


def compute_cascade(rol, horizon_idx=H50_IDX):
    eur_res, gbp_res = _align(rol.get_residuals("EURJPY"), rol.get_residuals("GBPJPY"))
    eur_sign = sign_strict(eur_res)
    gbp_sign = sign_strict(gbp_res)
    fut_gbp = rol._data["GBPJPY"]["fut_ret"][:len(gbp_res)]
    dir_gbp = sign_strict(fut_gbp[:, horizon_idx])
    n = len(eur_res)
    results = {}
    for lag in LAGS:
        if 1 + lag >= n:
            continue

        direct_pred = eur_sign[:n - 1 - lag]
        direct_target = dir_gbp[1 + lag:]
        m = min(len(direct_pred), len(direct_target))
        direct_pred, direct_target = direct_pred[:m], direct_target[:m]
        direct_acc, direct_n = directional_accuracy(direct_pred, direct_target)
        direct_mi = safe_mutual_info(direct_target, direct_pred)

        step1_pred = eur_sign[:n - 1]
        step1_target = gbp_sign[1:]
        m1 = min(len(step1_pred), len(step1_target))
        step1_acc, step1_total = directional_accuracy(step1_pred[:m1], step1_target[:m1])

        step2_pred = gbp_sign[:n - 1 - lag]
        step2_target = dir_gbp[1 + lag:]
        m2 = min(len(step2_pred), len(step2_target))
        step2_acc, step2_total = directional_accuracy(step2_pred[:m2], step2_target[:m2])

        min_len = min(n - 1 - lag, n - 1, len(dir_gbp) - 1 - lag)
        sp = eur_sign[:min_len]
        gr = gbp_sign[1:1 + min_len]
        dg = dir_gbp[1 + lag:1 + lag + min_len]
        valid = (sp != 0) & (gr != 0) & (dg != 0)
        if np.sum(valid) > 10:
            cascade_acc = float(np.mean(sp[valid] == dg[valid]))
            cascade_n = int(np.sum(valid))
        else:
            cascade_acc = 0.0
            cascade_n = 0

        results[str(lag)] = {
            "direct_accuracy": round(direct_acc, 4),
            "direct_n": direct_n,
            "direct_mutual_info": round(direct_mi, 4),
            "step1_eur_resid_to_gbp_resid_accuracy": round(step1_acc, 4),
            "step1_n": step1_total,
            "step2_gbp_resid_to_gbp_dir_accuracy": round(step2_acc, 4),
            "step2_n": step2_total,
            "cascade_accuracy": round(cascade_acc, 4),
            "cascade_n": cascade_n,
            "cascade_improves_over_direct": cascade_acc > direct_acc if cascade_n > 0 else False,
        }
    return results


def generate_markdown(report):
    lines = [
        "# ROL-4: Cross-Asset Residual Propagation",
        "",
        f"**Horizon:** H50  |  **Lags:** {LAGS}",
        "",
        "## Pairwise Results (Residual -> Direction @ H50)",
        "",
        "| Source | Target | Lag | Resid Acc | Price Acc | Resid MI | Resid Corr | Aligned Acc | Misaligned Acc | Resid Better? |",
        "|--------|--------|-----|-----------|-----------|----------|------------|-------------|----------------|---------------|",
    ]
    for p in report.get("pairs", []):
        for lag in sorted(p["metrics"].keys(), key=int):
            m = p["metrics"][lag]
            lines.append(
                f"| {p['source']} | {p['target']} | {lag} | "
                f"{m['residual_directional_accuracy']:.3f} | {m['price_directional_accuracy']:.3f} | "
                f"{m['residual_mutual_info']:.4f} | {m['residual_correlation']:.3f} | "
                f"{m['regime_aligned_accuracy']:.3f} | {m['regime_misaligned_accuracy']:.3f} | "
                f"{'YES' if m['residual_better_than_price'] else 'no'} |"
            )

    lines.extend([
        "",
        "## NAS100 Impact on JPY Crosses",
        "",
        "| Target | Corr | Lag1 Acc | Lag5 Acc | Lag10 Acc | Lag20 Acc |",
        "|--------|------|----------|----------|-----------|-----------|",
    ])
    for sym, m in report.get("nas100_impact", {}).items():
        corr = m.get("residual_correlation_with_NAS100", 0)
        l1 = m.get("1", {}).get("directional_accuracy", 0)
        l5 = m.get("5", {}).get("directional_accuracy", 0)
        l10 = m.get("10", {}).get("directional_accuracy", 0)
        l20 = m.get("20", {}).get("directional_accuracy", 0)
        lines.append(f"| {sym} | {corr:.3f} | {l1:.3f} | {l5:.3f} | {l10:.3f} | {l20:.3f} |")

    lines.extend([
        "",
        "## Cascade: EURJPY -> GBPJPY -> Direction",
        "",
        "| Lag | Direct Acc | Step1 (Resid->Resid) | Step2 (Resid->Dir) | Cascade Acc | Improves? |",
        "|-----|------------|---------------------|-------------------|-------------|-----------|",
    ])
    for lag in sorted(report.get("cascade", {}).keys(), key=int):
        m = report["cascade"][lag]
        lines.append(
            f"| {lag} | {m['direct_accuracy']:.3f} | {m['step1_eur_resid_to_gbp_resid_accuracy']:.3f} | "
            f"{m['step2_gbp_resid_to_gbp_dir_accuracy']:.3f} | {m['cascade_accuracy']:.3f} | "
            f"{'YES' if m['cascade_improves_over_direct'] else 'no'} |"
        )

    lines.extend([
        "",
        "## Summary",
        "",
    ])
    pairs = report.get("pairs", [])
    best_resid = max(
        ((p["source"], p["target"], lag, p["metrics"][lag]["residual_directional_accuracy"])
         for p in pairs for lag in p["metrics"]),
        key=lambda x: x[3], default=("", "", 0, 0.0)
    )
    lines.append(f"- **Best residual pair:** {best_resid[0]}->{best_resid[1]} lag={best_resid[2]} acc={best_resid[3]:.3f}")

    resid_wins = sum(1 for p in pairs for lag in p["metrics"] if p["metrics"][lag]["residual_better_than_price"])
    total = sum(len(p["metrics"]) for p in pairs)
    lines.append(f"- **Residual beats price:** {resid_wins}/{total} ({100*resid_wins/total:.1f}%)")
    cascade_improves = sum(1 for m in report.get("cascade", {}).values() if m["cascade_improves_over_direct"])
    lines.append(f"- **Cascade improves over direct:** {cascade_improves}/{len(report.get('cascade', {}))}")
    lines.append("")
    return "\n".join(lines)


def main():
    rol = ROLCore()
    rol.load_all()
    print("=" * 72)
    print("ROL-4: Cross-Asset Residual Propagation")
    print("=" * 72)

    report = {
        "type": "ROL-4 Cross-Asset Residual Propagation",
        "horizon": "H50",
        "lags_tested": LAGS,
    }

    print("\n--- Pairwise Residual -> Direction (H50) ---")
    pairs = []
    for sym_a in SYMBOLS:
        for sym_b in SYMBOLS:
            if sym_a == sym_b:
                continue
            metrics = compute_pair_metrics(rol, sym_a, sym_b)
            pairs.append({"source": sym_a, "target": sym_b, "metrics": metrics})
            for lag in sorted(metrics.keys(), key=int):
                m = metrics[lag]
                better = "RESIDUAL" if m["residual_better_than_price"] else "PRICE"
                print(f"  {sym_a}->{sym_b} lag={lag}: resid_acc={m['residual_directional_accuracy']:.3f} "
                      f"price_acc={m['price_directional_accuracy']:.3f} [{better}]")
    report["pairs"] = pairs

    print("\n--- NAS100 -> JPY Crosses ---")
    nas = compute_nas100_impact(rol)
    report["nas100_impact"] = nas
    for sym, m in nas.items():
        print(f"  NAS100->{sym}: corr={m.get('residual_correlation_with_NAS100', 'N/A')}")
        for lag in sorted([k for k in m.keys() if k != "residual_correlation_with_NAS100"], key=int):
            print(f"    lag={lag}: acc={m[lag]['directional_accuracy']:.3f} mi={m[lag]['mutual_info']:.4f}")

    print("\n--- Cascade: EURJPY -> GBPJPY -> Direction ---")
    cascade = compute_cascade(rol)
    report["cascade"] = cascade
    for lag in sorted(cascade.keys(), key=int):
        m = cascade[lag]
        impr = "CASCADE" if m["cascade_improves_over_direct"] else "DIRECT"
        print(f"  lag={lag}: direct={m['direct_accuracy']:.3f} step1={m['step1_eur_resid_to_gbp_resid_accuracy']:.3f} "
              f"step2={m['step2_gbp_resid_to_gbp_dir_accuracy']:.3f} cascade={m['cascade_accuracy']:.3f} [{impr}]")

    save_rol_report(report, "rol4_residual_propagation")

    md_path = Path(__file__).parent / "reports" / "ROL4_RESIDUAL_PROPAGATION.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md = generate_markdown(report)
    md_path.write_text(md)
    print(f"\nSaved {md_path}")

    print("\n" + "=" * 72)
    print("KEY FINDINGS")
    print("=" * 72)
    best_resid = max(
        ((p["source"], p["target"], lag, p["metrics"][lag]["residual_directional_accuracy"])
         for p in pairs for lag in p["metrics"]),
        key=lambda x: x[3], default=("", "", 0, 0.0)
    )
    print(f"  Best residual pair: {best_resid[0]}->{best_resid[1]} lag={best_resid[2]} acc={best_resid[3]:.3f}")
    resid_wins = sum(1 for p in pairs for lag in p["metrics"] if p["metrics"][lag]["residual_better_than_price"])
    total = sum(len(p["metrics"]) for p in pairs)
    print(f"  Residual beats price: {resid_wins}/{total} ({100*resid_wins/total:.1f}%)")
    cascade_improves = sum(1 for m in cascade.values() if m["cascade_improves_over_direct"])
    print(f"  Cascade improves over direct: {cascade_improves}/{len(cascade)}")


if __name__ == "__main__":
    main()
