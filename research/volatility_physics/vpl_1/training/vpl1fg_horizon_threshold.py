import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from core.target_engine import (
    load_m5, compute_returns, compute_crf, compute_crf_deciles,
    compute_expansion_labels, realized_variance_from_log_returns,
    compute_decile_expansion_profile, SYMBOLS_FULL,
)
from core.sit_engine import compute_sit
from core.vcm_engine import compute_vcm

np.set_printoptions(precision=4, suppress=True)


def build_features(symbol):
    data = load_m5(symbol)
    c, h, l, ts = data["close"], data["high"], data["low"], data["timestamp"]
    r = compute_returns(c)
    n_close = len(c)
    rv = realized_variance_from_log_returns(r, 24)
    sav = compute_crf(c, h, l)
    saf_val = sav["crf"]
    sit_out = compute_sit(c, h, l, r, rv, saf_val)
    I = sit_out["instability"]
    vcm_out = compute_vcm(r, rv)
    vem_val = vcm_out["vcm"]
    return c, h, l, ts, r, saf_val, I, vem_val, n_close


def run_horizon_test(symbol, c, saf_val, I, vem_val, ts):
    results = {}
    for N in [3, 6, 24]:
        exp = compute_expansion_labels(c, baseline_window=24, horizons=(N,), thresholds=(1.5,))
        label = exp[f"expand_1.5_{N}"]["label"]
        n_min = min(len(saf_val), len(label))
        saf, sit, vem, lab = saf_val[:n_min], I[:n_min], np.full(n_min, np.nan), label[:n_min]
        vem[1:] = vem_val[:n_min-1] if n_min > 1 else vem_val[:n_min]
        ts_min = ts[:n_min]
        keep = ~(np.isnan(saf) | np.isnan(sit) | np.isnan(lab))
        saf, sit, vem, lab, ts_f = saf[keep], sit[keep], vem[keep], lab[keep], ts_min[keep]
        vem = np.nan_to_num(vem, nan=0.0)
        saf_z = (saf - np.mean(saf)) / max(np.std(saf), 1e-12)
        sit_z = (sit - np.mean(sit)) / max(np.std(sit), 1e-12)
        vem_z = (vem - np.mean(vem)) / max(np.std(vem), 1e-12)
        uncond = float(np.mean(lab))
        buckets = np.array([datetime.fromtimestamp(t, tz=timezone.utc).year * 12 +
                            datetime.fromtimestamp(t, tz=timezone.utc).month - 1 for t in ts_f])
        unique = np.sort(np.unique(buckets))
        n_splits = min(8, len(unique) - 4)
        aucs, lifts = [], []
        for i in range(n_splits):
            train_end, test_start = unique[i + 2], unique[i + 3]
            test_end = unique[i + 4] if i + 4 < len(unique) else unique[-1] + 1
            tr, te = (buckets <= train_end), (buckets >= test_start) & (buckets < test_end)
            if np.sum(tr) < 500 or np.sum(te) < 200:
                continue
            X_tr = np.column_stack([saf_z[tr], sit_z[tr], vem_z[tr], saf_z[tr] * sit_z[tr]])
            X_te = np.column_stack([saf_z[te], sit_z[te], vem_z[te], saf_z[te] * sit_z[te]])
            try:
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf.fit(X_tr, lab[tr])
                prob = clf.predict_proba(X_te)[:, 1]
                auc = roc_auc_score(lab[te], prob)
                top_dec = np.mean(lab[te][np.argsort(prob)[-len(prob)//10:]]) if len(prob)//10 > 0 else 0
                aucs.append(auc)
                lifts.append(top_dec / max(uncond, 0.001))
            except:
                continue
        results[N] = {"mean_auc": float(np.mean(aucs)) if aucs else None,
                      "mean_top_lift": float(np.mean(lifts)) if lifts else None,
                      "unconditional": round(uncond, 4), "n_splits": len(aucs)}
    return results


def run_threshold_test(symbol, c, saf_val, I, vem_val, ts):
    results = {}
    for thresh in [1.25, 2.0, 3.0]:
        exp = compute_expansion_labels(c, baseline_window=24, horizons=(12,), thresholds=(thresh,))
        label = exp[f"expand_{thresh}_12"]["label"]
        n_min = min(len(saf_val), len(label))
        saf, sit, vem, lab = saf_val[:n_min], I[:n_min], np.full(n_min, np.nan), label[:n_min]
        vem[1:] = vem_val[:n_min-1] if n_min > 1 else vem_val[:n_min]
        ts_min = ts[:n_min]
        keep = ~(np.isnan(saf) | np.isnan(sit) | np.isnan(lab))
        saf, sit, vem, lab, ts_f = saf[keep], sit[keep], vem[keep], lab[keep], ts_min[keep]
        vem = np.nan_to_num(vem, nan=0.0)
        saf_z = (saf - np.mean(saf)) / max(np.std(saf), 1e-12)
        sit_z = (sit - np.mean(sit)) / max(np.std(sit), 1e-12)
        vem_z = (vem - np.mean(vem)) / max(np.std(vem), 1e-12)
        uncond = float(np.mean(lab))
        buckets = np.array([datetime.fromtimestamp(t, tz=timezone.utc).year * 12 +
                            datetime.fromtimestamp(t, tz=timezone.utc).month - 1 for t in ts_f])
        unique = np.sort(np.unique(buckets))
        n_splits = min(8, len(unique) - 4)
        aucs, lifts = [], []
        for i in range(n_splits):
            train_end, test_start = unique[i + 2], unique[i + 3]
            test_end = unique[i + 4] if i + 4 < len(unique) else unique[-1] + 1
            tr, te = (buckets <= train_end), (buckets >= test_start) & (buckets < test_end)
            if np.sum(tr) < 500 or np.sum(te) < 200:
                continue
            X_tr = np.column_stack([saf_z[tr], sit_z[tr], vem_z[tr], saf_z[tr] * sit_z[tr]])
            X_te = np.column_stack([saf_z[te], sit_z[te], vem_z[te], saf_z[te] * sit_z[te]])
            try:
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf.fit(X_tr, lab[tr])
                prob = clf.predict_proba(X_te)[:, 1]
                auc = roc_auc_score(lab[te], prob)
                top_dec = np.mean(lab[te][np.argsort(prob)[-len(prob)//10:]]) if len(prob)//10 > 0 else 0
                aucs.append(auc)
                lifts.append(top_dec / max(uncond, 0.001))
            except:
                continue
        results[thresh] = {"mean_auc": float(np.mean(aucs)) if aucs else None,
                           "mean_top_lift": float(np.mean(lifts)) if lifts else None,
                           "unconditional": round(uncond, 4), "n_splits": len(aucs)}
    return results


if __name__ == "__main__":
    from datetime import datetime, timezone

    all_results = {}
    for sym in SYMBOLS_FULL:
        print(f"\n=== VPL-1F/G: {sym} ===")
        t0 = time.time()
        c, h, l, ts, r, saf, I, vem, _ = build_features(sym)

        hz = run_horizon_test(sym, c, saf, I, vem, ts)
        print(f"  Horizon stability:")
        for N in sorted(hz.keys()):
            v = hz[N]
            if v["mean_auc"] is not None:
                print(f"    N={N}: AUC={v['mean_auc']:.4f}  top_lift={v['mean_top_lift']:.4f}x  uncond={v['unconditional']:.4f}  n={v['n_splits']}")
            else:
                print(f"    N={N}: FAILED")

        th = run_threshold_test(sym, c, saf, I, vem, ts)
        print(f"  Threshold stability:")
        for t in sorted(th.keys()):
            v = th[t]
            if v["mean_auc"] is not None:
                print(f"    {t}x: AUC={v['mean_auc']:.4f}  top_lift={v['mean_top_lift']:.4f}x  uncond={v['unconditional']:.4f}  n={v['n_splits']}")
            else:
                print(f"    {t}x: FAILED")

        all_results[sym] = {"horizon": hz, "threshold": th}
        print(f"  ({time.time()-t0:.1f}s)")

    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "vpl1fg_results.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")
