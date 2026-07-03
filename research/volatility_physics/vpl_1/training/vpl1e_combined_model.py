import sys, os, json, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from datetime import datetime, timezone
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

from core.target_engine import (
    load_m5, compute_returns, compute_crf, compute_crf_deciles,
    compute_expansion_labels, compute_decile_expansion_profile,
    realized_variance_from_log_returns, SYMBOLS_FULL,
)
from core.sit_engine import compute_sit
from core.vcm_engine import compute_vcm

np.set_printoptions(precision=4, suppress=True, linewidth=200)


def month_bucket(ts):
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.year * 12 + dt.month - 1


def build_feature_matrix(symbol, saf_norm=True, sit_norm=True, vem_norm=True):
    data = load_m5(symbol)
    c, h, l, ts = data["close"], data["high"], data["low"], data["timestamp"]
    r = compute_returns(c)
    n_close = len(c)
    n_ret = len(r)

    rv = realized_variance_from_log_returns(r, 24)
    rv_close = np.full(n_close, np.nan)
    rv_close[1:] = rv

    sav = compute_crf(c, h, l)
    saf_val = sav["crf"]

    sit_out = compute_sit(c, h, l, r, rv, saf_val)
    I = sit_out["instability"]

    vcm_out = compute_vcm(r, rv)
    vem_val = vcm_out["vcm"]

    exp = compute_expansion_labels(c, baseline_window=24, horizons=(12,), thresholds=(1.5,))
    label = exp["expand_1.5_12"]["label"]

    n_feat = len(saf_val)
    n_lab = len(label)
    n_min = min(n_feat, n_lab)
    saf = saf_val[:n_min]
    sit = I[:n_min]
    vem = np.full(n_min, np.nan)
    vem[1:] = vem_val[:n_min-1] if n_min > 1 else vem_val[:n_min]
    lab = label[:n_min]
    ts_feat = ts[:n_min]

    keep = ~(np.isnan(saf) | np.isnan(sit) | np.isnan(lab))
    saf = saf[keep]
    sit = sit[keep]
    vem = vem[keep]
    lab = lab[keep]
    ts_feat = ts_feat[keep]
    vem = np.nan_to_num(vem, nan=0.0)

    if saf_norm:
        saf = (saf - np.mean(saf)) / np.std(saf) if np.std(saf) > 1e-12 else saf
    if sit_norm:
        sit = (sit - np.mean(sit)) / np.std(sit) if np.std(sit) > 1e-12 else sit
    if vem_norm:
        vem = (vem - np.mean(vem)) / np.std(vem) if np.std(vem) > 1e-12 else vem

    return saf, sit, vem, lab, ts_feat


def make_walkforward_splits(ts, n_splits=8):
    buckets = np.array([month_bucket(t) for t in ts])
    unique = np.sort(np.unique(buckets))
    splits = []
    for i in range(n_splits):
        if i + 3 >= len(unique):
            break
        train_end = unique[i + 2]
        test_start = unique[i + 3]
        test_end = unique[i + 4] if i + 4 < len(unique) else unique[-1] + 1
        train_mask = (buckets <= train_end)
        test_mask = (buckets >= test_start) & (buckets < test_end)
        if np.sum(train_mask) < 1000 or np.sum(test_mask) < 500:
            continue
        splits.append((train_mask, test_mask, unique[i], test_start, test_end))
    return splits


def compute_logistic(saf, sit, vem, lab, train, test):
    X_train = np.column_stack([saf[train], sit[train], vem[train],
                                (saf[train] * sit[train]),
                                (saf[train] * vem[train]),
                                (sit[train] * vem[train]),
                                saf[train]**2, sit[train]**2])
    X_test = np.column_stack([saf[test], sit[test], vem[test],
                               (saf[test] * sit[test]),
                               (saf[test] * vem[test]),
                               (sit[test] * vem[test]),
                               saf[test]**2, sit[test]**2])
    y_train, y_test = lab[train], lab[test]
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    return prob, y_test, clf.coef_[0]


def compute_xgb(saf, sit, vem, lab, train, test):
    try:
        import xgboost as xgb
    except ImportError:
        return None, None, None
    X_train = np.column_stack([saf[train], sit[train], vem[train],
                                (saf[train] * sit[train]),
                                (saf[train] * vem[train]),
                                (sit[train] * vem[train])])
    X_test = np.column_stack([saf[test], sit[test], vem[test],
                               (saf[test] * sit[test]),
                               (saf[test] * vem[test]),
                               (sit[test] * vem[test])])
    y_train, y_test = lab[train], lab[test]
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=np.sum(y_train==0)/max(np.sum(y_train==1), 1),
                               use_label_encoder=False, eval_metric="logloss",
                               verbosity=0)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    return prob, y_test, model


def compute_dt(saf, sit, vem, lab, train, test):
    X_train = np.column_stack([saf[train], sit[train], vem[train]])
    X_test = np.column_stack([saf[test], sit[test], vem[test]])
    y_train, y_test = lab[train], lab[test]
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, class_weight="balanced")
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    return prob, y_test, clf


def compute_baseline(saf, sit, vem, lab, train, test, feature="saf"):
    if feature == "saf":
        X_train, X_test = saf[train].reshape(-1, 1), saf[test].reshape(-1, 1)
    elif feature == "sit":
        X_train, X_test = sit[train].reshape(-1, 1), sit[test].reshape(-1, 1)
    y_train, y_test = lab[train], lab[test]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1], y_test


def top_decile_precision(prob, y_true):
    order = np.argsort(prob)
    n = len(prob)
    top = order[-n // 10:]
    if len(top) == 0:
        return 0.0
    return np.mean(y_true[top])


def evaluate_symbol(symbol):
    saf, sit, vem, lab, ts = build_feature_matrix(symbol)
    splits = make_walkforward_splits(ts, n_splits=8)
    if len(splits) < 2:
        return {"error": "not enough splits", "symbol": symbol}

    models = OrderedDict()
    models["saf_baseline"] = lambda tr, te: compute_baseline(saf, sit, vem, lab, tr, te, "saf")
    models["sit_baseline"] = lambda tr, te: compute_baseline(saf, sit, vem, lab, tr, te, "sit")
    models["logistic"] = lambda tr, te: compute_logistic(saf, sit, vem, lab, tr, te)
    models["dt"] = lambda tr, te: compute_dt(saf, sit, vem, lab, tr, te)
    try:
        import xgboost as xgb
        models["xgb"] = lambda tr, te: compute_xgb(saf, sit, vem, lab, tr, te)
    except ImportError:
        pass

    results = {k: {"aucs": [], "top_deciles": [], "coeffs": [], "n_train": [], "n_test": []} for k in models}

    for train_mask, test_mask, train_m, test_s, test_e in splits:
        n_train, n_test = int(np.sum(train_mask)), int(np.sum(test_mask))
        for name, fn in models.items():
            result = fn(train_mask, test_mask)
            prob, y_test = result[0], result[1]
            extra = result[2] if len(result) > 2 else None
            if prob is None:
                continue
            try:
                auc = roc_auc_score(y_test, prob)
            except ValueError:
                continue
            top_dec = top_decile_precision(prob, y_test)
            results[name]["aucs"].append(auc)
            results[name]["top_deciles"].append(top_dec)
            results[name]["n_train"].append(n_train)
            results[name]["n_test"].append(n_test)
            if name == "logistic" and extra is not None:
                results[name]["coeffs"].append(extra.tolist())

    summary = {"symbol": symbol, "n": len(lab)}
    unconditional = np.mean(lab)
    summary["unconditional_expansion"] = round(float(unconditional), 4)

    for name in sorted(results.keys()):
        r = results[name]
        aucs = r["aucs"]
        topds = r["top_deciles"]
        if len(aucs) == 0:
            continue
        summary[name] = {
            "mean_auc": round(float(np.mean(aucs)), 4),
            "std_auc": round(float(np.std(aucs)), 4),
            "mean_top_decile": round(float(np.mean(topds)), 4),
            "lift_vs_uncond": round(float(np.mean(topds) / unconditional), 4) if unconditional > 0 else None,
            "n_splits": len(aucs),
        }
        if name == "logistic" and r["coeffs"]:
            avg_coeff = np.mean(r["coeffs"], axis=0)
            summary[name]["coeff_mean"] = {
                "saf": round(float(avg_coeff[0]), 4),
                "sit": round(float(avg_coeff[1]), 4),
                "vem": round(float(avg_coeff[2]), 4),
                "saf_sit": round(float(avg_coeff[3]), 4),
                "saf_vem": round(float(avg_coeff[4]), 4),
                "sit_vem": round(float(avg_coeff[5]), 4),
                "saf2": round(float(avg_coeff[6]), 4),
                "sit2": round(float(avg_coeff[7]), 4),
            }

    return summary


if __name__ == "__main__":
    all_results = {}
    for sym in SYMBOLS_FULL:
        print(f"\n=== VPL-1E: {sym} ===")
        t0 = time.time()
        res = evaluate_symbol(sym)
        all_results[sym] = res
        elapsed = time.time() - t0
        if "error" in res:
            print(f"  ERROR: {res['error']}")
            continue
        print(f"  N={res['n']} ({elapsed:.1f}s)")
        print(f"  Unconditional expansion: {res['unconditional_expansion']:.4f}")
        for name in ["saf_baseline", "sit_baseline", "logistic", "dt", "xgb"]:
            if name not in res:
                continue
            r = res[name]
            print(f"  {name}: AUC={r['mean_auc']:.4f}±{r['std_auc']:.4f}  topD={r['mean_top_decile']:.4f}  lift={r['lift_vs_uncond']:.4f}x  n_splits={r['n_splits']}")
            if "coeff_mean" in r:
                c = r["coeff_mean"]
                print(f"    coeffs: SAF={c['saf']} SIT={c['sit']} VEM={c['vem']} SAF×SIT={c['saf_sit']} SAF×VEM={c['saf_vem']} SIT×VEM={c['sit_vem']} SAF²={c['saf2']} SIT²={c['sit2']}")

    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "vpl1e_results.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")
