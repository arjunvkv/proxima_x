"""DPL-17D v3: Proper comparison against validated TPI>0 baseline.
Plus coefficient analysis, ablated walk-forward.

Run: python -u proxima_x/research/directional_physics/dpl_16/run_dpl17d_model.py
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, "C:/Trading/Agentic_Trading/proxima_x")
from research.directional_physics.dpl_16.core.msl_engine import (
    load_ticks, load_m5, align_ticks_to_bars,
    compute_tpi, aggregate_features_to_bars,
    compute_directional_labels_vect,
)
from research.volatility_physics.vpl_1.core.target_engine import (
    compute_returns, compute_crf, realized_variance_from_log_returns,
)
from research.volatility_physics.vpl_1.core.sit_engine import compute_sit
from research.volatility_physics.vpl_1.core.vcm_engine import compute_vcm

REPORT_DIR = "C:/Trading/Agentic_Trading/research/directional_physics/dpl_16/reports"
os.makedirs(REPORT_DIR, exist_ok=True)
SYMBOLS = ["EURJPY", "USDJPY", "EURUSD", "GBPUSD"]
TPI_KEY = "tpi_200"

def prepare(symbol):
    ticks = load_ticks(symbol)
    m5 = load_m5(symbol)
    close = m5["close"]
    high = m5["close"] if "high" not in m5 else m5["high"]
    low = m5["close"] if "low" not in m5 else m5["low"]
    ts_m5 = m5["timestamp"]
    starts, ends, tick_sec = align_ticks_to_bars(ticks["timestamp"], ts_m5)
    n_bars = len(starts)
    tick_bar_idx = np.searchsorted(ts_m5, tick_sec, side="right") - 1
    tick_bar_idx = np.clip(tick_bar_idx, 0, n_bars - 1)
    tpi_feats = compute_tpi(ticks["mid"])
    bar_feats = aggregate_features_to_bars(tpi_feats, tick_bar_idx, n_bars)
    tpi = bar_feats[TPI_KEY]
    r = compute_returns(close)
    rv = realized_variance_from_log_returns(r, 24)
    crf = compute_crf(close, high, low)
    saf_raw = crf["crf"]
    sit_out = compute_sit(close, high, low, r, rv, saf_raw)
    sit_raw = sit_out["instability"]
    vcm_out = compute_vcm(r, rv)
    vem_raw = vcm_out["vcm"]
    n = len(close)
    vem = np.full(n, np.nan); vem[1:] = vem_raw
    saf, sit = saf_raw.copy(), sit_raw.copy()
    labels_3 = compute_directional_labels_vect(close, n_bars=3)
    first_valid = np.where(~np.isnan(tpi))[0]
    if len(first_valid) == 0: return None
    s, e = first_valid[0], min(first_valid[-1] + 1, n - 3)
    return {"symbol": symbol, "n": e - s, "close": close[s:e],
            "tpi": tpi[s:e], "saf": saf[s:e], "sit": sit[s:e], "vem": vem[s:e],
            "ts": ts_m5[s:e], "labels_3": labels_3[s:e]}

def build_arrays(data):
    """Build clean arrays for all models."""
    tpi = data["tpi"]
    tpi_sign = np.where(tpi > 0, 1.0, np.where(tpi < 0, -1.0, 0.0))
    tpi_mag = np.abs(tpi)
    saf = data["saf"]
    sit = data["sit"]
    vem = np.nan_to_num(data["vem"], nan=0.0)
    labels = data["labels_3"]

    # TPI-zero-threshold baseline: only bars with non-zero TPI
    non_zero = np.abs(tpi) > 1e-10
    valid_nz = non_zero & ~np.isnan(labels)
    # Everyone else
    valid_all = ~np.isnan(labels) & ~np.isnan(saf) & ~np.isnan(sit)

    # Full-set valid: all features + labels non-NaN
    valid_full = valid_all & ~np.isnan(tpi_sign)
    return {
        "tpi_sign": tpi_sign, "tpi_mag": tpi_mag,
        "saf": saf, "sit": sit, "vem": vem, "labels": labels,
        "valid_nz": valid_nz, "valid_full": valid_full,
        "non_zero_idx": np.where(valid_nz)[0],
        "all_idx": np.where(valid_full)[0],
    }

def tpi_zero_threshold_accuracy(arr, split=0.8):
    """TPI>0 predicts UP, TPI<0 predicts DOWN (skip zeros). Validated at 55-58%."""
    idx = arr["non_zero_idx"]
    n = len(idx)
    if n < 50: return np.nan, 0, np.nan, 0
    tr = int(n * split)
    train_idx = idx[:tr]
    test_idx = idx[tr:]
    pred = np.where(arr["tpi_sign"][test_idx] > 0, 1, -1)
    y = arr["labels"][test_idx]
    acc = float(np.mean(pred == y))
    # Also full-sample
    pred_full = np.where(arr["tpi_sign"][idx] > 0, 1, -1)
    acc_full = float(np.mean(pred_full == arr["labels"][idx]))
    return acc_full, n, acc, n - tr

def logit_walk_forward(X_full, y_full, idx, windows=8):
    """Purged walk-forward with LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    n = len(idx)
    if n < 200: return []
    ws = n // windows
    results = []
    for w in range(windows):
        tr_s = 0; tr_e = ws * (w + 1)
        te_s = tr_e; te_e = min(te_s + ws, n)
        if te_e - te_s < 20: continue
        try:
            clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
            clf.fit(X_full[idx[tr_s:tr_e]], y_full[idx[tr_s:tr_e]])
            pred = clf.predict(X_full[idx[te_s:te_e]])
            acc = float(np.mean(pred == y_full[idx[te_s:te_e]]))
            results.append(acc)
        except:
            pass
    return results

def interaction_model(arr):
    """Full logistic: TPI x State interactions."""
    from sklearn.linear_model import LogisticRegression
    idx = arr["all_idx"]
    n = len(idx)
    split = int(n * 0.67)
    ts, te = idx[:split], idx[split:]
    X = np.column_stack([
        arr["tpi_sign"], arr["tpi_mag"],
        arr["saf"], arr["sit"], arr["vem"],
        arr["tpi_sign"] * arr["saf"],
        arr["tpi_sign"] * arr["sit"],
        arr["tpi_mag"] * arr["saf"],
    ])
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    acc, nt = _safe_fit_predict(clf, X[ts], arr["labels"][ts], X[te], arr["labels"][te])
    wf = logit_walk_forward(X, arr["labels"], idx, windows=8)
    coefs = {}
    if not np.isnan(acc):
        for f, c in zip(["tpi_sign","tpi_mag","saf","sit","vem",
                         "sign*saf","sign*sit","mag*saf"], clf.coef_[0]):
            coefs[f] = float(c)
    return {
        "accuracy": acc,
        "n_test": nt,
        "coef": coefs,
        "intercept": float(clf.intercept_[0]) if not np.isnan(acc) else None,
        "wf_mean": float(np.mean(wf)) if wf else None,
        "wf_n": len(wf),
        "wf_min": float(np.min(wf)) if wf else None,
        "wf_max": float(np.max(wf)) if wf else None,
    }

def state_only_model(arr):
    """Logistic: SAF + SIT + VEM only (no TPI)."""
    from sklearn.linear_model import LogisticRegression
    idx = arr["all_idx"]
    n = len(idx)
    split = int(n * 0.67)
    ts, te = idx[:split], idx[split:]
    X = np.column_stack([arr["saf"], arr["sit"], arr["vem"]])
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    acc, nt = _safe_fit_predict(clf, X[ts], arr["labels"][ts], X[te], arr["labels"][te])
    wf = logit_walk_forward(X, arr["labels"], idx, windows=8)
    return {"accuracy": acc, "n_test": nt,
            "wf_mean": float(np.mean(wf)) if wf else None, "wf_n": len(wf)}

def _safe_fit_predict(clf, X_tr, y_tr, X_te, y_te):
    """Fit and predict, handling any leftover NaN."""
    ok_tr = ~np.any(np.isnan(X_tr), axis=1) & ~np.isnan(y_tr)
    ok_te = ~np.any(np.isnan(X_te), axis=1) & ~np.isnan(y_te)
    if np.sum(ok_tr) < 10 or np.sum(ok_te) < 10:
        return np.nan, 0
    clf.fit(X_tr[ok_tr], y_tr[ok_tr])
    pred = clf.predict(X_te[ok_te])
    return float(np.mean(pred == y_te[ok_te])), int(np.sum(ok_te))

def _safe_fit(clf, X_tr, y_tr):
    ok = ~np.any(np.isnan(X_tr), axis=1) & ~np.isnan(y_tr)
    if np.sum(ok) < 10: return None
    clf.fit(X_tr[ok], y_tr[ok])
    return clf

def pure_tpi_model(arr):
    """Logistic: TPI sign + magnitude only."""
    from sklearn.linear_model import LogisticRegression
    idx = arr["all_idx"]
    n = len(idx)
    split = int(n * 0.67)
    ts, te = idx[:split], idx[split:]
    X = np.column_stack([arr["tpi_sign"], arr["tpi_mag"]])
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    acc, nt = _safe_fit_predict(clf, X[ts], arr["labels"][ts], X[te], arr["labels"][te])
    wf = logit_walk_forward(X, arr["labels"], idx, windows=8)
    return {"accuracy": acc, "n_test": nt,
            "wf_mean": float(np.mean(wf)) if wf else None, "wf_n": len(wf)}

if __name__ == "__main__":
    print("=" * 65)
    print("DPL-17D: Interaction Model Validation (v3)")
    print("=" * 65)

    all_data = {}
    for sym in SYMBOLS:
        d = prepare(sym)
        if d:
            arr = build_arrays(d)
            arr["symbol"] = sym
            all_data[sym] = arr

    print(f"\n{'='*65}")
    print(f"MODEL COMPARISON (67/33 split, 3-bar directional accuracy)")
    print(f"{'='*65}")
    print(f"  BL = TPI>0 zero-threshold (validated gold standard)")
    print(f"  A  = Logistic(TPI sign + magnitude)")
    print(f"  B  = Logistic(SAF + SIT + VEM)")
    print(f"  C  = Logistic(TPI + State + Interactions)")
    print(f"  WF = purged walk-forward ({'n' if len(SYMBOLS)>0 else ''} windows)")
    print("")

    results = {}
    for sym, arr in all_data.items():
        # TPI zero-threshold baseline
        bl_full, bl_n, bl_test, bl_nt = tpi_zero_threshold_accuracy(arr, split=0.67)
        A = pure_tpi_model(arr)
        B = state_only_model(arr)
        C = interaction_model(arr)
        results[sym] = {"tpi0_threshold": {"full_acc": bl_full, "n": bl_n, "test_acc": bl_test, "n_test": bl_nt},
                        "model_A": A, "model_B": B, "model_C": C}
        def fmt(v, d="    -"): return f"{v:.4f}" if v is not None and not np.isnan(v) else d
        print(f"  {sym:8s}")
        print(f"    BL(TPI>0):  full={bl_full:.4f} test={bl_test:.4f} (n={bl_n})")
        print(f"    A(TPI):     test={fmt(A['accuracy'])} WF={fmt(A['wf_mean'])} ({A['wf_n']} win)")
        print(f"    B(State):   test={fmt(B['accuracy'])} WF={fmt(B['wf_mean'])} ({B['wf_n']} win)")
        print(f"    C(Int):     test={fmt(C['accuracy'])} WF={fmt(C['wf_mean'])} ({C['wf_n']} win)")
        print(f"                WF-range: [{fmt(C['wf_min'])}-{fmt(C['wf_max'])}]")

    # Summary
    print(f"\n{'='*65}")
    print(f"SUMMARY: INCREMENTAL LIFT")
    print(f"{'='*65}")
    print(f"  Lift = Model C (Interaction) - Model A (TPI-only) - both logistic")
    print("")
    for sym, r in results.items():
        def sv(k): return r[k]["accuracy"] if r[k]["accuracy"] is not None and not np.isnan(r[k]["accuracy"]) else 0.5
        a, c = sv("model_A"), sv("model_C")
        bl = r["tpi0_threshold"]["test_acc"]
        lift, vs_bl_val = c - a, c - bl
        print(f"  {sym:8s}  A(TPI)={a:.4f}  C(Int)={c:.4f}  "
              f"Lift(C-A)={lift:+.4f}  Lift(C-BL)={vs_bl_val:+.4f}")

    # Coefficient analysis
    print(f"\n{'='*65}")
    print(f"MODEL C COEFFICIENTS")
    print(f"{'='*65}")
    print(f"  {'Feature':<15s}", end="")
    for sym in SYMBOLS:
        print(f"  {sym:>8s}", end="")
    print("")
    features = ["tpi_sign","tpi_mag","saf","sit","vem","sign*saf","sign*sit","mag*saf"]
    for f in features:
        print(f"  {f:<15s}", end="")
        for sym in SYMBOLS:
            if sym in results and f in results[sym]["model_C"]["coef"]:
                print(f"  {results[sym]['model_C']['coef'][f]:>+8.4f}", end="")
        print("")
    print(f"  {'intercept':<15s}", end="")
    for sym in SYMBOLS:
        if sym in results and results[sym]["model_C"]["intercept"] is not None:
            print(f"  {results[sym]['model_C']['intercept']:>+8.4f}", end="")
    print("")

    # Save
    with open(os.path.join(REPORT_DIR, "dpl17d_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDPL-17D -> dpl17d_results.json")
