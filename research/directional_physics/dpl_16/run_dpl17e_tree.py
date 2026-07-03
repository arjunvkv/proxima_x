"""DPL-17E: Tree-based interaction model (RandomForest).
Tests if nonlinear interactions (TPI × state) capture the DPL-17A cell effects.

Run: python -u proxima_x/research/directional_physics/dpl_16/run_dpl17e_tree.py
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
from sklearn.ensemble import RandomForestClassifier

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
    return {"symbol": symbol, "n": e - s, "ts": ts_m5[s:e], "close": close[s:e],
            "tpi": tpi[s:e], "saf": saf[s:e], "sit": sit[s:e], "vem": vem[s:e],
            "labels_3": labels_3[s:e]}

def build_clean(data):
    tpi = data["tpi"]
    tpi_sign = np.where(tpi > 0, 1.0, np.where(tpi < 0, -1.0, 0.0))
    tpi_mag = np.abs(tpi)
    saf = data["saf"]
    sit = data["sit"]
    vem = np.nan_to_num(data["vem"], nan=0.0)
    labels = data["labels_3"]
    y = labels.copy()
    X_a = np.column_stack([tpi_sign, tpi_mag])
    X_b = np.column_stack([saf, sit, vem])
    X_c = np.column_stack([tpi_sign, tpi_mag, saf, sit, vem,
                           tpi_sign * saf, tpi_sign * sit, tpi_mag * saf])
    # Filter NaN
    ok = ~np.any(np.isnan(X_c), axis=1) & ~np.isnan(y)
    X_a, X_b, X_c, y = X_a[ok], X_b[ok], X_c[ok], y[ok]
    # Remove any infinite
    ok2 = ~np.any(np.isinf(X_c), axis=1)
    X_a, X_b, X_c, y = X_a[ok2], X_b[ok2], X_c[ok2], y[ok2]
    return X_a, X_b, X_c, y

def tpi_baseline(X_a, y, train_frac=0.67):
    """TPI > 0 threshold baseline."""
    split = int(len(y) * train_frac)
    train_tpi, train_y = X_a[:split, 0], y[:split]
    test_tpi, test_y = X_a[split:, 0], y[split:]
    # Zero-threshold: TPI > 0 predicts UP
    nz = test_tpi != 0
    if np.sum(nz) < 10: return np.nan, 0
    pred = np.where(test_tpi[nz] > 0, 1, -1)
    acc = float(np.mean(pred == test_y[nz]))
    # Full sample too
    nz_all = train_tpi != 0
    if np.sum(nz_all) < 10: return np.nan, 0
    pred_all = np.where(train_tpi[nz_all] > 0, 1, -1)
    acc_all = float(np.mean(pred_all == train_y[nz_all]))
    return acc_all, acc

def rf_full(X, y, n_est=200, train_frac=0.67):
    """RandomForest full-sample 67/33 split."""
    split = int(len(y) * train_frac)
    clf = RandomForestClassifier(n_estimators=n_est, max_depth=8, min_samples_leaf=50,
                                  random_state=42, n_jobs=-1)
    clf.fit(X[:split], y[:split])
    pred = clf.predict(X[split:])
    acc = float(np.mean(pred == y[split:]))
    return acc, clf

def rf_walk_forward(X, y, windows=8, n_est=200):
    """Purged walk-forward with RF."""
    n = len(y)
    ws = n // windows
    results = []
    for w in range(windows):
        tr_s = 0; tr_e = ws * (w + 1)
        te_s = tr_e; te_e = min(te_s + ws, n)
        if te_e - te_s < 20: continue
        try:
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=8,
                                          min_samples_leaf=50,
                                          random_state=42, n_jobs=-1)
            clf.fit(X[tr_s:tr_e], y[tr_s:tr_e])
            pred = clf.predict(X[te_s:te_e])
            results.append(float(np.mean(pred == y[te_s:te_e])))
        except:
            pass
    return results

if __name__ == "__main__":
    print("=" * 65)
    print("DPL-17E: Tree-based Interaction Model")
    print("=" * 65)

    all_data = {}
    for sym in SYMBOLS:
        d = prepare(sym)
        if d:
            X_a, X_b, X_c, y = build_clean(d)
            all_data[sym] = (X_a, X_b, X_c, y)

    print(f"\n{'='*65}")
    print("COMPARISON (RF max_depth=8, min_samples_leaf=50)")
    print(f"{'='*65}")
    print(f"  A = RF(TPI sign + magnitude)")
    print(f"  B = RF(SAF + SIT + VEM)")
    print(f"  C = RF(TPI + State + Interactions)")
    print(f"  BL = TPI>0 zero-threshold")
    print("")

    results = {}
    for sym, (X_a, X_b, X_c, y) in all_data.items():
        n = len(y)
        # TPI baseline
        bl_train, bl_test = tpi_baseline(X_a, y)
        # RF models
        acc_a, cls_a = rf_full(X_a, y)
        acc_b, _ = rf_full(X_b, y)
        acc_c, cls_c = rf_full(X_c, y)
        # WF
        wf_a = rf_walk_forward(X_a, y)
        wf_c = rf_walk_forward(X_c, y)
        results[sym] = {
            "n": n,
            "baseline": {"train": bl_train, "test": bl_test},
            "rf_A": {"acc": acc_a, "wf_mean": float(np.mean(wf_a)) if wf_a else None, "wf_n": len(wf_a)},
            "rf_B": {"acc": acc_b},
            "rf_C": {"acc": acc_c, "wf_mean": float(np.mean(wf_c)) if wf_c else None, "wf_n": len(wf_c)},
        }
        def f(v): return f"{v:.4f}" if v is not None and not np.isnan(v) else "    -"
        print(f"  {sym:8s}  (n={n})")
        print(f"    BL(TPI>0):  train={f(bl_train)}  test={f(bl_test)}")
        print(f"    A(RF,TPI):  {f(acc_a)}  WF={f(np.mean(wf_a)) if wf_a else '    -'} ({len(wf_a)} win)")
        print(f"    B(RF,St):   {f(acc_b)}")
        print(f"    C(RF,Int):  {f(acc_c)}  WF={f(np.mean(wf_c)) if wf_c else '    -'} ({len(wf_c)} win)")

    print(f"\n{'='*65}")
    print("RF FEATURE IMPORTANCE (Model C)")
    print(f"{'='*65}")
    features = ["tpi_sign","tpi_mag","saf","sit","vem","sign*saf","sign*sit","mag*saf"]
    print(f"  {'Feature':<15s}", end="")
    for sym in SYMBOLS:
        print(f"  {sym:>8s}", end="")
    print("")
    for i, fn in enumerate(features):
        print(f"  {fn:<15s}", end="")
        for sym in SYMBOLS:
            if sym in all_data:
                _, _, _, y = all_data[sym]
                X_a, X_b, X_c, _ = all_data[sym]
                _, clf = rf_full(X_c, y)
                print(f"  {clf.feature_importances_[i]:>8.4f}", end="")
        print("")

    print(f"\n{'='*65}")
    print("INCREMENTAL LIFT (C vs A)")
    print(f"{'='*65}")
    for sym, r in results.items():
        lift = r["rf_C"]["acc"] - r["rf_A"]["acc"]
        vs_bl = r["rf_C"]["acc"] - r["baseline"]["test"]
        print(f"  {sym:8s}  A(RF)={r['rf_A']['acc']:.4f}  C(RF)={r['rf_C']['acc']:.4f}  "
              f"Lift(C-A)={lift:+.4f}  Lift(C-BL)={vs_bl:+.4f}")

    with open(os.path.join(REPORT_DIR, "dpl17e_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDPL-17E -> dpl17e_results.json")
