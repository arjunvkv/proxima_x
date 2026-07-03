import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import polars as pl
import numpy as np
from datetime import datetime
from collections import defaultdict

DATA_BASE = "C:/Trading/Agentic_Trading/data/intraday"
SYMBOLS = ["XAUUSD", "EURJPY", "USDJPY", "GBPJPY"]
HORIZONS = [12, 24, 48]


def load_m5(symbol):
    path = f"{DATA_BASE}/{symbol}_M5.parquet"
    df = pl.read_parquet(path)
    arr = df.to_numpy()
    ts = arr[:, 0].astype(np.int64)
    o, h, l, c, v = [arr[:, i].astype(np.float64) for i in range(1, 6)]
    return {"symbol": symbol, "timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v, "n": len(c)}


def year_from_ts(t):
    return datetime.fromtimestamp(t).year


def rolling_percentile(x, window=200, p=95):
    out = np.zeros(len(x))
    for i in range(window, len(x)):
        out[i] = float(np.percentile(x[i - window:i], p))
    return out


class ImpulseDetector:
    def __init__(self, window=200, pct=95, k=2.0):
        self.window = window
        self.pct = pct
        self.k = k

    def detect(self, data):
        n = data["n"]
        tr = np.abs(data["high"] - data["low"])
        ret = np.diff(np.log(data["close"]), prepend=np.log(data["close"][0]))
        displacement = np.abs(data["close"] - np.roll(data["close"], 1))
        displacement[0] = 0.0

        tr_p95 = rolling_percentile(tr, self.window, self.pct)
        ret_p95 = rolling_percentile(np.abs(ret), self.window, self.pct)
        disp_p95 = rolling_percentile(displacement, self.window, self.pct)

        impulses = np.zeros(n, dtype=bool)
        impulse_types = np.zeros(n, dtype=int)
        impulse_mags = np.zeros(n)

        for i in range(self.window, n):
            mag = 0.0
            typ = 0
            if tr[i] > tr_p95[i] and tr_p95[i] > 0:
                mag += tr[i] / tr_p95[i]
                typ = 1
            if abs(ret[i]) > ret_p95[i] and ret_p95[i] > 0:
                mag += abs(ret[i]) / ret_p95[i]
                typ = 2 if typ == 0 else typ
            if displacement[i] > disp_p95[i] * self.k and disp_p95[i] > 0:
                mag += displacement[i] / disp_p95[i]
                typ = 3 if typ == 0 else typ
            if typ > 0 and mag > 1.0:
                impulses[i] = True
                impulse_types[i] = typ
                impulse_mags[i] = mag

        return impulses, impulse_types, impulse_mags


class ResponseGeometry:
    def compute(self, data, idx, max_horizon=48):
        close = data["close"]
        n = len(close)
        end = min(idx + max_horizon, n)
        if end - idx < 5:
            return None
        entry = close[idx]
        future = close[idx + 1:end]
        rets = (future - entry) / entry
        if len(rets) == 0:
            return None
        mfe = float(np.max(rets))
        mae = float(np.min(rets))
        final_ret = float(rets[-1])
        direction = 1 if final_ret > 0 else -1
        half_life = None
        for h in range(1, len(rets)):
            if abs(rets[h]) >= abs(final_ret) * 0.5:
                half_life = h
                break
        return {
            "mfe": mfe, "mae": mae, "final_ret": final_ret,
            "direction": direction, "range": float(np.ptp(rets)),
            "half_life": half_life, "n_bars": len(rets)
        }


class PreStateFeatures:
    def extract(self, data, idx, lookback=20):
        close = data["close"]
        vol = data["volume"]
        if idx < lookback:
            return None
        wins = close[idx - lookback:idx]
        vol_w = vol[idx - lookback:idx]
        rets = np.diff(wins) / (wins[:-1] + 1e-8)
        features = {
            "volatility": float(np.std(rets)),
            "trend": float((wins[-1] - wins[0]) / (wins[0] + 1e-8)),
            "mean_vol": float(np.mean(vol_w)),
            "range_ratio": float((np.max(wins) - np.min(wins)) / (wins[0] + 1e-8)),
            "skew": float(np.mean(rets ** 3)) / (float(np.std(rets)) ** 3 + 1e-8),
        }
        return features


def compute_asymmetry(responses):
    if len(responses) < 10:
        return None
    cont = [r for r in responses if r["final_ret"] > 0]
    rev = [r for r in responses if r["final_ret"] < 0]
    n_cont = len(cont)
    n_rev = len(rev)
    p_cont = n_cont / len(responses)
    p_rev = n_rev / len(responses)
    avg_mfe = float(np.mean([r["mfe"] for r in responses]))
    avg_mae = float(np.mean([r["mae"] for r in responses]))
    return {
        "n_events": len(responses), "p_cont": round(p_cont, 4), "p_rev": round(p_rev, 4),
        "bias": round(p_cont - p_rev, 4), "avg_mfe": round(avg_mfe, 6), "avg_mae": round(avg_mae, 6),
        "avg_mag": round(avg_mfe - abs(avg_mae), 6)
    }


def main():
    print("=" * 80)
    print("  DPL-13 INTRADAY IMPULSE RESPONSE ASYMMETRY ENGINE")
    print("=" * 80)

    all_results = {}

    for sym in SYMBOLS:
        print(f"\n  Loading {sym}...", end=" ")
        data = load_m5(sym)
        print(f"{data['n']} bars")

        years = np.array([year_from_ts(t) for t in data["timestamp"]])
        unique_years = sorted(set(years))

        detector = ImpulseDetector(window=200, pct=95, k=2.0)
        impulses, imp_types, imp_mags = detector.detect(data)

        n_impulses = int(np.sum(impulses))
        print(f"  Impulses detected: {n_impulses} ({n_impulses/data['n']*100:.1f}% of bars)")

        geo = ResponseGeometry()
        pre = PreStateFeatures()

        event_features = []
        event_responses = []
        event_years = []

        for i in range(200, data["n"]):
            if not impulses[i]:
                continue
            resp = geo.compute(data, i, max_horizon=48)
            if resp is None:
                continue
            f = pre.extract(data, i, lookback=20)
            if f is None:
                continue
            f["impulse_type"] = int(imp_types[i])
            f["impulse_mag"] = float(imp_mags[i])
            f["symbol"] = sym
            event_features.append(f)
            event_responses.append(resp)
            event_years.append(years[i])

        print(f"  Valid events: {len(event_features)}")

        if len(event_features) < 50:
            print(f"  SKIP: too few events")
            continue

        feat_keys = ["volatility", "trend", "range_ratio", "skew", "impulse_mag"]
        X = np.array([[f[k] for k in feat_keys] for f in event_features])
        X_n = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

        n_clusters = min(5, len(X_n) // 20)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        labels = km.fit_predict(X_n)

        print(f"  Clusters: {n_clusters}")

        sym_results = {"overall": compute_asymmetry(event_responses), "by_cluster": {}, "by_year": {}} if compute_asymmetry(event_responses) else {}

        if not sym_results:
            continue

        for cid in range(n_clusters):
            mask = labels == cid
            c_responses = [event_responses[j] for j in range(len(event_responses)) if mask[j]]
            asym = compute_asymmetry(c_responses)
            if asym:
                sym_results["by_cluster"][f"c{cid}"] = asym

        by_year = defaultdict(list)
        for j, yr in enumerate(event_years):
            by_year[yr].append(event_responses[j])
        for yr, resps in sorted(by_year.items()):
            asym = compute_asymmetry(resps)
            if asym:
                sym_results["by_year"][str(yr)] = asym

        cluster_biases = [v["bias"] for v in sym_results["by_cluster"].values()]
        year_biases = [v["bias"] for v in sym_results["by_year"].values()]

        asym_stable = False
        if len(year_biases) >= 2:
            same_sign = all(b > 0 for b in year_biases) or all(b < 0 for b in year_biases)
            asym_stable = same_sign

        sym_results["asymmetry_stable"] = asym_stable
        all_results[sym] = sym_results

        print(f"  Overall: bias={sym_results['overall']['bias']:.4f}  MFE={sym_results['overall']['avg_mfe']:.6f}  MAE={sym_results['overall']['avg_mae']:.6f}")
        if asym_stable:
            print(f"  *** ASYMMETRY STABLE ACROSS YEARS ***")
        for cid_str, casym in sym_results["by_cluster"].items():
            print(f"  {cid_str}: bias={casym['bias']:.4f}  n={casym['n_events']}  MFE={casym['avg_mfe']:.6f}")

    print(f"\n{'=' * 60}")
    print("  AGGREGATE ASYMMETRY SUMMARY")
    print(f"{'=' * 60}")

    n_stable = 0
    for sym, res in all_results.items():
        bias = res["overall"]["bias"]
        stable = res.get("asymmetry_stable", False)
        marker = " *** STABLE ***" if stable else ""
        print(f"  {sym}: bias={bias:+.4f}  MFE={res['overall']['avg_mfe']:.6f}  MAE={res['overall']['avg_mae']:.6f}{marker}")
        if stable:
            n_stable += 1

    print(f"\n  Stable asymmetries: {n_stable}/{len(all_results)}")

    if n_stable > 0:
        print("\n  VERDICT: ASYMMETRY DETECTED (some symbols have stable directional bias)")
    else:
        print("\n  VERDICT: NO STABLE ASYMMETRY (no symbol has year-consistent directional bias)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    import json
    serializable = {}
    for sym, res in all_results.items():
        serializable[sym] = str(res)
    with open(os.path.join(out_dir, "dpl13_results.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved")


if __name__ == "__main__":
    main()
