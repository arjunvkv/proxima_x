import polars as pl
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

DATA_BASE = "C:/Trading/Agentic_Trading/data/intraday"
SYMBOLS = ["XAUUSD", "EURJPY", "USDJPY", "GBPJPY"]
MAX_HORIZON = 48

SESSION_DEFS = {
    "asia": (0, 8),
    "london": (8, 16),
    "ny": (13, 22),
}


def load_m5(symbol):
    df = pl.read_parquet(f"{DATA_BASE}/{symbol}_M5.parquet")
    arr = df.to_numpy()
    ts = arr[:, 0].astype(np.int64)
    o, h, l, c, v = [arr[:, i].astype(np.float64) for i in range(1, 6)]
    return {"symbol": symbol, "timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v, "n": len(c)}


def session_hour(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).hour


def compute(data):
    close = data["close"]
    n = data["n"]
    tr = np.abs(data["high"] - data["low"])
    rets = np.diff(np.log(close), prepend=np.log(close[0]))
    ts = data["timestamp"]

    tr_p95 = np.zeros(n)
    for i in range(200, n):
        tr_p95[i] = float(np.percentile(tr[i - 200:i], 95))

    impulses = np.zeros(n, dtype=bool)
    impulse_dir = np.zeros(n)
    impulse_mag = np.zeros(n)

    for i in range(200, n):
        if tr[i] > tr_p95[i] and tr_p95[i] > 0:
            impulses[i] = True
            impulse_dir[i] = np.sign(rets[i])
            impulse_mag[i] = tr[i] / tr_p95[i]

    return impulses, impulse_dir, impulse_mag


def signed_metrics(data, idx, horizon=48):
    close = data["close"]
    n = len(close)
    end = min(idx + horizon, n)
    if end - idx < 5:
        return None
    entry = close[idx]
    future = close[idx + 1:end]
    returns = (future - entry) / entry
    dir_imp = np.sign(data["close"][idx] - data["close"][idx - 1]) if idx > 0 else 0
    if dir_imp == 0:
        return None
    scr = float(np.mean(returns * dir_imp))
    cum_dir = returns * dir_imp
    smfe = float(np.max(cum_dir)) if len(cum_dir) > 0 else 0.0
    smae = float(np.min(cum_dir)) if len(cum_dir) > 0 else 0.0
    final_dir = float(np.sign(returns[-1])) if len(returns) > 0 else 0
    cont = 1 if final_dir == dir_imp else 0
    rev = 1 if final_dir == -dir_imp else 0
    er = abs(smfe) / (abs(smae) + 1e-8)
    return {
        "scr": scr, "smfe": smfe, "smae": smae, "er": er,
        "continuation": cont, "reversal": rev,
        "dir_imp": int(dir_imp), "final_dir": int(final_dir),
        "final_ret": float(returns[-1]),
        "n_bars": len(returns)
    }


def main():
    print("=" * 80)
    print("  DPL-14 SIGNED IMPULSE CONTINUATION AUDIT")
    print("=" * 80)
    print("  Metric: Signed Continuation Return (SCR = dir * future_return)")
    print("  Signed MFE/MAE relative to impulse direction")
    print("  Efficiency Ratio = SMFE / SMAE")
    print()

    all_results = {}

    for sym in SYMBOLS:
        print(f"\n{'=' * 60}")
        print(f"  {sym}")
        print(f"{'=' * 60}")

        data = load_m5(sym)
        impulses, imp_dir, imp_mag = compute(data)
        n_imp = int(np.sum(impulses))
        print(f"  Bars: {data['n']}  Impulses: {n_imp} ({n_imp/data['n']*100:.1f}%)")

        all_events = []
        for i in range(200, data["n"]):
            if impulses[i]:
                m = signed_metrics(data, i, MAX_HORIZON)
                if m:
                    hr = session_hour(data["timestamp"][i])
                    m["session"] = "asia" if hr < 8 else "london" if hr < 16 else "ny" if hr < 22 else "asia"
                    m["ts"] = data["timestamp"][i]
                    all_events.append(m)

        if not all_events:
            continue

        scrs = [e["scr"] for e in all_events]
        smfes = [e["smfe"] for e in all_events]
        smaes = [e["smae"] for e in all_events]
        ers = [e["er"] for e in all_events]
        conts = sum(e["continuation"] for e in all_events)
        revs = sum(e["reversal"] for e in all_events)
        n = len(all_events)

        print(f"\n  *** GLOBAL ***")
        print(f"    n={n}  SCR={np.mean(scrs):.6f}  SMFE={np.mean(smfes):.6f}  SMAE={np.mean(smaes):.6f}")
        print(f"    ER={np.mean(ers):.4f}  Cont={conts} ({conts/n*100:.1f}%)  Rev={revs} ({revs/n*100:.1f}%)")

        results = {"global": {
            "n": n, "scr": round(float(np.mean(scrs)), 6),
            "smfe": round(float(np.mean(smfes)), 6),
            "smae": round(float(np.mean(smaes)), 6),
            "er": round(float(np.mean(ers)), 4),
            "cont_pct": round(conts / n * 100, 1),
            "rev_pct": round(revs / n * 100, 1),
            "bias": round((conts - revs) / n * 100, 1),
        }}

        print(f"\n  *** BY SESSION ***")
        sessions = defaultdict(list)
        for e in all_events:
            sessions[e["session"]].append(e)
        for sname in ["asia", "london", "ny"]:
            evts = sessions.get(sname, [])
            if len(evts) < 20:
                continue
            s_scrs = [e["scr"] for e in evts]
            s_ers = [e["er"] for e in evts]
            s_cont = sum(e["continuation"] for e in evts)
            print(f"    {sname}: n={len(evts)}  SCR={np.mean(s_scrs):.6f}  ER={np.mean(s_ers):.4f}  Cont={s_cont/len(evts)*100:.1f}%")
            results[f"session_{sname}"] = {
                "n": len(evts), "scr": round(float(np.mean(s_scrs)), 6),
                "er": round(float(np.mean(s_ers)), 4),
                "cont_pct": round(s_cont / len(evts) * 100, 1),
            }

        vol_features = []
        for i in range(200, data["n"]):
            ret_window = np.diff(np.log(data["close"][i - 20:i]))
            vol_features.append(float(np.std(ret_window)))
        vol_features = np.array(vol_features)
        vol_qs = np.percentile(vol_features, [25, 50, 75])

        print(f"\n  *** BY VOLATILITY QUARTILE ***")
        vol_idx = 0
        vol_events = defaultdict(list)
        for i in range(200, data["n"]):
            if impulses[i]:
                vf = vol_features[vol_idx]
                if vf <= vol_qs[0]:
                    vol_events["q1_low"].append(all_events[len(vol_events["q1_low"]) + len(vol_events.get("q2", [])) + len(vol_events.get("q3", [])) + len(vol_events.get("q4_high", []))])
                elif vf <= vol_qs[1]:
                    pass
                vol_idx += 1

        vol_events2 = defaultdict(list)
        for i, e in enumerate(all_events):
            vf = vol_features[i]
            if vf <= vol_qs[0]:
                vol_events2["q1_low"].append(e)
            elif vf <= vol_qs[1]:
                vol_events2["q2"].append(e)
            elif vf <= vol_qs[2]:
                vol_events2["q3"].append(e)
            else:
                vol_events2["q4_high"].append(e)
        for qname in ["q1_low", "q2", "q3", "q4_high"]:
            evts = vol_events2[qname]
            if len(evts) < 10:
                continue
            q_scrs = [e["scr"] for e in evts]
            q_ers = [e["er"] for e in evts]
            q_cont = sum(e["continuation"] for e in evts)
            print(f"    {qname}: n={len(evts)}  SCR={np.mean(q_scrs):.6f}  ER={np.mean(q_ers):.4f}  Cont={q_cont/len(evts)*100:.1f}%")
            results[f"vol_{qname}"] = {
                "n": len(evts), "scr": round(float(np.mean(q_scrs)), 6),
                "er": round(float(np.mean(q_ers)), 4),
                "cont_pct": round(q_cont / len(evts) * 100, 1),
            }

        all_results[sym] = results

    print(f"\n{'=' * 60}")
    print("  DPL-14 SUMMARY (Signed Continuation)")
    print(f"{'=' * 60}")
    print(f"\n  {'Symbol':10s} {'n':>6s} {'SCR':>10s} {'SMFE':>10s} {'SMAE':>10s} {'ER':>6s} {'Cont%':>6s} {'Bias%':>6s}")
    print(f"  {'-'*60}")
    for sym, res in all_results.items():
        g = res["global"]
        print(f"  {sym:10s} {g['n']:6d} {g['scr']:10.6f} {g['smfe']:10.6f} {g['smae']:10.6f} {g['er']:6.2f} {g['cont_pct']:5.1f}% {g['bias']:5.1f}%")

    pos_scr = sum(1 for r in all_results.values() if r["global"]["scr"] > 0)
    pos_er = sum(1 for r in all_results.values() if r["global"]["er"] > 1.0)
    print(f"\n  SCR > 0: {pos_scr}/{len(all_results)}")
    print(f"  ER > 1.0: {pos_er}/{len(all_results)}")

    if pos_scr >= 3 and pos_er >= 3:
        print("\n  VERDICT: SIGNED CONTINUATION EDGE DETECTED (multiple symbols have positive SCR and ER > 1)")
    elif pos_scr >= 2:
        print("\n  VERDICT: MARGINAL (some symbols show signed continuation but not consistent)")
    else:
        print("\n  VERDICT: NO EDGE (signed continuation not detectable after drift removal)")

    import json
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    serializable = {}
    for sym, res in all_results.items():
        serializable[sym] = {k: v for k, v in res.items() if isinstance(v, dict)}
    with open(os.path.join(out_dir, "dpl14_results.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved")


if __name__ == "__main__":
    main()
