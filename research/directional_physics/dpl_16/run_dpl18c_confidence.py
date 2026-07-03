"""DPL-18C: Confidence Gating + Session Analysis.
Test directional accuracy by |TPI| quartile, session, and combined filters.

Run: python -u proxima_x/research/directional_physics/dpl_16/run_dpl18c_confidence.py
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, "C:/Trading/Agentic_Trading/proxima_x")
from research.directional_physics.dpl_16.core.msl_engine import (
    load_ticks, load_m5, align_ticks_to_bars,
    compute_tpi, aggregate_to_bars_vectorized,
    compute_directional_labels_vect,
)

REPORT_DIR = "C:/Trading/Agentic_Trading/research/directional_physics/dpl_16/reports"
os.makedirs(REPORT_DIR, exist_ok=True)
SYMBOLS = ["EURJPY", "USDJPY", "EURUSD", "GBPUSD"]

SESSION = {
    "asia":   (0, 9),
    "london": (9, 17),
    "ny":     (13, 22),
}

def load_data(symbol):
    ticks = load_ticks(symbol)
    m5 = load_m5(symbol)
    close = m5["close"]
    ts_m5 = m5["timestamp"]
    n = len(close)
    starts, ends, tick_sec = align_ticks_to_bars(ticks["timestamp"], ts_m5)
    tick_bar_idx = np.searchsorted(ts_m5, tick_sec, side="right") - 1
    tick_bar_idx = np.clip(tick_bar_idx, 0, n - 1)
    tpi_feats = compute_tpi(ticks["mid"])
    bar_tpi = aggregate_to_bars_vectorized(tpi_feats["tpi_200"], tick_bar_idx, n)
    first_valid = np.where(~np.isnan(bar_tpi))[0]
    if len(first_valid) == 0: return None
    s = first_valid[0]; e = min(first_valid[-1] + 1, n - 1)
    return {"symbol": symbol, "close": close[s:e], "tpi": bar_tpi[s:e],
            "ts": ts_m5[s:e], "n": e - s}

def session_hour(ts_sec):
    """Extract UTC hour from timestamp in seconds."""
    return (ts_sec // 3600) % 24

if __name__ == "__main__":
    print("=" * 65)
    print("DPL-18C: Confidence Gating + Session Analysis")
    print("=" * 65)

    all_data = {}
    for sym in SYMBOLS:
        d = load_data(sym)
        if d: all_data[sym] = d

    results = {}
    for sym, data in all_data.items():
        tpi = data["tpi"]
        close = data["close"]
        ts = data["ts"]
        n = len(tpi)
        tpi_mag = np.abs(tpi)
        tpi_sign = np.where(tpi > 0, 1, np.where(tpi < 0, -1, 0)).astype(float)
        labels = compute_directional_labels_vect(close)

        # 3-bar directional label
        valid = (tpi != 0) & ~np.isnan(tpi) & ~np.isnan(labels)
        if np.sum(valid) < 50:
            continue

        # Overall baseline
        pred = np.where(tpi_sign[valid] > 0, 1, -1)
        base_acc = float(np.mean(pred == labels[valid]))

        sym_res = {"baseline": {"acc": base_acc, "n": int(np.sum(valid))}}

        # By confidence quartile
        mag_valid = tpi_mag[valid]
        qs = np.percentile(mag_valid, [25, 50, 75])
        sym_res["quartiles"] = {}
        for qi, (lo, hi, name) in enumerate([
            (-np.inf, qs[0], "Q1"),
            (qs[0], qs[1], "Q2"),
            (qs[1], qs[2], "Q3"),
            (qs[2], np.inf, "Q4"),
        ]):
            mask = (tpi_mag[valid] >= lo) & (tpi_mag[valid] < hi)
            if np.sum(mask) < 5: continue
            p = pred[mask]
            l = labels[valid][mask]
            sym_res["quartiles"][name] = {
                "acc": float(np.mean(p == l)),
                "n": int(np.sum(mask)),
                "mag_range": [float(lo) if np.isfinite(lo) else None, float(hi) if np.isfinite(hi) else None],
            }

        # By session
        hours = session_hour(ts[valid])
        sym_res["sessions"] = {}
        for sname, (h_lo, h_hi) in SESSION.items():
            mask = (hours >= h_lo) & (hours < h_hi)
            if np.sum(mask) < 5: continue
            p = pred[mask]
            l = labels[valid][mask]
            sym_res["sessions"][sname] = {
                "acc": float(np.mean(p == l)),
                "n": int(np.sum(mask)),
            }

        # By session x Q4
        for sname, (h_lo, h_hi) in SESSION.items():
            q4_mask = (tpi_mag[valid] >= qs[2]) & (hours >= h_lo) & (hours < h_hi)
            if np.sum(q4_mask) < 5: continue
            p = pred[q4_mask]
            l = labels[valid][q4_mask]
            sym_res.setdefault("session_x_q4", {})[sname] = {
                "acc": float(np.mean(p == l)),
                "n": int(np.sum(q4_mask)),
            }

        # By filtered threshold
        for thresh_pct in [75, 80, 85, 90]:
            th = np.percentile(mag_valid, thresh_pct)
            mask = tpi_mag[valid] >= th
            if np.sum(mask) < 5: continue
            p = pred[mask]
            l = labels[valid][mask]
            sym_res.setdefault("thresholds", {})[f"p{thresh_pct}"] = {
                "acc": float(np.mean(p == l)),
                "n": int(np.sum(mask)),
                "threshold": float(th),
            }

        results[sym] = sym_res

        # Print
        print(f"\n  {sym:8s}  baseline={base_acc:.4f} (n={np.sum(valid)})")
        print(f"  Confidence Quartiles:")
        for qn, qd in sym_res["quartiles"].items():
            print(f"    {qn:4s}  acc={qd['acc']:.4f}  n={qd['n']}")
        print(f"  Sessions:")
        for sn, sd in sym_res["sessions"].items():
            print(f"    {sn:8s}  acc={sd['acc']:.4f}  n={sd['n']}")
        if "session_x_q4" in sym_res:
            print(f"  Session_x_Q4:")
            for sn, sd in sym_res["session_x_q4"].items():
                print(f"    {sn:8s}  acc={sd['acc']:.4f}  n={sd['n']}")
        if "thresholds" in sym_res:
            print(f"  Percentile Thresholds:")
            for tk, td in sym_res["thresholds"].items():
                print(f"    {tk:4s}  acc={td['acc']:.4f}  n={td['n']}  thresh={td['threshold']:.6f}")

    # Summary
    print(f"\n{'='*65}")
    print(f"SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Symbol':>6s}  {'Base':>6s}  {'Q1':>6s}  {'Q2':>6s}  {'Q3':>6s}  {'Q4':>6s}  {'Asia':>6s}  {'Lon':>6s}  {'NY':>6s}")
    for sym in SYMBOLS:
        r = results.get(sym, {})
        bl = r.get("baseline", {}).get("acc", 0)
        q1 = r.get("quartiles", {}).get("Q1", {}).get("acc", 0)
        q2 = r.get("quartiles", {}).get("Q2", {}).get("acc", 0)
        q3 = r.get("quartiles", {}).get("Q3", {}).get("acc", 0)
        q4 = r.get("quartiles", {}).get("Q4", {}).get("acc", 0)
        asia = r.get("sessions", {}).get("asia", {}).get("acc", 0)
        lon = r.get("sessions", {}).get("london", {}).get("acc", 0)
        ny = r.get("sessions", {}).get("ny", {}).get("acc", 0)
        print(f"  {sym:>6s}  {bl:>6.4f}  {q1:>6.4f}  {q2:>6.4f}  {q3:>6.4f}  {q4:>6.4f}  {asia:>6.4f}  {lon:>6.4f}  {ny:>6.4f}")

    with open(os.path.join(REPORT_DIR, "dpl18c_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDPL-18C -> dpl18c_results.json")
