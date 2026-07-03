"""ROL-3: Residual Pressure Surface Analysis.

Questions:
1. Do long-duration small-magnitude residuals release differently than short-duration large-magnitude residuals?
2. Is there a "pressure threshold" where directional release becomes inevitable?
3. Does residual magnitude x duration predict direction BETTER than sign alone? (OOS only)
4. What is the pressure-release function? Linear? Threshold? Exponential?
"""

import sys, json
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.residual_origin.rol_core import ROLCore, SYMBOLS, save_rol_report
from research.directional_state.dsr_core import WalkForwardValidator

HORIZON_MAP = {"H5": 1, "H20": 2, "H50": 3}
PRESSURE_HORIZONS = ["H5", "H20", "H50"]


def extract_run_features(rol, symbol):
    d = rol.dsr._data[symbol]
    pressure = d["residual_pressure"]
    residual = d["residual"]
    fut_ret = d["fut_ret"]
    regime = d["regime"]
    runs = rol.residual_run_lengths(symbol)

    records = []
    for sign, length, start_idx in runs:
        end_idx = start_idx + length - 1
        if end_idx >= len(pressure) or end_idx >= fut_ret.shape[0]:
            continue

        start_p = float(pressure[start_idx - 1]) if start_idx > 0 else 0.0
        cum_pressure = float(pressure[end_idx] - start_p)
        avg_magnitude = abs(cum_pressure) / length if length > 0 else 0.0

        reg = int(regime[end_idx]) if regime[end_idx] >= 0 else -1

        record = {
            "sign": int(sign), "length": int(length),
            "start_idx": int(start_idx), "end_idx": int(end_idx),
            "cum_pressure": round(cum_pressure, 6),
            "avg_magnitude": round(avg_magnitude, 8),
            "regime": reg,
        }
        for hk in PRESSURE_HORIZONS:
            hi = HORIZON_MAP[hk]
            fr = fut_ret[end_idx, hi]
            if np.isnan(fr):
                record[f"up_{hk}"] = None
            else:
                record[f"up_{hk}"] = 1.0 if fr > 0 else 0.0
        records.append(record)
    return records


def _quintile_bins(arr):
    valid = ~np.isnan(arr)
    if np.sum(valid) < 10:
        return None
    return [float(np.nanpercentile(arr[valid], p)) for p in [20, 40, 60, 80]]


def _apply_quintiles(arr, bins):
    labels = np.full(len(arr), -1, dtype=int)
    valid = ~np.isnan(arr)
    if bins is None or np.sum(valid) < 3:
        return labels
    labels[valid & (arr <= bins[0])] = 0
    labels[valid & (arr > bins[0]) & (arr <= bins[1])] = 1
    labels[valid & (arr > bins[1]) & (arr <= bins[2])] = 2
    labels[valid & (arr > bins[2]) & (arr <= bins[3])] = 3
    labels[valid & (arr > bins[3])] = 4
    return labels


def _state_id(mag_q, dur_q, sign):
    if mag_q < 0 or dur_q < 0:
        return -1
    s = 0 if sign < 0 else 1
    return s * 100 + mag_q * 10 + dur_q


def _cell_stats(state_ids, up, min_samples=3):
    stats = {}
    ids = np.array(state_ids)
    u = np.array(up, dtype=float)
    valid = (ids >= 0) & ~np.isnan(u)
    for sid in np.unique(ids[valid]):
        mask = ids == sid
        cnt = int(np.sum(mask))
        if cnt < min_samples:
            continue
        n_up = int(np.sum(u[mask]))
        stats[int(sid)] = {"p_up": n_up / cnt, "count": cnt, "n_up": n_up}
    return stats


def _evaluate(state_ids, up, stats):
    ids = np.array(state_ids)
    u = np.array(up, dtype=float)
    valid = (ids >= 0) & ~np.isnan(u)
    n = int(np.sum(valid))
    if n < 5:
        return None
    preds = np.full(n, np.nan)
    probs = np.full(n, np.nan)
    actuals = u[valid]
    sids = ids[valid].astype(int)
    for i, sid in enumerate(sids):
        if sid in stats:
            probs[i] = stats[sid]["p_up"]
            preds[i] = 1.0 if stats[sid]["p_up"] > 0.5 else 0.0
    ok = ~np.isnan(preds)
    if np.sum(ok) < 5:
        return None
    p, a, pr = preds[ok], actuals[ok], probs[ok]
    nv = int(np.sum(ok))
    correct = float(np.sum(p == a))
    acc = correct / nv
    tp = float(np.sum((p == 1.0) & (a == 1.0)))
    fp = float(np.sum((p == 1.0) & (a == 0.0)))
    fn = float(np.sum((p == 0.0) & (a == 1.0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "n_valid": nv, "accuracy": round(acc, 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4),
        "brier_score": round(float(np.mean((pr - a) ** 2)), 4),
        "base_p_up": round(float(np.mean(a)), 4),
    }


def build_pressure_surface(features):
    surface = {}
    for label, sign_val in [("positive", 1), ("negative", -1)]:
        runs = [f for f in features if f["sign"] == sign_val]
        if len(runs) < 15:
            surface[label] = {"n_runs": len(runs), "cells": []}
            continue
        mags = np.array([f["avg_magnitude"] for f in runs])
        durs = np.array([f["length"] for f in runs])
        mb = _quintile_bins(mags)
        db = _quintile_bins(durs)
        if mb is None or db is None:
            surface[label] = {"n_runs": len(runs), "cells": []}
            continue
        mq = _apply_quintiles(mags, mb)
        dq = _apply_quintiles(durs, db)
        cells = []
        for mi in range(5):
            for di in range(5):
                cell = {}
                for hk in PRESSURE_HORIZONS:
                    ups = [runs[j][f"up_{hk}"] for j in range(len(runs))
                           if mq[j] == mi and dq[j] == di and runs[j][f"up_{hk}"] is not None]
                    if len(ups) >= 3:
                        cell[hk] = {"p_up": round(float(np.mean(ups)), 4), "count": len(ups)}
                if cell:
                    mag_low = 0.0 if mi == 0 else mb[mi - 1]
                    mag_high = mb[mi] if mi < 4 else float(mags.max())
                    dur_low = 0 if di == 0 else int(db[di - 1])
                    dur_high = int(db[di]) if di < 4 else int(durs.max())
                    cells.append({
                        "mag_quintile": mi, "dur_quintile": di,
                        "mag_range": [round(mag_low, 6), round(mag_high, 6)],
                        "dur_range": [dur_low, dur_high],
                        "results": cell,
                    })
        surface[label] = {
            "n_runs": len(runs),
            "mag_bins": [round(b, 8) for b in mb],
            "dur_bins": [int(b) for b in db],
            "cells": cells,
        }
    return surface


def pressure_threshold_analysis(features):
    """Find pressure thresholds where directional release becomes inevitable."""
    result = {}
    for label, sign_val in [("positive", 1), ("negative", -1)]:
        runs = [f for f in features if f["sign"] == sign_val]
        if len(runs) < 20:
            continue
        cps = np.array([abs(f["cum_pressure"]) for f in runs])
        mb = _quintile_bins(cps)
        if mb is None:
            continue
        mq = _apply_quintiles(cps, mb)
        thresholds = {}
        for hk in PRESSURE_HORIZONS:
            quintile_ps = []
            for qi in range(5):
                ups = [runs[j][f"up_{hk}"] for j in range(len(runs))
                       if mq[j] == qi and runs[j][f"up_{hk}"] is not None]
                if len(ups) >= 3:
                    p_up = float(np.mean(ups))
                    quintile_ps.append({
                        "quintile": qi,
                        "p_up": round(p_up, 4),
                        "count": len(ups),
                        "pressure_range": [0.0 if qi == 0 else float(mb[qi - 1]),
                                           float(mb[qi]) if qi < 4 else float(cps.max())],
                    })
            # Find where p_up crosses 0.5 for positive, or below 0.5 for negative
            thr = None
            for qp in quintile_ps:
                if sign_val > 0 and qp["p_up"] > 0.5:
                    thr = qp
                    break
                if sign_val < 0 and qp["p_up"] < 0.5:
                    thr = qp
                    break
            thresholds[hk] = {
                "quintiles": quintile_ps,
                "threshold_crossed": thr is not None,
                "threshold_quintile": thr["quintile"] + 1 if thr else None,
                "threshold_p_up": thr["p_up"] if thr else None,
                "threshold_pressure_max": thr["pressure_range"][1] if thr else None,
            }
        result[label] = thresholds
    return result


def pressure_release_function(features):
    """Fit pressure-release curves: linear, threshold, exponential."""
    result = {}
    for label, sign_val in [("positive", 1), ("negative", -1)]:
        runs = [f for f in features if f["sign"] == sign_val]
        if len(runs) < 30:
            continue
        result[label] = {}
        for hk in PRESSURE_HORIZONS:
            valid = [(abs(f["cum_pressure"]), f[f"up_{hk}"])
                     for f in runs if f[f"up_{hk}"] is not None]
            if len(valid) < 30:
                continue
            x = np.array([v[0] for v in valid], dtype=float)
            y = np.array([v[1] for v in valid], dtype=float)
            # Bin by deciles
            xbins = np.nanpercentile(x, np.arange(10, 100, 10))
            bin_ps = []
            prev = 0.0
            for bi, bv in enumerate(xbins):
                if bi == 0:
                    mask = x <= bv
                else:
                    mask = (x > xbins[bi - 1]) & (x <= bv)
                cnt = int(np.sum(mask))
                if cnt >= 3:
                    bin_ps.append({
                        "bin": bi, "pressure_max": float(bv),
                        "p_up": round(float(np.mean(y[mask])), 4),
                        "count": cnt,
                    })
            result[hk] = {"decile_bins": bin_ps}
    return result


def run_rol3():
    print("=" * 72)
    print("ROL-3: RESIDUAL PRESSURE SURFACE ANALYSIS")
    print("=" * 72)

    print("\nLoading data...")
    rol = ROLCore()
    rol.load_all()
    wfv = WalkForwardValidator(rol.dsr)

    report = {
        "title": "ROL-3: Residual Pressure Surface Analysis",
        "splits": [f"{a}->{b}" for a, b in WalkForwardValidator.SPLITS],
        "horizons": PRESSURE_HORIZONS,
        "results": {}, "per_symbol": {},
        "pressure_surface": {}, "threshold_analysis": {},
        "release_function": {}, "summary": {},
    }

    for sym in SYMBOLS:
        print(f"\n--- {sym} ---")
        wfv.prepare(sym)
        years_arr = wfv._year_ranges[sym]

        features = extract_run_features(rol, sym)
        for f in features:
            ei = f["end_idx"]
            f["year"] = int(years_arr[ei]) if ei < len(years_arr) else None

        n_runs = len(features)
        print(f"  Runs: {n_runs}")
        report["per_symbol"][sym] = {
            "n_runs": n_runs,
            "sign_dist": {1: sum(1 for f in features if f["sign"] == 1),
                          -1: sum(1 for f in features if f["sign"] == -1)},
            "avg_run_length": round(float(np.mean([f["length"] for f in features])), 2) if features else 0,
        }

        # Pressure surface (full data)
        report["pressure_surface"][sym] = build_pressure_surface(features)

        # Threshold analysis
        report["threshold_analysis"][sym] = pressure_threshold_analysis(features)

        # Release function
        report["release_function"][sym] = pressure_release_function(features)

        # Walk-forward validation per horizon
        for hk in PRESSURE_HORIZONS:
            if hk not in report["results"]:
                report["results"][hk] = {}
            report["results"][hk][sym] = {}

            for train_name, test_name in WalkForwardValidator.SPLITS:
                split_key = f"{train_name}->{test_name}"
                train_year_end = int(test_name) - 1
                test_year = int(test_name)

                train_runs = [f for f in features if f["year"] is not None
                              and int(train_name[:4]) <= f["year"] <= train_year_end
                              and f[f"up_{hk}"] is not None]
                test_runs = [f for f in features if f["year"] is not None
                             and f["year"] == test_year
                             and f[f"up_{hk}"] is not None]

                if len(train_runs) < 20 or len(test_runs) < 5:
                    continue

                # --- Method 1: Sign alone ---
                sign_train_ids = np.array([0 if f["sign"] < 0 else 1 for f in train_runs])
                sign_train_up = np.array([f[f"up_{hk}"] for f in train_runs])
                sign_stats = _cell_stats(sign_train_ids, sign_train_up)

                sign_test_ids = np.array([0 if f["sign"] < 0 else 1 for f in test_runs])
                sign_test_up = np.array([f[f"up_{hk}"] for f in test_runs])
                sign_eval = _evaluate(sign_test_ids, sign_test_up, sign_stats)

                # --- Method 2: Magnitude x Duration ---
                train_mag = np.array([f["avg_magnitude"] for f in train_runs])
                train_dur = np.array([f["length"] for f in train_runs])
                mag_bins = _quintile_bins(train_mag)
                dur_bins = _quintile_bins(train_dur)

                if mag_bins and dur_bins:
                    train_mq = _apply_quintiles(train_mag, mag_bins)
                    train_dq = _apply_quintiles(train_dur, dur_bins)
                    train_s = np.array([f["sign"] for f in train_runs])
                    train_md_ids = np.array([_state_id(train_mq[i], train_dq[i], train_s[i])
                                             for i in range(len(train_runs))])
                    md_stats = _cell_stats(train_md_ids, sign_train_up)

                    test_mag = np.array([f["avg_magnitude"] for f in test_runs])
                    test_dur = np.array([f["length"] for f in test_runs])
                    test_mq = _apply_quintiles(test_mag, mag_bins)
                    test_dq = _apply_quintiles(test_dur, dur_bins)
                    test_s = np.array([f["sign"] for f in test_runs])
                    test_md_ids = np.array([_state_id(test_mq[i], test_dq[i], test_s[i])
                                            for i in range(len(test_runs))])
                    md_eval = _evaluate(test_md_ids, sign_test_up, md_stats)
                else:
                    md_eval = None

                entry = {}
                if sign_eval:
                    entry["sign_only"] = sign_eval
                if md_eval:
                    entry["mag_x_dur"] = md_eval
                if sign_eval and md_eval and md_eval.get("accuracy") is not None and sign_eval.get("accuracy") is not None:
                    entry["improvement"] = round(md_eval["accuracy"] - sign_eval["accuracy"], 4)
                if entry:
                    report["results"][hk][sym][split_key] = entry

    # ── Summaries ──
    report["summary"] = compute_summary(report)
    save_rol_report(report, "rol3_pressure_surface")
    print_summary(report)
    return report


def compute_summary(report):
    splits = report["splits"]
    summary = {}

    for hk in PRESSURE_HORIZONS:
        h = {"splits": {}, "accuracy_comparison": {}}
        for sk in splits:
            sign_accs, md_accs = [], []
            for sym in SYMBOLS:
                e = report["results"].get(hk, {}).get(sym, {}).get(sk, {})
                s = e.get("sign_only", {})
                m = e.get("mag_x_dur", {})
                if s.get("n_valid", 0) > 0:
                    sign_accs.append(s["accuracy"])
                if m.get("n_valid", 0) > 0:
                    md_accs.append(m["accuracy"])
            h["splits"][sk] = {
                "sign_only_mean_acc": round(float(np.mean(sign_accs)), 4) if sign_accs else None,
                "mag_x_dur_mean_acc": round(float(np.mean(md_accs)), 4) if md_accs else None,
                "sign_only_n_sym": len(sign_accs),
                "mag_x_dur_n_sym": len(md_accs),
            }

        # Cross-split comparison
        all_sign = []
        all_md = []
        for sk in splits:
            for sym in SYMBOLS:
                e = report["results"].get(hk, {}).get(sym, {}).get(sk, {})
                s = e.get("sign_only", {})
                m = e.get("mag_x_dur", {})
                if s.get("n_valid", 0) > 0:
                    all_sign.append(s["accuracy"])
                if m.get("n_valid", 0) > 0:
                    all_md.append(m["accuracy"])
        if all_sign and all_md:
            h["accuracy_comparison"] = {
                "sign_only_mean": round(float(np.mean(all_sign)), 4),
                "sign_only_std": round(float(np.std(all_sign)), 4),
                "mag_x_dur_mean": round(float(np.mean(all_md)), 4),
                "mag_x_dur_std": round(float(np.std(all_md)), 4),
                "improvement": round(float(np.mean(all_md)) - float(np.mean(all_sign)), 4),
                "sign_outperforms": float(np.mean(all_sign)) > float(np.mean(all_md)),
            }
        summary[hk] = h

    # Across horizons
    all_sign_all = []
    all_md_all = []
    for hk in PRESSURE_HORIZONS:
        for sk in splits:
            for sym in SYMBOLS:
                e = report["results"].get(hk, {}).get(sym, {}).get(sk, {})
                s = e.get("sign_only", {})
                m = e.get("mag_x_dur", {})
                if s.get("n_valid", 0) > 0:
                    all_sign_all.append(s["accuracy"])
                if m.get("n_valid", 0) > 0:
                    all_md_all.append(m["accuracy"])
    if all_sign_all and all_md_all:
        summary["overall"] = {
            "sign_only_mean": round(float(np.mean(all_sign_all)), 4),
            "sign_only_std": round(float(np.std(all_sign_all)), 4),
            "mag_x_dur_mean": round(float(np.mean(all_md_all)), 4),
            "mag_x_dur_std": round(float(np.std(all_md_all)), 4),
            "improvement": round(float(np.mean(all_md_all)) - float(np.mean(all_sign_all)), 4),
            "n_sign_evals": len(all_sign_all),
            "n_md_evals": len(all_md_all),
        }

    # Pressure threshold summary
    thr_summary = {}
    for sym in SYMBOLS:
        ta = report.get("threshold_analysis", {}).get(sym, {})
        for label in ["positive", "negative"]:
            hk_data = ta.get(label, {})
            for hk in PRESSURE_HORIZONS:
                hd = hk_data.get(hk, {})
                if hd.get("threshold_crossed"):
                    key = f"{sym}/{label}/{hk}"
                    thr_summary[key] = {
                        "threshold_quintile": hd["threshold_quintile"],
                        "threshold_p_up": hd["threshold_p_up"],
                        "threshold_pressure_max": hd["threshold_pressure_max"],
                    }
    summary["pressure_thresholds"] = thr_summary

    return summary


def print_summary(report):
    summary = report["summary"]
    print("\n" + "=" * 72)
    print("ROL-3 RESULTS SUMMARY")
    print("=" * 72)

    # Overall comparison
    ov = summary.get("overall", {})
    if ov:
        print(f"\n  OVERALL: Sign Alone vs Magnitude x Duration")
        print(f"    Sign Only:       {ov['sign_only_mean']:.4f} +/- {ov['sign_only_std']:.4f} (n={ov['n_sign_evals']})")
        print(f"    Mag x Dur:       {ov['mag_x_dur_mean']:.4f} +/- {ov['mag_x_dur_std']:.4f} (n={ov['n_md_evals']})")
        print(f"    Improvement:     {ov['improvement']:+.4f}")
        if ov['improvement'] > 0:
            print(f"    -> Mag x Dur outperforms sign alone")
        else:
            print(f"    -> Sign alone is at least as good as Mag x Dur")

    # Per horizon
    print(f"\n  PER HORIZON:")
    for hk in PRESSURE_HORIZONS:
        h = summary.get(hk, {}).get("accuracy_comparison", {})
        if h:
            print(f"    {hk}: Sign={h['sign_only_mean']:.4f} vs MagxD={h['mag_x_dur_mean']:.4f} "
                  f"(delta={h['improvement']:+.4f})")

    # Per split
    print(f"\n  PER SPLIT:")
    for sk in report["splits"]:
        print(f"\n    {sk}:")
        for hk in PRESSURE_HORIZONS:
            sd = summary.get(hk, {}).get("splits", {}).get(sk, {})
            if sd.get("sign_only_mean") is not None:
                print(f"      {hk}: Sign={sd['sign_only_mean']:.4f} ({sd['sign_only_n_sym']} syms) "
                      f"| MagxD={sd['mag_x_dur_mean']:.4f} ({sd['mag_x_dur_n_sym']} syms)")

    # Per symbol detail
    print(f"\n  PER SYMBOL:")
    for sym in SYMBOLS:
        ps = report.get("per_symbol", {}).get(sym, {})
        print(f"\n    {sym}: {ps.get('n_runs', 0)} runs, "
              f"sign_dist={ps.get('sign_dist', {})}, "
              f"avg_len={ps.get('avg_run_length', 0):.1f}")
        for hk in PRESSURE_HORIZONS:
            parts = []
            for sk in report["splits"]:
                e = report["results"].get(hk, {}).get(sym, {}).get(sk, {})
                s = e.get("sign_only", {})
                m = e.get("mag_x_dur", {})
                imp = e.get("improvement")
                s_acc = s.get("accuracy")
                m_acc = m.get("accuracy")
                s_str = f"{s_acc:.3f}" if s_acc is not None else "N/A"
                m_str = f"{m_acc:.3f}" if m_acc is not None else "N/A"
                imp_str = f"{imp:+.3f}" if imp is not None else ""
                parts.append(f"{sk}: S={s_str} MD={m_str} {imp_str}")
            if parts:
                print(f"      {hk}: {' | '.join(parts)}")

    # Pressure thresholds
    thr = summary.get("pressure_thresholds", {})
    if thr:
        print(f"\n  PRESSURE THRESHOLDS (directional release becomes inevitable):")
        for key, val in sorted(thr.items()):
            print(f"    {key}: Q{val['threshold_quintile']} p_up={val['threshold_p_up']:.3f} "
                  f"pressure<={val['threshold_pressure_max']:.4f}")

    # Surface highlights
    print(f"\n  PRESSURE SURFACE HIGHLIGHTS:")
    for sym in SYMBOLS:
        surf = report.get("pressure_surface", {}).get(sym, {})
        for label in ["positive", "negative"]:
            cells = surf.get(label, {}).get("cells", [])
            if not cells:
                continue
            for hk in PRESSURE_HORIZONS:
                max_cell = None
                max_p = -1
                min_cell = None
                min_p = 2
                for c in cells:
                    r = c.get("results", {}).get(hk, {})
                    p = r.get("p_up")
                    if p is not None:
                        if p > max_p:
                            max_p = p
                            max_cell = c
                        if p < min_p:
                            min_p = p
                            min_cell = c
                if max_cell:
                    print(f"    {sym} {label}/{hk}: max P(up)={max_p:.3f} at "
                          f"magQ={max_cell['mag_quintile']} durQ={max_cell['dur_quintile']}")

    # Answer research questions
    print(f"\n{'=' * 72}")
    print("  RESEARCH QUESTIONS")
    print(f"{'=' * 72}")

    # RQ3: Does magnitude x duration beat sign alone?
    ov = summary.get("overall", {})
    if ov:
        print(f"\n  RQ3: Does magnitude x duration predict direction BETTER than sign alone? (OOS)")
        if ov["improvement"] > 0.005:
            print(f"      YES - Mag x Dur outperforms sign alone by {ov['improvement']:.4f}")
        elif ov["improvement"] > 0:
            print(f"      WEAK YES - Mag x Dur marginally outperforms ({ov['improvement']:.4f})")
        else:
            print(f"      NO - Sign alone matches or beats Mag x Dur ({ov['improvement']:.4f})")

    # RQ4: What is the pressure-release function?
    print(f"\n  RQ4: What is the pressure-release function?")
    linear_count, threshold_count, exp_count = 0, 0, 0
    for sym in SYMBOLS:
        rf = report.get("release_function", {}).get(sym, {})
        for label in ["positive", "negative"]:
            for hk in PRESSURE_HORIZONS:
                bins = rf.get(label, {}).get(hk, {}).get("decile_bins", [])
                if len(bins) >= 4:
                    ps = [b["p_up"] for b in bins]
                    if ps[-1] - ps[0] > 0.2:
                        # Check if threshold-like
                        diffs = np.diff(ps)
                        if np.max(np.abs(diffs)) > 0.1:
                            threshold_count += 1
                        elif ps[-1] > ps[0] * 1.5:
                            exp_count += 1
                        else:
                            linear_count += 1
    total = linear_count + threshold_count + exp_count
    if total > 0:
        print(f"      Linear-like: {linear_count}/{total}")
        print(f"      Threshold-like: {threshold_count}/{total}")
        print(f"      Exponential-like: {exp_count}/{total}")

    # RQ2: Pressure threshold
    thr = summary.get("pressure_thresholds", {})
    print(f"\n  RQ2: Is there a pressure threshold where directional release becomes inevitable?")
    if thr:
        print(f"      YES - {len(thr)} threshold crossings detected across symbol/horizon pairs")
        thr_by_q = {}
        for v in thr.values():
            q = v["threshold_quintile"]
            thr_by_q[q] = thr_by_q.get(q, 0) + 1
        for q in sorted(thr_by_q):
            print(f"        Quintile {q}: {thr_by_q[q]} crossings")
    else:
        print(f"      NO - No clear threshold detected")

    # RQ1: Long-duration small-magnitude vs short-duration large-magnitude
    print(f"\n  RQ1: Do long-duration small-magnitude residuals release differently?")
    diff_count = 0
    total_cell_pairs = 0
    for sym in SYMBOLS:
        surf = report.get("pressure_surface", {}).get(sym, {})
        for label in ["positive", "negative"]:
            cells = surf.get(label, {}).get("cells", [])
            for hk in PRESSURE_HORIZONS:
                # Find long-dur low-mag cell (dur_q=3-4, mag_q=0-1)
                long_low = [c for c in cells if c["dur_quintile"] >= 3 and c["mag_quintile"] <= 1]
                # Find short-dur high-mag cell (dur_q=0-1, mag_q=3-4)
                short_high = [c for c in cells if c["dur_quintile"] <= 1 and c["mag_quintile"] >= 3]
                for ll in long_low:
                    for sh in short_high:
                        ll_p = ll.get("results", {}).get(hk, {}).get("p_up")
                        sh_p = sh.get("results", {}).get(hk, {}).get("p_up")
                        if ll_p is not None and sh_p is not None:
                            total_cell_pairs += 1
                            if abs(ll_p - sh_p) > 0.05:
                                diff_count += 1
    if total_cell_pairs > 0:
        pct_diff = diff_count / total_cell_pairs * 100
        print(f"      {diff_count}/{total_cell_pairs} cell pairs differ by >5% ({pct_diff:.0f}%)")
        if pct_diff > 50:
            print(f"      -> YES - long/short release patterns differ materially")
        else:
            print(f"      -> Patterns are similar regardless of duration x magnitude tradeoff")

    print(f"\n{'=' * 72}")
    print("  ROL-3 COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    run_rol3()
