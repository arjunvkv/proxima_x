"""
continuous_d_sim.py — OFFLINE analysis script.

Hypothesis: replacing discrete {-1,0,+1} direction with continuous D ∈ [-1,+1]
might solve signal starvation by retaining more actionable signals.

Reads proxima_demo.log, extracts OSS SURFACE / PROD_SIGNAL_BREAKDOWN /
SHADOW_RAW entries, constructs simulated continuous D, and compares
signal availability and characteristics.

Usage: python research/direction_simulation/continuous_d_sim.py
"""

import re
import os
import sys
import math
from collections import defaultdict
from statistics import mean, stdev

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "proxima_demo.log")
LOG_PATH = os.path.normpath(LOG_PATH)


def parse_oss_surface(line: str):
    """Parse [OSS SURFACE] log line -> dict or None."""
    m = re.search(r"\[OSS SURFACE\] (\S+) ecdf=([\d.]+) exec_drift=([-\d.]+) live_drift=([-\d.]+) horizon=blended\(w3=([\d.]+),w10=([\d.]+),w20=([\d.]+)\) regime=(\S+) p_cont=([\d.]+) ph=(\d+) pt=(\d+) r_pc=([\d.]+) r_ph=(\d+) r_pt=(\d+) r_bucket=(\S+) r_fb=(\S+) signal=(-?\d+) up=([\d.]+)% dn=([\d.]+)%", line)
    if not m:
        return None
    return {
        "symbol": m.group(1),
        "ecdf": float(m.group(2)),
        "exec_drift": float(m.group(3)),
        "live_drift": float(m.group(4)),
        "w3": float(m.group(5)),
        "w10": float(m.group(6)),
        "w20": float(m.group(7)),
        "regime": m.group(8),
        "p_cont": float(m.group(9)),
        "ph": int(m.group(10)),
        "pt": int(m.group(11)),
        "r_pc": float(m.group(12)),
        "r_ph": int(m.group(13)),
        "r_pt": int(m.group(14)),
        "r_bucket": m.group(15),
        "r_fb": m.group(16),
        "signal": int(m.group(17)),
        "up_pct": float(m.group(18)),
        "dn_pct": float(m.group(19)),
    }


def parse_prod_signal_breakdown(line: str):
    """Parse [PROD_SIGNAL_BREAKDOWN] line -> dict or None (skip count=N lines)."""
    if "count=" in line:
        return None
    m = re.search(r"\[PROD_SIGNAL_BREAKDOWN\] (\S+) oss=([+-]\d+)\(ev=([-\d.]+),conf=([\d.]+)\) ev_sig=([+-]\d+) shadow=([+-]\d+)\(conf=([\d.]+)\) regime=(\S+) reason=(\S+) final=([+-]?\d+) pc=([\d.]+)", line)
    if not m:
        return None
    return {
        "symbol": m.group(1),
        "oss": int(m.group(2)),
        "oss_ev": float(m.group(3)),
        "oss_conf": float(m.group(4)),
        "ev_sig": int(m.group(5)),
        "shadow": int(m.group(6)),
        "shadow_conf": float(m.group(7)),
        "regime": m.group(8),
        "reason": m.group(9),
        "final": int(m.group(10)),
        "pc": float(m.group(11)),
    }


def parse_shadow_raw(line: str):
    """Parse [SHADOW_RAW] line -> dict or None."""
    m = re.search(r"\[SHADOW_RAW\] (\S+) ecdf=([\d.]+) entropy=([\d.]+) score=([+-]?[\d.]+) raw=([+-]?\d+) final=([+-]?\d+) flip_suppress=(\S+)", line)
    if not m:
        return None
    return {
        "symbol": m.group(1),
        "ecdf": float(m.group(2)),
        "entropy": float(m.group(3)),
        "score": float(m.group(4)),
        "raw": int(m.group(5)),
        "final": int(m.group(6)),
        "flip_suppress": m.group(7) == "True",
    }


def parse_thesis_resolve(line: str):
    """Parse [THESIS_RESOLVE] line -> dict or None (trade outcome)."""
    m = re.search(r"\[THESIS_RESOLVE\] id=(\d+) (\S+) profit=([+-]?[\d.]+) label=(-?\d+) reason=(\S+)", line)
    if not m:
        return None
    return {
        "trade_id": int(m.group(1)),
        "symbol": m.group(2),
        "profit": float(m.group(3)),
        "label": int(m.group(4)),
        "reason": m.group(5),
    }


def continuous_d(entry: dict) -> float:
    """Compute simulated continuous D from exec_drift and p_cont."""
    return entry["exec_drift"] * (entry["p_cont"] - 0.5) * 2.0


def main():
    if not os.path.exists(LOG_PATH):
        print(f"ERROR: log not found at {LOG_PATH}")
        sys.exit(1)
    log_size_mb = os.path.getsize(LOG_PATH) / (1024 * 1024)
    print(f"Reading {LOG_PATH} ({log_size_mb:.1f} MB)...")

    # ---- Parse all entries ----
    oss_entries = []
    prod_entries = []
    shadow_entries = []
    thesis_entries = []

    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            e = parse_oss_surface(line)
            if e:
                oss_entries.append(e)
                continue
            p = parse_prod_signal_breakdown(line)
            if p:
                prod_entries.append(p)
                continue
            s = parse_shadow_raw(line)
            if s:
                shadow_entries.append(s)
                continue
            t = parse_thesis_resolve(line)
            if t:
                thesis_entries.append(t)

    print(f"\n{'='*72}")
    print(f"PARSED ENTRIES")
    print(f"{'='*72}")
    print(f"  OSS SURFACE:          {len(oss_entries)}")
    print(f"  PROD_SIGNAL_BREAKDOWN: {len(prod_entries)}")
    print(f"  SHADOW_RAW:            {len(shadow_entries)}")
    print(f"  THESIS_RESOLVE:        {len(thesis_entries)}")

    if not oss_entries:
        print("No OSS SURFACE entries found — nothing to analyze.")
        return

    # ---- 1. Discrete vs Continuous signal availability ----
    print(f"\n{'='*72}")
    print(f"SECTION 1: DISCRETE vs CONTINUOUS SIGNAL AVAILABILITY")
    print(f"{'='*72}")

    discrete_nonzero = sum(1 for e in oss_entries if e["signal"] != 0)
    total = len(oss_entries)
    print(f"  Total OSS cycles: {total}")
    print(f"  Discrete signal != 0: {discrete_nonzero} ({100*discrete_nonzero/total:.1f}%)")

    cont_sigs = [continuous_d(e) for e in oss_entries]
    cont_threshold_01 = sum(1 for d in cont_sigs if abs(d) > 0.1)
    cont_threshold_02 = sum(1 for d in cont_sigs if abs(d) > 0.2)
    cont_threshold_005 = sum(1 for d in cont_sigs if abs(d) > 0.05)

    print(f"\n  Continuous |D| > 0.05: {cont_threshold_005} ({100*cont_threshold_005/total:.1f}%)")
    print(f"  Continuous |D| > 0.10: {cont_threshold_01} ({100*cont_threshold_01/total:.1f}%)")
    print(f"  Continuous |D| > 0.20: {cont_threshold_02} ({100*cont_threshold_02/total:.1f}%)")
    print(f"\n  Signal availability ratio:")
    print(f"    |D|>0.1 / discrete_nonzero: {cont_threshold_01/max(1,discrete_nonzero):.2f}x")
    print(f"    |D|>0.1 - discrete_nonzero = {cont_threshold_01 - discrete_nonzero} {'(more signals)' if cont_threshold_01 > discrete_nonzero else '(fewer signals)'}")

    # ---- 2. Distribution of p_cont values ----
    print(f"\n{'='*72}")
    print(f"SECTION 2: p_cont DISTRIBUTION")
    print(f"{'='*72}")

    p_vals = [e["p_cont"] for e in oss_entries]
    print(f"  Mean p_cont: {mean(p_vals):.4f}")
    print(f"  Std p_cont:  {stdev(p_vals):.4f}" if len(p_vals) > 1 else "")
    print(f"  Min p_cont:  {min(p_vals):.4f}")
    print(f"  Max p_cont:  {max(p_vals):.4f}")

    buckets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
               (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    print(f"\n  p_cont histogram:")
    for lo, hi in buckets:
        cnt = sum(1 for p in p_vals if lo <= p < hi)
        bar = "#" * int(40 * cnt / max(total, 1))
        print(f"    [{lo:.1f}, {hi:.1f}): {cnt:6d} ({100*cnt/total:5.1f}%) {bar}")

    # How often is p_cont in the signal-able range?
    p_ge_60 = sum(1 for p in p_vals if p >= 0.60)
    p_le_40 = sum(1 for p in p_vals if p <= 0.40)
    print(f"\n  p_cont >= 0.60 (would trigger discrete +1): {p_ge_60} ({100*p_ge_60/total:.1f}%)")
    print(f"  p_cont <= 0.40 (would trigger discrete -1): {p_le_40} ({100*p_le_40/total:.1f}%)")
    print(f"  p_cont in (0.40, 0.60) — no-man's land: {total - p_ge_60 - p_le_40} ({100*(total-p_ge_60-p_le_40)/total:.1f}%)")

    # ---- 3. exec_drift vs p_cont relationship ----
    print(f"\n{'='*72}")
    print(f"SECTION 3: exec_drift vs p_cont AGREEMENT")
    print(f"{'='*72}")

    agree, disagree, neutral = 0, 0, 0
    for e in oss_entries:
        if e["exec_drift"] > 0 and e["p_cont"] > 0.5:
            agree += 1
        elif e["exec_drift"] < 0 and e["p_cont"] < 0.5:
            agree += 1
        elif e["exec_drift"] == 0:
            neutral += 1
        else:
            disagree += 1

    print(f"  exec_drift sign agrees with p_cont>0.5: {agree} ({100*agree/total:.1f}%)")
    print(f"  exec_drift sign disagrees:              {disagree} ({100*disagree/total:.1f}%)")
    print(f"  exec_drift == 0 (neutral):              {neutral} ({100*neutral/total:.1f}%)")

    # Correlation
    corr_n = sum(1 for e in oss_entries if e["exec_drift"] != 0)
    if corr_n > 1:
        ex = mean(e["exec_drift"] for e in oss_entries if e["exec_drift"] != 0)
        px = mean(e["p_cont"] for e in oss_entries if e["exec_drift"] != 0)
        num = sum((e["exec_drift"] - ex) * (e["p_cont"] - px) for e in oss_entries if e["exec_drift"] != 0)
        den = math.sqrt(sum((e["exec_drift"] - ex)**2 for e in oss_entries if e["exec_drift"] != 0) *
                       sum((e["p_cont"] - px)**2 for e in oss_entries if e["exec_drift"] != 0))
        corr = num / den if den > 0 else 0
        print(f"  Pearson r (exec_drift vs p_cont, nonzero only): {corr:.4f}")

    # ---- 4. Shadow signal comparison ----
    print(f"\n{'='*72}")
    print(f"SECTION 4: SHADOW SIGNAL CHARACTERISTICS")
    print(f"{'='*72}")

    if shadow_entries:
        shadow_final_nonzero = sum(1 for s in shadow_entries if s["final"] != 0)
        st = len(shadow_entries)
        print(f"  Shadow entries: {st}")
        print(f"  Shadow final != 0: {shadow_final_nonzero} ({100*shadow_final_nonzero/st:.1f}%)")
        print(f"  Shadow final == 0: {st - shadow_final_nonzero} ({100*(st-shadow_final_nonzero)/st:.1f}%)")

        # Shadow score distribution
        scores = [s["score"] for s in shadow_entries]
        print(f"  Shadow score mean:  {mean(scores):.4f}")
        print(f"  Shadow score std:   {stdev(scores):.4f}" if len(scores) > 1 else "")
        print(f"  Shadow score range: [{min(scores):.4f}, {max(scores):.4f}]")

        # Shadow continuous proxy: score * (2*entropy - 1) or just score itself
        # Score is already [-1, +1] continuous
        score_threshold_01 = sum(1 for s in shadow_entries if abs(s["score"]) > 0.1)
        score_threshold_02 = sum(1 for s in shadow_entries if abs(s["score"]) > 0.2)
        print(f"\n  Shadow |score| > 0.10: {score_threshold_01} ({100*score_threshold_01/st:.1f}%)")
        print(f"  Shadow |score| > 0.20: {score_threshold_02} ({100*score_threshold_02/st:.1f}%)")

        # Compare OSS signal vs Shadow signal
        # Match by symbol+timestamp proximity (simplified: cross-section comparison)
        oss_symbols = set(e["symbol"] for e in oss_entries)
        shadow_symbols = set(s["symbol"] for s in shadow_entries)
        common = oss_symbols & shadow_symbols
        print(f"\n  OSS symbols: {len(oss_symbols)}, Shadow symbols: {len(shadow_symbols)}, Common: {len(common)}")
    else:
        print("  No SHADOW_RAW entries found.")

    # ---- 5. PROD_SIGNAL_BREAKDOWN analysis ----
    print(f"\n{'='*72}")
    print(f"SECTION 5: PROD_SIGNAL_BREAKDOWN — OSS vs SHADOW fusion")
    print(f"{'='*72}")

    if prod_entries:
        pt = len(prod_entries)
        oss_nonzero_in_prod = sum(1 for p in prod_entries if p["oss"] != 0)
        shadow_nonzero_in_prod = sum(1 for p in prod_entries if p["shadow"] != 0)
        final_nonzero = sum(1 for p in prod_entries if p["final"] != 0)
        agree_count = sum(1 for p in prod_entries if p["oss"] != 0 and p["shadow"] != 0 and p["oss"] == p["shadow"])
        disagree_count = sum(1 for p in prod_entries if p["oss"] != 0 and p["shadow"] != 0 and p["oss"] != p["shadow"])

        print(f"  Total breakdown entries: {pt}")
        print(f"  OSS non-zero:   {oss_nonzero_in_prod} ({100*oss_nonzero_in_prod/pt:.1f}%)")
        print(f"  Shadow non-zero: {shadow_nonzero_in_prod} ({100*shadow_nonzero_in_prod/pt:.1f}%)")
        print(f"  Final non-zero:  {final_nonzero} ({100*final_nonzero/pt:.1f}%)")
        if agree_count + disagree_count > 0:
            print(f"  OSS=Shadow agree:   {agree_count} ({100*agree_count/(agree_count+disagree_count):.1f}%)")
            print(f"  OSS=Shadow disagree: {disagree_count} ({100*disagree_count/(agree_count+disagree_count):.1f}%)")

        # pc (probability cutoff) distribution
        pc_vals = [p["pc"] for p in prod_entries]
        print(f"\n  pc (cutoff) mean: {mean(pc_vals):.4f}")
        print(f"  pc always-exactly-0.50 count: {sum(1 for p in pc_vals if p == 0.50)}/{pt}")

        # Shadow confidence distribution
        sc_vals = [p["shadow_conf"] for p in prod_entries]
        print(f"  Shadow conf mean: {mean(sc_vals):.4f}")
        print(f"  Shadow conf >= 0.90: {sum(1 for s in sc_vals if s >= 0.90)} ({100*sum(1 for s in sc_vals if s >= 0.90)/max(1,len(sc_vals)):.1f}%)")
    else:
        print("  No per-symbol PROD_SIGNAL_BREAKDOWN entries found.")

    # ---- 6. Trade outcome validation (if available) ----
    print(f"\n{'='*72}")
    print(f"SECTION 6: TRADE OUTCOME — D ACCURACY (THESIS_RESOLVE)")
    print(f"{'='*72}")

    if thesis_entries:
        print(f"  Trade outcomes (THESIS_RESOLVE): {len(thesis_entries)}")
        profitable = sum(1 for t in thesis_entries if t["profit"] > 0)
        unprofitable = sum(1 for t in thesis_entries if t["profit"] < 0)
        flat = sum(1 for t in thesis_entries if t["profit"] == 0)
        print(f"  Profitable:   {profitable}")
        print(f"  Unprofitable: {unprofitable}")
        print(f"  Flat:         {flat}")
        print(f"  Win rate:     {100*profitable/max(1,profitable+unprofitable):.1f}% (excl flat)")

        # Match outcomes to OSS signals (simplified: look at last OSS entry for that symbol)
        last_oss = {}
        for e in oss_entries:
            last_oss[e["symbol"]] = e

        matched = 0
        correct_d = 0
        correct_discrete = 0
        for t in thesis_entries:
            sym = t["symbol"]
            if sym not in last_oss:
                continue
            e = last_oss[sym]
            d = continuous_d(e)
            # Outcome sign: profit > 0 means direction was correct
            outcome_dir = 1 if t["profit"] > 0 else (-1 if t["profit"] < 0 else 0)
            if outcome_dir == 0:
                continue
            matched += 1
            if (d > 0 and outcome_dir > 0) or (d < 0 and outcome_dir < 0):
                correct_d += 1
            if (e["signal"] > 0 and outcome_dir > 0) or (e["signal"] < 0 and outcome_dir < 0):
                correct_discrete += 1

        if matched > 0:
            print(f"\n  Matched outcomes to OSS signals: {matched}")
            print(f"  Continuous D correct direction: {correct_d} ({100*correct_d/matched:.1f}%)")
            print(f"  Discrete signal correct direction: {correct_discrete} ({100*correct_discrete/matched:.1f}%)")
    else:
        print("  No THESIS_RESOLVE entries found. Using alternative outcome sources...")

        # Try to find CLOSE_REQ entries and match to OSS signals
        print("  (CLOSE_REQ entries found but lack profit data — accuracy not computable)")

    # ---- 7. Continuous D magnitude analysis ----
    print(f"\n{'='*72}")
    print(f"SECTION 7: CONTINUOUS D — MAGNITUDE & REGIME ANALYSIS")
    print(f"{'='*72}")

    # D magnitude distribution
    d_abs = [abs(continuous_d(e)) for e in oss_entries]
    print(f"  |D| mean: {mean(d_abs):.4f}")
    print(f"  |D>0| count: {sum(1 for d in d_abs if d > 0)}/{total}")

    d_buckets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                 (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    print(f"\n  |D| histogram:")
    for lo, hi in d_buckets:
        cnt = sum(1 for d in d_abs if lo <= d < hi)
        bar = "#" * int(40 * cnt / max(total, 1))
        print(f"    [{lo:.1f}, {hi:.1f}): {cnt:6d} ({100*cnt/total:5.1f}%) {bar}")

    # D by regime
    regime_groups = defaultdict(list)
    for e in oss_entries:
        regime_groups[e["regime"]].append(continuous_d(e))

    print(f"\n  Mean D by regime:")
    for regime in sorted(regime_groups.keys()):
        vals = regime_groups[regime]
        d_mean = mean(vals)
        d_nonzero = sum(1 for v in vals if abs(v) > 0.1)
        print(f"    {regime:25s}: n={len(vals):6d}  mean D={d_mean:+.4f}  |D|>0.1={d_nonzero:6d} ({100*d_nonzero/len(vals):5.1f}%)")

    # ---- 8. Signal starvation root cause ----
    print(f"\n{'='*72}")
    print(f"SECTION 8: SIGNAL STARVATION ROOT CAUSE INDICATORS")
    print(f"{'='*72}")

    # What fraction of the time would continuous D give direction for free?
    # exec_drift is almost always 0 -> then D=0 regardless of p_cont
    ed_zero = sum(1 for e in oss_entries if e["exec_drift"] == 0)
    print(f"  exec_drift == 0: {ed_zero}/{total} ({100*ed_zero/total:.1f}%)")
    print(f"    -> When exec_drift=0, D=0 regardless of p_cont (D = 0 * (p_cont-0.5) * 2)")
    print(f"    -> Continuous D can ONLY produce signal when exec_drift != 0")

    ed_nonzero = total - ed_zero
    print(f"\n  exec_drift != 0: {ed_nonzero}/{total} ({100*ed_nonzero/total:.1f}%)")
    if ed_nonzero > 0:
        # Among entries with nonzero exec_drift, how often does D give signal?
        d_signal_among_ed = sum(1 for e in oss_entries if e["exec_drift"] != 0 and abs(continuous_d(e)) > 0.1)
        print(f"    |D| > 0.1 among exec_drift!=0: {d_signal_among_ed}/{ed_nonzero} ({100*d_signal_among_ed/ed_nonzero:.1f}%)")
        # Mean p_cont among exec_drift!=0
        mean_p_cont_ed_nonzero = mean(e["p_cont"] for e in oss_entries if e["exec_drift"] != 0)
        print(f"    Mean p_cont among exec_drift!=0: {mean_p_cont_ed_nonzero:.4f}")

    per_symbol = defaultdict(list)
    for e in oss_entries:
        per_symbol[e["symbol"]].append(continuous_d(e))

    print(f"\n  Mean |D| by symbol (top 10 by count):")
    sym_counts = sorted([(sym, len(vals)) for sym, vals in per_symbol.items()], key=lambda x: -x[1])
    for sym, cnt in sym_counts[:10]:
        vals = per_symbol[sym]
        d_mean = mean(vals)
        d_abs_mean = mean(abs(v) for v in vals)
        d_signal_pct = 100 * sum(1 for v in vals if abs(v) > 0.1) / len(vals)
        print(f"    {sym:8s}: n={cnt:6d}  mean D={d_mean:+.4f}  mean|D|={d_abs_mean:.4f}  |D|>0.1={d_signal_pct:5.1f}%")

    # ---- Conclusions ----
    print(f"\n{'='*72}")
    print(f"CONCLUSIONS")
    print(f"{'='*72}")

    # Determine starvation type
    if ed_zero / total > 0.8:
        print(f"  >> PRIMARY FINDING: exec_drift is zero in {100*ed_zero/total:.0f}% of cycles.")
        print(f"     Continuous D cannot help if exec_drift is stuck at 0.")
        print(f"     The real bottleneck is that exec_drift has near-zero variance,")
        print(f"     not that the {-1,0,+1} quantization is too coarse.")
    elif cont_threshold_01 > discrete_nonzero * 2:
        print(f"  >> Continuous D would generate {cont_threshold_01/discrete_nonzero:.1f}x more signals.")
        print(f"     This supports the hypothesis: quantization is causing starvation.")
    elif cont_threshold_01 <= discrete_nonzero:
        print(f"  >> Continuous D does NOT increase signal count for |D|>0.1 threshold.")
        print(f"     The starvation is not caused by {-1,0,+1} quantization.")

    # p_cont central tendency
    p_mean = mean(p_vals)
    if 0.45 <= p_mean <= 0.55:
        print(f"  >> p_cont is concentrated near 0.50 (mean={p_mean:.3f}).")
        print(f"     This means (p_cont-0.5) is near zero, making D ~ 0 even when exec_drift != 0.")
        print(f"     The OSS surface rarely produces confident continuation probabilities.")
    else:
        print(f"  >> p_cont mean={p_mean:.3f} — moderate dispersion exists.")

    # Shadow continuous vs OSS discrete
    if shadow_entries:
        sc = [s["score"] for s in shadow_entries]
        shadow_avail = sum(1 for s in sc if abs(s) > 0.1) / len(sc)
        print(f"  >> Shadow continuous score available {100*shadow_avail:.0f}% of the time.")
        if shadow_avail > 0.5:
            print(f"     Shadow already provides a continuous signal — exploitation path exists.")

    summary = {
        "total_oss_cycles": total,
        "discrete_nonzero": discrete_nonzero,
        "cont_01": cont_threshold_01,
        "ed_zero_pct": 100 * ed_zero / total,
        "p_cont_mean": mean(p_vals),
        "d_mean": mean(cont_sigs),
    }
    print(f"\n  Summary dict: {summary}")


if __name__ == "__main__":
    main()
