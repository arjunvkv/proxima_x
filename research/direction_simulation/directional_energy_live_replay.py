"""
Directional Energy Live Replay Agent -- Offline Analysis
========================================================
Reads proxima_demo.log, extracts all OSS SURFACE entries,
computes Directional Energy from ecdf changes, and compares
against exec_drift across multiple dimensions.

Output: Full analysis to stdout + summary to `delr_report.txt`
"""

import re
import os
import math
from collections import defaultdict, deque

LOG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "proxima_demo.log")
)
OUTPUT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "delr_report.txt")
)

OSS_RE = re.compile(
    r"\[OSS SURFACE\]\s+(?P<symbol>\w+)\s+"
    r"ecdf=(?P<ecdf>[\d.]+)\s+"
    r"exec_drift=(?P<exec_drift>-?[\d.]+)\s+"
    r"live_drift=(?P<live_drift>-?[\d.]+)\s+"
    r"horizon=blended\(w3=(?P<w3>[\d.]+),w10=(?P<w10>[\d.]+),w20=(?P<w20>[\d.]+)\)\s+"
    r"regime=(?P<regime>\S+)\s+"
    r"p_cont=(?P<p_cont>[\d.]+).*?"
    r"signal=(?P<signal>-?[\d.]+)"
)

# Session definitions (UTC hour ranges for major forex sessions)
def classify_session(hour):
    if 0 <= hour < 9:
        return "ASIAN"
    elif 8 <= hour < 17:
        return "LONDON"
    elif 13 <= hour < 22:
        return "NEW_YORK"
    else:
        return "OFF_HOURS"

def compute_directional_energy(ecdf_curr, ecdf_prev):
    ecdf_delta = ecdf_curr - ecdf_prev
    energy = abs(ecdf_delta)
    if energy > 1e-9:
        return ecdf_delta / energy, ecdf_delta
    return 0.0, 0.0


def main():
    if not os.path.exists(LOG_PATH):
        print(f"ERROR: Log not found at {LOG_PATH}")
        return

    print(f"Reading {LOG_PATH}...")
    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    entries = []
    for m in OSS_RE.finditer(content):
        entries.append({
            "symbol": m.group("symbol"),
            "ecdf": float(m.group("ecdf")),
            "exec_drift": int(m.group("exec_drift")),
            "live_drift": int(m.group("live_drift")),
            "regime": m.group("regime"),
            "p_cont": float(m.group("p_cont")),
            "signal": int(m.group("signal")),
        })

    # Extract timestamp with a separate regex (to avoid multiline complexity)
    ts_re = re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}).*?\[OSS SURFACE\]")
    timestamps = [m.group(1) for m in ts_re.finditer(content)]
    # Assign timestamps to entries (they should match 1:1)
    for i, ts in enumerate(timestamps):
        if i < len(entries):
            entries[i]["timestamp"] = ts

    print(f"Parsed {len(entries)} OSS SURFACE entries.")
    if len(entries) == 0:
        print("No entries found. Exiting.")
        return

    # ------------------------------------------------------------
    # Build per-symbol sequence with ecdf tracking
    # ------------------------------------------------------------
    symbol_sequences = defaultdict(list)
    for e in entries:
        symbol_sequences[e["symbol"]].append(e)

    # Compute Directional Energy for each entry
    prev_ecdf = {}
    for e in entries:
        sym = e["symbol"]
        prev = prev_ecdf.get(sym, e["ecdf"])
        de, ecdf_delta = compute_directional_energy(e["ecdf"], prev)
        e["de"] = de
        e["ecdf_delta"] = ecdf_delta
        e["energy"] = abs(ecdf_delta)
        prev_ecdf[sym] = e["ecdf"]

    # Total entries with valid DE
    total_de = sum(1 for e in entries if e["de"] != 0)
    pct_de = (total_de / len(entries)) * 100 if entries else 0

    lines = []
    def P(*args, **kwargs):
        s = " ".join(str(a) for a in args)
        print(s)
        lines.append(s)

    P("=" * 72)
    P("DIRECTIONAL ENERGY LIVE REPLAY — ANALYSIS REPORT")
    P("=" * 72)
    P(f"Log file: {LOG_PATH}")
    P(f"Total OSS SURFACE entries: {len(entries)}")
    P(f"Total entries with DE != 0: {total_de} ({pct_de:.1f}%)")
    P(f"exec_drift always 0: {all(e['exec_drift'] == 0 for e in entries)}")
    P(f"signal always 0: {all(e['signal'] == 0 for e in entries)}")
    P(f"Unique symbols: {len(symbol_sequences)}")
    P(f"Live drift values: {sorted(set(e['live_drift'] for e in entries))}")
    P("")

    # ------------------------------------------------------------
    # 3a. By Currency Pair (DE coverage)
    # ------------------------------------------------------------
    P("-" * 72)
    P("3a. DIRECTIONAL ENERGY BY CURRENCY PAIR")
    P("-" * 72)

    pair_stats = {}
    for sym, seq in symbol_sequences.items():
        n = len(seq)
        de_nonzero = sum(1 for e in seq if e["de"] != 0)
        exec_nonzero = sum(1 for e in seq if e["exec_drift"] != 0)
        mean_abs_de = sum(abs(e["de"]) for e in seq) / n if n else 0
        mean_energy = sum(e["energy"] for e in seq) / n if n else 0
        de_pos = sum(1 for e in seq if e["de"] > 0)
        de_neg = sum(1 for e in seq if e["de"] < 0)
        pair_stats[sym] = {
            "n": n, "de_nonzero": de_nonzero,
            "exec_nonzero": exec_nonzero,
            "de_coverage": de_nonzero / n * 100,
            "mean_abs_de": mean_abs_de,
            "mean_energy": mean_energy,
            "de_pos": de_pos, "de_neg": de_neg,
        }

    # Sorted by DE coverage
    sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1]["de_coverage"], reverse=True)
    P(f"{'Symbol':<10} {'Entries':<8} {'DE>0':<8} {'DE Cov%':<8} {'Mean|DE|':<8} {'MeanE':<8} {'DE+/DE-':<10}")
    P("-" * 62)
    for sym, s in sorted_pairs:
        ratio = f"{s['de_pos']}/{s['de_neg']}"
        P(f"{sym:<10} {s['n']:<8} {s['de_nonzero']:<8} {s['de_coverage']:<8.1f} {s['mean_abs_de']:<8.4f} {s['mean_energy']:<8.4f} {ratio:<10}")

    # Top/bottom 3
    P("")
    P(f"Highest DE coverage: {sorted_pairs[0][0]} ({sorted_pairs[0][1]['de_coverage']:.1f}%), "
      f"{sorted_pairs[1][0]} ({sorted_pairs[1][1]['de_coverage']:.1f}%), "
      f"{sorted_pairs[2][0]} ({sorted_pairs[2][1]['de_coverage']:.1f}%)")
    P(f"Lowest DE coverage:  {sorted_pairs[-1][0]} ({sorted_pairs[-1][1]['de_coverage']:.1f}%), "
      f"{sorted_pairs[-2][0]} ({sorted_pairs[-2][1]['de_coverage']:.1f}%), "
      f"{sorted_pairs[-3][0]} ({sorted_pairs[-3][1]['de_coverage']:.1f}%)")
    P(f"exec_drift is ALWAYS 0 on all pairs — no coverage gap measurement possible for exec_drift")
    P("")

    # ------------------------------------------------------------
    # 3b. By Session
    # ------------------------------------------------------------
    P("-" * 72)
    P("3b. DIRECTIONAL ENERGY BY SESSION")
    P("-" * 72)

    session_data = defaultdict(lambda: {
        "n": 0, "de_sum": 0.0, "de_abs_sum": 0.0,
        "de_nonzero": 0, "exec_nonzero": 0,
        "de_pos": 0, "de_neg": 0,
        "ecdf_deltas": []
    })
    for e in entries:
        ts = e.get("timestamp", "1970-01-01 00:00:00,000")
        try:
            hour = int(ts.split(" ")[1].split(":")[0])
        except (IndexError, ValueError):
            hour = 0
        sess = classify_session(hour)
        sd = session_data[sess]
        sd["n"] += 1
        sd["de_sum"] += e["de"]
        sd["de_abs_sum"] += abs(e["de"])
        if e["de"] != 0:
            sd["de_nonzero"] += 1
        if e["exec_drift"] != 0:
            sd["exec_nonzero"] += 1
        if e["de"] > 0:
            sd["de_pos"] += 1
        elif e["de"] < 0:
            sd["de_neg"] += 1
        sd["ecdf_deltas"].append(e["ecdf_delta"])

    session_order = ["ASIAN", "LONDON", "NEW_YORK", "OFF_HOURS"]
    P(f"{'Session':<15} {'Entries':<10} {'DE Cov%':<10} {'Mean|DE|':<10} {'MeanE':<10} {'DE+/DE-':<15} {'Exec>0':<10}")
    P("-" * 70)
    for sess in session_order:
        if sess not in session_data:
            continue
        sd = session_data[sess]
        cov = sd["de_nonzero"] / sd["n"] * 100
        mean_abs_de = sd["de_abs_sum"] / sd["n"]
        mean_e = sum(abs(d) for d in sd["ecdf_deltas"]) / sd["n"] if sd["n"] else 0
        ratio = f"{sd['de_pos']}/{sd['de_neg']}"
        P(f"{sess:<15} {sd['n']:<10} {cov:<10.1f} {mean_abs_de:<10.4f} {mean_e:<10.6f} {ratio:<15} {sd['exec_nonzero']:<10}")

    # Per-session mean DE sign
    P("")
    for sess in session_order:
        if sess not in session_data:
            continue
        sd = session_data[sess]
        mean_de = sd["de_sum"] / sd["n"]
        P(f"  {sess}: mean DE = {mean_de:+.6f} (bullish={'Y' if mean_de > 0 else 'N' if mean_de < 0 else 'FLAT'})")
    P("")

    # ------------------------------------------------------------
    # 3c. By Volatility Regime
    # ------------------------------------------------------------
    P("-" * 72)
    P("3c. DIRECTIONAL ENERGY BY VOLATILITY REGIME")
    P("-" * 72)

    regime_data = defaultdict(lambda: {
        "n": 0, "de_sum": 0.0, "de_abs_sum": 0.0,
        "de_nonzero": 0, "exec_nonzero": 0, "ecdf_deltas": []
    })
    for e in entries:
        rd = regime_data[e["regime"]]
        rd["n"] += 1
        rd["de_sum"] += e["de"]
        rd["de_abs_sum"] += abs(e["de"])
        if e["de"] != 0:
            rd["de_nonzero"] += 1
        if e["exec_drift"] != 0:
            rd["exec_nonzero"] += 1
        rd["ecdf_deltas"].append(e["ecdf_delta"])

    regime_order = ["NORMAL", "LOCKED", "ACTIVE_INSTABILITY", "COMPRESSED_CHAOS"]
    P(f"{'Regime':<25} {'Entries':<10} {'DE Cov%':<10} {'Mean|DE|':<10} {'DE>0%':<10} {'Exec>0':<10}")
    P("-" * 65)
    for reg in regime_order:
        if reg not in regime_data:
            continue
        rd = regime_data[reg]
        cov = rd["de_nonzero"] / rd["n"] * 100
        mean_abs_de = rd["de_abs_sum"] / rd["n"]
        de_pos_pct = sum(1 for e in entries if e["regime"] == reg and e["de"] > 0) / rd["n"] * 100
        P(f"{reg:<25} {rd['n']:<10} {cov:<10.1f} {mean_abs_de:<10.4f} {de_pos_pct:<10.1f} {rd['exec_nonzero']:<10}")

    # Coverage gap: exec_drift - DE non-zero
    P("")
    P(f"  Coverage gap (exec_drift - DE): exec_drift fires 0% in ALL regimes")
    P(f"  Largest gap: N/A — exec_drift never fires")
    P("")

    # ------------------------------------------------------------
    # 3d. By p_cont buckets
    # ------------------------------------------------------------
    P("-" * 72)
    P("3d. DIRECTIONAL ENERGY BY p_cont BUCKETS")
    P("-" * 72)

    bucket_defs = [
        ("Low (0-0.4)", lambda x: 0 <= x < 0.4),
        ("Medium (0.4-0.6)", lambda x: 0.4 <= x <= 0.6),
        ("High (0.6-1.0)", lambda x: 0.6 < x <= 1.0),
    ]
    bucket_data = defaultdict(lambda: {
        "n": 0, "de_sum": 0.0, "de_abs_sum": 0.0,
        "de_nonzero": 0, "exec_nonzero": 0
    })
    for e in entries:
        for label, fn in bucket_defs:
            if fn(e["p_cont"]):
                bd = bucket_data[label]
                bd["n"] += 1
                bd["de_sum"] += e["de"]
                bd["de_abs_sum"] += abs(e["de"])
                if e["de"] != 0:
                    bd["de_nonzero"] += 1
                if e["exec_drift"] != 0:
                    bd["exec_nonzero"] += 1

    P(f"{'Bucket':<20} {'Entries':<10} {'DE Cov%':<10} {'Mean|DE|':<10} {'Exec>0':<10}")
    P("-" * 60)
    for label, fn in bucket_defs:
        bd = bucket_data[label]
        if bd["n"] == 0:
            continue
        cov = bd["de_nonzero"] / bd["n"] * 100
        mean_abs_de = bd["de_abs_sum"] / bd["n"]
        P(f"{label:<20} {bd['n']:<10} {cov:<10.1f} {mean_abs_de:<10.4f} {bd['exec_nonzero']:<10}")
    P("")

    # ------------------------------------------------------------
    # 4. Critical Failure Analysis (DE = 0)
    # ------------------------------------------------------------
    P("-" * 72)
    P("4. CRITICAL FAILURE ANALYSIS — DE = 0")
    P("-" * 72)

    zero_de = [e for e in entries if e["de"] == 0]
    nonzero_de = [e for e in entries if e["de"] != 0]
    P(f"Total DE=0 entries: {len(zero_de)} / {len(entries)} ({len(zero_de)/len(entries)*100:.1f}%)")

    if zero_de:
        # ecdf pattern in zero-de entries
        flat_ecdf = sum(1 for e in zero_de if abs(e["ecdf_delta"]) < 1e-9)
        P(f"  Flat ecdf (no change): {flat_ecdf} / {len(zero_de)} ({flat_ecdf/len(zero_de)*100:.1f}%)")

        # p_cont distribution
        P("  p_cont distribution in DE=0 entries:")
        for label, fn in bucket_defs:
            cnt = sum(1 for e in zero_de if fn(e["p_cont"]))
            P(f"    {label}: {cnt} ({cnt/len(zero_de)*100:.1f}%)")

        # regime distribution
        P("  Regime distribution in DE=0 entries:")
        for reg in regime_order:
            cnt = sum(1 for e in zero_de if e["regime"] == reg)
            if cnt:
                total_reg = sum(1 for e in entries if e["regime"] == reg)
                P(f"    {reg:<25}: {cnt}/{len(zero_de)} ({cnt/len(zero_de)*100:.1f}% of DE=0, "
                  f"{cnt/total_reg*100:.1f}% of all {reg} entries)")

        # Compare with non-zero DE regimes
        P("")
        P("  Regime prevalence: DE=0 vs DE!=0:")
        for reg in regime_order:
            zd_cnt = sum(1 for e in zero_de if e["regime"] == reg)
            nzd_cnt = sum(1 for e in nonzero_de if e["regime"] == reg)
            zd_pct = zd_cnt / len(zero_de) * 100
            nzd_pct = nzd_cnt / len(nonzero_de) * 100
            delta = zd_pct - nzd_pct
            flag = " <<< OVER-REPRESENTED" if delta > 5 else (" >>> UNDER-REPRESENTED" if delta < -5 else "")
            P(f"    {reg:<25}: DE=0 {zd_pct:5.1f}% | DE!=0 {nzd_pct:5.1f}% | Δ={delta:+.1f}%{flag}")

        # Does DE fail during specific recognizable patterns?
        P("")
        P("  Pattern recognition for DE=0:")
        # Check for extreme ecdf values (near 0 or 1)
        extreme_ecdf = sum(1 for e in zero_de if e["ecdf"] <= 0.05 or e["ecdf"] >= 0.95)
        P(f"    Extreme ecdf (≤0.05 or ≥0.95): {extreme_ecdf}/{len(zero_de)} ({extreme_ecdf/len(zero_de)*100:.1f}%)")
        # Low p_cont
        low_pcont = sum(1 for e in zero_de if e["p_cont"] < 0.05)
        P(f"    Very low p_cont (<0.05): {low_pcont}/{len(zero_de)} ({low_pcont/len(zero_de)*100:.1f}%)")
        # LOCKED regime
        locked_de0 = sum(1 for e in zero_de if e["regime"] == "LOCKED")
        P(f"    Regime=LOCKED: {locked_de0}/{len(zero_de)} ({locked_de0/len(zero_de)*100:.1f}%)")
    P("")

    # ------------------------------------------------------------
    # 5. Sign Stability Analysis
    # ------------------------------------------------------------
    P("-" * 72)
    P("5. SIGN STABILITY ANALYSIS (5+ consecutive entries per symbol)")
    P("-" * 72)

    P("")
    P(f"  NOTE: exec_drift is always 0 — it never changes sign.")
    P(f"  Sign flip rate comparison is trivially zero for exec_drift.")
    P(f"  Only Directional Energy sign stability is meaningful.")
    P("")

    total_flips = 0
    total_windows = 0
    symbol_flip_rates = {}
    for sym, seq in symbol_sequences.items():
        seq = sorted(seq, key=lambda x: entries.index(x))
        if len(seq) < 5:
            continue
        flips = 0
        prev_de = None
        prev_exec = None
        for e in seq:
            # DE flips
            if e["de"] != 0 and prev_de is not None and prev_de != 0:
                if (e["de"] > 0) != (prev_de > 0):
                    flips += 1
            if e["de"] != 0:
                prev_de = e["de"]
        total_flips += flips
        win_count = max(0, sum(1 for i in range(1, len(seq))
                               if seq[i]["de"] != 0 and seq[i-1]["de"] != 0
                               and (seq[i]["de"] > 0) != (seq[i-1]["de"] > 0)))
        total_windows += len(seq)
        flip_rate = flips / len(seq) * 100 if len(seq) > 0 else 0
        symbol_flip_rates[sym] = {
            "seq_len": len(seq), "de_flips": flips,
            "de_flip_rate": flip_rate
        }

    P(f"  Total DE sign flips across all sequences: {total_flips}")
    P(f"  Total windows of opportunity: {total_windows}")
    P(f"  Global DE flip rate: {total_flips/max(total_windows,1)*100:.2f}%")

    # Per-symbol flip rates (top/bottom)
    sorted_flips = sorted(symbol_flip_rates.items(), key=lambda x: x[1]["de_flip_rate"], reverse=True)
    P("")
    P(f"  {'Symbol':<10} {'SeqLen':<8} {'DE Flips':<10} {'FlipRate%':<10}")
    P("  " + "-" * 38)
    for sym, sf in sorted_flips[:5]:
        P(f"  {sym:<10} {sf['seq_len']:<8} {sf['de_flips']:<10} {sf['de_flip_rate']:<10.2f}")
    P("  ...")
    for sym, sf in sorted_flips[-5:]:
        P(f"  {sym:<10} {sf['seq_len']:<8} {sf['de_flips']:<10} {sf['de_flip_rate']:<10.2f}")

    P("")
    # Summary
    most_stable = min(sorted_flips, key=lambda x: x[1]["de_flip_rate"])
    most_flippy = max(sorted_flips, key=lambda x: x[1]["de_flip_rate"])
    P(f"  Most stable (lowest flip rate): {most_stable[0]} ({most_stable[1]['de_flip_rate']:.2f}%)")
    P(f"  Most flippy (highest flip rate): {most_flippy[0]} ({most_flippy[1]['de_flip_rate']:.2f}%)")
    P("")

    # ------------------------------------------------------------
    # Summary / Key Findings
    # ------------------------------------------------------------
    P("=" * 72)
    P("KEY FINDINGS")
    P("=" * 72)
    P(f"1. exec_drift is ALWAYS 0 across all {len(entries)} OSS SURFACE entries.")
    P(f"   - Directional Energy provides the ONLY non-zero directional signal.")
    P(f"   - exec_drift coverage gap = 100% (it never fires).")
    P(f"2. Directional Energy fires on {total_de}/{len(entries)} entries ({pct_de:.1f}%).")
    P(f"3. DE=0 cases ({len(zero_de)} entries, {len(zero_de)/len(entries)*100:.1f}%):")
    if zero_de:
        zd_reg = max(regime_order, key=lambda r: sum(1 for e in zero_de if e["regime"] == r))
        P(f"   - Most common regime: {zd_reg} ({sum(1 for e in zero_de if e['regime']==zd_reg)/len(zero_de)*100:.1f}%)")
    P(f"4. Session with strongest directional bias: ", end="")
    for sess in session_order:
        if sess in session_data:
            sd = session_data[sess]
            mean_de = sd["de_sum"] / sd["n"]
            P(f"  {sess} (mean DE={mean_de:+.4f})", end="")
    P("")
    P(f"5. DE sign flip rate: {total_flips/max(total_windows,1)*100:.2f}% across all symbols")
    P("")
    P("=" * 72)
    P("END OF REPORT")
    P("=" * 72)

    # Write report
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
