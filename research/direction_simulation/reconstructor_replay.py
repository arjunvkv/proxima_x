"""
reconstructor_replay.py — OFFLINE DirectionField reconstruction from log data.

Reads proxima_demo.log, extracts OSS_SURFACE / PROD_SIGNAL_BREAKDOWN /
SHADOW_RAW / TPI_SOURCE entries, constructs both Phase A (discrete -1/0/1)
and proposed DirectionField (continuous D, uncertainty, conflict, regime),
then compares them across 5 dimensions.

Usage: python research/direction_simulation/reconstructor_replay.py
"""

import re
import os
import sys
import math
from datetime import datetime, timedelta
from collections import defaultdict
from statistics import mean, stdev, median

LOG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "proxima_demo.log"))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_timestamp(line: str):
    """Extract ISO-like timestamp from log line."""
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
    if m:
        return m.group(1)
    return ""


def ts_to_seconds(ts: str) -> float:
    """Convert timestamp string to seconds (relative reference) for window matching."""
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")
        ref = datetime(2026, 1, 1)
        return (dt - ref).total_seconds()
    except (ValueError, OSError):
        return 0.0


def parse_oss_surface(line: str):
    """Parse [OSS SURFACE] log line.

    Handles both with and without live_drift field (format changed mid-log).
    """
    m = re.search(
        r"\[OSS SURFACE\] (\S+) ecdf=([\d.]+) exec_drift=([-\d.]+?)"
        r"(?: live_drift=[-\d.]+)?"
        r" horizon=blended\(w3=([\d.]+),w10=([\d.]+),w20=([\d.]+)\)"
        r" regime=(\S+) p_cont=([\d.]+)"
        r" ph=(\d+) pt=(\d+) r_pc=([\d.]+) r_ph=(\d+) r_pt=(\d+)"
        r" r_bucket=(\S+) r_fb=(\S+) signal=(-?\d+) up=([\d.]+)% dn=([\d.]+)%",
        line,
    )
    if not m:
        return None
    return {
        "symbol": m.group(1),
        "ecdf": float(m.group(2)),
        "exec_drift": float(m.group(3)),
        "w3": float(m.group(4)),
        "w10": float(m.group(5)),
        "w20": float(m.group(6)),
        "regime": m.group(7),
        "p_cont": float(m.group(8)),
        "ph": int(m.group(9)),
        "pt": int(m.group(10)),
        "r_pc": float(m.group(11)),
        "r_ph": int(m.group(12)),
        "r_pt": int(m.group(13)),
        "r_bucket": m.group(14),
        "r_fb": m.group(15),
        "signal": int(m.group(16)),
        "up_pct": float(m.group(17)),
        "dn_pct": float(m.group(18)),
    }


def parse_prod_signal_breakdown(line: str):
    """Parse per-symbol [PROD_SIGNAL_BREAKDOWN] line. Skip count=N headers."""
    if "count=" in line:
        return None
    m = re.search(
        r"\[PROD_SIGNAL_BREAKDOWN\] (\S+)"
        r" oss=([+-]?\d+)\(ev=([-\d.]+),conf=([\d.]+)\)"
        r" ev_sig=([+-]?\d+)"
        r" shadow=([+-]?\d+)\(conf=([\d.]+)\)"
        r" regime=(\S+) reason=(\S+) final=([+-]?\d+) pc=([\d.]+)",
        line,
    )
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
    """Parse [SHADOW_RAW] line."""
    m = re.search(
        r"\[SHADOW_RAW\] (\S+)"
        r" ecdf=([\d.]+) entropy=([\d.]+) score=([+-]?[\d.]+)"
        r" raw=([+-]?\d+) final=([+-]?\d+) flip_suppress=(\S+)",
        line,
    )
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


def parse_tpi_source(line: str):
    """Parse [TPI_SOURCE] line."""
    m = re.search(
        r"\[TPI_SOURCE\] (\S+)"
        r" source=(\S+)"
        r" direction=(\S+)"
        r" conf=([\d.]+)"
        r" n_ticks=(\d+)",
        line,
    )
    if not m:
        return None
    return {
        "symbol": m.group(1),
        "source": m.group(2),
        "direction": m.group(3),
        "conf": float(m.group(4)),
        "n_ticks": int(m.group(5)),
    }


# ---------------------------------------------------------------------------
# Matching engine
# ---------------------------------------------------------------------------

MATCH_WINDOW = 3.0  # seconds — entries within this window + same symbol = same cycle


def build_cycles(oss_entries, prod_entries, shadow_entries, tpi_entries):
    """Group all parsed entries into cycles keyed by (rounded_time, symbol)."""
    cycles = defaultdict(lambda: {"oss": None, "prod": None, "shadow": None, "tpis": []})

    for e in oss_entries:
        key = (round(e["ts"]), e["symbol"])
        cycles[key]["oss"] = e
        cycles[key]["ts"] = e["ts"]

    for e in prod_entries:
        key = (round(e["ts"]), e["symbol"])
        cycles[key]["prod"] = e
        cycles[key]["ts"] = e["ts"]

    for e in shadow_entries:
        key = (round(e["ts"]), e["symbol"])
        cycles[key]["shadow"] = e
        cycles[key]["ts"] = e["ts"]

    for e in tpi_entries:
        key = (round(e["ts"]), e["symbol"])
        cycles[key]["tpis"].append(e)
        cycles[key]["ts"] = e["ts"]

    return dict(cycles)


# ---------------------------------------------------------------------------
# DirectionField constructor
# ---------------------------------------------------------------------------

def tpi_signal_to_value(direction: str, conf: float) -> float:
    """Map TPI direction+conf to continuous value."""
    if direction == "FLAT":
        return 0.0
    v = 1.0 if direction == "LONG" else -1.0
    return v * conf


def compute_direction_field(oss, prod, shadow, tpis, weights: dict = None):
    """Construct DirectionField from matched cycle entries.

    Returns dict with:
      D             - continuous -1..+1
      uncertainty   - 0..1
      regime        - from OSS
      conflict      - probe disagreement measure
      contributors  - dict of probe contributions
    """
    if weights is None:
        weights = {"oss": 0.35, "shadow": 0.35, "tpi": 0.15, "mcv": 0.15}

    # --- Contributor values ---
    oss_contrib = 0.0
    if oss:
        oss_contrib = oss["exec_drift"] * (oss["p_cont"] - 0.5) * 2.0

    shadow_contrib = 0.0
    if shadow:
        shadow_contrib = shadow["score"]

    tpi_contrib = 0.0
    if tpis:
        tpi_contrib = mean(
            tpi_signal_to_value(t["direction"], t["conf"]) for t in tpis
        )

    mcv_contrib = 0.0  # placeholder — not in logs yet

    contributors = {
        "oss": oss_contrib,
        "shadow": shadow_contrib,
        "tpi": tpi_contrib,
        "mcv": mcv_contrib,
    }

    # --- D: weighted blend ---
    D = (
        weights["oss"] * oss_contrib
        + weights["shadow"] * shadow_contrib
        + weights["tpi"] * tpi_contrib
        + weights["mcv"] * mcv_contrib
    )
    D = max(-1.0, min(1.0, D))

    # --- Probe agreement ---
    probes = [oss_contrib, shadow_contrib, tpi_contrib, mcv_contrib]
    non_zero_signs = [1 if p > 0.01 else (-1 if p < -0.01 else 0) for p in probes]
    # Agreement = fraction of non-zero probes with same sign as D
    # (simplified: max agreement score)
    if D == 0.0:
        agreement = 1.0
    else:
        d_sign = 1 if D > 0 else -1
        aligned = sum(1 for s in non_zero_signs if s == d_sign)
        total_nonzero = sum(1 for s in non_zero_signs if s != 0)
        agreement = aligned / max(total_nonzero, 1)

    uncertainty = 1.0 - agreement
    uncertainty = max(0.0, min(1.0, uncertainty))

    # --- Conflict: mean absolute deviation from D ---
    active = [p for p in probes if abs(p) > 0.001]
    if active:
        conflict = sum(abs(p - D) for p in active) / len(active)
    else:
        conflict = 0.0

    regime = oss["regime"] if oss else (prod["regime"] if prod else "UNKNOWN")

    phase_a = prod["final"] if prod else 0

    return {
        "D": D,
        "uncertainty": uncertainty,
        "regime": regime,
        "conflict": conflict,
        "contributors": contributors,
        "phase_a": phase_a,
        "oss_signal": oss["signal"] if oss else 0,
        "oss_contrib": oss_contrib,
        "shadow_contrib": shadow_contrib,
        "tpi_contrib": tpi_contrib,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(LOG_PATH):
        print(f"ERROR: log not found at {LOG_PATH}")
        sys.exit(1)
    log_size_mb = os.path.getsize(LOG_PATH) / (1024 * 1024)
    print(f"Reading {LOG_PATH} ({log_size_mb:.1f} MB)...")

    # ---- Parse all entries with timestamps ----
    oss_entries = []
    prod_entries = []
    shadow_entries = []
    tpi_entries = []

    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ts = ts_to_seconds(parse_timestamp(line))

            e = parse_oss_surface(line)
            if e:
                e["ts"] = ts
                oss_entries.append(e)
                continue

            p = parse_prod_signal_breakdown(line)
            if p:
                p["ts"] = ts
                prod_entries.append(p)
                continue

            s = parse_shadow_raw(line)
            if s:
                s["ts"] = ts
                shadow_entries.append(s)
                continue

            t = parse_tpi_source(line)
            if t:
                t["ts"] = ts
                tpi_entries.append(t)
                continue

    print(f"\n{'='*72}")
    print(f"PARSED ENTRIES")
    print(f"{'='*72}")
    print(f"  OSS SURFACE:           {len(oss_entries)}")
    print(f"  PROD_SIGNAL_BREAKDOWN:  {len(prod_entries)}")
    print(f"  SHADOW_RAW:            {len(shadow_entries)}")
    print(f"  TPI_SOURCE:            {len(tpi_entries)}")

    if not oss_entries:
        print("No OSS SURFACE entries found — nothing to analyze.")
        return

    # ---- Build matched cycles ----
    cycles = build_cycles(oss_entries, prod_entries, shadow_entries, tpi_entries)
    matched_keys = list(cycles.keys())
    print(f"\n  Matched cycles (rounded_t, symbol): {len(matched_keys)}")

    # ---- Construct DirectionField for each cycle ----
    fields = []
    incomplete = {"no_oss": 0, "no_prod": 0, "no_shadow": 0}
    for key, c in sorted(cycles.items(), key=lambda x: x[1].get("ts", 0)):
        if c["oss"] is None:
            incomplete["no_oss"] += 1
            continue
        field = compute_direction_field(
            oss=c["oss"],
            prod=c["prod"],
            shadow=c["shadow"],
            tpis=c["tpis"],
        )
        field["ts"] = c.get("ts", 0)
        field["symbol"] = c["oss"]["symbol"]
        field["key"] = key
        fields.append(field)

    print(f"  DirectionField constructed: {len(fields)}")
    print(f"  Skipped — no OSS: {incomplete['no_oss']}")
    print(f"  Skipped — no PROD: {incomplete['no_prod']}")
    print(f"  Skipped — no shadow: {incomplete['no_shadow']}")

    if not fields:
        print("No fields constructed — nothing to analyze.")
        return

    # ===================================================================
    # SECTION 1: Signal frequency — Phase A vs DirectionField
    # ===================================================================
    print(f"\n{'='*72}")
    print(f"SECTION 1: SIGNAL FREQUENCY — Phase A vs DirectionField D")
    print(f"{'='*72}")

    total = len(fields)
    phase_a_nonzero = sum(1 for f in fields if f["phase_a"] != 0)
    d_gt_02 = sum(1 for f in fields if abs(f["D"]) > 0.2)
    d_gt_01 = sum(1 for f in fields if abs(f["D"]) > 0.1)
    d_gt_03 = sum(1 for f in fields if abs(f["D"]) > 0.3)

    print(f"  Total cycles: {total}")
    print(f"  Phase A non-zero:        {phase_a_nonzero} ({100*phase_a_nonzero/total:.1f}%)")
    print(f"  DirectionField |D|>0.10: {d_gt_01} ({100*d_gt_01/total:.1f}%)")
    print(f"  DirectionField |D|>0.20: {d_gt_02} ({100*d_gt_02/total:.1f}%)")
    print(f"  DirectionField |D|>0.30: {d_gt_03} ({100*d_gt_03/total:.1f}%)")

    if phase_a_nonzero > 0:
        print(f"\n  Signal ratio (|D|>0.2 / Phase-A): {d_gt_02/phase_a_nonzero:.2f}x")
    print(f"  Net new signals (|D|>0.2 - Phase-A): {d_gt_02 - phase_a_nonzero}")

    # Phase A vs D sign agreement
    both_nonzero = [(f["phase_a"], f["D"]) for f in fields if f["phase_a"] != 0 and abs(f["D"]) > 0.1]
    sign_agree = sum(1 for pa, d in both_nonzero if (pa > 0 and d > 0) or (pa < 0 and d < 0))
    sign_disagree = sum(1 for pa, d in both_nonzero if (pa > 0 and d < 0) or (pa < 0 and d > 0))
    if both_nonzero:
        print(f"\n  When both non-zero — sign agreement:")
        print(f"    Agree:   {sign_agree} ({100*sign_agree/len(both_nonzero):.1f}%)")
        print(f"    Disagree: {sign_disagree} ({100*sign_disagree/len(both_nonzero):.1f}%)")

    # ===================================================================
    # SECTION 2: Directional stability — rolling window
    # ===================================================================
    print(f"\n{'='*72}")
    print(f"SECTION 2: DIRECTIONAL STABILITY (rolling window = 10)")
    print(f"{'='*72}")

    per_symbol_ts = defaultdict(list)
    for f in fields:
        per_symbol_ts[f["symbol"]].append((f["ts"], f["D"], f["phase_a"]))

    ROLLING_W = 10
    d_stabilities = []
    pa_stabilities = []
    for sym, entries in per_symbol_ts.items():
        sorted_e = sorted(entries, key=lambda x: x[0])
        # D stability = mean absolute change between consecutive D values within window
        d_vals = [e[1] for e in sorted_e]
        pa_vals = [e[2] for e in sorted_e]
        for i in range(0, len(d_vals) - ROLLING_W + 1):
            chunk_d = d_vals[i:i + ROLLING_W]
            chunk_pa = pa_vals[i:i + ROLLING_W]
            # D stability: inverse of stdev (smaller stdev = more stable)
            d_stabilities.append(stdev(chunk_d))
            # Phase A stability: fraction of non-flips within window
            nonflips = sum(1 for j in range(1, len(chunk_pa)) if chunk_pa[j] == chunk_pa[j - 1])
            pa_stabilities.append(nonflips / max(1, len(chunk_pa) - 1))

    if d_stabilities:
        print(f"  D stdev (window={ROLLING_W}): mean={mean(d_stabilities):.4f} median={median(d_stabilities):.4f}")
        print(f"  Phase A stability (non-flip fraction): mean={mean(pa_stabilities):.4f} median={median(pa_stabilities):.4f}")
        d_flip_rate = sum(1 for s in d_stabilities if s > 0.3) / len(d_stabilities)
        print(f"  D high-variance windows (>0.3 stdev): {100*d_flip_rate:.1f}%")

    # ===================================================================
    # SECTION 3: Conflict frequency
    # ===================================================================
    print(f"\n{'='*72}")
    print(f"SECTION 3: CONFLICT ANALYSIS")
    print(f"{'='*72}")

    conflicts = [f["conflict"] for f in fields]
    high_conflict = sum(1 for c in conflicts if c > 0.5)
    print(f"  Mean conflict: {mean(conflicts):.4f}")
    print(f"  Median conflict: {median(conflicts):.4f}")
    print(f"  Max conflict: {max(conflicts):.4f}")
    print(f"  Conflict > 0.5: {high_conflict} ({100*high_conflict/total:.1f}%)")
    print(f"  Conflict > 0.3: {sum(1 for c in conflicts if c > 0.3)} ({100*sum(1 for c in conflicts if c > 0.3)/total:.1f}%)")

    # Conflict by regime
    regime_conflict = defaultdict(list)
    for f in fields:
        regime_conflict[f["regime"]].append(f["conflict"])
    print(f"\n  Mean conflict by regime:")
    for regime in sorted(regime_conflict.keys()):
        vals = regime_conflict[regime]
        print(f"    {regime:25s}: n={len(vals):6d}  mean conflict={mean(vals):.4f}  >0.5={sum(1 for v in vals if v>0.5):5d}")

    # ===================================================================
    # SECTION 4: Trade blocking comparison
    # ===================================================================
    print(f"\n{'='*72}")
    print(f"SECTION 4: TRADE BLOCKING COMPARISON")
    print(f"{'='*72}")

    # Would D have blocked what Phase A blocked?
    # Phase A blocked when final == 0 (no trade)
    # D would block when |D| <= 0.2 (below threshold)
    pa_blocked = total - phase_a_nonzero
    d_blocked = total - d_gt_02
    both_blocked = sum(1 for f in fields if f["phase_a"] == 0 and abs(f["D"]) <= 0.2)
    d_allowed_pa_blocked = sum(1 for f in fields if f["phase_a"] == 0 and abs(f["D"]) > 0.2)
    pa_allowed_d_blocked = sum(1 for f in fields if f["phase_a"] != 0 and abs(f["D"]) <= 0.2)

    print(f"  Phase A blocked count:   {pa_blocked} ({100*pa_blocked/total:.1f}%)")
    print(f"  D blocked count (|D|<=0.2): {d_blocked} ({100*d_blocked/total:.1f}%)")
    print(f"  Both blocked:              {both_blocked}")
    print(f"  D allowed while Phase A blocked: {d_allowed_pa_blocked} — opportunities D would capture")
    print(f"  Phase A allowed while D blocked:  {pa_allowed_d_blocked} — trades D would reject")

    # Phase A trade quality if matched to outcomes
    print(f"\n  Trade opportunity analysis (from log):")
    print(f"    D-recaptured opportunities: {d_allowed_pa_blocked} cycles where Phase A=0 but |D|>0.2")
    print(f"    D-rejected trades:          {pa_allowed_d_blocked} cycles where Phase A!=0 but |D|<=0.2")

    # ===================================================================
    # SECTION 5: Distribution statistics
    # ===================================================================
    print(f"\n{'='*72}")
    print(f"SECTION 5: DISTRIBUTION STATISTICS")
    print(f"{'='*72}")

    # D distribution
    d_vals = [f["D"] for f in fields]
    print(f"\n  D distribution:")
    print(f"    Mean: {mean(d_vals):.4f}")
    print(f"    Median: {median(d_vals):.4f}")
    print(f"    Std:  {stdev(d_vals):.4f}")
    print(f"    Min:  {min(d_vals):.4f}")
    print(f"    Max:  {max(d_vals):.4f}")
    d_buckets = [(-1.0, -0.8), (-0.8, -0.6), (-0.6, -0.4), (-0.4, -0.2),
                 (-0.2, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6),
                 (0.6, 0.8), (0.8, 1.0)]
    print(f"  D histogram:")
    for lo, hi in d_buckets:
        cnt = sum(1 for d in d_vals if lo <= d < hi)
        bar = "#" * int(40 * cnt / max(total, 1))
        print(f"    [{lo:+.1f}, {hi:+.1f}): {cnt:6d} ({100*cnt/total:5.1f}%) {bar}")

    # Uncertainty distribution
    unc_vals = [f["uncertainty"] for f in fields]
    print(f"\n  Uncertainty distribution:")
    print(f"    Mean: {mean(unc_vals):.4f}")
    print(f"    Median: {median(unc_vals):.4f}")
    unc_buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for lo, hi in unc_buckets:
        cnt = sum(1 for u in unc_vals if lo <= u < hi)
        bar = "#" * int(40 * cnt / max(total, 1))
        print(f"    [{lo:.1f}, {hi:.1f}): {cnt:6d} ({100*cnt/total:5.1f}%) {bar}")

    # Contributor variance
    oss_conts = [f["contributors"]["oss"] for f in fields]
    shadow_conts = [f["contributors"]["shadow"] for f in fields]
    tpi_conts = [f["contributors"]["tpi"] for f in fields]
    mcv_conts = [f["contributors"]["mcv"] for f in fields]

    print(f"\n  Contributor statistics:")
    print(f"    OSS:    mean={mean(oss_conts):+.4f} std={stdev(oss_conts):.4f} nonzero={sum(1 for v in oss_conts if abs(v)>0.01):6d}")
    print(f"    Shadow: mean={mean(shadow_conts):+.4f} std={stdev(shadow_conts):.4f} nonzero={sum(1 for v in shadow_conts if abs(v)>0.01):6d}")
    print(f"    TPI:    mean={mean(tpi_conts):+.4f} std={stdev(tpi_conts):.4f} nonzero={sum(1 for v in tpi_conts if abs(v)>0.01):6d}")
    print(f"    MCV:    mean={mean(mcv_conts):+.4f} std={stdev(mcv_conts):.4f} nonzero={sum(1 for v in mcv_conts if abs(v)>0.01):6d}")

    # Contributor correlation matrix
    print(f"\n  Contributor correlation (Pearson r):")
    contrib_names = ["oss", "shadow", "tpi", "mcv"]
    contrib_arrays = {"oss": oss_conts, "shadow": shadow_conts, "tpi": tpi_conts, "mcv": mcv_conts}
    for i, n1 in enumerate(contrib_names):
        a1 = contrib_arrays[n1]
        for n2 in contrib_names[i + 1:]:
            a2 = contrib_arrays[n2]
            m1, m2 = mean(a1), mean(a2)
            num = sum((a1[j] - m1) * (a2[j] - m2) for j in range(len(a1)))
            den = math.sqrt(sum((a1[j] - m1)**2 for j in range(len(a1))) *
                            sum((a2[j] - m2)**2 for j in range(len(a2))))
            r = num / den if den > 0 else 0
            print(f"    {n1:8s} x {n2:8s}: r={r:+.4f}")

    # ===================================================================
    # SECTION 6: Regime correlation
    # ===================================================================
    print(f"\n{'='*72}")
    print(f"SECTION 6: REGIME CORRELATION ANALYSIS")
    print(f"{'='*72}")

    regime_d = defaultdict(list)
    regime_unc = defaultdict(list)
    for f in fields:
        regime_d[f["regime"]].append(f["D"])
        regime_unc[f["regime"]].append(f["uncertainty"])

    print(f"  Mean D and uncertainty by regime:")
    for regime in sorted(regime_d.keys()):
        d_v = regime_d[regime]
        u_v = regime_unc[regime]
        print(f"    {regime:25s}: n={len(d_v):6d}  mean D={mean(d_v):+.4f}  mean unc={mean(u_v):.4f}")
        # Within-regime |D|>0.2 rate
        signal_rate = sum(1 for d in d_v if abs(d) > 0.2) / max(len(d_v), 1)
        print(f"                          |D|>0.2 rate={100*signal_rate:5.1f}%")

    # ===================================================================
    # SECTION 7: What-If — Opportunity mapping
    # ===================================================================
    print(f"\n{'='*72}")
    print(f"SECTION 7: WHAT-IF OPPORTUNITY MAPPING")
    print(f"{'='*72}")

    # When Phase A is zero but D is strong
    missed_ops = [f for f in fields if f["phase_a"] == 0 and abs(f["D"]) > 0.2]
    print(f"  Phase A missed opportunities (Phase A=0, |D|>0.2): {len(missed_ops)}")
    if missed_ops:
        print(f"  Top regimes for missed ops:")
        miss_regimes = defaultdict(int)
        for f in missed_ops:
            miss_regimes[f["regime"]] += 1
        for regime in sorted(miss_regimes, key=miss_regimes.get, reverse=True)[:5]:
            print(f"    {regime:25s}: {miss_regimes[regime]}")

    # When Phase A is active but D is weak
    false_alarms = [f for f in fields if f["phase_a"] != 0 and abs(f["D"]) <= 0.2]
    print(f"\n  False alarms (Phase A!=0, |D|<=0.2): {len(false_alarms)}")
    if false_alarms:
        print(f"  Top regimes for false alarms:")
        fa_regimes = defaultdict(int)
        for f in false_alarms:
            fa_regimes[f["regime"]] += 1
        for regime in sorted(fa_regimes, key=fa_regimes.get, reverse=True)[:5]:
            print(f"    {regime:25s}: {fa_regimes[regime]}")

    # ===================================================================
    # SECTION 8: DirectionField vs Phase A direction concurrence
    # ===================================================================
    print(f"\n{'='*72}")
    print(f"SECTION 8: DIRECTION CONCURRENCE MATRIX")
    print(f"{'='*72}")

    # Compute concurrence across regimes
    print(f"  {'Regime':25s} {'n':>6s} {'D_agree':>9s} {'D_flip':>8s} {'conflict_hi':>12s}")
    print(f"  {'-'*25} {'-'*6} {'-'*9} {'-'*8} {'-'*12}")
    for regime in sorted(regime_d.keys()):
        r_fields = [f for f in fields if f["regime"] == regime]
        rn = len(r_fields)
        d_agree = sum(1 for f in r_fields if f["phase_a"] != 0 and abs(f["D"]) > 0.1
                      and ((f["phase_a"] > 0 and f["D"] > 0) or (f["phase_a"] < 0 and f["D"] < 0)))
        d_flip = sum(1 for f in r_fields if f["phase_a"] != 0 and abs(f["D"]) > 0.1
                     and ((f["phase_a"] > 0 and f["D"] < 0) or (f["phase_a"] < 0 and f["D"] > 0)))
        conf_hi = sum(1 for f in r_fields if f["conflict"] > 0.5)
        print(f"  {regime:25s} {rn:6d} {d_agree:9d} {d_flip:8d} {conf_hi:12d}")

    # ===================================================================
    # CONCLUSIONS
    # ===================================================================
    print(f"\n{'='*72}")
    print(f"CONCLUSIONS")
    print(f"{'='*72}")

    # Signal availability
    if phase_a_nonzero == 0:
        print(f"  >> Phase A produced ZERO non-zero signals across {total} cycles.")
        print(f"     DirectionField would capture {d_gt_02} signals (|D|>0.2) from same data.")
        print(f"     This strongly supports the continuous direction hypothesis.")
    elif d_gt_02 > phase_a_nonzero * 1.5:
        print(f"  >> DirectionField produces {d_gt_02/phase_a_nonzero:.1f}x more signals than Phase A.")
        print(f"     Continuous D would reduce signal starvation significantly.")
    else:
        print(f"  >> DirectionField produces comparable or fewer signals than Phase A.")
        print(f"     The bottleneck may not be quantization alone.")

    # Uncertainty vs conflict
    if mean(unc_vals) > 0.3:
        print(f"  >> Uncertainty is high (mean={mean(unc_vals):.3f}). Probes frequently disagree.")
        print(f"     This means the continuous D blend has genuine ambiguity.")
    else:
        print(f"  >> Uncertainty is low (mean={mean(unc_vals):.3f}). Probes tend to agree.")

    if high_conflict / total > 0.2:
        print(f"  >> High conflict in {100*high_conflict/total:.0f}% of cycles — probes strongly diverge.")
    else:
        print(f"  >> Conflict is rare (>{100*high_conflict/total:.0f}%). Probes largely consistent.")

    # Regime-specific findings
    high_unc_regimes = [(reg, mean(regime_unc[reg])) for reg in regime_unc]
    if high_unc_regimes:
        highest = max(high_unc_regimes, key=lambda x: x[1])
        print(f"  >> Highest uncertainty regime: {highest[0]} (mean unc={highest[1]:.3f})")

    # Trade-off assessment
    print(f"\n  Trade-off: D would allow {d_allowed_pa_blocked} more signals but reject {pa_allowed_d_blocked} existing ones.")
    net_gain = d_allowed_pa_blocked - pa_allowed_d_blocked
    if net_gain > 0:
        print(f"  Net opportunity gain: +{net_gain} signals")
    else:
        print(f"  Net opportunity loss: {net_gain} signals")

    print(f"\n  Done. Fields analyzed: {len(fields)}")
    print(f"  Source log: {LOG_PATH} ({log_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
