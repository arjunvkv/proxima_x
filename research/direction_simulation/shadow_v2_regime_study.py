"""
shadow_v2_regime_study.py — OFFLINE Shadow V2 Regime Study.

Designs a regime-conditioned entropy interpretation for Shadow that avoids
the destructive behavior of Candidate B (z-score normalization).

Reads proxima_demo.log, extracts SHADOW_RAW / OSS SURFACE / TOPOLOGY_DBG,
builds a 4-state Entropy State Machine, and tests regime-conditioned thresholds.

Usage: python research/direction_simulation/shadow_v2_regime_study.py
"""

import re
import os
import sys
import math
from collections import defaultdict, deque
from statistics import mean, stdev

LOG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "proxima_demo.log"))

# ── Entropy State Machine boundaries ──
ENTROPY_LOW_THRESH = 0.70
ENTROPY_CHAOS_THRESH = 0.85

# ── Proposed regime-conditioned thresholds ──
THRESHOLDS = {
    "LOW_INFORMATION": 0.05,
    "TRANSITION": 0.10,
    "DIRECTIONAL_PRESSURE": 0.05,
    "CHAOS": 0.20,
}

BASE_THRESHOLD = 0.05


# ── Parsing ──

def parse_timestamp(line):
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
    return m.group(1) if m else ""


def ts_to_seconds(ts):
    from datetime import datetime
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")
        ref = datetime(2026, 1, 1)
        return (dt - ref).total_seconds()
    except (ValueError, OSError):
        return 0.0


def parse_shadow_raw(line):
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


def parse_oss_surface(line):
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
        "regime": m.group(7),
        "p_cont": float(m.group(8)),
        "signal": int(m.group(16)),
    }


def parse_prod_signal_breakdown(line):
    if "count=" in line:
        return None
    m = re.search(
        r"\[PROD_SIGNAL_BREAKDOWN\] (\S+)"
        r" oss=([+-]\d+)\(ev=([-\d.]+),conf=([\d.]+)\)"
        r" ev_sig=([+-]\d+)"
        r" shadow=([+-]\d+)\(conf=([\d.]+)\)"
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


def parse_topology_dbg(line):
    m = re.search(
        r"\[TOPOLOGY_DBG\] (\S+)"
        r" decompose=\{.*?'normalized_entropy': np\.float64\(([\d.]+)\).*?"
        r"'dominant_prob': ([\d.]+).*?"
        r"'occupied_bins': (\d+).*?"
        r"'status': '(\w+)'",
        line,
    )
    if not m:
        return None
    return {
        "symbol": m.group(1),
        "normalized_entropy": float(m.group(2)),
        "dominant_prob": float(m.group(3)),
        "occupied_bins": int(m.group(4)),
        "status": m.group(5),
    }


# ── Helpers ──

def signal_from_score(score, threshold=0.05):
    if score > threshold:
        return 1
    elif score < -threshold:
        return -1
    return 0


def classify_entropy_state(entropy, d_entropy):
    if entropy < ENTROPY_LOW_THRESH:
        return "LOW_INFORMATION"
    elif entropy > ENTROPY_CHAOS_THRESH:
        return "CHAOS"
    else:
        if d_entropy > 0:
            return "TRANSITION"
        else:
            return "DIRECTIONAL_PRESSURE"


def compute_direction_pct(entries, key="signal"):
    total = len(entries)
    if total == 0:
        return {"BUY": 0, "SELL": 0, "FLAT": 0, "count": 0}
    buy = sum(1 for e in entries if e[key] == 1)
    sell = sum(1 for e in entries if e[key] == -1)
    flat = sum(1 for e in entries if e[key] == 0)
    return {"BUY": buy / total * 100, "SELL": sell / total * 100, "FLAT": flat / total * 100, "count": total}


def pearson_r(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    mx = mean(x)
    my = mean(y)
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)) * sum((y[i] - my) ** 2 for i in range(n)))
    return num / den if den > 0 else 0.0


# ── Main ──

def main():
    if not os.path.exists(LOG_PATH):
        print(f"ERROR: log not found at {LOG_PATH}")
        sys.exit(1)
    log_size_mb = os.path.getsize(LOG_PATH) / (1024 * 1024)
    print(f"Reading {LOG_PATH} ({log_size_mb:.1f} MB)...\n")

    # ── Parse ──
    shadow_entries = []
    oss_entries = []
    prod_entries = []
    topo_entries = []

    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ts = parse_timestamp(line)
            ts_s = ts_to_seconds(ts)

            s = parse_shadow_raw(line)
            if s:
                s["ts"] = ts_s
                s["ts_str"] = ts
                shadow_entries.append(s)
                continue

            o = parse_oss_surface(line)
            if o:
                o["ts"] = ts_s
                o["ts_str"] = ts
                oss_entries.append(o)
                continue

            p = parse_prod_signal_breakdown(line)
            if p:
                p["ts"] = ts_s
                p["ts_str"] = ts
                prod_entries.append(p)
                continue

            t = parse_topology_dbg(line)
            if t:
                t["ts"] = ts_s
                t["ts_str"] = ts
                topo_entries.append(t)
                continue

    print(f"{'=' * 72}")
    print("PARSED ENTRIES")
    print(f"{'=' * 72}")
    print(f"  SHADOW_RAW:            {len(shadow_entries)}")
    print(f"  OSS SURFACE:           {len(oss_entries)}")
    print(f"  PROD_SIGNAL_BREAKDOWN: {len(prod_entries)}")
    print(f"  TOPOLOGY_DBG:          {len(topo_entries)}")

    if not shadow_entries:
        print("No SHADOW_RAW entries — nothing to analyze.")
        return

    # ── Build Entropy State Machine ──
    # Compute d_entropy per symbol via rolling window
    by_symbol = defaultdict(list)
    for s in shadow_entries:
        by_symbol[s["symbol"]].append(s)

    ROLLING_WINDOW = 20

    for sym, entries in by_symbol.items():
        sorted_e = sorted(entries, key=lambda x: x["ts"])
        buffer = deque(maxlen=ROLLING_WINDOW)
        for e in sorted_e:
            buffer.append(e["entropy"])
            if len(buffer) >= 2:
                e["d_entropy"] = buffer[-1] - buffer[-2]
            else:
                e["d_entropy"] = 0.0
            e["state"] = classify_entropy_state(e["entropy"], e["d_entropy"])

    # ── Match OSS SURFACE to Shadow entries ──
    oss_lookup = {}
    for o in oss_entries:
        key = (round(o["ts"]), o["symbol"])
        oss_lookup[key] = o

    for e in shadow_entries:
        key = (round(e["ts"]), e["symbol"])
        if key in oss_lookup:
            o = oss_lookup[key]
            e["p_cont"] = o["p_cont"]
            e["oss_regime"] = o["regime"]
            e["oss_signal"] = o["signal"]
            e["exec_drift"] = o["exec_drift"]
        else:
            e["p_cont"] = None
            e["oss_regime"] = None
            e["oss_signal"] = None
            e["exec_drift"] = None

    # ── Match PROD_SIGNAL_BREAKDOWN to Shadow entries ──
    prod_lookup = {}
    for p in prod_entries:
        key = (round(p["ts"]), p["symbol"])
        prod_lookup[key] = p

    for e in shadow_entries:
        key = (round(e["ts"]), e["symbol"])
        if key in prod_lookup:
            p = prod_lookup[key]
            e["prod_oss"] = p["oss"]
            e["oss_ev"] = p["oss_ev"]
            e["prod_shadow"] = p["shadow"]
            e["shadow_conf"] = p["shadow_conf"]
            e["prod_regime"] = p["regime"]
            e["prod_reason"] = p["reason"]
            e["prod_final"] = p["final"]
        else:
            e["prod_oss"] = None
            e["oss_ev"] = None
            e["prod_shadow"] = None
            e["shadow_conf"] = None
            e["prod_regime"] = None
            e["prod_reason"] = None
            e["prod_final"] = None

    # ── Compute Shadow V2 scores ──
    # shadow_raw = ecdf - entropy
    # signal = threshold-based from raw
    for e in shadow_entries:
        e["shadow_raw_v2"] = e["ecdf"] - e["entropy"]
        e["shadow_signal_v2"] = signal_from_score(e["shadow_raw_v2"], BASE_THRESHOLD)

    # ── Filter to entries with full data ──
    full_entries = [e for e in shadow_entries if e["p_cont"] is not None and e["prod_shadow"] is not None]
    print(f"\n  Entries with full OSS+PROD match: {len(full_entries)}")

    if not full_entries:
        print("No matched entries — cannot proceed.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: ENTROPY STATE MACHINE — State Distribution
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("SECTION 1: ENTROPY STATE MACHINE — STATE DISTRIBUTION")
    print(f"{'=' * 72}")

    state_entries = defaultdict(list)
    for e in full_entries:
        state_entries[e["state"]].append(e)

    print(f"\n  {'State':25s} {'Count':>8s} {'Pct':>8s} {'Mean_Entropy':>13s} {'Mean_dEntropy':>14s} {'Dominant?':>10s}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 13} {'-' * 14} {'-' * 10}")

    for state in ["LOW_INFORMATION", "TRANSITION", "DIRECTIONAL_PRESSURE", "CHAOS"]:
        entries = state_entries.get(state, [])
        n = len(entries)
        pct = n / len(full_entries) * 100
        if entries:
            m_e = mean(e["entropy"] for e in entries)
            m_de = mean(e["d_entropy"] for e in entries)
        else:
            m_e = m_de = 0
        dom = "YES" if pct > 25 else "no"
        print(f"  {state:25s} {n:>8d} {pct:>7.2f}% {m_e:>12.4f} {m_de:>+13.6f} {dom:>10s}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: PER-STATE SHADOW SIGNAL DISTRIBUTION
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("SECTION 2: SHADOW SIGNAL DISTRIBUTION BY ENTROPY STATE")
    print(f"{'=' * 72}")

    header = f"  {'State':25s} {'n':>6s} {'BUY%':>7s} {'SELL%':>7s} {'FLAT%':>7s} {'Mean_p_cont':>12s} {'Mean|drift|':>12s} {'Agree%':>8s}"
    print(f"\n  {header}")
    print(f"  {'-' * len(header)}")

    for state in ["LOW_INFORMATION", "TRANSITION", "DIRECTIONAL_PRESSURE", "CHAOS"]:
        entries = state_entries.get(state, [])
        if not entries:
            continue
        d = compute_direction_pct(entries, "shadow_signal_v2")
        m_pc = mean(e["p_cont"] for e in entries)
        m_drift = mean(abs(e["exec_drift"]) for e in entries if e["exec_drift"] is not None)

        # Agreement: Shadow signal == OSS signal
        agree = sum(1 for e in entries if e["prod_oss"] is not None and e["shadow_signal_v2"] == e["prod_oss"])
        agree_pct = agree / len(entries) * 100

        print(f"  {state:25s} {d['count']:>6d} {d['BUY']:>6.2f}% {d['SELL']:>6.2f}% {d['FLAT']:>6.2f}% {m_pc:>11.4f} {m_drift:>11.4f} {agree_pct:>7.1f}%")

    # ── SELL bias by state ──
    print(f"\n  SELL bias (SELL% - BUY%) by state:")
    print(f"  {'State':25s} {'BUY%':>7s} {'SELL%':>7s} {'Bias':>8s}")
    print(f"  {'-' * 25} {'-' * 7} {'-' * 7} {'-' * 8}")
    for state in ["LOW_INFORMATION", "TRANSITION", "DIRECTIONAL_PRESSURE", "CHAOS"]:
        entries = state_entries.get(state, [])
        if not entries:
            continue
        d = compute_direction_pct(entries, "shadow_signal_v2")
        bias = d["SELL"] - d["BUY"]
        print(f"  {state:25s} {d['BUY']:>6.2f}% {d['SELL']:>6.2f}% {bias:>+7.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: SHADOW SIGNAL QUALITY BY STATE (p_cont alignment)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("SECTION 3: SHADOW SIGNAL QUALITY vs p_cont")
    print(f"{'=' * 72}")

    for state in ["LOW_INFORMATION", "TRANSITION", "DIRECTIONAL_PRESSURE", "CHAOS"]:
        entries = state_entries.get(state, [])
        if not entries:
            continue
        # For BUY signals: p_cont > 0.5 supports BUY → information
        # For SELL signals: p_cont < 0.5 supports SELL → information
        buy_entries = [e for e in entries if e["shadow_signal_v2"] == 1]
        sell_entries = [e for e in entries if e["shadow_signal_v2"] == -1]

        buy_info = sum(1 for e in buy_entries if e["p_cont"] > 0.5)
        sell_info = sum(1 for e in sell_entries if e["p_cont"] < 0.5)
        directional = len(buy_entries) + len(sell_entries)

        info_signals = buy_info + sell_info
        noise_signals = directional - info_signals
        info_ratio = info_signals / max(directional, 1) * 100

        print(f"\n  {state}:")
        print(f"    Directional signals: {directional}")
        print(f"    Aligned with p_cont: {info_signals} ({info_ratio:.1f}%) -> INFORMATION")
        print(f"    Misaligned:          {noise_signals} ({100 - info_ratio:.1f}%) -> NOISE")

        if buy_entries:
            mean_buy_pc = mean(e["p_cont"] for e in buy_entries)
            print(f"    BUY mean p_cont:     {mean_buy_pc:.4f}")
        if sell_entries:
            mean_sell_pc = mean(e["p_cont"] for e in sell_entries)
            print(f"    SELL mean p_cont:    {mean_sell_pc:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: REGIME-CONDITIONED THRESHOLD ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("SECTION 4: REGIME-CONDITIONED THRESHOLD ANALYSIS")
    print(f"{'=' * 72}")

    # Compute signals with base threshold (0.05) vs regime-conditioned thresholds
    for e in full_entries:
        e["signal_base"] = signal_from_score(e["shadow_raw_v2"], BASE_THRESHOLD)
        threshold = THRESHOLDS.get(e["state"], BASE_THRESHOLD)
        e["signal_regime"] = signal_from_score(e["shadow_raw_v2"], threshold)

    total = len(full_entries)

    for state in ["LOW_INFORMATION", "TRANSITION", "DIRECTIONAL_PRESSURE", "CHAOS"]:
        entries = state_entries.get(state, [])
        if not entries:
            continue
        n = len(entries)
        thresh = THRESHOLDS[state]

        # Count signal changes
        flat_to_dir = sum(
            1 for e in entries if e["signal_base"] == 0 and e["signal_regime"] != 0
        )
        dir_to_flat = sum(
            1 for e in entries if e["signal_base"] != 0 and e["signal_regime"] == 0
        )
        flip = sum(
            1
            for e in entries
            if e["signal_base"] != 0
            and e["signal_regime"] != 0
            and e["signal_base"] != e["signal_regime"]
        )

        # Direction of FLAT→directional changes
        flat_to_buy = sum(
            1 for e in entries if e["signal_base"] == 0 and e["signal_regime"] == 1
        )
        flat_to_sell = sum(
            1 for e in entries if e["signal_base"] == 0 and e["signal_regime"] == -1
        )

        # Direction of directional→FLAT changes
        dir_to_flat_was_buy = sum(
            1 for e in entries if e["signal_base"] == 1 and e["signal_regime"] == 0
        )
        dir_to_flat_was_sell = sum(
            1 for e in entries if e["signal_base"] == -1 and e["signal_regime"] == 0
        )

        # p_cont alignment for regime-threshold signals
        buy_info = sum(
            1
            for e in entries
            if e["signal_regime"] == 1 and e["p_cont"] > 0.5
        )
        sell_info = sum(
            1
            for e in entries
            if e["signal_regime"] == -1 and e["p_cont"] < 0.5
        )
        regime_directional = sum(1 for e in entries if e["signal_regime"] != 0)
        regime_info = buy_info + sell_info
        regime_info_ratio = regime_info / max(regime_directional, 1) * 100

        # Compare with base threshold quality
        base_buy_info = sum(
            1
            for e in entries
            if e["signal_base"] == 1 and e["p_cont"] > 0.5
        )
        base_sell_info = sum(
            1
            for e in entries
            if e["signal_base"] == -1 and e["p_cont"] < 0.5
        )
        base_directional = sum(1 for e in entries if e["signal_base"] != 0)
        base_info = base_buy_info + base_sell_info
        base_info_ratio = base_info / max(base_directional, 1) * 100

        directionals_base = compute_direction_pct(entries, "signal_base")
        directionals_regime = compute_direction_pct(entries, "signal_regime")

        print(f"\n  {state} (threshold: {thresh}, n={n}):")
        print(f"    FLAT → directional: {flat_to_dir} ({flat_to_dir/n*100:.1f}%)")
        print(f"      → BUY:  {flat_to_buy}")
        print(f"      → SELL: {flat_to_sell}")
        print(f"    Directional → FLAT:  {dir_to_flat} ({dir_to_flat/n*100:.1f}%)")
        print(f"      was BUY:  {dir_to_flat_was_buy}")
        print(f"      was SELL: {dir_to_flat_was_sell}")
        print(f"    BUY↔SELL flips:      {flip} ({flip/n*100:.1f}%)")
        print(f"    ---")
        print(f"    Base (thr={BASE_THRESHOLD}): BUY={directionals_base['BUY']:.1f}% "
              f"SELL={directionals_base['SELL']:.1f}% FLAT={directionals_base['FLAT']:.1f}%")
        print(f"    Regime (thr={thresh}):   BUY={directionals_regime['BUY']:.1f}% "
              f"SELL={directionals_regime['SELL']:.1f}% FLAT={directionals_regime['FLAT']:.1f}%")
        print(f"    ---")
        print(f"    Base p_cont alignment:   {base_info}/{base_directional} = {base_info_ratio:.1f}%")
        print(f"    Regime p_cont alignment: {regime_info}/{regime_directional} = {regime_info_ratio:.1f}%")
        print(f"    Alignment impact:        {'IMPROVES' if regime_info_ratio > base_info_ratio else 'DEGRADES'}"
              f" ({regime_info_ratio - base_info_ratio:+.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: OVERALL IMPACT OF REGIME-CONDITIONED THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("SECTION 5: OVERALL IMPACT — BASE vs REGIME THRESHOLDS")
    print(f"{'=' * 72}")

    all_base = compute_direction_pct(full_entries, "signal_base")
    all_regime = compute_direction_pct(full_entries, "signal_regime")

    print(f"\n  {'':25s} {'Base':>10s} {'Regime':>10s} {'Δ':>10s}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f"  {'BUY%':25s} {all_base['BUY']:>9.2f}% {all_regime['BUY']:>9.2f}% {all_regime['BUY'] - all_base['BUY']:>+9.2f}%")
    print(f"  {'SELL%':25s} {all_base['SELL']:>9.2f}% {all_regime['SELL']:>9.2f}% {all_regime['SELL'] - all_base['SELL']:>+9.2f}%")
    print(f"  {'FLAT%':25s} {all_base['FLAT']:>9.2f}% {all_regime['FLAT']:>9.2f}% {all_regime['FLAT'] - all_base['FLAT']:>+9.2f}%")

    sell_bias_base = all_base["SELL"] - all_base["BUY"]
    sell_bias_regime = all_regime["SELL"] - all_regime["BUY"]
    print(f"\n  SELL bias (SELL% - BUY%):")
    print(f"    Base:   {sell_bias_base:+.2f}%")
    print(f"    Regime: {sell_bias_regime:+.2f}%")
    bias_reduction = abs(sell_bias_base) - abs(sell_bias_regime)
    print(f"    Change: {sell_bias_regime - sell_bias_base:+.2f}%")
    print(f"    Bias reduction: {bias_reduction:+.2f}% {'YES ✓' if bias_reduction > 0 else 'NO ✗'}")

    # Overall p_cont alignment
    all_base_info = sum(
        1
        for e in full_entries
        if e["signal_base"] == 1 and e["p_cont"] > 0.5
    ) + sum(
        1
        for e in full_entries
        if e["signal_base"] == -1 and e["p_cont"] < 0.5
    )
    all_base_dir = sum(1 for e in full_entries if e["signal_base"] != 0)
    all_base_info_ratio = all_base_info / max(all_base_dir, 1) * 100

    all_regime_info = sum(
        1
        for e in full_entries
        if e["signal_regime"] == 1 and e["p_cont"] > 0.5
    ) + sum(
        1
        for e in full_entries
        if e["signal_regime"] == -1 and e["p_cont"] < 0.5
    )
    all_regime_dir = sum(1 for e in full_entries if e["signal_regime"] != 0)
    all_regime_info_ratio = all_regime_info / max(all_regime_dir, 1) * 100

    print(f"\n  Overall p_cont alignment:")
    print(f"    Base:   {all_base_info}/{all_base_dir} = {all_base_info_ratio:.1f}%")
    print(f"    Regime: {all_regime_info}/{all_regime_dir} = {all_regime_info_ratio:.1f}%")
    print(f"    Impact: {'IMPROVES' if all_regime_info_ratio > all_base_info_ratio else 'DEGRADES'}"
          f" ({all_regime_info_ratio - all_base_info_ratio:+.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: CHAOS STATE DEEP DIVE — SELL BIAS ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("SECTION 6: CHAOS STATE DEEP DIVE — SELL BIAS ORIGIN")
    print(f"{'=' * 72}")

    chaos_entries = state_entries.get("CHAOS", [])
    if chaos_entries:
        chaos_raw_scores = [e["shadow_raw_v2"] for e in chaos_entries]
        print(f"\n  CHAOS raw score stats:")
        print(f"    Mean: {mean(chaos_raw_scores):+.4f}")
        print(f"    Std:  {stdev(chaos_raw_scores):.4f}")
        print(f"    Min:  {min(chaos_raw_scores):+.4f}")
        print(f"    Max:  {max(chaos_raw_scores):+.4f}")

        # Distribution of ecdf vs entropy in CHAOS
        chaos_ecdf = [e["ecdf"] for e in chaos_entries]
        chaos_entropy = [e["entropy"] for e in chaos_entries]
        print(f"\n  CHAOS ecdf stats:")
        print(f"    Mean: {mean(chaos_ecdf):.4f}  Std: {stdev(chaos_ecdf):.4f}")
        print(f"    Median ecdf < 0.5: {sum(1 for e in chaos_ecdf if e < 0.5)}/{len(chaos_ecdf)}")
        print(f"  CHAOS entropy stats:")
        print(f"    Mean: {mean(chaos_entropy):.4f}  Std: {stdev(chaos_entropy):.4f}")

        # Entropy in CHAOS is always > 0.85, so shadow_raw = ecdf - high_entropy
        # This means shadow_raw is dominated by -entropy, pushing signals to SELL
        # ecdf is uniform in [0,1], so ecdf - 0.875 on average = large negative
        print(f"\n  CHAOS score decomposition:")
        print(f"    Mean(ecdf) - Mean(entropy) = {mean(chaos_ecdf):.4f} - {mean(chaos_entropy):.4f} = {mean(chaos_ecdf) - mean(chaos_entropy):.4f}")
        print(f"    This means SIGNAL is dominated by entropy penalty in CHAOS")
        print(f"    ecdf would need to be > {THRESHOLDS['CHAOS'] + mean(chaos_entropy):.4f} to produce BUY")
        print(f"    ecdf would need to be < {mean(chaos_entropy) - THRESHOLDS['CHAOS']:.4f} to produce SELL")

        # How does regime threshold help?
        base_dir_chaos = compute_direction_pct(chaos_entries, "signal_base")
        regime_dir_chaos = compute_direction_pct(chaos_entries, "signal_regime")
        print(f"\n  CHAOS signal distribution:")
        print(f"    Base threshold ({BASE_THRESHOLD}):  "
              f"BUY={base_dir_chaos['BUY']:.1f}% SELL={base_dir_chaos['SELL']:.1f}% FLAT={base_dir_chaos['FLAT']:.1f}%")
        print(f"    Regime threshold ({THRESHOLDS['CHAOS']}): "
              f"BUY={regime_dir_chaos['BUY']:.1f}% SELL={regime_dir_chaos['SELL']:.1f}% FLAT={regime_dir_chaos['FLAT']:.1f}%")

        # CHAOS-specific p_cont alignment comparison
        c_base_info = sum(1 for e in chaos_entries if e["signal_base"] == 1 and e["p_cont"] > 0.5) + \
                      sum(1 for e in chaos_entries if e["signal_base"] == -1 and e["p_cont"] < 0.5)
        c_base_dir = sum(1 for e in chaos_entries if e["signal_base"] != 0)
        c_regime_info = sum(1 for e in chaos_entries if e["signal_regime"] == 1 and e["p_cont"] > 0.5) + \
                        sum(1 for e in chaos_entries if e["signal_regime"] == -1 and e["p_cont"] < 0.5)
        c_regime_dir = sum(1 for e in chaos_entries if e["signal_regime"] != 0)

        print(f"\n  CHAOS p_cont alignment:")
        print(f"    Base:   {c_base_info}/{c_base_dir} = {c_base_info / max(c_base_dir, 1) * 100:.1f}%")
        print(f"    Regime: {c_regime_info}/{c_regime_dir} = {c_regime_info / max(c_regime_dir, 1) * 100:.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7: ANSWER — Does regime-conditioned preserve useful signals?
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("SECTION 7: ANSWER — REGIME-CONDITIONED THRESHOLD VERDICT")
    print(f"{'=' * 72}")

    # Useful signals preserved: directional signals that are p_cont aligned
    # False SELL bias: SELL signals that have p_cont >= 0.5 (contradicts SELL)

    # Base false SELL count
    base_false_sell = sum(1 for e in full_entries if e["signal_base"] == -1 and e["p_cont"] >= 0.5)
    regime_false_sell = sum(1 for e in full_entries if e["signal_regime"] == -1 and e["p_cont"] >= 0.5)

    # Useful signals preserved (directional AND p_cont aligned)
    base_useful = sum(1 for e in full_entries if e["signal_base"] == 1 and e["p_cont"] > 0.5) + \
                  sum(1 for e in full_entries if e["signal_base"] == -1 and e["p_cont"] < 0.5)
    regime_useful = sum(1 for e in full_entries if e["signal_regime"] == 1 and e["p_cont"] > 0.5) + \
                    sum(1 for e in full_entries if e["signal_regime"] == -1 and e["p_cont"] < 0.5)

    print(f"\n  {'Metric':40s} {'Base':>10s} {'Regime':>10s} {'Δ':>10s}")
    print(f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f"  {'Total directional signals':40s} {all_base_dir:>10d} {all_regime_dir:>10d} {all_regime_dir - all_base_dir:>+10d}")
    print(f"  {'Useful (p_cont aligned)':40s} {base_useful:>10d} {regime_useful:>10d} {regime_useful - base_useful:>+10d}")
    print(f"  {'False SELL (p_cont >= 0.5)':40s} {base_false_sell:>10d} {regime_false_sell:>10d} {regime_false_sell - base_false_sell:>+10d}")
    print(f"  {'Useful ratio':40s} {all_base_info_ratio:>9.1f}% {all_regime_info_ratio:>9.1f}% {all_regime_info_ratio - all_base_info_ratio:>+9.1f}%")
    print(f"  {'False SELL ratio':40s} {base_false_sell / max(all_base_dir, 1) * 100:>9.1f}% "
          f"{regime_false_sell / max(all_regime_dir, 1) * 100:>9.1f}% "
          f"{(regime_false_sell / max(all_regime_dir, 1) - base_false_sell / max(all_base_dir, 1)) * 100:>+9.1f}%")

    print(f"\n  Answer: Does regime-conditioned threshold preserve useful signals "
          f"while reducing false SELL bias?")
    false_sell_reduced = regime_false_sell < base_false_sell
    useful_preserved = regime_useful >= base_useful * 0.8  # preserve at least 80% of useful signals
    if false_sell_reduced and useful_preserved:
        print(f"  >>> YES — regime-conditioned thresholds reduce false SELL bias "
              f"({base_false_sell} -> {regime_false_sell}) "
              f"while preserving {regime_useful}/{base_useful} useful signals")
    elif false_sell_reduced:
        print(f"  >>> PARTIAL — reduces false SELL but reduces useful signals "
              f"({base_useful} -> {regime_useful})")
    else:
        print(f"  >>> NO — regime-conditioned thresholds do not improve false SELL bias")

    # ── CHAOS-specific answer ──
    if chaos_entries:
        chaos_base_false_sell = sum(1 for e in chaos_entries if e["signal_base"] == -1 and e["p_cont"] >= 0.5)
        chaos_regime_false_sell = sum(1 for e in chaos_entries if e["signal_regime"] == -1 and e["p_cont"] >= 0.5)
        chaos_base_dir = sum(1 for e in chaos_entries if e["signal_base"] != 0)
        chaos_regime_dir = sum(1 for e in chaos_entries if e["signal_regime"] != 0)

        print(f"\n  CHAOS-specific sell bias:")
        print(f"    Base false SELL:   {chaos_base_false_sell}/{chaos_base_dir} "
              f"({chaos_base_false_sell / max(chaos_base_dir, 1) * 100:.1f}%)")
        print(f"    Regime false SELL: {chaos_regime_false_sell}/{chaos_regime_dir} "
              f"({chaos_regime_false_sell / max(chaos_regime_dir, 1) * 100:.1f}%)")
        if chaos_regime_false_sell < chaos_base_false_sell:
            print(f"    >>> CHAOS regime threshold ({THRESHOLDS['CHAOS']}) effectively filters "
                  f"high-entropy SELL bias")
        else:
            print(f"    >>> CHAOS regime threshold does NOT reduce sell bias effectively")

    print(f"\n{'=' * 72}")
    print("END OF ANALYSIS")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
