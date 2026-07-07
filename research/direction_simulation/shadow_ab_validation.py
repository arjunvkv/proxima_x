"""
shadow_ab_validation.py — OFFLINE Shadow A/B formula comparison.

Reads proxima_demo.log, extracts SHADOW_RAW and OSS SURFACE entries,
computes Current vs Candidate B (entropy z-score) Shadow formulas,
and compares direction distribution, regime impact, and information quality.

Usage: python research/direction_simulation/shadow_ab_validation.py
"""

import re
import os
import sys
from datetime import datetime
from collections import defaultdict, deque
from statistics import mean, stdev

LOG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "proxima_demo.log"))


def parse_timestamp(line: str) -> str:
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
    return m.group(1) if m else ""


def ts_to_seconds(ts: str) -> float:
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")
        ref = datetime(2026, 1, 1)
        return (dt - ref).total_seconds()
    except (ValueError, OSError):
        return 0.0


def parse_shadow_raw(line: str):
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


def parse_oss_surface(line: str):
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


MATCH_WINDOW = 3.0


def signal_from_score(score: float) -> int:
    if score > 0.05:
        return 1
    elif score < -0.05:
        return -1
    return 0


def compute_direction_pct(entries, key="signal"):
    total = len(entries)
    if total == 0:
        return {"BUY": 0, "SELL": 0, "FLAT": 0}
    buy = sum(1 for e in entries if e[key] == 1)
    sell = sum(1 for e in entries if e[key] == -1)
    flat = sum(1 for e in entries if e[key] == 0)
    return {
        "BUY": buy / total * 100,
        "SELL": sell / total * 100,
        "FLAT": flat / total * 100,
        "count": total,
    }


def main():
    if not os.path.exists(LOG_PATH):
        print(f"ERROR: log not found at {LOG_PATH}")
        sys.exit(1)
    log_size_mb = os.path.getsize(LOG_PATH) / (1024 * 1024)
    print(f"Reading {LOG_PATH} ({log_size_mb:.1f} MB)...")

    # ---- Parse all entries ----
    shadow_entries = []
    oss_entries = []

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

    print(f"\nParsed entries:")
    print(f"  SHADOW_RAW:   {len(shadow_entries)}")
    print(f"  OSS SURFACE:  {len(oss_entries)}")

    if not shadow_entries:
        print("No SHADOW_RAW entries found — nothing to analyze.")
        return

    # ---- Compute Candidate B (entropy z-score) for Shadow entries ----
    # Group by symbol, sort by timestamp, compute rolling stats
    by_symbol = defaultdict(list)
    for s in shadow_entries:
        by_symbol[s["symbol"]].append(s)

    WINDOW = 50

    b_entries = []  # entries with both A and B scores computed
    for sym, entries in by_symbol.items():
        sorted_e = sorted(entries, key=lambda x: x["ts"])
        # Rolling window stats for entropy
        rolling_buffer = deque(maxlen=WINDOW)

        for e in sorted_e:
            entropy = e["entropy"]
            rolling_buffer.append(entropy)

            # Current formula
            ecdf = e["ecdf"]
            score_current = ecdf - entropy
            signal_current = signal_from_score(score_current)

            # Candidate B
            if len(rolling_buffer) >= 2:
                r_mean = mean(rolling_buffer)
                r_std = stdev(rolling_buffer)
            else:
                r_mean = entropy
                r_std = 0.0

            entropy_z = (entropy - r_mean) / max(r_std, 1e-6)
            entropy_z = max(-2.0, min(2.0, entropy_z))
            adjusted = entropy_z / 4.0 + 0.5
            score_new = ecdf - adjusted
            signal_new = signal_from_score(score_new)

            b_entries.append({
                "symbol": sym,
                "ts": e["ts"],
                "ts_str": e["ts_str"],
                "ecdf": ecdf,
                "entropy": entropy,
                "raw": e["raw"],
                "final": e["final"],
                "flip_suppress": e["flip_suppress"],
                "score_current": score_current,
                "signal_current": signal_current,
                "score_new": score_new,
                "signal_new": signal_new,
                "entropy_z": entropy_z,
                "adjusted_entropy": adjusted,
                "rolling_n": len(rolling_buffer),
            })

    print(f"  Shadow entries with AB scores: {len(b_entries)}")

    # ---- Match OSS surface to Shadow entries ----
    # Build OSS lookup by (rounded_ts, symbol)
    oss_lookup = {}
    for o in oss_entries:
        key = (round(o["ts"]), o["symbol"])
        oss_lookup[key] = o

    for e in b_entries:
        key = (round(e["ts"]), e["symbol"])
        if key in oss_lookup:
            o = oss_lookup[key]
            e["p_cont"] = o["p_cont"]
            e["regime"] = o["regime"]
            e["oss_signal"] = o["signal"]
            e["exec_drift"] = o["exec_drift"]
        else:
            e["p_cont"] = None
            e["regime"] = None
            e["oss_signal"] = None
            e["exec_drift"] = None

    matched = [e for e in b_entries if e["p_cont"] is not None]
    unmatched = [e for e in b_entries if e["p_cont"] is None]
    print(f"  Matched with OSS SURFACE: {len(matched)}")
    print(f"  Unmatched:                {len(unmatched)}")

    # Use matched for regime/p_cont analysis, all for direction distribution
    all_entries = b_entries
    match_entries = matched

    # ===========================================================================
    # SECTION 1: Overall Direction Distribution
    # ===========================================================================
    print(f"\n{'=' * 72}")
    print("SECTION 1: DIRECTION DISTRIBUTION — ALL ENTRIES")
    print(f"{'=' * 72}")

    cur_dist = compute_direction_pct(all_entries, "signal_current")
    new_dist = compute_direction_pct(all_entries, "signal_new")

    print(f"\n  Current formula (score = ecdf - entropy):")
    print(f"    BUY:  {cur_dist['BUY']:.2f}%")
    print(f"    SELL: {cur_dist['SELL']:.2f}%")
    print(f"    FLAT: {cur_dist['FLAT']:.2f}%")
    print(f"    Total: {cur_dist['count']}")

    print(f"\n  Candidate B (entropy z-score):")
    print(f"    BUY:  {new_dist['BUY']:.2f}%")
    print(f"    SELL: {new_dist['SELL']:.2f}%")
    print(f"    FLAT: {new_dist['FLAT']:.2f}%")
    print(f"    Total: {new_dist['count']}")

    # ===========================================================================
    # SECTION 2: Score statistics
    # ===========================================================================
    print(f"\n{'=' * 72}")
    print("SECTION 2: SCORE STATISTICS")
    print(f"{'=' * 72}")

    cur_scores = [e["score_current"] for e in all_entries]
    new_scores = [e["score_new"] for e in all_entries]

    print(f"\n  Current formula scores:")
    print(f"    Mean: {mean(cur_scores):+.4f}")
    print(f"    Std:  {stdev(cur_scores):.4f}")
    print(f"    Min:  {min(cur_scores):+.4f}")
    print(f"    Max:  {max(cur_scores):+.4f}")

    print(f"\n  Candidate B scores:")
    print(f"    Mean: {mean(new_scores):+.4f}")
    print(f"    Std:  {stdev(new_scores):.4f}")
    print(f"    Min:  {min(new_scores):+.4f}")
    print(f"    Max:  {max(new_scores):+.4f}")

    # ===========================================================================
    # SECTION 3: Disagreement Analysis
    # ===========================================================================
    print(f"\n{'=' * 72}")
    print("SECTION 3: SIGNAL DISAGREEMENT ANALYSIS")
    print(f"{'=' * 72}")

    total = len(all_entries)
    disagree = sum(1 for e in all_entries if e["signal_current"] != e["signal_new"])
    agree = total - disagree
    print(f"\n  Total entries: {total}")
    print(f"  Agree (same signal):   {agree} ({100 * agree / total:.2f}%)")
    print(f"  Disagree (different):  {disagree} ({100 * disagree / total:.2f}%)")

    # FLAT transitions
    flat_to_dir = sum(1 for e in all_entries if e["signal_current"] == 0 and e["signal_new"] != 0)
    dir_to_flat = sum(1 for e in all_entries if e["signal_current"] != 0 and e["signal_new"] == 0)
    flat_both = sum(1 for e in all_entries if e["signal_current"] == 0 and e["signal_new"] == 0)
    dir_both = sum(1 for e in all_entries if e["signal_current"] != 0 and e["signal_new"] != 0)

    print(f"\n  FLAT -> directional (under B): {flat_to_dir} ({100 * flat_to_dir / total:.2f}%)")
    print(f"  Directional -> FLAT (under B):  {dir_to_flat} ({100 * dir_to_flat / total:.2f}%)")
    print(f"  FLAT in both formulas:          {flat_both} ({100 * flat_both / total:.2f}%)")
    print(f"  Directional in both formulas:   {dir_both} ({100 * dir_both / total:.2f}%)")

    # FLAT -> BUY vs FLAT -> SELL
    flat_to_buy = sum(1 for e in all_entries if e["signal_current"] == 0 and e["signal_new"] == 1)
    flat_to_sell = sum(1 for e in all_entries if e["signal_current"] == 0 and e["signal_new"] == -1)
    print(f"\n  Of FLAT->directional: BUY={flat_to_buy}, SELL={flat_to_sell}")

    # Directional under A that become FLAT under B: what were they?
    dir_to_flat_buy = sum(1 for e in all_entries if e["signal_current"] == 1 and e["signal_new"] == 0)
    dir_to_flat_sell = sum(1 for e in all_entries if e["signal_current"] == -1 and e["signal_new"] == 0)
    print(f"  Of directional->FLAT: was BUY={dir_to_flat_buy}, was SELL={dir_to_flat_sell}")

    # Flipped signals (BUY<->SELL)
    flipped = sum(1 for e in all_entries if e["signal_current"] != 0 and e["signal_new"] != 0 and e["signal_current"] != e["signal_new"])
    print(f"\n  Signal flipped (BUY<->SELL): {flipped} ({100 * flipped / total:.2f}%)")

    # Net bias change
    cur_buy_sell_diff = cur_dist["BUY"] - cur_dist["SELL"]
    new_buy_sell_diff = new_dist["BUY"] - new_dist["SELL"]
    print(f"\n  SELL bias (BUY% - SELL%):")
    print(f"    Current:  {cur_buy_sell_diff:+.2f}%")
    print(f"    Candidate B: {new_buy_sell_diff:+.2f}%")
    print(f"    Change:   {new_buy_sell_diff - cur_buy_sell_diff:+.2f}%")

    # ===========================================================================
    # SECTION 4: Per-Regime Breakdown
    # ===========================================================================
    print(f"\n{'=' * 72}")
    print("SECTION 4: PER-REGIME BREAKDOWN (matched entries only)")
    print(f"{'=' * 72}")

    regime_groups = defaultdict(list)
    for e in match_entries:
        regime_groups[e["regime"]].append(e)

    print(f"\n  {'Regime':25s} {'n':>6s} {'Cur_Score':>10s} {'New_Score':>10s} {'Cur_B%':>7s} {'Cur_S%':>7s} {'Cur_F%':>7s} {'New_B%':>7s} {'New_S%':>7s} {'New_F%':>7s}")
    print(f"  {'-' * 25} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")

    for regime in sorted(regime_groups.keys(), key=lambda r: -len(regime_groups[r])):
        entries = regime_groups[regime]
        n = len(entries)
        cur_scores_r = [e["score_current"] for e in entries]
        new_scores_r = [e["score_new"] for e in entries]

        cur_m = mean(cur_scores_r)
        new_m = mean(new_scores_r)

        cur_d = compute_direction_pct(entries, "signal_current")
        new_d = compute_direction_pct(entries, "signal_new")

        print(f"  {regime:25s} {n:6d} {cur_m:>+10.4f} {new_m:>+10.4f} {cur_d['BUY']:>6.2f}% {cur_d['SELL']:>6.2f}% {cur_d['FLAT']:>6.2f}% {new_d['BUY']:>6.2f}% {new_d['SELL']:>6.2f}% {new_d['FLAT']:>6.2f}%")

    # Regime-specific bias reduction
    print(f"\n  SELL bias (BUY% - SELL%) by regime:")
    print(f"  {'Regime':25s} {'Cur':>8s} {'New':>8s} {'Δ':>8s} {'SELL_reduced?':>14s}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 14}")
    for regime in sorted(regime_groups.keys(), key=lambda r: -len(regime_groups[r])):
        entries = regime_groups[regime]
        n = len(entries)
        cur_d = compute_direction_pct(entries, "signal_current")
        new_d = compute_direction_pct(entries, "signal_new")
        cur_bias = cur_d["BUY"] - cur_d["SELL"]
        new_bias = new_d["BUY"] - new_d["SELL"]
        delta = new_bias - cur_bias
        reduced = "YES ✓" if abs(new_bias) < abs(cur_bias) else ("NO ✗" if abs(new_bias) > abs(cur_bias) else "SAME")
        print(f"  {regime:25s} {cur_bias:>+7.2f}% {new_bias:>+7.2f}% {delta:>+7.2f}% {reduced:>14s}")

    # ===========================================================================
    # SECTION 5: Correlation with p_cont
    # ===========================================================================
    print(f"\n{'=' * 72}")
    print("SECTION 5: SIGNAL STRENGTH vs p_cont CORRELATION")
    print(f"{'=' * 72}")

    # For entries with p_cont available (matched)
    p_cont_vals = [e["p_cont"] for e in match_entries if e["p_cont"] is not None]
    cur_scores_p = [e["score_current"] for e in match_entries if e["p_cont"] is not None]
    new_scores_p = [e["score_new"] for e in match_entries if e["p_cont"] is not None]

    if len(p_cont_vals) > 1:
        # Pearson correlation: p_cont vs score_current
        m_p = mean(p_cont_vals)
        m_cur = mean(cur_scores_p)
        m_new = mean(new_scores_p)

        num_cur = sum((p_cont_vals[i] - m_p) * (cur_scores_p[i] - m_cur) for i in range(len(p_cont_vals)))
        num_new = sum((p_cont_vals[i] - m_p) * (new_scores_p[i] - m_new) for i in range(len(p_cont_vals)))

        den_p = sum((p_cont_vals[i] - m_p) ** 2 for i in range(len(p_cont_vals)))
        den_cur = sum((cur_scores_p[i] - m_cur) ** 2 for i in range(len(p_cont_vals)))
        den_new = sum((new_scores_p[i] - m_new) ** 2 for i in range(len(p_cont_vals)))

        r_cur = num_cur / (den_p * den_cur) ** 0.5 if den_p > 0 and den_cur > 0 else 0
        r_new = num_new / (den_p * den_new) ** 0.5 if den_p > 0 and den_new > 0 else 0

        print(f"\n  Pearson r with p_cont:")
        print(f"    Current formula:  r = {r_cur:+.4f}")
        print(f"    Candidate B:      r = {r_new:+.4f}")
        print(f"    {'B improves correlation' if abs(r_new) > abs(r_cur) else 'Current better correlated'}")

    # p_cont bins
    p_bins = [(0.0, 0.4, "low (0-0.4)"),
              (0.4, 0.6, "mid (0.4-0.6)"),
              (0.6, 1.0, "high (0.6-1.0)")]

    print(f"\n  Signal distribution by p_cont bucket:")
    print(f"  {'p_cont bucket':20s} {'n':>6s} {'Cur_B%':>7s} {'Cur_S%':>7s} {'New_B%':>7s} {'New_S%':>7s} {'ΔBUY':>7s} {'ΔSELL':>7s} {'ΔFLAT':>7s}")
    print(f"  {'-' * 20} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")

    for lo, hi, label in p_bins:
        bucket = [e for e in match_entries if e["p_cont"] is not None and lo <= e["p_cont"] < hi]
        if not bucket:
            continue
        n = len(bucket)
        cur_d = compute_direction_pct(bucket, "signal_current")
        new_d = compute_direction_pct(bucket, "signal_new")
        db = new_d["BUY"] - cur_d["BUY"]
        ds = new_d["SELL"] - cur_d["SELL"]
        df = new_d["FLAT"] - cur_d["FLAT"]
        print(f"  {label:20s} {n:6d} {cur_d['BUY']:>6.2f}% {cur_d['SELL']:>6.2f}% {new_d['BUY']:>6.2f}% {new_d['SELL']:>6.2f}% {db:>+6.2f}% {ds:>+6.2f}% {df:>+6.2f}%")

    # ===========================================================================
    # SECTION 6: Critical Question — Information or Noise?
    # ===========================================================================
    print(f"\n{'=' * 72}")
    print("SECTION 6: INFORMATION vs NOISE ANALYSIS")
    print(f"{'=' * 72}")

    # Hypothesis: If Candidate B flips many signals to BUY that have low p_cont -> noise
    #             If Candidate B flips to BUY when p_cont is high -> information

    # Case 1: FLAT -> BUY under B
    flat_to_buy_entries = [e for e in match_entries if e["signal_current"] == 0 and e["signal_new"] == 1]
    flat_to_sell_entries = [e for e in match_entries if e["signal_current"] == 0 and e["signal_new"] == -1]

    # For BUY flips, high p_cont (>0.5) supports BUY direction -> information
    # For SELL flips, low p_cont (<0.5) supports SELL direction -> information
    if flat_to_buy_entries:
        fb_pcont = [e["p_cont"] for e in flat_to_buy_entries if e["p_cont"] is not None]
        fb_info = sum(1 for p in fb_pcont if p > 0.5)
        fb_noise = sum(1 for p in fb_pcont if p <= 0.5)
        fb_mean_p = mean(fb_pcont) if fb_pcont else 0
        print(f"\n  FLAT->BUY transitions: {len(flat_to_buy_entries)}")
        print(f"    Mean p_cont: {fb_mean_p:.4f}")
        print(f"    p_cont > 0.5 (supports BUY): {fb_info} ({100 * fb_info / max(len(fb_pcont), 1):.1f}%) -> INFORMATION")
        print(f"    p_cont <= 0.5 (contradicts BUY): {fb_noise} ({100 * fb_noise / max(len(fb_pcont), 1):.1f}%) -> NOISE")

    if flat_to_sell_entries:
        fs_pcont = [e["p_cont"] for e in flat_to_sell_entries if e["p_cont"] is not None]
        fs_info = sum(1 for p in fs_pcont if p < 0.5)
        fs_noise = sum(1 for p in fs_pcont if p >= 0.5)
        fs_mean_p = mean(fs_pcont) if fs_pcont else 0
        print(f"\n  FLAT->SELL transitions: {len(flat_to_sell_entries)}")
        print(f"    Mean p_cont: {fs_mean_p:.4f}")
        print(f"    p_cont < 0.5 (supports SELL): {fs_info} ({100 * fs_info / max(len(fs_pcont), 1):.1f}%) -> INFORMATION")
        print(f"    p_cont >= 0.5 (contradicts SELL): {fs_noise} ({100 * fs_noise / max(len(fs_pcont), 1):.1f}%) -> NOISE")

    # Case 2: Directional -> FLAT under B (signal suppression)
    dir_to_flat_entries = [e for e in match_entries if e["signal_current"] != 0 and e["signal_new"] == 0]
    if dir_to_flat_entries:
        df_pcont = [e["p_cont"] for e in dir_to_flat_entries if e["p_cont"] is not None]
        df_mean_p = mean(df_pcont) if df_pcont else 0
        # For each, was the original signal supported by p_cont?
        df_good_suppress = 0  # suppressed signal that was NOT supported by p_cont -> good suppress
        df_bad_suppress = 0   # suppressed signal that WAS supported by p_cont -> bad suppress
        for e in dir_to_flat_entries:
            if e["p_cont"] is None:
                continue
            if e["signal_current"] == 1 and e["p_cont"] <= 0.5:
                df_good_suppress += 1  # BUY with low p_cont -> good to suppress
            elif e["signal_current"] == -1 and e["p_cont"] >= 0.5:
                df_good_suppress += 1  # SELL with high p_cont -> good to suppress
            else:
                df_bad_suppress += 1   # signal supported by p_cont but suppressed -> bad

        print(f"\n  Directional->FLAT (suppressed): {len(dir_to_flat_entries)}")
        print(f"    Mean p_cont: {df_mean_p:.4f}")
        print(f"    Good suppression (p_cont contradicts signal): {df_good_suppress}")
        print(f"    Bad suppression (p_cont supported signal): {df_bad_suppress}")

    # Case 3: Flipped signals (BUY <-> SELL)
    flipped_entries = [e for e in match_entries if
                       e["signal_current"] != 0 and e["signal_new"] != 0
                       and e["signal_current"] != e["signal_new"]]
    if flipped_entries:
        flip_pcont = [e["p_cont"] for e in flipped_entries if e["p_cont"] is not None]
        flip_mean_p = mean(flip_pcont) if flip_pcont else 0
        # BUY->SELL: new SELL is supported if p_cont < 0.5
        # SELL->BUY: new BUY is supported if p_cont > 0.5
        flip_info = 0
        flip_noise = 0
        for e in flipped_entries:
            if e["p_cont"] is None:
                continue
            if e["signal_current"] == 1 and e["signal_new"] == -1:
                # Was BUY, now SELL: good if p_cont < 0.5
                if e["p_cont"] < 0.5:
                    flip_info += 1
                else:
                    flip_noise += 1
            elif e["signal_current"] == -1 and e["signal_new"] == 1:
                # Was SELL, now BUY: good if p_cont > 0.5
                if e["p_cont"] > 0.5:
                    flip_info += 1
                else:
                    flip_noise += 1

        print(f"\n  Flipped signals (BUY<->SELL): {len(flipped_entries)}")
        print(f"    Mean p_cont: {flip_mean_p:.4f}")
        print(f"    Flip improves alignment with p_cont: {flip_info}")
        print(f"    Flip worsens alignment with p_cont: {flip_noise}")

    # ===========================================================================
    # SECTION 7: Entropy z-score distribution
    # ===========================================================================
    print(f"\n{'=' * 72}")
    print("SECTION 7: ENTROPY Z-SCORE DIAGNOSTICS")
    print(f"{'=' * 72}")

    z_scores = [e["entropy_z"] for e in all_entries]
    adj_vals = [e["adjusted_entropy"] for e in all_entries]

    print(f"\n  Entropy z-score:")
    print(f"    Mean: {mean(z_scores):.4f}")
    print(f"    Std:  {stdev(z_scores):.4f}")
    raw_ent = [e["entropy"] for e in all_entries]
    print(f"  Raw entropy:")
    print(f"    Mean: {mean(raw_ent):.4f}")
    print(f"    Std:  {stdev(raw_ent):.4f}")

    print(f"\n  Adjusted entropy (z/4 + 0.5):")
    print(f"    Mean: {mean(adj_vals):.4f}")
    print(f"    Range: [{min(adj_vals):.4f}, {max(adj_vals):.4f}]")

    # ===========================================================================
    # CONCLUSIONS
    # ===========================================================================
    print(f"\n{'=' * 72}")
    print("CONCLUSIONS")
    print(f"{'=' * 72}")

    # Bias reduction check
    cur_sell_bias = cur_dist["SELL"] - cur_dist["BUY"]
    new_sell_bias = new_dist["SELL"] - new_dist["BUY"]
    print(f"\n  SELL bias (SELL% - BUY%): Current={cur_sell_bias:+.2f}% -> B={new_sell_bias:+.2f}%")
    if abs(new_sell_bias) < abs(cur_sell_bias):
        print(f"  >> Candidate B REDUCES SELL bias by {abs(cur_sell_bias) - abs(new_sell_bias):.2f}%")
    else:
        print(f"  >> Candidate B DOES NOT reduce SELL bias (Δ={new_sell_bias - cur_sell_bias:+.2f}%)")

    # Information quality
    total_flat_to_dir = flat_to_buy + flat_to_sell
    if total_flat_to_dir > 0:
        info_transitions = 0
        if flat_to_buy_entries:
            info_transitions += sum(1 for e in flat_to_buy_entries if e["p_cont"] is not None and e["p_cont"] > 0.5)
        if flat_to_sell_entries:
            info_transitions += sum(1 for e in flat_to_sell_entries if e["p_cont"] is not None and e["p_cont"] < 0.5)
        info_ratio = info_transitions / max(total_flat_to_dir, 1) * 100
        print(f"\n  FLAT->directional transitions: {total_flat_to_dir}")
        print(f"  Aligned with p_cont: {info_ratio:.1f}%")
        if info_ratio > 60:
            print(f"  >> Candidate B new signals are MOSTLY information (>60% p_cont aligned)")
        elif info_ratio > 50:
            print(f"  >> Candidate B new signals are SLIGHTLY above random (50-60%)")
        else:
            print(f"  >> Candidate B new signals are MOSTLY noise (<=50% p_cont aligned)")

    # Net directional signal change
    net_dir_change = (flat_to_dir - dir_to_flat)
    print(f"\n  Net directional change: {net_dir_change:+.0f} entries (FLAT->dir minus dir->FLAT)")
    if net_dir_change > 0:
        print(f"  >> Candidate B is MORE active (generates more directional signals)")
    elif net_dir_change < 0:
        print(f"  >> Candidate B is LESS active (more restrictive than current)")
    else:
        print(f"  >> Candidate B has similar activity level")

    # Overall verdict
    info_noise_ratio = 0
    if total_flat_to_dir > 0:
        info_noise_ratio = info_transitions / max(total_flat_to_dir - info_transitions, 1)
    if abs(new_sell_bias) < abs(cur_sell_bias) and info_ratio > 55 if total_flat_to_dir > 0 else False:
        print(f"\n  >>> VERDICT: Candidate B is BENEFICIAL — reduces bias AND adds information")
    elif abs(new_sell_bias) < abs(cur_sell_bias):
        print(f"\n  >>> VERDICT: Candidate B reduces bias but at information quality cost — MIXED")
    elif info_ratio > 55 if total_flat_to_dir > 0 else False:
        print(f"\n  >>> VERDICT: Candidate B adds information but does not fix bias — PARTIAL")
    else:
        print(f"\n  >>> VERDICT: Candidate B does NOT improve either bias or information — REJECT")

    print(f"\n  Source log: {LOG_PATH} ({log_size_mb:.1f} MB)")
    print(f"  Shadow entries analyzed: {len(all_entries)}")
    print(f"  Matched with OSS: {len(match_entries)}")


if __name__ == "__main__":
    main()
