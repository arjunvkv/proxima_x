"""
sensor_independence.py — OFFLINE correlation analysis of direction sensors.

Reads proxima_demo.log, extracts OSS SURFACE / PROD_SIGNAL_BREAKDOWN /
TPI_SOURCE / SHADOW_RAW entries, aligns them into cycles, and computes:

1. Pairwise Pearson/Spearman/%agreement/%disagreement for all sensor pairs
2. Same analysis for simulated "repaired" sensors (Shadow z-score, Energy proxy)
3. Text-based correlation matrix heatmap
4. Per-regime correlation breakdown

Usage: python research/direction_simulation/sensor_independence.py
"""

import re
import os
import sys
import math
from datetime import datetime
from collections import defaultdict
from statistics import mean, stdev

LOG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "proxima_demo.log")
)

MATCH_WINDOW = 5.0  # seconds — entries within this window + same symbol = same cycle

REPAIR_ZSCORE_WINDOW = 30  # rolling window for shadow z-score normalization
ENERGY_HISTORY_WINDOW = 20  # rolling window for ecdf change baseline


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_timestamp(line: str):
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
    return m.group(1) if m else ""


def ts_to_seconds(ts: str) -> float:
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")
        ref = datetime(2026, 1, 1)
        return (dt - ref).total_seconds()
    except (ValueError, OSError):
        return 0.0


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
        "signal": int(m.group(16)),
    }


def parse_prod_signal_breakdown(line: str):
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
# Direction helpers
# ---------------------------------------------------------------------------

TPI_DIR_MAP = {"BUY": 1, "SELL": -1, "FLAT": 0, "LONG": 1, "SHORT": -1}


def tpi_to_direction(direction_str: str) -> int:
    return TPI_DIR_MAP.get(direction_str, 0)


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def rolling_zscore(values, window=REPAIR_ZSCORE_WINDOW):
    """Compute rolling z-score of a sequence."""
    result = [0.0] * len(values)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        if len(chunk) < 3:
            result[i] = 0.0
        else:
            mu = mean(chunk)
            sigma = stdev(chunk)
            result[i] = (values[i] - mu) / sigma if sigma > 1e-12 else 0.0
    return result


def directional_energy(ecdf_vals, window=ENERGY_HISTORY_WINDOW):
    """Compute |ecdf_change| - mean(|ecdf_change|) over rolling window."""
    if len(ecdf_vals) < 2:
        return [0.0] * len(ecdf_vals)
    changes = [abs(ecdf_vals[i] - ecdf_vals[i - 1]) for i in range(1, len(ecdf_vals))]
    changes = [0.0] + changes  # first entry has no prior

    result = [0.0] * len(ecdf_vals)
    for i in range(len(ecdf_vals)):
        start = max(0, i - window + 1)
        chunk = changes[start:i + 1]
        mu = mean(chunk)
        sigma = stdev(chunk) if len(chunk) > 1 else 0.0
        # Energy = excess over baseline (positive = more displacement than usual)
        diff = changes[i] - mu
        if sigma > 1e-12 and len(chunk) > 2:
            diff = diff / sigma  # z-score normalize
        result[i] = diff
    return result


# ---------------------------------------------------------------------------
# Correlation functions
# ---------------------------------------------------------------------------

def pearson_r(x, y):
    n = len(x)
    if n < 3:
        return 0.0
    mx = mean(x)
    my = mean(y)
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)) *
                    sum((y[i] - my) ** 2 for i in range(n)))
    return num / den if den > 1e-12 else 0.0


def spearman_r(x, y):
    n = len(x)
    if n < 3:
        return 0.0

    def rank(vals):
        sorted_pairs = sorted([(v, i) for i, v in enumerate(vals)])
        ranks = [0] * n
        for pos, (_, idx) in enumerate(sorted_pairs):
            ranks[idx] = pos + 1
        return ranks

    rx = rank(x)
    ry = rank(y)
    return pearson_r(rx, ry)


def agreement(x, y):
    """% same sign when both non-zero."""
    both = [(x[i], y[i]) for i in range(len(x)) if x[i] != 0 and y[i] != 0]
    if not both:
        return 0.0, 0.0, 0
    agree = sum(1 for a, b in both if (a > 0 and b > 0) or (a < 0 and b < 0))
    total = len(both)
    return 100.0 * agree / total, 100.0 * (total - agree) / total, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    tpi_entries = []

    line_count = 0
    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            e = parse_oss_surface(line)
            if e:
                ts = parse_timestamp(line)
                e["ts"] = ts
                e["ts_sec"] = ts_to_seconds(ts)
                oss_entries.append(e)
                continue
            p = parse_prod_signal_breakdown(line)
            if p:
                ts = parse_timestamp(line)
                p["ts"] = ts
                p["ts_sec"] = ts_to_seconds(ts)
                prod_entries.append(p)
                continue
            s = parse_shadow_raw(line)
            if s:
                ts = parse_timestamp(line)
                s["ts"] = ts
                s["ts_sec"] = ts_to_seconds(ts)
                shadow_entries.append(s)
                continue
            t = parse_tpi_source(line)
            if t:
                ts = parse_timestamp(line)
                t["ts"] = ts
                t["ts_sec"] = ts_to_seconds(ts)
                tpi_entries.append(t)

    print(f"  Lines read: {line_count:,}")
    print(f"  OSS SURFACE:          {len(oss_entries):,}")
    print(f"  PROD_SIGNAL_BREAKDOWN: {len(prod_entries):,}")
    print(f"  SHADOW_RAW:           {len(shadow_entries):,}")
    print(f"  TPI_SOURCE:           {len(tpi_entries):,}")

    if not prod_entries:
        print("ERROR: No PROD_SIGNAL_BREAKDOWN entries found.")
        return
    if not oss_entries:
        print("ERROR: No OSS SURFACE entries found.")
        return

    # ---- Match entries into cycles ----
    # For each PROD_SIGNAL_BREAKDOWN entry, find the closest OSS SURFACE and TPI_SOURCE
    # within MATCH_WINDOW seconds for the same symbol

    # Index OSS by (symbol, approx_time)
    oss_by_sym = defaultdict(list)
    for e in oss_entries:
        oss_by_sym[e["symbol"]].append(e)

    tpi_by_sym = defaultdict(list)
    for t in tpi_entries:
        tpi_by_sym[t["symbol"]].append(t)

    shadow_by_sym = defaultdict(list)
    for s in shadow_entries:
        shadow_by_sym[s["symbol"]].append(s)

    def find_closest(entries, ts_sec, window=MATCH_WINDOW):
        """Find the closest entry within window seconds."""
        best = None
        best_delta = window
        for entry in entries:
            delta = abs(entry["ts_sec"] - ts_sec)
            if delta < best_delta:
                best_delta = delta
                best = entry
        return best

    cycles = []
    for p in prod_entries:
        sym = p["symbol"]
        ts = p["ts_sec"]

        oss = find_closest(oss_by_sym.get(sym, []), ts)
        tpi = find_closest(tpi_by_sym.get(sym, []), ts)
        shadow = find_closest(shadow_by_sym.get(sym, []), ts)

        if oss is None:
            continue

        cycles.append({
            "symbol": sym,
            "ts": p["ts"],
            "regime": p["regime"],
            "oss_sig": p["oss"],
            "oss_conf": p["oss_conf"],
            "oss_ev": p["oss_ev"],
            "ev_sig": p["ev_sig"],
            "shadow_sig": p["shadow"],
            "shadow_conf": p["shadow_conf"],
            "final": p["final"],
            "exec_drift": oss["exec_drift"],
            "p_cont": oss["p_cont"],
            "ecdf": oss["ecdf"],
            "oss_signal": oss["signal"],
            "tpi_direction": tpi_to_direction(tpi["direction"]) if tpi else None,
            "tpi_conf": tpi["conf"] if tpi else None,
            "shadow_score": shadow["score"] if shadow else None,
            "shadow_entropy": shadow["entropy"] if shadow else None,
            "shadow_raw": shadow["raw"] if shadow else None,
            "shadow_final": shadow["final"] if shadow else None,
        })

    print(f"\n  Matched cycles: {len(cycles):,}")
    if not cycles:
        print("ERROR: No matched cycles.")
        return

    # ---- Build sensor arrays (filtered to non-None pairs) ----
    def array_for(getter, non_zero_check=False):
        vals = []
        for c in cycles:
            v = getter(c)
            if v is None:
                continue
            if non_zero_check and v == 0:
                vals.append(v)
            else:
                vals.append(v)
        return vals

    # Current sensors
    oss_sig_arr = array_for(lambda c: c["oss_sig"])
    shadow_sig_arr = array_for(lambda c: c["shadow_sig"])
    exec_drift_arr = array_for(lambda c: c["exec_drift"])
    p_cont_arr = array_for(lambda c: c["p_cont"])
    shadow_conf_arr = array_for(lambda c: c["shadow_conf"])
    tpi_dir_arr = array_for(lambda c: c["tpi_direction"])
    tpi_conf_arr = array_for(lambda c: c["tpi_conf"])

    # ---- Repaired sensors ----
    print("\n  Computing repaired sensors...")

    # Shadow z-score: rolling z-score of shadow_score
    # Group by symbol first
    cycles_by_sym = defaultdict(list)
    for c in cycles:
        cycles_by_sym[c["symbol"]].append(c)

    # Sort each symbol's cycles by timestamp
    for sym in cycles_by_sym:
        cycles_by_sym[sym].sort(key=lambda x: x["ts"])

    shadow_z_scores = []
    energy_proxies = []
    for sym in sorted(cycles_by_sym.keys()):
        cc = cycles_by_sym[sym]
        scores = [c["shadow_score"] if c["shadow_score"] is not None else 0.0 for c in cc]
        ecdf_vals = [c["ecdf"] for c in cc]
        zs = rolling_zscore(scores, REPAIR_ZSCORE_WINDOW)
        en = directional_energy(ecdf_vals, ENERGY_HISTORY_WINDOW)
        shadow_z_scores.extend(zs)
        energy_proxies.extend(en)

    # Ensure same length as cycles
    if len(shadow_z_scores) != len(cycles):
        print(f"  WARNING: shadow_z_scores length {len(shadow_z_scores)} != cycles {len(cycles)}")
    if len(energy_proxies) != len(cycles):
        print(f"  WARNING: energy_proxies length {len(energy_proxies)} != cycles {len(cycles)}")

    # Also compute directional energy for ecdf change (non-normalized)
    ecdf_changes = [0.0]
    for i in range(1, len(cycles)):
        ecdf_changes.append(cycles[i]["ecdf"] - cycles[i - 1]["ecdf"])
    ecdf_change_abs = [abs(v) for v in ecdf_changes]

    # ---- Correlation analysis ----
    sensors_current = {
        "OSS": ("oss_sig", lambda c: c["oss_sig"]),
        "Shadow": ("shadow_sig", lambda c: c["shadow_sig"]),
        "TPI": ("tpi_direction", lambda c: c["tpi_direction"]),
        "exec_drift": ("exec_drift", lambda c: c["exec_drift"]),
    }

    # For repaired sensors, we add shadow_z and energy
    # Build the repaired arrays
    repaired_arr = {
        "ShadowZ": shadow_z_scores,
        "Energy": energy_proxies,
    }

    # Current sensor arrays (aligned across all cycles)
    sensor_arr = {}
    for name, (_, getter) in sensors_current.items():
        arr = [getter(c) for c in cycles]
        # Replace None with 0 for correlation calculations
        arr = [0.0 if v is None else v for v in arr]
        sensor_arr[name] = arr

    # Add repaired
    for name, arr in repaired_arr.items():
        sensor_arr[name] = arr

    # Also add Shadow score (raw continuous) as a baseline
    sensor_arr["ShadowScore"] = [c["shadow_score"] if c["shadow_score"] is not None else 0.0 for c in cycles]

    # ---- 1. Pairwise correlations (ALL sensors) ----
    all_sensors = ["OSS", "Shadow", "ShadowScore", "TPI", "exec_drift", "ShadowZ", "Energy"]

    print(f"\n{'='*72}")
    print(f"SENSOR INDEPENDENCE ANALYSIS")
    print(f"  Total matched cycles: {len(cycles):,}")
    print(f"  Match window: {MATCH_WINDOW}s")
    print(f"  Shadow z-score window: {REPAIR_ZSCORE_WINDOW}")
    print(f"  Energy window: {ENERGY_HISTORY_WINDOW}")
    print(f"{'='*72}")

    print(f"\n{'='*72}")
    print(f"CORRELATION MATRIX (Pearson r)")
    print(f"{'='*72}")
    print(f"{'':>14s}", end="")
    for name in all_sensors:
        print(f"{name:>12s}", end="")
    print()
    for name1 in all_sensors:
        print(f"{name1:>14s}", end="")
        for name2 in all_sensors:
            a1 = sensor_arr[name1]
            a2 = sensor_arr[name2]
            r = pearson_r(a1, a2)
            print(f"{r:>12.4f}", end="")
        print()

    print(f"\n{'='*72}")
    print(f"CORRELATION MATRIX (Spearman rank)")
    print(f"{'='*72}")
    print(f"{'':>14s}", end="")
    for name in all_sensors:
        print(f"{name:>12s}", end="")
    print()
    for name1 in all_sensors:
        print(f"{name1:>14s}", end="")
        for name2 in all_sensors:
            a1 = sensor_arr[name1]
            a2 = sensor_arr[name2]
            r = spearman_r(a1, a2)
            print(f"{r:>12.4f}", end="")
        print()

    # ---- 2. Pairwise directional agreement ----
    print(f"\n{'='*72}")
    print(f"DIRECTIONAL AGREEMENT (same sign, both non-zero)")
    print(f"{'='*72}")
    dir_pairs = [
        ("OSS", "Shadow"),
        ("OSS", "TPI"),
        ("Shadow", "TPI"),
        ("OSS", "exec_drift"),
        ("exec_drift", "TPI"),
    ]
    for n1, n2 in dir_pairs:
        a1 = sensor_arr[n1]
        a2 = sensor_arr[n2]
        ag, disag, n = agreement(a1, a2)
        print(f"  {n1:12s} vs {n2:12s}: agree={ag:5.1f}% disagree={disag:5.1f}% (n={n})")

    # ---- 3. p_cont vs shadow confidence ----
    print(f"\n{'='*72}")
    print(f"CROSS-DOMAIN CORRELATIONS")
    print(f"{'='*72}")
    r_p = pearson_r(p_cont_arr, shadow_conf_arr)
    r_s = spearman_r(p_cont_arr, shadow_conf_arr)
    print(f"  p_cont vs shadow_conf:  Pearson={r_p:.4f}  Spearman={r_s:.4f}")

    # OSS confidence vs shadow confidence
    oss_conf_arr = [c["oss_conf"] for c in cycles]
    r_p = pearson_r(oss_conf_arr, shadow_conf_arr)
    r_s = spearman_r(oss_conf_arr, shadow_conf_arr)
    print(f"  oss_conf vs shadow_conf: Pearson={r_p:.4f}  Spearman={r_s:.4f}")

    # OSS EV vs shadow score
    oss_ev_arr = [c["oss_ev"] for c in cycles]
    shadow_score_arr = [c["shadow_score"] if c["shadow_score"] is not None else 0.0 for c in cycles]
    r_p = pearson_r(oss_ev_arr, shadow_score_arr)
    r_s = spearman_r(oss_ev_arr, shadow_score_arr)
    print(f"  oss_ev vs shadow_score:  Pearson={r_p:.4f}  Spearman={r_s:.4f}")

    # ---- 4. Repaired sensor analysis ----
    print(f"\n{'='*72}")
    print(f"REPAIRED SENSOR ANALYSIS (ShadowZ and Energy)")
    print(f"{'='*72}")

    repaired_pairs = [
        ("OSS", "ShadowZ"),
        ("Shadow", "ShadowZ"),
        ("ShadowScore", "ShadowZ"),
        ("TPI", "ShadowZ"),
        ("exec_drift", "ShadowZ"),
        ("OSS", "Energy"),
        ("Shadow", "Energy"),
        ("ShadowScore", "Energy"),
        ("TPI", "Energy"),
        ("exec_drift", "Energy"),
        ("ShadowZ", "Energy"),
    ]
    for n1, n2 in repaired_pairs:
        a1 = sensor_arr[n1]
        a2 = sensor_arr[n2]
        r_p = pearson_r(a1, a2)
        r_s = spearman_r(a1, a2)
        ag, disag, n = agreement(
            [1 if v > 0 else (-1 if v < 0 else 0) for v in a1],
            [1 if v > 0 else (-1 if v < 0 else 0) for v in a2],
        )
        print(f"  {n1:12s} vs {n2:12s}:")
        print(f"    Pearson={r_p:+.4f}  Spearman={r_s:+.4f}  agree={ag:5.1f}%  disagree={disag:5.1f}%  (n={n})")

    # ---- 5. Independence verdict ----
    print(f"\n{'='*72}")
    print(f"VERDICT: ARE SENSORS STILL INDEPENDENT AFTER REPAIRS?")
    print(f"{'='*72}")

    # Check mean absolute correlation within current sensors (excl self)
    corr_current = []
    curr_list = ["OSS", "Shadow", "TPI", "exec_drift"]
    for i, n1 in enumerate(curr_list):
        for n2 in curr_list[i + 1:]:
            corr_current.append(abs(pearson_r(sensor_arr[n1], sensor_arr[n2])))
    mean_corr_current = mean(corr_current)
    print(f"  Current sensors mean |Pearson|: {mean_corr_current:.4f}")

    # Check mean absolute correlation of repaired sensors with others
    corr_repaired = []
    for n1 in ["ShadowZ", "Energy"]:
        for n2 in ["OSS", "Shadow", "TPI", "exec_drift"]:
            corr_repaired.append(abs(pearson_r(sensor_arr[n1], sensor_arr[n2])))
    mean_corr_repaired = mean(corr_repaired)
    print(f"  Repaired sensors mean |Pearson| with others: {mean_corr_repaired:.4f}")

    # Check ShadowZ vs Energy
    r_rep_pair = abs(pearson_r(sensor_arr["ShadowZ"], sensor_arr["Energy"]))
    print(f"  ShadowZ vs Energy |Pearson|: {r_rep_pair:.4f}")

    if mean_corr_current < 0.20 and mean_corr_repaired < 0.20:
        print(f"\n  >>> SENSORS REMAIN INDEPENDENT (mean |r| < 0.20)")
        print(f"      Repairs do not introduce convergence.")
    elif mean_corr_repaired < 0.20 < mean_corr_current:
        print(f"\n  >>> REPAIRS REDUCE CORRELATION (repairs are MORE independent)")
    elif mean_corr_repaired > 0.35:
        print(f"\n  >>> WARNING: Repaired sensors show MODERATE convergence")
        print(f"      (mean |r| = {mean_corr_repaired:.3f})")
        if r_rep_pair > 0.35:
            print(f"      ShadowZ and Energy also correlate — may measure same latent factor")
    else:
        print(f"\n  >>> BORDERLINE — mean |r| = {mean_corr_repaired:.3f}")
        print(f"      Further investigation recommended.")

    # Check ShadowZ vs ShadowScore (should be decorrelated if z-score worked)
    r_sz_ss = abs(pearson_r(sensor_arr["ShadowZ"], sensor_arr["ShadowScore"]))
    r_sz_sh = abs(pearson_r(sensor_arr["ShadowZ"], sensor_arr["Shadow"]))
    print(f"\n  ShadowZ vs ShadowScore (raw continuous): |r|={r_sz_ss:.4f}")
    print(f"  ShadowZ vs Shadow (discrete):           |r|={r_sz_sh:.4f}")
    if r_sz_ss < 0.3:
        print(f"    -> z-score successfully decorrelated Shadow from itself")
    else:
        print(f"    -> z-score still correlates with raw Shadow — check window size")

    # ---- 6. Per-regime analysis ----
    print(f"\n{'='*72}")
    print(f"PER-REGIME CORRELATION ANALYSIS")
    print(f"{'='*72}")

    regimes = set(c["regime"] for c in cycles)
    regime_data = defaultdict(lambda: {name: [] for name in all_sensors})

    for i, c in enumerate(cycles):
        r = c["regime"]
        for name in all_sensors:
            regime_data[r][name].append(sensor_arr[name][i])

    regime_order = ["TRENDING", "TRANSITION", "NORMAL", "LOCKED", "ACTIVE_INSTABILITY",
                    "COMPRESSED_CHAOS", "CHAOTIC"]

    print(f"\n  Pearson r by regime (key pairs):")
    regime_pairs = [("OSS", "Shadow"), ("OSS", "TPI"), ("Shadow", "TPI"),
                    ("ShadowScore", "ShadowZ"), ("ShadowZ", "Energy"),
                    ("OSS", "exec_drift")]
    header = f"{'Regime':>22s}"
    for n1, n2 in regime_pairs:
        header += f" {n1[:4]}/{n2[:4]:>8s}"
    print(f"  {header}")
    print(f"  {'-' * (22 + 14 * len(regime_pairs))}")

    regime_stats = {}
    for regime in regime_order:
        if regime not in regime_data:
            continue
        rd = regime_data[regime]
        line = f"  {regime:>22s}"
        for n1, n2 in regime_pairs:
            a1 = rd[n1]
            a2 = rd[n2]
            r = abs(pearson_r(a1, a2))
            line += f" {r:>12.4f}"
        print(line)
        regime_stats[regime] = {
            "count": len(rd["OSS"]),
            "mean_r_oss_shadow": abs(pearson_r(rd["OSS"], rd["Shadow"])),
            "mean_r_oss_tpi": abs(pearson_r(rd["OSS"], rd["TPI"])),
            "mean_r_shadow_tpi": abs(pearson_r(rd["Shadow"], rd["TPI"])),
        }

    # Regime agreement analysis
    print(f"\n  Directional agreement by regime:")
    print(f"  {'Regime':>22s} {'n':>6s} {'OSS/Shad':>10s} {'OSS/TPI':>10s} {'Shad/TPI':>10s}")
    print(f"  {'-' * 60}")
    for regime in regime_order:
        if regime not in regime_data:
            continue
        rd = regime_data[regime]
        n = len(rd["OSS"])
        ag_os, _, _ = agreement(rd["OSS"], rd["Shadow"])
        ag_ot, _, _ = agreement(rd["OSS"], rd["TPI"])
        ag_st, _, _ = agreement(rd["Shadow"], rd["TPI"])
        print(f"  {regime:>22s} {n:>6d} {ag_os:>9.1f}% {ag_ot:>9.1f}% {ag_st:>9.1f}%")

    # ---- 7. TRENDING / CHAOTIC comparison ----
    print(f"\n{'='*72}")
    print(f"TRENDING vs CHAOTIC: DO SENSORS CONVERGE OR DIVERGE?")
    print(f"{'='*72}")

    trending_like = {"TRENDING", "TRANSITION"}
    chaotic_like = {"CHAOTIC", "COMPRESSED_CHAOS", "ACTIVE_INSTABILITY"}

    trending_data = {name: [] for name in all_sensors}
    chaotic_data = {name: [] for name in all_sensors}
    for i, c in enumerate(cycles):
        r = c["regime"]
        for name in all_sensors:
            if r in trending_like:
                trending_data[name].append(sensor_arr[name][i])
            elif r in chaotic_like:
                chaotic_data[name].append(sensor_arr[name][i])

    print(f"  Trending regimes: {trending_like}")
    print(f"  Chaotic regimes:  {chaotic_like}")

    def regime_corr_block(label, data):
        print(f"\n  --- {label} ---")
        print(f"  {'Pair':>20s}  {'Pearson':>8s}  {'Spearman':>8s}  {'Agree%':>7s}")
        print(f"  {'-' * 48}")
        for n1, n2 in [("OSS", "Shadow"), ("OSS", "TPI"), ("Shadow", "TPI"),
                        ("ShadowScore", "ShadowZ"), ("ShadowZ", "Energy")]:
            a1 = data[n1]
            a2 = data[n2]
            if len(a1) < 3 or len(a2) < 3:
                continue
            r_p = pearson_r(a1, a2)
            r_s = spearman_r(a1, a2)
            ag, _, _ = agreement(
                [1 if v > 0 else (-1 if v < 0 else 0) for v in a1],
                [1 if v > 0 else (-1 if v < 0 else 0) for v in a2],
            )
            print(f"  {n1:>10s} vs {n2:10s}  {r_p:>+8.4f}  {r_s:>+8.4f}  {ag:>6.1f}%")

    regime_corr_block("TRENDING / TRANSITION", trending_data)
    regime_corr_block("CHAOTIC / COMPRESSED_CHAOS / ACTIVE_INSTABILITY", chaotic_data)

    # Comparative query
    if len(trending_data["OSS"]) > 2 and len(chaotic_data["OSS"]) > 2:
        t_r = abs(pearson_r(trending_data["OSS"], trending_data["Shadow"]))
        c_r = abs(pearson_r(chaotic_data["OSS"], chaotic_data["Shadow"]))
        print(f"\n  OSS-Shadow |r| in trending: {t_r:.4f}  chaotic: {c_r:.4f}")
        if t_r > c_r + 0.1:
            print(f"  >> Sensors AGREE MORE in trending regimes (convergence)")
        elif c_r > t_r + 0.1:
            print(f"  >> Sensors AGREE MORE in chaotic regimes (unexpected)")
        else:
            print(f"  >> No significant regime-dependent convergence difference")

        # TPI dispersion
        t_tpi = abs(pearson_r(trending_data["TPI"], trending_data["Shadow"]))
        c_tpi = abs(pearson_r(chaotic_data["TPI"], chaotic_data["Shadow"]))
        print(f"  TPI-Shadow |r| in trending: {t_tpi:.4f}  chaotic: {c_tpi:.4f}")
        if t_tpi > c_tpi + 0.1:
            print(f"  >> TPI aligns more with Shadow in trending (flow matches conviction)")
        elif c_tpi > t_tpi + 0.1:
            print(f"  >> TPI aligns more in chaotic (microstructure dominates)")

    # ---- 8. Summary statistics ----
    print(f"\n{'='*72}")
    print(f"SENSOR STATISTICS OVERVIEW")
    print(f"{'='*72}")
    for name in all_sensors:
        arr = sensor_arr[name]
        n = len(arr)
        mu = mean(arr)
        nonzero = sum(1 for v in arr if v != 0)
        print(f"  {name:>14s}: n={n:>6d}  mean={mu:+.4f}  nonzero={nonzero:>6d} ({100*nonzero/n:5.1f}%)")

    # TPI flat rate
    tpi_flat = sum(1 for c in cycles if c["tpi_direction"] is not None and c["tpi_direction"] == 0)
    tpi_total = sum(1 for c in cycles if c["tpi_direction"] is not None)
    print(f"           TPI FLAT rate: {tpi_flat}/{tpi_total} ({100*tpi_flat/max(1,tpi_total):.1f}%)")

    # exec_drift zero rate
    ed_zero = sum(1 for c in cycles if c["exec_drift"] == 0)
    print(f"           exec_drift==0 rate: {ed_zero}/{len(cycles)} ({100*ed_zero/len(cycles):.1f}%)")

    print(f"\n{'='*72}")
    print(f"CONCLUSION")
    print(f"{'='*72}")

    # Final verdict
    os_d = agreement(sensor_arr["OSS"], sensor_arr["Shadow"])
    os_t = agreement(sensor_arr["OSS"], sensor_arr["TPI"])
    st_d = agreement(sensor_arr["Shadow"], sensor_arr["TPI"])

    print(f"  CURRENT SENSORS:")
    print(f"    OSS vs Shadow:     Pearson r={pearson_r(sensor_arr['OSS'], sensor_arr['Shadow']):+.4f}, "
          f"agree={os_d[0]:.1f}%")
    print(f"    OSS vs TPI:        Pearson r={pearson_r(sensor_arr['OSS'], sensor_arr['TPI']):+.4f}, "
          f"agree={os_t[0]:.1f}%")
    print(f"    Shadow vs TPI:     Pearson r={pearson_r(sensor_arr['Shadow'], sensor_arr['TPI']):+.4f}, "
          f"agree={st_d[0]:.1f}%")

    print(f"\n  REPAIRED SENSORS:")
    os_z = pearson_r(sensor_arr["OSS"], sensor_arr["ShadowZ"])
    sh_z = pearson_r(sensor_arr["Shadow"], sensor_arr["ShadowZ"])
    ss_z = pearson_r(sensor_arr["ShadowScore"], sensor_arr["ShadowZ"])
    en_os = pearson_r(sensor_arr["Energy"], sensor_arr["OSS"])
    en_sh = pearson_r(sensor_arr["Energy"], sensor_arr["Shadow"])
    en_tpi = pearson_r(sensor_arr["Energy"], sensor_arr["TPI"])

    print(f"    ShadowZ vs OSS:      r={os_z:+.4f}  (desired: near 0)")
    print(f"    ShadowZ vs Shadow:   r={sh_z:+.4f}  (desired: near 0 means independent)")
    print(f"    ShadowZ vs ShadowSc: r={ss_z:+.4f}  (desired: near 0 means decorrelated)")
    print(f"    Energy vs OSS:       r={en_os:+.4f}  (desired: near 0)")
    print(f"    Energy vs Shadow:    r={en_sh:+.4f}  (desired: near 0)")
    print(f"    Energy vs TPI:       r={en_tpi:+.4f}  (desired: near 0)")

    # Final independence test
    all_corrs = []
    for i, n1 in enumerate(all_sensors):
        for n2 in all_sensors[i + 1:]:
            all_corrs.append(abs(pearson_r(sensor_arr[n1], sensor_arr[n2])))
    mean_ac = mean(all_corrs)
    max_ac = max(all_corrs)
    print(f"\n  Mean absolute pairwise |r| across ALL {len(all_sensors)} sensors: {mean_ac:.4f}")
    print(f"  Max absolute pairwise |r|: {max_ac:.4f}")

    if max_ac < 0.20:
        print(f"\n  >>> ALL SENSORS ARE INDEPENDENT (max |r| < 0.20)")
        print(f"      Each sensor measures a distinct facet of market direction.")
    elif mean_ac < 0.15 and max_ac < 0.40:
        print(f"\n  >>> SENSORS MOSTLY INDEPENDENT (minor overlap)")
        print(f"      Some pairs show weak convergence but overall independence holds.")
    else:
        print(f"\n  >>> PARTIAL CONVERGENCE DETECTED")
        print(f"      Some sensor pairs may be measuring the same latent factor.")
        print(f"      Consider Dimensionality Reduction (PCA/ICA) on sensor space.")

    # Print regime convergence table
    print(f"\n  REGIME CONVERGENCE TABLE:")
    print(f"  {'Regime':>22s} {'n':>6s} {'mean|r|':>8s} {'max|r|':>8s}")
    print(f"  {'-' * 46}")
    for regime in regime_order:
        if regime not in regime_data:
            continue
        rd = regime_data[regime]
        n = len(rd["OSS"])
        c_list = []
        for i, n1 in enumerate(curr_list):
            for n2 in curr_list[i + 1:]:
                c_list.append(abs(pearson_r(rd[n1], rd[n2])))
        m = mean(c_list) if c_list else 0
        x = max(c_list) if c_list else 0
        print(f"  {regime:>22s} {n:>6d} {m:>8.4f} {x:>8.4f}")


if __name__ == "__main__":
    main()
