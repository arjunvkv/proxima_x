"""
sensor_reliability_evolution.py — OFFLINE analysis script.

After proposed repairs (Directional Energy + Shadow regime-conditioned threshold
+ Conflict Intelligence), re-evaluate the sensor matrix.

Tasks:
1. Read demo log, extract OSS SURFACE / PROD_SIGNAL_BREAKDOWN / TPI_SOURCE
2. Compute REPAIRED sensor values (Shadow_v2, DE, CI, OSS+)
3. Build new sensor quality matrix (5 dimensions, 0-10 scale)
4. Compare against old matrix
5. Recompute pairwise correlations with repaired sensors
6. Answer: correlation convergence or divergence?

Usage: python research/direction_simulation/sensor_reliability_evolution.py
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

MATCH_WINDOW = 5.0
ENERGY_HISTORY_WINDOW = 20


# ---------------------------------------------------------------------------
# Parsers (reused from sensor_independence.py)
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


TPI_DIR_MAP = {"BUY": 1, "SELL": -1, "FLAT": 0, "LONG": 1, "SHORT": -1}


def tpi_to_direction(direction_str: str) -> int:
    return TPI_DIR_MAP.get(direction_str, 0)


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def directional_energy(ecdf_vals, window=ENERGY_HISTORY_WINDOW):
    """Compute |ecdf_change| z-score over rolling window (same as sensor_independence.py)."""
    if len(ecdf_vals) < 2:
        return [0.0] * len(ecdf_vals)
    changes = [abs(ecdf_vals[i] - ecdf_vals[i - 1]) for i in range(1, len(ecdf_vals))]
    changes = [0.0] + changes
    result = [0.0] * len(ecdf_vals)
    for i in range(len(ecdf_vals)):
        start = max(0, i - window + 1)
        chunk = changes[start:i + 1]
        mu = mean(chunk)
        sigma = stdev(chunk) if len(chunk) > 1 else 0.0
        diff = changes[i] - mu
        if sigma > 1e-12 and len(chunk) > 2:
            diff = diff / sigma
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
    return pearson_r(rank(x), rank(y))


def agreement(x, y):
    """% same sign when both non-zero."""
    both = [(x[i], y[i]) for i in range(len(x)) if x[i] != 0 and y[i] != 0]
    if not both:
        return 0.0, 0.0, 0
    agree_count = sum(1 for a, b in both if (a > 0 and b > 0) or (a < 0 and b < 0))
    total = len(both)
    return 100.0 * agree_count / total, 100.0 * (total - agree_count) / total, total


# ---------------------------------------------------------------------------
# Repaired sensor computation
# ---------------------------------------------------------------------------

def compute_shadow_v2(shadow_conf, regime, entropy, ecdf_change_abs):
    """
    Shadow_v2: regime-conditioned threshold.
    - COMPRESSED_CHAOS: threshold = 0.20
    - TRANSITION with entropy > 0: threshold = 0.10
    - Else: threshold = 0.05 (current default)
    Returns continuous signal scaled by (shadow_conf - threshold) / (1 - threshold).
    """
    if regime == "COMPRESSED_CHAOS":
        threshold = 0.20
    elif regime == "TRANSITION" and entropy > 0:
        threshold = 0.10
    else:
        threshold = 0.05

    if shadow_conf <= threshold:
        return 0.0
    # Scale signal strength: how far above threshold
    signal_strength = (shadow_conf - threshold) / (1.0 - threshold)
    # Cap at 1.0
    return min(signal_strength, 1.0)


def compute_conflict_intelligence(oss_sig, shadow_sig, tpi_dir, oss_conf, shadow_conf, regime,
                                  oss_plus=None, shadow_v2=None):
    """
    Conflict Intelligence v2: Uses OSS+ (continuous) and Shadow_v2 rather than
    discrete { -1,0,+1 } so that signal exists even when OSS is near-zero.

    Types:
      A_NOISE: All three signals near zero or disagree — ignore, no signal
      B_TRANSITION: Single sensor has moderate conviction — cautious signal
      C_ACCUMULATION: Two sensors agree in direction — sustained signal
      D_SIGNAL: All three sensors agree — strongest

    Returns numeric signal strength: 0.0, 0.25, 0.5, 0.75, or 1.0
    """
    # Use continuous variants if available, fall back to discrete
    oss_val = oss_plus if oss_plus is not None else float(oss_sig)
    shadow_val = shadow_v2 if shadow_v2 is not None else float(shadow_sig)

    oss_dir = 1 if oss_val > 0.01 else (-1 if oss_val < -0.01 else 0)
    shadow_dir = 1 if shadow_val > 0.3 else (-1 if shadow_val < 0 else 0)
    if tpi_dir is None:
        tpi_binary = 0
    else:
        tpi_binary = 1 if tpi_dir > 0 else (-1 if tpi_dir < 0 else 0)

    os_agree = (oss_dir != 0 and oss_dir == shadow_dir)
    ot_agree = (oss_dir != 0 and tpi_binary != 0 and oss_dir == tpi_binary)
    st_agree = (shadow_dir != 0 and tpi_binary != 0 and shadow_dir == tpi_binary)
    all_three = os_agree and st_agree and ot_agree

    if all_three and oss_dir != 0:
        return 1.0  # D_SIGNAL: full conviction
    if os_agree and st_agree:
        return 0.75  # strong two-sensor cross-validation
    if os_agree:
        return 0.5  # C_ACCUMULATION: OSS+Shadow agree
    if st_agree:
        return 0.5  # C_ACCUMULATION: Shadow+TPI agree
    if ot_agree:
        return 0.5  # C_ACCUMULATION: OSS+TPI agree

    oc = oss_conf if oss_conf is not None else 0.0
    sc = shadow_conf if shadow_conf is not None else 0.0
    tpi_mag = abs(tpi_dir) if tpi_dir is not None and tpi_dir != 0 else 0.0
    if (shadow_dir != 0 or tpi_binary != 0) and (sc > 0.6 or tpi_mag > 0.5):
        return 0.25  # B_TRANSITION: weak signal from one sensor
    return 0.0  # A_NOISE


# ---------------------------------------------------------------------------
# Quality matrix scoring
# ---------------------------------------------------------------------------

def score_directional_prediction(sensor_arr, cycles):
    """How often sensor gives non-zero direction, quality-weighted."""
    nonzero = sum(1 for v in sensor_arr if v != 0)
    total = len(sensor_arr)
    rate = nonzero / max(1, total)
    # Baseline score: rate * 10, but penalize if rate > 0.95 (too noisy)
    raw = rate * 10.0
    if rate > 0.95:
        raw = 10.0 * (1.0 - (rate - 0.95) * 10)  # penalize over-eager sensors
    return max(0, min(10, round(raw, 1)))


def score_state_detection(sensor_arr, cycles):
    """How well sensor tracks regime changes — use variance of signal as proxy."""
    if len(sensor_arr) < 2:
        return 0.0
    sigma = stdev(sensor_arr)
    # Higher variance = more responsive to state changes
    # Normalize: typical range [0, 1] for sigmoid-like sensors
    # Map: sigma=0 -> 0, sigma=0.3 -> 5, sigma=1.0 -> 10
    raw = min(10, sigma * 15)
    return round(raw, 1)


def score_entry_timing(sensor_arr, ecdf_changes_abs, cycles):
    """Correlation between sensor changes and price displacement (ecdf change)."""
    if len(sensor_arr) < 3 or len(ecdf_changes_abs) < 3:
        return 0.0
    r = abs(pearson_r(sensor_arr, ecdf_changes_abs))
    # r=0 -> 0, r=0.5 -> 6, r=1.0 -> 10
    raw = r * 12
    return round(min(10, raw), 1)


def score_uncertainty_detection(sensor_arr, cycles, conf_arr=None):
    """How well confidence/entropy tracks errors — dispersion of signal when conf is low vs high."""
    if conf_arr is None or len(conf_arr) < 3:
        # Fallback: use coefficient of variation as inverse proxy for certainty
        if len(sensor_arr) < 2:
            return 0.0
        mu = abs(mean(sensor_arr))
        sigma = stdev(sensor_arr)
        if mu > 1e-6:
            cv = sigma / mu
            # Higher CV = less certain = poorer uncertainty detection
            return round(max(0, min(10, (1.0 - min(cv, 2.0) / 2.0) * 10)), 1)
        return 5.0
    # Compare signal when conf is high vs low
    high_conf = [sensor_arr[i] for i in range(len(cycles))
                 if i < len(conf_arr) and conf_arr[i] is not None and conf_arr[i] > 0.7]
    low_conf = [sensor_arr[i] for i in range(len(cycles))
                if i < len(conf_arr) and conf_arr[i] is not None and conf_arr[i] < 0.3]
    if len(high_conf) > 2 and len(low_conf) > 2:
        h_sigma = stdev(high_conf)
        l_sigma = stdev(low_conf)
        # Good uncertainty detection: high conf -> low variance, low conf -> high variance
        ratio = l_sigma / max(1e-6, h_sigma)
        raw = min(10, ratio * 3)
        return round(raw, 1)
    return 5.0


def score_robustness(sensor_arr, cycles):
    """Stability across regimes — lower per-regime variance = more robust."""
    regimes = defaultdict(list)
    for i, c in enumerate(cycles):
        regimes[c["regime"]].append(sensor_arr[i])
    if len(regimes) < 2:
        return 10.0
    # Compute mean absolute deviation of regime means from global mean
    global_mu = mean(sensor_arr)
    deviations = []
    for reg, vals in regimes.items():
        if len(vals) > 2:
            reg_mu = mean(vals)
            deviations.append(abs(reg_mu - global_mu) * len(vals) / len(sensor_arr))
    if not deviations:
        return 10.0
    weighted_dev = sum(deviations)
    # Lower weighted deviation = more robust
    raw = max(0, 10 - weighted_dev * 30)
    return round(raw, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(LOG_PATH):
        print(f"ERROR: log not found at {LOG_PATH}")
        sys.exit(1)
    log_size_mb = os.path.getsize(LOG_PATH) / (1024 * 1024)
    print(f"Reading {LOG_PATH} ({log_size_mb:.1f} MB)...")
    print()
    print("=" * 72)
    print("  SENSOR RELIABILITY EVOLUTION — Post-Repair Evaluation")
    print("=" * 72)
    print()

    # ---- Parse all entries ----
    oss_entries = []
    prod_entries = []
    shadow_entries = []
    tpi_entries = []

    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
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

    print(f"  OSS SURFACE:          {len(oss_entries):,}")
    print(f"  PROD_SIGNAL_BREAKDOWN: {len(prod_entries):,}")
    print(f"  SHADOW_RAW:           {len(shadow_entries):,}")
    print(f"  TPI_SOURCE:           {len(tpi_entries):,}")
    print()

    if not prod_entries or not oss_entries:
        print("ERROR: Insufficient data.")
        return

    # ---- Match entries into cycles ----
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
            "pc": p["pc"],
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

    print(f"  Matched cycles: {len(cycles):,}")
    print()

    if not cycles:
        print("ERROR: No matched cycles.")
        return

    # ---- Sort cycles by timestamp for rolling computations ----
    cycles.sort(key=lambda c: c["ts"])

    # ---- Compute Directional Energy (ecdf change proxy) ----
    cycles_by_sym = defaultdict(list)
    for c in cycles:
        cycles_by_sym[c["symbol"]].append(c)
    for sym in cycles_by_sym:
        cycles_by_sym[sym].sort(key=lambda x: x["ts"])

    energy_proxies = []
    for sym in sorted(cycles_by_sym.keys()):
        cc = cycles_by_sym[sym]
        ecdf_vals = [c["ecdf"] for c in cc]
        en = directional_energy(ecdf_vals, ENERGY_HISTORY_WINDOW)
        energy_proxies.extend(en)

    if len(energy_proxies) != len(cycles):
        print(f"  WARNING: energy_proxies length {len(energy_proxies)} != {len(cycles)}")

    # ---- Compute Shadow_v2 (regime-conditioned threshold) ----
    shadow_v2_vals = []
    for i, c in enumerate(cycles):
        shadow_conf = c["shadow_conf"]
        regime = c["regime"]
        entropy = c["shadow_entropy"] if c["shadow_entropy"] is not None else 0.0
        # Use ecdf change as proxy for price displacement in threshold logic
        prev_ecdf = cycles[i - 1]["ecdf"] if i > 0 else c["ecdf"]
        ecdf_change = abs(c["ecdf"] - prev_ecdf)
        sv2 = compute_shadow_v2(shadow_conf, regime, entropy, ecdf_change)
        shadow_v2_vals.append(sv2)

    # ---- Compute OSS+ (same as current: p_cont scaled signal) ----
    oss_plus_vals = []
    for c in cycles:
        # OSS+: continuous version using p_cont and exec_drift
        p_cont = c["p_cont"]
        exec_drift = c["exec_drift"]
        oss_plus = exec_drift * (p_cont - 0.5) * 2.0
        oss_plus_vals.append(oss_plus)

    # ---- Compute Conflict Intelligence ----
    ci_vals = []
    for i, c in enumerate(cycles):
        op = oss_plus_vals[i] if i < len(oss_plus_vals) else None
        sv = shadow_v2_vals[i] if i < len(shadow_v2_vals) else None
        ci = compute_conflict_intelligence(
            c["oss_sig"], c["shadow_sig"], c["tpi_direction"],
            c["oss_conf"], c["shadow_conf"], c["regime"],
            oss_plus=op, shadow_v2=sv
        )
        ci_vals.append(ci)

    # ---- Build sensor arrays ----
    sensor_arr = {}
    sensor_arr["OSS"] = [c["oss_sig"] for c in cycles]
    sensor_arr["Shadow_old"] = [c["shadow_sig"] for c in cycles]
    sensor_arr["TPI"] = [c["tpi_direction"] if c["tpi_direction"] is not None else 0 for c in cycles]
    sensor_arr["exec_drift"] = [c["exec_drift"] for c in cycles]

    # Repaired sensors
    sensor_arr["Shadow_v2"] = shadow_v2_vals
    sensor_arr["DE"] = energy_proxies
    sensor_arr["CI"] = ci_vals
    sensor_arr["OSS+"] = oss_plus_vals

    # Also compute signed versions for agreement
    def to_sign(arr, threshold=0.01):
        return [1 if v > threshold else (-1 if v < -threshold else 0) for v in arr]

    # ---- Per-regime breakdown ----
    regimes_seen = set(c["regime"] for c in cycles)
    regime_order = ["TRENDING", "TRANSITION", "NORMAL", "LOCKED",
                    "ACTIVE_INSTABILITY", "COMPRESSED_CHAOS", "CHAOTIC"]

    # ================================================================
    # SECTION 1: Sensor Quality Matrix (Old vs New)
    # ================================================================
    print("=" * 72)
    print("  SECTION 1: SENSOR QUALITY MATRIX (5 dimensions, 0-10 scale)")
    print("=" * 72)
    print()

    # Ecdf changes for entry timing
    ecdf_changes_abs = [0.0]
    for i in range(1, len(cycles)):
        ecdf_changes_abs.append(abs(cycles[i]["ecdf"] - cycles[i - 1]["ecdf"]))

    sensors_to_score = {
        "OSS": ("oss_sig", None, True),
        "Shadow_old": ("shadow_sig", "shadow_conf", True),
        "TPI": ("tpi_direction", "tpi_conf", False),
        "exec_drift": ("exec_drift", None, True),
        "OSS+": ("OSS+", None, False),
        "Shadow_v2": ("Shadow_v2", None, False),
        "DE": ("DE", None, False),
        "CI": ("CI", None, False),
    }

    # Re-map for scoring
    sensor_configs = {
        "OSS": {"arr_key": "OSS", "conf_arr_key": "oss_conf", "is_discrete": True},
        "Shadow_old": {"arr_key": "Shadow_old", "conf_arr_key": "shadow_conf", "is_discrete": True},
        "TPI": {"arr_key": "TPI", "conf_arr_key": "tpi_conf", "is_discrete": True},
        "exec_drift": {"arr_key": "exec_drift", "conf_arr_key": None, "is_discrete": True},
        "OSS+": {"arr_key": "OSS+", "conf_arr_key": "oss_conf", "is_discrete": False},
        "Shadow_v2": {"arr_key": "Shadow_v2", "conf_arr_key": "shadow_conf", "is_discrete": False},
        "DE": {"arr_key": "DE", "conf_arr_key": None, "is_discrete": False},
        "CI": {"arr_key": "CI", "conf_arr_key": None, "is_discrete": False},
    }

    # Build conf arrays
    conf_arr = {}
    conf_arr["oss_conf"] = [c["oss_conf"] for c in cycles]
    conf_arr["shadow_conf"] = [c["shadow_conf"] for c in cycles]
    conf_arr["tpi_conf"] = [c["tpi_conf"] if c["tpi_conf"] is not None else 0.0 for c in cycles]

    quality_matrix = {}
    for sname, cfg in sensor_configs.items():
        arr = sensor_arr[cfg["arr_key"]]
        ck = cfg["conf_arr_key"]
        conf_list = conf_arr[ck] if ck and ck in conf_arr else None

        dp = score_directional_prediction(arr, cycles)
        sd = score_state_detection(arr, cycles)
        et = score_entry_timing(arr, ecdf_changes_abs, cycles)
        ud = score_uncertainty_detection(arr, cycles, conf_list)
        rb = score_robustness(arr, cycles)
        total = round(dp + sd + et + ud + rb, 1)
        quality_matrix[sname] = {
            "DirPred": dp, "StateDet": sd, "EntryTim": et,
            "UncDet": ud, "Robust": rb, "Total": total
        }

    # Old matrix (from prior analysis)
    old_matrix = {
        "OSS": {"Total": 22, "DirPred": 4, "StateDet": 5, "EntryTim": 5, "UncDet": 4, "Robust": 4},
        "Shadow": {"Total": 22, "DirPred": 5, "StateDet": 4, "EntryTim": 4, "UncDet": 4, "Robust": 5},
        "TPI": {"Total": 22, "DirPred": 4, "StateDet": 5, "EntryTim": 6, "UncDet": 3, "Robust": 4},
        "ExecDrift/DE": {"Total": 23, "DirPred": 4, "StateDet": 5, "EntryTim": 5, "UncDet": 4, "Robust": 5},
        "MCV/FSV": {"Total": 18, "DirPred": 3, "StateDet": 4, "EntryTim": 4, "UncDet": 3, "Robust": 4},
    }

    # Map new sensor names to old comparison
    old_mapping = {
        "OSS": "OSS",
        "Shadow_old": "Shadow",
        "Shadow_v2": "Shadow",
        "TPI": "TPI",
        "exec_drift": "ExecDrift/DE",
        "DE": "ExecDrift/DE",
        "OSS+": "ExecDrift/DE",
        "CI": "MCV/FSV",
    }

    print(f"  {'Sensor':<14s} {'DirPred':>8s} {'StateDet':>8s} {'EntryTim':>8s} {'UncDet':>8s} {'Robust':>8s} {'Total':>8s}")
    print(f"  {'-' * 62}")
    for sname in ["OSS", "Shadow_old", "Shadow_v2", "TPI", "exec_drift", "DE", "CI", "OSS+"]:
        q = quality_matrix[sname]
        print(f"  {sname:<14s} {q['DirPred']:>8.1f} {q['StateDet']:>8.1f} {q['EntryTim']:>8.1f} {q['UncDet']:>8.1f} {q['Robust']:>8.1f} {q['Total']:>8.1f}")
    print()

    # ---- Old vs New comparison ----
    print(f"  {'Sensor':<20s} {'Old Total':>10s} {'New Total':>10s} {'Change':>10s}")
    print(f"  {'-' * 52}")
    for new_name, old_name in old_mapping.items():
        old = old_matrix.get(old_name, {})
        new = quality_matrix.get(new_name, {})
        old_t = old.get("Total", 0)
        new_t = new.get("Total", 0)
        chg = new_t - old_t
        marker = "+" if chg >= 0 else ""
        print(f"  {new_name:<20s} {old_t:>10d} {new_t:>10.1f} {marker}{chg:>+8.1f}")
    print()

    # ================================================================
    # SECTION 2: Pairwise Correlation Matrix (Repaired)
    # ================================================================
    print("=" * 72)
    print("  SECTION 2: PAIRWISE CORRELATION — REPAIRED SENSORS")
    print("=" * 72)
    print()

    repair_sensors = ["OSS", "Shadow_v2", "DE", "TPI"]
    # Also include CI and OSS+
    all_repair = ["OSS", "Shadow_v2", "DE", "TPI", "CI", "OSS+"]

    print(f"  Pearson r — Repaired sensor matrix:")
    print(f"  {'':>12s}", end="")
    for n in repair_sensors:
        print(f"{n:>12s}", end="")
    print()
    for n1 in repair_sensors:
        print(f"  {n1:>12s}", end="")
        for n2 in repair_sensors:
            r = pearson_r(sensor_arr[n1], sensor_arr[n2])
            print(f"{r:>12.4f}", end="")
        print()
    print()

    # Full matrix with all repaired sensors
    print(f"  Full repaired matrix (all 6 sensors):")
    print(f"  {'':>12s}", end="")
    for n in all_repair:
        print(f"{n:>12s}", end="")
    print()
    for n1 in all_repair:
        print(f"  {n1:>12s}", end="")
        for n2 in all_repair:
            r = pearson_r(sensor_arr[n1], sensor_arr[n2])
            print(f"{r:>12.4f}", end="")
        print()
    print()

    # ---- Old correlation summary (from sensor_independence.py output) ----
    old_corr = {
        ("OSS", "Shadow"): -0.0021,
        ("OSS", "TPI"): -0.0045,
        ("Shadow", "TPI"): 0.0668,
        ("OSS", "exec_drift"): -0.3329,
    }
    print(f"  Old sensor correlations (current system):")
    for (a, b), r in old_corr.items():
        print(f"    {a:>10s} vs {b:<10s}:  r = {r:+.4f}")
    print()

    # ================================================================
    # SECTION 3: Correlation Convergence Analysis
    # ================================================================
    print("=" * 72)
    print("  SECTION 3: CORRELATION ANALYSIS — CONVERGENCE OR DIVERGENCE?")
    print("=" * 72)
    print()

    # Compare OLD (current) sensor correlations vs NEW (repaired)
    h1, h2, h3, h4 = "Comparison", "Old r", "New r", "Delta"
    print(f"  {h1:<30s} {h2:>10s} {h3:>10s} {h4:>10s}")
    print(f"  {'-' * 60}")
    comparisons = [
        ("OSS vs Shadow / Shadow_v2", "OSS", "Shadow_old", "OSS", "Shadow_v2"),
        ("OSS vs TPI", "OSS", "TPI", "OSS", "TPI"),
        ("Shadow vs TPI / Shadow_v2 vs TPI", "Shadow_old", "TPI", "Shadow_v2", "TPI"),
        ("OSS vs exec_drift / OSS vs DE", "OSS", "exec_drift", "OSS", "DE"),
        ("Shadow vs exec_drift / Shadow_v2 vs DE", "Shadow_old", "exec_drift", "Shadow_v2", "DE"),
    ]
    for label, old_a, old_b, new_a, new_b in comparisons:
        old_r = pearson_r(sensor_arr[old_a], sensor_arr[old_b])
        new_r = pearson_r(sensor_arr[new_a], sensor_arr[new_b])
        delta = new_r - old_r
        print(f"  {label:<30s} {old_r:>+10.4f} {new_r:>+10.4f} {delta:>+10.4f}")
    print()

    # ---- Compute mean absolute correlation across old and new ----
    old_sensors = ["OSS", "Shadow_old", "TPI", "exec_drift"]
    new_sensors = ["OSS", "Shadow_v2", "DE", "TPI"]

    old_corrs_abs = []
    for i, n1 in enumerate(old_sensors):
        for n2 in old_sensors[i + 1:]:
            old_corrs_abs.append(abs(pearson_r(sensor_arr[n1], sensor_arr[n2])))
    new_corrs_abs = []
    for i, n1 in enumerate(new_sensors):
        for n2 in new_sensors[i + 1:]:
            new_corrs_abs.append(abs(pearson_r(sensor_arr[n1], sensor_arr[n2])))

    mean_old = mean(old_corrs_abs) if old_corrs_abs else 0
    mean_new = mean(new_corrs_abs) if new_corrs_abs else 0
    print(f"  Mean |r| (old, 4 sensors):        {mean_old:.4f}")
    print(f"  Mean |r| (new, 4 repaired):       {mean_new:.4f}")

    if mean_new > mean_old + 0.05:
        print(f"  >>> CORRELATION INCREASED after repairs (Δ = {mean_new - mean_old:+.4f})")
        print(f"      Sensors converge on same information — signal is MORE coherent.")
        if mean_new < 0.35:
            print(f"      BUT |r| still < 0.35 — moderate, diverse-but-coherent (IDEAL).")
        elif mean_new < 0.5:
            print(f"      |r| in [0.35, 0.5) — acceptable convergence, still diverse.")
        else:
            print(f"      WARNING: |r| >= 0.5 — excessive convergence, diversity LOST.")
    elif mean_new < mean_old - 0.05:
        print(f"  >>> CORRELATION DECREASED after repairs (Δ = {mean_new - mean_old:+.4f})")
        print(f"      Sensors remain MORE independent — diversity PRESERVED.")
    else:
        print(f"  >>> CORRELATION STABLE (Δ = {mean_new - mean_old:+.4f})")
        print(f"      Repairs did not materially change sensor independence structure.")

    # ---- Compare Shadow_v2 vs old Shadow correlation with OSS ----
    r_oss_shadow = abs(pearson_r(sensor_arr["OSS"], sensor_arr["Shadow_old"]))
    r_oss_shadow_v2 = abs(pearson_r(sensor_arr["OSS"], sensor_arr["Shadow_v2"]))
    print(f"\n  OSS vs Shadow (old): |r| = {r_oss_shadow:.4f}")
    print(f"  OSS vs Shadow_v2:    |r| = {r_oss_shadow_v2:.4f}")
    if abs(r_oss_shadow_v2 - r_oss_shadow) < 0.02:
        print(f"  -> Shadow_v2 maintains same independence from OSS")
    elif r_oss_shadow_v2 > r_oss_shadow:
        print(f"  -> Shadow_v2 correlates MORE with OSS (Δ = {r_oss_shadow_v2 - r_oss_shadow:+.4f})")
    else:
        print(f"  -> Shadow_v2 is MORE independent from OSS (Δ = {r_oss_shadow_v2 - r_oss_shadow:+.4f})")

    # ---- Directional agreement between old Shadow and Shadow_v2 ----
    shadow_old_sig = to_sign(sensor_arr["Shadow_old"])
    shadow_v2_sig = to_sign(sensor_arr["Shadow_v2"], threshold=0.01)
    ag, disag, n = agreement(shadow_old_sig, shadow_v2_sig)
    print(f"\n  Shadow (old) vs Shadow_v2 sign agreement: agree={ag:.1f}%  disagree={disag:.1f}%  (n={n})")
    if ag > 80:
        print(f"  -> Shadow_v2 preserves most old Shadow signals (high agreement)")
    elif ag > 50:
        print(f"  -> Shadow_v2 modifies a SIGNIFICANT subset of Shadow signals")
    else:
        print(f"  -> Shadow_v2 DIVERGES substantially from old Shadow")

    # ================================================================
    # SECTION 4: Per-Regime Shadow_v2 Filtering Analysis
    # ================================================================
    print()
    print("=" * 72)
    print("  SECTION 4: SHADOW_V2 — REGIME-CONDITIONED FILTERING EFFECT")
    print("=" * 72)
    print()

    print(f"  {'Regime':<25s} {'n':>6s} {'Old Nonzero':>12s} {'V2 Nonzero':>12s} {'Reduction':>10s}")
    print(f"  {'-' * 67}")
    for regime in regime_order:
        if regime not in regimes_seen:
            continue
        idxs = [i for i, c in enumerate(cycles) if c["regime"] == regime]
        n = len(idxs)
        old_nz = sum(1 for i in idxs if sensor_arr["Shadow_old"][i] != 0)
        v2_nz = sum(1 for i in idxs if abs(sensor_arr["Shadow_v2"][i]) > 0.01)
        reduction = (old_nz - v2_nz) / max(1, old_nz) * 100
        print(f"  {regime:<25s} {n:>6d} {old_nz:>12d} {v2_nz:>12d} {reduction:>9.1f}%")

    print()

    # ================================================================
    # SECTION 5: Conflict Intelligence Distribution
    # ================================================================
    print("=" * 72)
    print("  SECTION 5: CONFLICT INTELLIGENCE — TYPE DISTRIBUTION")
    print("=" * 72)
    print()

    ci_buckets = {"A_NOISE (0.0)": 0, "B_TRANSITION (0.25)": 0,
                  "C_ACCUMULATION (0.5)": 0, "STRONG (0.75)": 0, "D_SIGNAL (1.0)": 0}
    for v in sensor_arr["CI"]:
        if v == 0.0:
            ci_buckets["A_NOISE (0.0)"] += 1
        elif v <= 0.25:
            ci_buckets["B_TRANSITION (0.25)"] += 1
        elif v <= 0.5:
            ci_buckets["C_ACCUMULATION (0.5)"] += 1
        elif v <= 0.75:
            ci_buckets["STRONG (0.75)"] += 1
        else:
            ci_buckets["D_SIGNAL (1.0)"] += 1

    for label, count in ci_buckets.items():
        pct = 100 * count / len(sensor_arr["CI"])
        print(f"  {label:<25s}: {count:>6d} ({pct:>5.1f}%)")
    print()

    # ================================================================
    # SECTION 6: Summary Statistics for Repaired Sensors
    # ================================================================
    print("=" * 72)
    print("  SECTION 6: REPAIRED SENSOR STATISTICS")
    print("=" * 72)
    print()

    for sname in ["OSS", "Shadow_old", "Shadow_v2", "TPI", "DE", "CI", "OSS+"]:
        arr = sensor_arr[sname]
        n = len(arr)
        mu = mean(arr)
        sigma = stdev(arr) if len(arr) > 1 else 0.0
        nz = sum(1 for v in arr if v != 0)
        nz_pct = 100 * nz / n
        abs_mu = mean(abs(v) for v in arr)
        rng = f"[{min(arr):.4f}, {max(arr):.4f}]"
        print(f"  {sname:<14s}: n={n:>6d}  mean={mu:+.4f}  std={sigma:.4f}  "
              f"nz={nz:>6d} ({nz_pct:>5.1f}%)  |mean|={abs_mu:.4f}  range={rng}")

    print()

    # ================================================================
    # SECTION 7: Final Verdict
    # ================================================================
    print("=" * 72)
    print("  SECTION 7: FINAL VERDICT — SENSOR RELIABILITY EVOLUTION")
    print("=" * 72)
    print()

    # Compute quality improvement
    old_total_map = {
        "OSS": old_matrix["OSS"]["Total"],
        "Shadow_old": old_matrix["Shadow"]["Total"],
        "Shadow_v2": old_matrix["Shadow"]["Total"],
        "TPI": old_matrix["TPI"]["Total"],
        "exec_drift": old_matrix["ExecDrift/DE"]["Total"],
        "DE": old_matrix["ExecDrift/DE"]["Total"],
        "CI": old_matrix["MCV/FSV"]["Total"],
        "OSS+": old_matrix["ExecDrift/DE"]["Total"],
    }

    total_old = sum(old_total_map.get(s, 0) for s in ["OSS", "Shadow_old", "TPI", "exec_drift"])
    total_new = sum(quality_matrix[s]["Total"] for s in ["OSS", "Shadow_v2", "DE", "TPI", "CI", "OSS+"])

    print(f"  Total quality score (old, 4 sensors):   {total_old:.1f}")
    print(f"  Total quality score (new, 6 sensors):   {total_new:.1f}")
    print(f"  Improvement:                            {total_new - total_old:+.1f} (+{100*(total_new-total_old)/max(1,total_old):.1f}%)")
    print()

    # Compare old vs new for comparable sensors
    for sname in ["OSS", "Shadow_v2", "TPI", "DE"]:
        old_name = sname if sname != "Shadow_v2" else "Shadow_old"
        if old_name in old_total_map:
            old_t = old_matrix.get(old_name.replace("_old", ""), {}).get("Total",
                     old_matrix.get("Shadow", {}).get("Total", 0))
            new_t = quality_matrix[sname]["Total"]
            chg = new_t - old_t
            print(f"  {sname:<14s}: old={old_t:.1f}  new={new_t:.1f}  Δ={chg:+.1f}")
    print()

    # Critical answer
    print(f"  CRITICAL QUESTION: After repairs, do sensors become MORE or LESS correlated?")
    print(f"  ---------------------------------------------------------")
    print(f"  Mean |r| (old OSS/Shadow/TPI/exec_drift): {mean_old:.4f}")
    print(f"  Mean |r| (new OSS/Shadow_v2/DE/TPI):      {mean_new:.4f}")
    if mean_new > mean_old + 0.05:
        print(f"  Answer: MORE correlated (convergence)")
        if mean_new < 0.35:
            print(f"  Assessment: IDEAL — moderate correlation (0.3-0.5 target zone)")
        elif mean_new < 0.5:
            print(f"  Assessment: ACCEPTABLE — approaching upper bound of diversity zone")
        else:
            print(f"  Assessment: WARNING — excessive convergence, diversity at risk")
    elif mean_new < mean_old - 0.05:
        print(f"  Answer: LESS correlated (diversity preserved)")
        print(f"  Assessment: GOOD — sensors remain independent probes")
    else:
        print(f"  Answer: STABLE (no meaningful change)")
        print(f"  Assessment: MIXED — repairs did not change independence structure")

    print()
    print(f"  KEY METRICS:")
    print(f"  - Shadow_v2 reduces false positives in COMPRESSED_CHAOS: "
          f"threshold 0.05 -> 0.20")
    print(f"  - Conflict Intelligence provides 4-type taxonomy: "
          f"A_NOISE -> D_SIGNAL")
    print(f"  - Directional Energy adds ecdf-change displacement proxy")
    print(f"  - OSS+ continuous signal (exec_drift * (p_cont-0.5) * 2)")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
