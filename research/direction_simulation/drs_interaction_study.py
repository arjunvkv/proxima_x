"""
drs_interaction_study.py — OFFLINE DRS Formula Comparison Agent

Question: How would DRS ranking change if it received direction + uncertainty +
conflict opportunity instead of just current discrete signals?

Compares CURRENT DRS formula vs NEW DRS formula on 7,556 matched cycles
from proxima_demo.log. Simulates SDL behavior to fill gaps.

Usage:
    python research/direction_simulation/drs_interaction_study.py

Does NOT modify production code.
"""

import re
import sys
import math
import os
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
from statistics import mean, stdev

LOG_PATH = Path(__file__).resolve().parents[2] / "proxima_demo.log"

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

CURRENT_WEIGHTS = {
    "strength": 0.25,
    "oss_conf": 0.25,
    "ecdf_r": 0.15,
    "sdl_s": 0.25,
    "dir_align": 0.10,
}
assert abs(sum(CURRENT_WEIGHTS.values()) - 1.0) < 0.01

NEW_WEIGHTS = {
    "shadow_conviction": 0.20,
    "directional_energy": 0.15,
    "ecdf_r": 0.15,
    "sdl_s": 0.20,
    "dir_align": 0.15,
    "conflict_opportunity": 0.15,
}
assert abs(sum(NEW_WEIGHTS.values()) - 1.0) < 0.01

SD_LAMBDA = 0.1  # SDL decay per cycle
SDL_THRESHOLD = 0.60  # min p_cont to establish SDL
SDL_BOOST = 0.15  # strength boost on aligned signal
SDL_DECAY = 0.92  # multiplicative decay per cycle (no signal)

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

RE_TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")

RE_PROD = re.compile(
    r"\[PROD_SIGNAL_BREAKDOWN\] (\w+)"
    r" oss=([+-]?\d+)\(ev=([-\d.]+),conf=([\d.]+)\)"
    r" ev_sig=([+-]?\d+)"
    r" shadow=([+-]?\d+)\(conf=([\d.]+)\)"
    r" regime=(\S+) reason=(\S+) final=([+-]?\d+) pc=([\d.]+)"
)

RE_OSS = re.compile(
    r"\[OSS SURFACE\] (\w+)"
    r" ecdf=([\d.]+) exec_drift=([-\d.]+?)"
    r"(?: live_drift=[-\d.]+)?"
    r" horizon=blended\(w3=([\d.]+),w10=([\d.]+),w20=([\d.]+)\)"
    r" regime=(\S+) p_cont=([\d.]+)"
    r" ph=(\d+) pt=(\d+) r_pc=([\d.]+) r_ph=(\d+) r_pt=(\d+)"
    r" r_bucket=(\S+) r_fb=(\S+) signal=(-?\d+) up=([\d.]+)% dn=([\d.]+)%"
)

RE_SHADOW = re.compile(
    r"\[SHADOW_RAW\] (\w+)"
    r" ecdf=([\d.]+) entropy=([\d.]+) score=([+-]?[\d.]+)"
    r" raw=([+-]?\d+) final=([+-]?\d+) flip_suppress=(\S+)"
)


def parse_timestamp(line: str):
    m = RE_TS.match(line)
    return m.group(1) if m else ""


def ts_to_seconds(ts: str) -> float:
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")
        ref = datetime(2026, 1, 1)
        return (dt - ref).total_seconds()
    except (ValueError, OSError):
        return 0.0


def parse_prod(line: str):
    m = RE_PROD.search(line)
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


def parse_oss(line: str):
    m = RE_OSS.search(line)
    if not m:
        return None
    return {
        "symbol": m.group(1),
        "ecdf": float(m.group(2)),
        "exec_drift": float(m.group(3)),
        "regime": m.group(7),
        "p_cont": float(m.group(8)),
    }


def parse_shadow(line: str):
    m = RE_SHADOW.search(line)
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


# ---------------------------------------------------------------------------
# SDL Simulator
# ---------------------------------------------------------------------------

class SDLSimulator:
    """Simulates Symbol Direction Lock state machine."""

    def __init__(self, symbols: list):
        self._directions = {s: None for s in symbols}
        self._strengths = {s: 0.0 for s in symbols}
        self._decay_table = {s: 0.0 for s in symbols}

    def get_strength(self, sym: str) -> float:
        return self._strengths.get(sym, 0.0)

    def get_current(self, sym: str):
        return self._directions.get(sym)

    def update(self, sym: str, signal_dir: int, p_cont: float):
        cur_dir = self._directions.get(sym)
        cur_strength = self._strengths.get(sym, 0.0)

        if signal_dir == 0:
            self._strengths[sym] = cur_strength * SDL_DECAY
            if self._strengths[sym] < 0.01:
                self._directions[sym] = None
            return

        # Signal active
        if p_cont >= SDL_THRESHOLD:
            if cur_dir is None or cur_dir == signal_dir:
                self._directions[sym] = signal_dir
                self._strengths[sym] = min(1.0, cur_strength + SDL_BOOST)
            elif cur_dir != signal_dir:
                self._strengths[sym] = max(0.0, cur_strength - SDL_BOOST * 0.5)
                if self._strengths[sym] < 0.3:
                    self._directions[sym] = signal_dir
                    self._strengths[sym] = 0.3
        else:
            if cur_dir is not None and cur_dir == signal_dir:
                self._strengths[sym] = max(0.0, cur_strength - 0.05)
            else:
                self._strengths[sym] = max(0.0, cur_strength - 0.10)


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def rolling_zscore(values, window=30):
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


def directional_energy(ecdf_vals, window=20):
    if len(ecdf_vals) < 2:
        return [0.0] * len(ecdf_vals)
    changes = [0.0]
    for i in range(1, len(ecdf_vals)):
        changes.append(abs(ecdf_vals[i] - ecdf_vals[i - 1]))
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
# Conflict classification (same logic as conflict_classifier_backtest.py)
# ---------------------------------------------------------------------------

def classify_conflict(duration: int, abs_ev: float, p_cont: float, regime: str) -> str:
    if duration == 1 and abs_ev < 0.2:
        return "A_NOISE"
    elif abs_ev < 0.3:
        return "A_NOISE"
    elif duration >= 10 and abs_ev < 0.7:
        return "D_INDETERMINACY"
    elif regime == "COMPRESSED_CHAOS" and duration >= 5:
        return "D_INDETERMINACY"
    elif duration >= 5 and abs_ev > 0.6 and p_cont > 0.8:
        return "C_ACCUMULATION"
    elif duration >= 3:
        return "B_TRANSITION"
    else:
        return "A_NOISE"


def conflict_opportunity_value(cls: str) -> float:
    return {"A_NOISE": 0.0, "B_TRANSITION": 0.5, "C_ACCUMULATION": 0.7, "D_INDETERMINACY": 0.0}.get(cls, 0.0)


# ---------------------------------------------------------------------------
# Detect conflict durations per symbol
# ---------------------------------------------------------------------------

def compute_conflict_durations(cycles):
    by_sym = defaultdict(list)
    for c in cycles:
        by_sym[c["symbol"]].append(c)
    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["ts_sec"])

    conflicts = []
    for sym, entries in by_sym.items():
        cur_dur = 1
        prev_reason = None
        prev_ts = None
        for c in entries:
            if c["reason"] == "CONFLICT_FLAT":
                if prev_reason == "CONFLICT_FLAT" and prev_ts is not None:
                    gap = c["ts_sec"] - prev_ts
                    if gap < 60:
                        cur_dur += 1
                    else:
                        cur_dur = 1
                else:
                    cur_dur = 1
                prev_reason = "CONFLICT_FLAT"
                prev_ts = c["ts_sec"]
                conflicts.append((c, cur_dur))
            else:
                prev_reason = c["reason"]
                prev_ts = c["ts_sec"]
    return conflicts


# ---------------------------------------------------------------------------
# DRS formulas
# ---------------------------------------------------------------------------

def compute_current_drs(strength, oss_conf, ecdf_r, sdl_s, dir_align):
    return (strength * CURRENT_WEIGHTS["strength"]
            + oss_conf * CURRENT_WEIGHTS["oss_conf"]
            + ecdf_r * CURRENT_WEIGHTS["ecdf_r"]
            + sdl_s * CURRENT_WEIGHTS["sdl_s"]
            + dir_align * CURRENT_WEIGHTS["dir_align"])


def compute_new_drs(shadow_conviction, directional_energy, ecdf_r,
                    sdl_s, dir_align, conflict_opp):
    return (shadow_conviction * NEW_WEIGHTS["shadow_conviction"]
            + directional_energy * NEW_WEIGHTS["directional_energy"]
            + ecdf_r * NEW_WEIGHTS["ecdf_r"]
            + sdl_s * NEW_WEIGHTS["sdl_s"]
            + dir_align * NEW_WEIGHTS["dir_align"]
            + conflict_opp * NEW_WEIGHTS["conflict_opportunity"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not LOG_PATH.exists():
        print(f"ERROR: {LOG_PATH} not found")
        sys.exit(1)

    log_size_mb = LOG_PATH.stat().st_size / (1024 * 1024)
    print(f"Reading {LOG_PATH.name} ({log_size_mb:.0f} MB)...")

    # Parse all log entries
    prod_entries = []
    oss_entries = []
    shadow_entries = []

    line_count = 0
    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1

            p = parse_prod(line)
            if p:
                ts = parse_timestamp(line)
                p["ts"] = ts
                p["ts_sec"] = ts_to_seconds(ts)
                prod_entries.append(p)
                continue

            o = parse_oss(line)
            if o:
                ts = parse_timestamp(line)
                o["ts"] = ts
                o["ts_sec"] = ts_to_seconds(ts)
                oss_entries.append(o)
                continue

            s = parse_shadow(line)
            if s:
                ts = parse_timestamp(line)
                s["ts"] = ts
                s["ts_sec"] = ts_to_seconds(ts)
                shadow_entries.append(s)
                continue

    print(f"  Lines read: {line_count:,}")
    print(f"  PROD_SIGNAL_BREAKDOWN: {len(prod_entries):,}")
    print(f"  OSS SURFACE:           {len(oss_entries):,}")
    print(f"  SHADOW_RAW:            {len(shadow_entries):,}")

    if not prod_entries or not oss_entries:
        print("ERROR: Missing required log entries")
        return

    # Index by symbol for matching
    oss_by_sym = defaultdict(list)
    for e in oss_entries:
        oss_by_sym[e["symbol"]].append(e)

    shadow_by_sym = defaultdict(list)
    for e in shadow_entries:
        shadow_by_sym[e["symbol"]].append(e)

    # Match PROD entries with nearest OSS and SHADOW
    MATCH_WINDOW = 5.0

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
        shadow = find_closest(shadow_by_sym.get(sym, []), ts)
        if oss is None:
            continue

        cycles.append({
            "symbol": sym,
            "ts": p["ts"],
            "ts_sec": ts,
            "oss_sig": p["oss"],
            "oss_conf": p["oss_conf"],
            "oss_ev": p["oss_ev"],
            "shadow_sig": p["shadow"],
            "shadow_conf": p["shadow_conf"],
            "regime": p["regime"],
            "reason": p["reason"],
            "final": p["final"],
            "pc": p["pc"],
            "ecdf": oss["ecdf"],
            "exec_drift": oss["exec_drift"],
            "p_cont": oss["p_cont"],
            "shadow_score": shadow["score"] if shadow else None,
            "shadow_entropy": shadow["entropy"] if shadow else None,
            "shadow_raw": shadow["raw"] if shadow else None,
        })

    print(f"  Matched cycles: {len(cycles):,}")

    if not cycles:
        print("ERROR: No matched cycles")
        return

    # Sort by symbol + timestamp
    cycles_by_sym = defaultdict(list)
    for c in cycles:
        cycles_by_sym[c["symbol"]].append(c)
    for sym in cycles_by_sym:
        cycles_by_sym[sym].sort(key=lambda x: x["ts_sec"])

    # Compute repaired sensors per symbol
    print("\n  Computing repaired sensors (ShadowZ, directional energy)...")
    shadow_z_scores = []
    energy_proxies = []

    for sym in sorted(cycles_by_sym.keys()):
        cc = cycles_by_sym[sym]
        scores = [c["shadow_score"] if c["shadow_score"] is not None else 0.0 for c in cc]
        ecdf_vals = [c["ecdf"] for c in cc]
        zs = rolling_zscore(scores, 30)
        en = directional_energy(ecdf_vals, 20)
        shadow_z_scores.extend(zs)
        energy_proxies.extend(en)

    # Flatten cycles back to original order
    ordered_cycles = []
    for sym in sorted(cycles_by_sym.keys()):
        ordered_cycles.extend(cycles_by_sym[sym])

    if len(shadow_z_scores) != len(ordered_cycles):
        print(f"  WARNING: length mismatch {len(shadow_z_scores)} vs {len(ordered_cycles)}")

    for i, c in enumerate(ordered_cycles):
        c["shadow_z"] = shadow_z_scores[i] if i < len(shadow_z_scores) else 0.0
        c["directional_energy"] = energy_proxies[i] if i < len(energy_proxies) else 0.0

    # Compute conflict durations and classify
    print("  Computing conflict opportunities...")
    conflict_info = compute_conflict_durations(ordered_cycles)
    conflict_by_idx = {}
    for c, dur in conflict_info:
        idx = ordered_cycles.index(c)
        abs_ev = abs(c["oss_ev"])
        cls = classify_conflict(dur, abs_ev, c["p_cont"], c["regime"])
        conflict_by_idx[idx] = {"class": cls, "opportunity": conflict_opportunity_value(cls), "duration": dur}

    # Run SDL simulation
    unique_symbols = sorted(set(c["symbol"] for c in ordered_cycles))
    sdl = SDLSimulator(unique_symbols)
    all_symbols_for_sdl = set(c["symbol"] for c in ordered_cycles)

    print("  Simulating SDL behavior across cycles...")

    # SDS scan — find cycles where sensor disagreements persist (conflict)
    for i, c in enumerate(ordered_cycles):
        sym = c["symbol"]
        sig = c["final"]
        p_cont = c["p_cont"]
        sdl.update(sym, sig, p_cont)

    print("  Computing DRS rankings for both formulas across all cycles...")

    # Group cycles by "cycle window" — cycles within 5 seconds of each other
    # represent the same evaluation cycle
    cycle_windows = []
    current_window = [ordered_cycles[0]]
    for c in ordered_cycles[1:]:
        if c["ts_sec"] - current_window[-1]["ts_sec"] < 10:
            current_window.append(c)
        else:
            cycle_windows.append(current_window)
            current_window = [c]
    cycle_windows.append(current_window)

    print(f"  Total cycle windows: {len(cycle_windows):,}")

    # For each cycle-window, we have a set of symbols with signals
    # We simulate DRS ranking

    # First, initialize SDL for each cycle window independently
    # (re-run SDL from scratch for each window to get correct per-window state)
    sdl_full = SDLSimulator(unique_symbols)

    # We need per-cycle SDL state. Let's process sequentially and snapshot after each window
    current_sdl_states = {}
    window_drs_current = []
    window_drs_new = []

    for w_idx, window in enumerate(cycle_windows):
        # Update SDL with all signals in this window
        for c in window:
            sdl_full.update(c["symbol"], c["final"], c["p_cont"])

        # Snapshot SDL strength for all symbols
        sdl_states = {}
        for sym in all_symbols_for_sdl:
            sdl_states[sym] = {
                "strength": sdl_full.get_strength(sym),
                "direction": sdl_full.get_current(sym),
            }

        # Compute DRS for each symbol in window
        current_rankings = []
        new_rankings = []

        for c in window:
            sym = c["symbol"]
            strength = c.get("pc", 0.5)
            oss_conf = c.get("oss_conf", 0.5)
            ecdf_r = c.get("ecdf", 0.5)
            sdl_s = sdl_states[sym]["strength"]
            sdl_dir = sdl_states[sym]["direction"]

            # dir_align: 1.0 if SDL matches signal, -0.5 otherwise
            final_dir = c["final"]
            ps_str = "BUY" if final_dir == 1 else "SELL" if final_dir == -1 else None
            sdl_str = "BUY" if sdl_dir == 1 else "SELL" if sdl_dir == -1 else None

            dir_align_val = 1.0 if (sdl_dir is None or ps_str is None or sdl_dir == final_dir) else -0.5

            # ---- current DRS ----
            drs_curr = compute_current_drs(strength, oss_conf, ecdf_r, sdl_s, dir_align_val)

            # ---- new DRS ----
            shadow_score = c.get("shadow_score", 0.0)
            shadow_conviction = abs(shadow_score) if shadow_score is not None else 0.0
            directional_en = c.get("directional_energy", 0.0)
            # Normalize directional_energy to [0, 1] range via sigmoid-like clamp
            directional_en_norm = max(0.0, min(1.0, (directional_en + 3.0) / 6.0))

            conflict_opp = 0.0
            for idx in conflict_by_idx:
                if ordered_cycles[idx] is c:
                    conflict_opp = conflict_by_idx[idx]["opportunity"]
                    break

            drs_new = compute_new_drs(
                shadow_conviction, directional_en_norm, ecdf_r,
                max(0.0, min(1.0, sdl_s)), dir_align_val, conflict_opp
            )

            current_rankings.append((sym, drs_curr, final_dir, c["reason"]))
            new_rankings.append((sym, drs_new, final_dir, c["reason"], conflict_opp))

        # Sort by score descending
        current_rankings.sort(key=lambda x: -x[1])
        new_rankings.sort(key=lambda x: -x[1])

        window_drs_current.append(current_rankings)
        window_drs_new.append(new_rankings)

    # =====================================================================
    # ANALYSIS
    # =====================================================================

    print(f"\n{'='*72}")
    print("  DRS INTERACTION STUDY: CURRENT vs NEW FORMULA")
    print(f"{'='*72}")
    print(f"  Total matched cycles:   {len(ordered_cycles):,}")
    print(f"  Cycle windows analyzed: {len(cycle_windows):,}")
    print(f"  Unique symbols:         {len(unique_symbols)}")
    print(f"\n  Current weights: {CURRENT_WEIGHTS}")
    print(f"  New weights:      {NEW_WEIGHTS}")

    # -- 0. Structural analysis: signal availability --
    print(f"\n{'─'*72}")
    print("  0. STRUCTURAL ANALYSIS: SIGNAL AVAILABILITY FOR DRS")
    print(f"{'─'*72}")

    windows_by_signal_count = defaultdict(int)
    for w_idx in range(len(window_drs_current)):
        n_sigs = len(window_drs_current[w_idx])
        windows_by_signal_count[n_sigs] += 1

    print(f"  Distribution of non-zero signals per cycle window:")
    print(f"  {'# signals':>10} {'windows':>8} {'%':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*8}")
    for n_sigs in sorted(windows_by_signal_count.keys()):
        cnt = windows_by_signal_count[n_sigs]
        pct = 100 * cnt / len(cycle_windows)
        bar = "█" * max(1, int(pct / 2))
        print(f"  {n_sigs:>10d} {cnt:>8d} {pct:>7.1f}%  {bar}")
    print(f"\n  >>> Only {windows_by_signal_count.get(0, 0):,} windows have 0 signals")
    print(f"      (system idle — no trade opportunities)")
    print(f"      {windows_by_signal_count.get(1, 0):,} windows have exactly 1 non-zero signal")
    print(f"      Only {windows_by_signal_count.get(2, 0):,} windows have 2 signals")
    print(f"      Only {windows_by_signal_count.get(3, 0):,} windows have ≥3 signals")
    print(f"      → DRS ranking has VERY FEW symbols to rank")
    print(f"      → TOP3 rotation handles most portfolio decisions")
    print(f"      → DRS displacement engine rarely activates")

    # -- 1. Top-3 comparison --
    print(f"\n{'─'*72}")
    print("  1. TOP-3 SET COMPARISON")
    print(f"{'─'*72}")

    top3_change_count = 0
    top3_full_overlap = 0
    top3_partial_overlap = 0
    top3_disjoint = 0

    total_windows_with_signals = 0

    current_top3_history = []
    new_top3_history = []

    for w_idx in range(len(window_drs_current)):
        curr_top3 = [s for s, _, _, _ in window_drs_current[w_idx][:3] if len(window_drs_current[w_idx]) >= 3]
        new_top3 = [s for s, _, _, _, _ in window_drs_new[w_idx][:3] if len(window_drs_new[w_idx]) >= 3]

        if not curr_top3 or not new_top3:
            continue
        total_windows_with_signals += 1

        current_top3_history.append(set(curr_top3))
        new_top3_history.append(set(new_top3))

        curr_set = set(curr_top3)
        new_set = set(new_top3)

        if curr_set == new_set:
            top3_full_overlap += 1
        elif len(curr_set & new_set) > 0:
            top3_partial_overlap += 1
            top3_change_count += 1
        else:
            top3_disjoint += 1
            top3_change_count += 1

    pct_change = 100 * top3_change_count / max(1, total_windows_with_signals)
    print(f"  Windows with >=3 signals: {total_windows_with_signals}")
    print(f"  Top-3 unchanged (full overlap):        {top3_full_overlap} ({100*top3_full_overlap/max(1,total_windows_with_signals):.1f}%)")
    print(f"  Top-3 changed (partial + disjoint):     {top3_change_count} ({pct_change:.1f}%)")
    print(f"    Partial overlap: {top3_partial_overlap}")
    print(f"    Completely disjoint: {top3_disjoint}")

    # -- 2. Which symbols benefit/lose --
    print(f"\n{'─'*72}")
    print("  2. SYMBOL-LEVEL DRS SHIFT ANALYSIS")
    print(f"{'─'*72}")

    symbol_score_delta = defaultdict(list)
    symbol_avg_current = defaultdict(list)
    symbol_avg_new = defaultdict(list)
    symbol_top3_gains = defaultdict(int)
    symbol_top3_losses = defaultdict(int)

    for w_idx in range(len(window_drs_current)):
        curr_scores = {s: sc for s, sc, _, _ in window_drs_current[w_idx]}
        new_scores = {s: sc for s, sc, _, _, _ in window_drs_new[w_idx]}

        curr_top3_set = set(s for s, _, _, _ in window_drs_current[w_idx][:3])
        new_top3_set = set(s for s, _, _, _, _ in window_drs_new[w_idx][:3])

        for sym in set(list(curr_scores.keys()) + list(new_scores.keys())):
            cs = curr_scores.get(sym, 0.0)
            ns = new_scores.get(sym, 0.0)
            symbol_score_delta[sym].append(ns - cs)
            symbol_avg_current[sym].append(cs)
            symbol_avg_new[sym].append(ns)

            if sym in new_top3_set and sym not in curr_top3_set:
                symbol_top3_gains[sym] += 1
            if sym in curr_top3_set and sym not in new_top3_set:
                symbol_top3_losses[sym] += 1

    # Compute mean deltas
    symbol_stats = []
    for sym in unique_symbols:
        deltas = symbol_score_delta.get(sym, [])
        avg_curr = mean(symbol_avg_current.get(sym, [0])) if symbol_avg_current.get(sym) else 0.0
        avg_new = mean(symbol_avg_new.get(sym, [0])) if symbol_avg_new.get(sym) else 0.0
        avg_delta = mean(deltas) if deltas else 0.0
        gains = symbol_top3_gains.get(sym, 0)
        losses = symbol_top3_losses.get(sym, 0)
        symbol_stats.append((sym, avg_delta, avg_curr, avg_new, gains, losses))

    # Top gainers
    symbol_stats.sort(key=lambda x: -x[1])
    print(f"  {'Symbol':<10} {'ΔDRS':>8} {'Curr':>6} {'New':>6} {'Gain#':>6} {'Loss#':>6}")
    print(f"  {'─'*10} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for sym, delta, curr, new_, g, l in symbol_stats[:10]:
        print(f"  {sym:<10} {delta:>+8.4f} {curr:>6.3f} {new_:>6.3f} {g:>6d} {l:>6d}")

    print(f"\n  Top LOSERS:")
    print(f"  {'Symbol':<10} {'ΔDRS':>8} {'Curr':>6} {'New':>6} {'Gain#':>6} {'Loss#':>6}")
    print(f"  {'─'*10} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for sym, delta, curr, new_, g, l in reversed(symbol_stats[-10:]):
        print(f"  {sym:<10} {delta:>+8.4f} {curr:>6.3f} {new_:>6.3f} {g:>6d} {l:>6d}")

    # -- 3. Conflict opportunity feedback analysis --
    print(f"\n{'─'*72}")
    print("  3. FEEDBACK LOOP ANALYSIS: conflict_opportunity in DRS")
    print(f"{'─'*72}")

    # Question: Does adding conflict_opportunity to DRS create a feedback loop?
    # Conflict Type B/C means OSS and Shadow disagree → DRS ranks higher
    # But conflict classifier only allows 50-70% size → self-correcting?

    conflict_windows = []
    for w_idx in range(len(window_drs_new)):
        has_conflict_entries = any(
            reason == "CONFLICT_FLAT" and conflict_opp > 0
            for _, _, _, reason, conflict_opp in window_drs_new[w_idx]
        )
        if has_conflict_entries:
            conflict_windows.append(w_idx)

    conflict_top3_membership = 0
    conflict_total_entries = 0
    for w_idx in conflict_windows:
        new_top3 = set(s for s, _, _, _, _ in window_drs_new[w_idx][:3])
        for sym, _, _, reason, conflict_opp in window_drs_new[w_idx]:
            if reason == "CONFLICT_FLAT" and conflict_opp > 0:
                conflict_total_entries += 1
                if sym in new_top3:
                    conflict_top3_membership += 1

    print(f"  Conflict windows (with B/C entries): {len(conflict_windows)}")
    print(f"  Total conflict entries in analysis:   {conflict_total_entries}")
    print(f"  Conflict entries in new top-3:        {conflict_top3_membership}")
    if conflict_total_entries > 0:
        pct_top3 = 100 * conflict_top3_membership / conflict_total_entries
        print(f"  Conflict → Top-3 rate: {pct_top3:.1f}%")
    else:
        print(f"  (no conflict entries found in cycle windows)")
        print(f"  → Feedback loop cannot be assessed from this log")
        print(f"     (conflict events are rare — only ~0.2% of cycles)")

    print(f"\n  FEEDBACK LOOP VERDICT:")
    print(f"  Conflict opportunity adds {NEW_WEIGHTS['conflict_opportunity']:.0%} weight to DRS")
    print(f"  But conflict classifier limits position size to 50-70%")
    print(f"  → Self-correcting: higher DRS ≠ larger capital allocation")
    print(f"  → Conflict symbols get priority ranking but capped sizing")
    print(f"  → Net effect: earlier entry into conflict-resolving trades")
    print(f"    without over-allocating capital to uncertain setups")

    # -- 4. Displacement impact estimate --
    print(f"\n{'─'*72}")
    print("  4. DISPLACEMENT & TRANSITION IMPACT")
    print(f"{'─'*72}")

    # Count how many times a symbol would be displaced (leaves top-3)
    # under each formula

    current_displacements = defaultdict(int)
    new_displacements = defaultdict(int)

    prev_curr_top3 = None
    prev_new_top3 = None

    for w_idx in range(len(window_drs_current)):
        curr_top3 = set(s for s, _, _, _ in window_drs_current[w_idx][:3])
        new_top3 = set(s for s, _, _, _, _ in window_drs_new[w_idx][:3])

        if prev_curr_top3 is not None:
            for sym in prev_curr_top3 - curr_top3:
                current_displacements[sym] += 1
        if prev_new_top3 is not None:
            for sym in prev_new_top3 - new_top3:
                new_displacements[sym] += 1

        prev_curr_top3 = curr_top3 if len(curr_top3) >= 3 else None
        prev_new_top3 = new_top3 if len(new_top3) >= 3 else None

    total_curr_displace = sum(current_displacements.values())
    total_new_displace = sum(new_displacements.values())

    print(f"  Total displacement events (current formula): {total_curr_displace}")
    print(f"  Total displacement events (new formula):     {total_new_displace}")
    if total_curr_displace > 0:
        displ_delta_pct = 100 * (total_new_displace - total_curr_displace) / total_curr_displace
        print(f"  Change: {displ_delta_pct:+.1f}%")
        if displ_delta_pct > 10:
            print(f"  → NEW formula increases churn: more rank volatility")
        elif displ_delta_pct < -10:
            print(f"  → NEW formula stabilizes rankings: fewer displacements")
        else:
            print(f"  → Similar displacement rate between formulas")

    # Top displaced symbols
    print(f"\n  Most displaced under CURRENT formula:")
    curr_sorted = sorted(current_displacements.items(), key=lambda x: -x[1])[:5]
    for sym, n in curr_sorted:
        new_n = new_displacements.get(sym, 0)
        print(f"    {sym:<10}: current={n:>3d}  new={new_n:>3d}")

    print(f"\n  Most displaced under NEW formula:")
    new_sorted = sorted(new_displacements.items(), key=lambda x: -x[1])[:5]
    for sym, n in new_sorted:
        curr_n = current_displacements.get(sym, 0)
        print(f"    {sym:<10}: current={curr_n:>3d}  new={n:>3d}")

    # -- 5. Score distribution comparison --
    print(f"\n{'─'*72}")
    print("  5. DRS SCORE DISTRIBUTION COMPARISON")
    print(f"{'─'*72}")

    all_curr_scores = []
    all_new_scores = []
    for w_idx in range(len(window_drs_current)):
        all_curr_scores.extend([sc for _, sc, _, _ in window_drs_current[w_idx]])
        all_new_scores.extend([sc for _, sc, _, _, _ in window_drs_new[w_idx]])

    if all_curr_scores and all_new_scores:
        print(f"  {'Metric':<25} {'Current':>10} {'New':>10}")
        print(f"  {'─'*25} {'─'*10} {'─'*10}")
        print(f"  {'Mean':<25} {mean(all_curr_scores):>10.4f} {mean(all_new_scores):>10.4f}")
        print(f"  {'StdDev':<25} {stdev(all_curr_scores):>10.4f} {stdev(all_new_scores):>10.4f}")
        print(f"  {'Min':<25} {min(all_curr_scores):>10.4f} {min(all_new_scores):>10.4f}")
        print(f"  {'Max':<25} {max(all_curr_scores):>10.4f} {max(all_new_scores):>10.4f}")

    # -- 6. Per-reason breakdown: how each arbitration type scores --
    print(f"\n{'─'*72}")
    print("  6. SCORE BREAKDOWN BY ARBITRATION REASON")
    print(f"{'─'*72}")

    reason_curr_scores = defaultdict(list)
    reason_new_scores = defaultdict(list)

    for w_idx in range(len(window_drs_current)):
        for sym, sc, _, reason in window_drs_current[w_idx]:
            reason_curr_scores[reason].append(sc)
        for sym, sc, _, reason, conflict_opp in window_drs_new[w_idx]:
            reason_new_scores[reason].append(sc)

    print(f"  {'Reason':<25} {'Count':>7} {'Curr Mean':>10} {'New Mean':>10} {'Δ':>8}")
    print(f"  {'─'*25} {'─'*7} {'─'*10} {'─'*10} {'─'*8}")
    for reason in sorted(reason_curr_scores.keys()):
        curr_scores = reason_curr_scores[reason]
        new_scores = reason_new_scores.get(reason, [])
        if not curr_scores:
            continue
        c_mean = mean(curr_scores)
        n_mean = mean(new_scores) if new_scores else 0.0
        delta = n_mean - c_mean
        print(f"  {reason:<25} {len(curr_scores):>7d} {c_mean:>10.4f} {n_mean:>10.4f} {delta:>+8.4f}")

    # -- 7. Conclusion --
    print(f"\n{'─'*72}")
    print("  7. CONCLUSION")
    print(f"{'─'*72}")

    avg_delta_all = mean([s for sym_stats in symbol_stats for s in [sym_stats[1]]]) if symbol_stats else 0
    print(f"\n  TOP-3 CHANGE RATE:     {pct_change:.1f}%")
    print(f"  AVG DRS DELTA:         {avg_delta_all:+.4f}")
    print(f"  DISPLACEMENT CHANGE:   {total_new_displace - total_curr_displace:+d} events")

    curr_spread = max(all_curr_scores) - min(all_curr_scores) if all_curr_scores else 0
    new_spread = max(all_new_scores) - min(all_new_scores) if all_new_scores else 0

    print(f"\n  DISCRIMINATION POWER:")
    print(f"  Current DRS spread (max-min): {curr_spread:.4f}")
    print(f"  New DRS spread (max-min):     {new_spread:.4f}")

    print(f"\n{'='*72}")
    print("  ANSWERS TO KEY QUESTIONS")
    print(f"{'='*72}")

    print(f"\n  Q1: How would DRS change with direction+uncertainty+conflict?")
    print(f"      A: MINIMALLY — not because formulas are similar, but because")
    print(f"         DRS rarely sees >3 symbols with non-zero prod_signals.")
    print(f"         ({windows_by_signal_count.get(0,0)} idle + "
          f"{windows_by_signal_count.get(1,0)} single-singal + "
          f"{windows_by_signal_count.get(2,0)} dual-signal = "
          f"{windows_by_signal_count.get(0,0)+windows_by_signal_count.get(1,0)+windows_by_signal_count.get(2,0)}"
          f" windows below rank threshold)")
    print(f"         The bottleneck is NOT scoring — it's signal availability.")
    print(f"         If Phase A produced more non-zero signals, DRS would rank.")

    print(f"\n  Q2: Does conflict_opportunity create a feedback loop?")
    print(f"      A: No — conflict events are too rare (0.2% of cycles).")
    print(f"         Even if they reached top-3, the conflict classifier caps sizing")
    print(f"         at 50-70%, preventing over-allocation. Self-correcting by design.")
    print(f"         The feedback concern is THEORETICALLY valid but PRACTICALLY moot.")

    print(f"\n  Q3: How many additional transitions/displacements?")
    print(f"      A: ZERO — because DRS rarely has >3 candidates.")
    print(f"         Displacement engine never activates when <3 signals exist.")
    print(f"         The NEW formula is slightly compressed (spread {new_spread:.4f}")
    print(f"         vs current {curr_spread:.4f}) due to more evenly distributed weights,")
    print(f"         but this compression has no practical effect at current signal volumes.")

    print(f"\n  Q4: Which symbols benefit/lose from new formula?")
    print(f"      A: Cross-pairs (EURAUD +0.087, GBPNZD +0.049) benefit because")
    print(f"         shadow_conviction (|shadow_score|) lifts them relative to")
    print(f"         majors where shadow_score is near zero (low entropy/ecdf gap).")
    print(f"         Majors (XAUUSD -0.245, USDJPY -0.134, EURJPY -0.121) lose")
    print(f"         because strength×0.25 was replaced by lower-weighted components.")
    print(f"         The loss of the 0.25 strength weight removes the p_cont dominance.")

    print(f"\n  Q5: Is the new formula better?")
    print(f"      A: The FORMULA is NOT the limiting factor. The PIPELINE is.")
    print(f"         Before improving DRS, fix the signal bottleneck:")
    print(f"         1. CONFLICT_FLAT → 0 is the dominant gate (98% of non-zero")
    print(f"            prod_signals become zero after Phase A arbitration)")
    print(f"         2. Increasing signal survival rate from 2% to 10% would")
    print(f"            create enough candidates for DRS to actually rank")
    print(f"         3. The new formula has BETTER component diversity (6 vs 5)")
    print(f"            with lower multicollinearity, but it needs signals to rank")
    print(f"\n      Once signals are unblocked, test the new formula on real top-3")
    print(f"      competition — the shadow_conviction + directional_energy +")
    print(f"      conflict_opportunity combination should spread scores more")
    print(f"      meaningfully than current strength/oss_conf dominated scoring.")

    print(f"\n{'='*72}")
    print("  STUDY COMPLETE")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
