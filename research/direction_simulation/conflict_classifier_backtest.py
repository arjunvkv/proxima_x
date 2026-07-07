"""
Conflict Classifier Backtest Agent
====================================
OFFLINE analysis of CONFLICT_FLAT events from proxima_demo.log.
Does NOT modify production code.

Usage:
    python research/direction_simulation/conflict_classifier_backtest.py
"""

import re
import sys
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
from statistics import mean, median, stdev

LOG_PATH = Path(__file__).resolve().parents[2] / "proxima_demo.log"

# ── Regex patterns ──────────────────────────────────────────────────

# PROD_SIGNAL_BREAKDOWN with CONFLICT_FLAT
# Example: "EURUSD oss=+1(ev=0.5113) shadow=-1 reason=CONFLICT_FLAT final=+0 pc=0.50"
RE_CONFLICT = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? \[PROD_SIGNAL_BREAKDOWN\] "
    r"(\w+) oss=([+-]?\d+)\(ev=([-0-9.]+)\) shadow=([+-]?\d+) "
    r"reason=CONFLICT_FLAT final=([+-]?\d+) pc=([0-9.]+)"
)

# OSS SURFACE
# Example: "EURUSD ecdf=0.7549 exec_drift=0.9999 ... regime=COMPRESSED_CHAOS p_cont=0.70"
RE_OSS_SURFACE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? \[OSS SURFACE\] "
    r"(\w+) ecdf=([0-9.]+) exec_drift=([-0-9.]+) .*? "
    r"regime=(\w+) p_cont=([0-9.]+)"
)

# EXPLORATION entry (forced triggered)
RE_EXPLORATION = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? \[EXPLORATION\] "
    r"(\w+): forced triggered=True"
)


def parse_timestamp(ts_str: str) -> datetime:
    return datetime.strptime(ts_str.split(",")[0], "%Y-%m-%d %H:%M:%S")


def load_log():
    """Load and parse the log file, returning three lists."""
    if not LOG_PATH.exists():
        print(f"ERROR: {LOG_PATH} not found", file=sys.stderr)
        sys.exit(1)

    conflicts = []
    oss_surfaces = []
    explorations = []

    size_mb = LOG_PATH.stat().st_size / 1_024 / 1_024
    print(f"Reading {LOG_PATH.name} ({size_mb:.0f} MB)...")

    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_CONFLICT.match(line)
            if m:
                ts = parse_timestamp(m.group(1))
                conflicts.append({
                    "timestamp": ts,
                    "symbol": m.group(2),
                    "oss_sig": int(m.group(3)),
                    "oss_ev": float(m.group(4)),
                    "shadow_sig": int(m.group(5)),
                    "final": int(m.group(6)),
                    "pc": float(m.group(7)),
                })
                continue

            m = RE_OSS_SURFACE.match(line)
            if m:
                ts = parse_timestamp(m.group(1))
                oss_surfaces.append({
                    "timestamp": ts,
                    "symbol": m.group(2),
                    "ecdf": float(m.group(3)),
                    "exec_drift": float(m.group(4)),
                    "regime": m.group(5),
                    "p_cont": float(m.group(6)),
                })
                continue

            m = RE_EXPLORATION.match(line)
            if m:
                ts = parse_timestamp(m.group(1))
                explorations.append({
                    "timestamp": ts,
                    "symbol": m.group(2),
                })

    print(f"  Loaded: {len(conflicts)} CONFLICT_FLAT, "
          f"{len(oss_surfaces)} OSS SURFACE, "
          f"{len(explorations)} EXPLORATION events")
    return conflicts, oss_surfaces, explorations


def build_oss_index(oss_surfaces):
    """Build a dict: symbol -> sorted list of (timestamp, record)."""
    idx = defaultdict(list)
    for rec in oss_surfaces:
        idx[rec["symbol"]].append((rec["timestamp"], rec))
    for sym in idx:
        idx[sym].sort(key=lambda x: x[0])
    return idx


def find_nearest_oss(oss_idx, symbol, ts, max_seconds=30):
    """Find the nearest OSS SURFACE entry for symbol within max_seconds."""
    entries = oss_idx.get(symbol, [])
    if not entries:
        return None
    # Binary search
    lo, hi = 0, len(entries) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if entries[mid][0] < ts:
            lo = mid + 1
        elif entries[mid][0] > ts:
            hi = mid - 1
        else:
            return entries[mid][1]
    # Check nearest candidates
    candidates = []
    for idx in (hi, lo):
        if 0 <= idx < len(entries):
            delta = abs((entries[idx][0] - ts).total_seconds())
            if delta <= max_seconds:
                candidates.append((delta, entries[idx][1]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def classify_event(conflict, duration, oss_rec):
    """Apply the proposed classifier."""
    oss_ev = conflict["oss_ev"]
    p_cont = oss_rec["p_cont"] if oss_rec else conflict["pc"]
    regime = oss_rec["regime"] if oss_rec else "UNKNOWN"
    abs_ev = abs(oss_ev)

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


def compute_durations(conflicts):
    """Group by symbol, sort by timestamp, return list of (event, duration)."""
    by_sym = defaultdict(list)
    for c in conflicts:
        by_sym[c["symbol"]].append(c)
    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["timestamp"])

    result = []
    # Track consecutive CONFLICT_FLAT runs per symbol
    for sym, entries in by_sym.items():
        if not entries:
            continue
        current_dur = 1
        for i in range(len(entries)):
            if i == 0:
                current_dur = 1
            else:
                gap = (entries[i]["timestamp"] - entries[i-1]["timestamp"]).total_seconds()
                if gap < 60:  # same cycle cluster (within 1 min)
                    current_dur += 1
                else:
                    current_dur = 1
            result.append((entries[i], current_dur, sym))
    return result


# ── Analysis ────────────────────────────────────────────────────────

TYPE_DESC = {
    "A_NOISE": "A_NOISE — transient noise, no action",
    "B_TRANSITION": "B_TRANSITION — regime shift 50% entry allowed",
    "C_ACCUMULATION": "C_ACCUMULATION — smart-money 70% entry allowed",
    "D_INDETERMINACY": "D_INDETERMINACY — persistent deadlock, no action",
}
TYPE_ORDER = ["A_NOISE", "B_TRANSITION", "C_ACCUMULATION", "D_INDETERMINACY"]


def main():
    conflicts, oss_surfaces, explorations = load_log()
    oss_idx = build_oss_index(oss_surfaces)

    # Build exploration set for quick lookup: (symbol, date) -> list of timestamps
    expl_by_sym = defaultdict(list)
    for e in explorations:
        expl_by_sym[e["symbol"]].append(e["timestamp"])

    # Compute durations and enrich with OSS data
    enriched = []
    for event, duration, sym in compute_durations(conflicts):
        oss_rec = find_nearest_oss(oss_idx, sym, event["timestamp"])
        cls = classify_event(event, duration, oss_rec)
        enriched.append({
            **event,
            "duration": duration,
            "class": cls,
            "oss_regime": oss_rec["regime"] if oss_rec else "UNKNOWN",
            "oss_p_cont": oss_rec["p_cont"] if oss_rec else None,
            "oss_ecdf": oss_rec["ecdf"] if oss_rec else None,
            "oss_exec_drift": oss_rec["exec_drift"] if oss_rec else None,
        })

    # Filter for unique (symbol, cycle) — use one event per durational
    type_counts = defaultdict(int)
    type_durations = defaultdict(list)
    type_regimes = defaultdict(lambda: defaultdict(int))
    type_symbols = defaultdict(lambda: defaultdict(int))
    type_evs = defaultdict(list)

    for rec in enriched:
        cls = rec["class"]
        type_counts[cls] += 1
        type_durations[cls].append(rec["duration"])
        type_regimes[cls][rec["oss_regime"]] += 1
        type_symbols[cls][rec["symbol"]] += 1
        type_evs[cls].append(abs(rec["oss_ev"]))

    # ── Type B & C: check if exploration eventually triggered ──────
    bc_entries = [r for r in enriched if r["class"] in ("B_TRANSITION", "C_ACCUMULATION")]

    # For each B/C conflict, check if an exploration exists within next 30min for same symbol
    bc_explored = 0
    bc_explored_details = []
    for rec in bc_entries:
        sym = rec["symbol"]
        t0 = rec["timestamp"]
        t1 = datetime.fromtimestamp(t0.timestamp() + 1800)  # 30min window
        matches = [t for t in expl_by_sym.get(sym, []) if t0 <= t <= t1]
        if matches:
            bc_explored += 1
            bc_explored_details.append({
                "symbol": sym,
                "timestamp": t0,
                "class": rec["class"],
                "oss_ev": rec["oss_ev"],
                "shadow_sig": rec["shadow_sig"],
                "regime": rec["oss_regime"],
                "p_cont": rec["oss_p_cont"],
                "exploration_ts": min(matches),
            })

    bc_total = len(bc_entries)
    bc_explored_pct = (bc_explored / bc_total * 100) if bc_total else 0.0

    # ── Summary stats ──────────────────────────────────────────────
    total = sum(type_counts.values())

    print(f"\n{'='*70}")
    print(f"  CONFLICT CLASSIFIER BACKTEST ANALYSIS")
    print(f"  Log: {LOG_PATH.name} ({LOG_PATH.stat().st_size / 1_024 / 1_024:.0f} MB)")
    print(f"  Date range: {conflicts[0]['timestamp'].date() if conflicts else 'N/A'} "
          f"to {conflicts[-1]['timestamp'].date() if conflicts else 'N/A'}")
    print(f"{'='*70}\n")

    # 1. Type distribution
    print("─" * 70)
    print("  1. TYPE DISTRIBUTION")
    print("─" * 70)
    print(f"  {'Type':<30} {'Count':>6} {'%':>7}")
    print(f"  {'─'*30} {'─'*6} {'─'*7}")
    for t in TYPE_ORDER:
        cnt = type_counts.get(t, 0)
        pct = cnt / total * 100 if total else 0
        print(f"  {TYPE_DESC[t]:<30} {cnt:>6} {pct:>6.1f}%")
    print(f"  {'─'*30} {'─'*6} {'─'*7}")
    print(f"  {'TOTAL':<30} {total:>6} 100.0%")

    # 2. Duration distribution per type
    print(f"\n{'─'*70}")
    print("  2. DURATION DISTRIBUTION PER TYPE")
    print("─" * 70)
    print(f"  {'Type':<20} {'Count':>6} {'Mean Dur':>9} {'Median':>8} {'Max':>5} {'StdDev':>8}")
    print(f"  {'─'*20} {'─'*6} {'─'*9} {'─'*8} {'─'*5} {'─'*8}")
    for t in TYPE_ORDER:
        durs = type_durations.get(t, [])
        if durs:
            m = mean(durs)
            med = median(durs)
            mx = max(durs)
            sd = stdev(durs) if len(durs) > 1 else 0.0
            print(f"  {t:<20} {len(durs):>6} {m:>8.1f} {med:>7.0f} {mx:>5} {sd:>7.1f}")
        else:
            print(f"  {t:<20} {0:>6} {'—':>9} {'—':>8} {'—':>5} {'—':>8}")

    # 3. Symbols most prone to each type
    print(f"\n{'─'*70}")
    print("  3. SYMBOLS MOST PRONE TO EACH TYPE (top 5)")
    print("─" * 70)
    for t in TYPE_ORDER:
        sym_counts = type_symbols.get(t, {})
        if not sym_counts:
            continue
        sorted_syms = sorted(sym_counts.items(), key=lambda x: -x[1])[:5]
        top5 = ", ".join(f"{s}({c})" for s, c in sorted_syms)
        t_total = sum(sym_counts.values())
        print(f"  {t:<20} ({t_total} total) → {top5}")

    # 4. Regime correlation per type
    print(f"\n{'─'*70}")
    print("  4. REGIME CORRELATION PER TYPE")
    print("─" * 70)
    all_regimes = set()
    for t in TYPE_ORDER:
        all_regimes.update(type_regimes[t].keys())
    all_regimes = sorted(all_regimes)

    header = f"  {'Type':<20}"
    for r in all_regimes:
        header += f" {r:>22}"
    print(header)
    print(f"  {'─'*20} {'─' * (22 * len(all_regimes))}")
    for t in TYPE_ORDER:
        reg_counts = type_regimes.get(t, {})
        t_total = sum(reg_counts.values()) or 1
        line = f"  {t:<20}"
        for r in all_regimes:
            cnt = reg_counts.get(r, 0)
            pct = cnt / t_total * 100
            line += f" {cnt:>4}({pct:>4.0f}%)"
        print(line)

    # 5. Type B & C analysis
    print(f"\n{'─'*70}")
    print("  5. TYPE B & C — EXPLORATION / PROFITABILITY ANALYSIS")
    print("─" * 70)
    for bc_cls in ("B_TRANSITION", "C_ACCUMULATION"):
        sub = [r for r in bc_entries if r["class"] == bc_cls]
        if not sub:
            continue
        sub_explored = sum(1 for r in bc_explored_details if r["class"] == bc_cls)
        sub_pct = sub_explored / len(sub) * 100
        sizing = {"B_TRANSITION": "50%", "C_ACCUMULATION": "70%"}[bc_cls]
        print(f"\n  {bc_cls} ({len(sub)} events, suggested sizing={sizing}):")
        print(f"    Events where exploration triggered within 30min: "
              f"{sub_explored}/{len(sub)} ({sub_pct:.1f}%)")

        if sub_explored:
            print(f"\n    {'Symbol':<10} {'OSS_ev':>8} {'Shadow':>7} {'Regime':<20} "
                  f"{'p_cont':>6} {'Decision':>10}")
            print(f"    {'─'*10} {'─'*8} {'─'*7} {'─'*20} {'─'*6} {'─'*10}")
            for d in bc_explored_details:
                if d["class"] != bc_cls:
                    continue
                # Estimate direction agreement: OSS vs Shadow
                oss_dir = "LONG" if d["oss_ev"] > 0 else "SHORT"
                shd_dir = "LONG" if d["shadow_sig"] > 0 else "SHORT" if d["shadow_sig"] < 0 else "FLAT"
                decision = "FOLLOW_OSS" if abs(d["oss_ev"]) > 0.5 else "FOLLOW_SHADOW"
                print(f"    {d['symbol']:<10} {d['oss_ev']:>+8.4f} {shd_dir:>7} "
                      f"{d['regime']:<20} {d['p_cont']:>6.2f} {decision:>10}")

    # 6. Additional trades estimation
    print(f"\n{'─'*70}")
    print("  6. ESTIMATED IMPACT — ADDITIONAL TRADES ALLOWED")
    print("─" * 70)

    # Count unique conflict cycles (grouped by symbol + ~1min window)
    # Each durational run represents one potential trade opportunity
    bc_cycles = set()
    for r in bc_entries:
        cycle_key = (r["symbol"], r["timestamp"].strftime("%Y-%m-%d %H:%M"))
        bc_cycles.add(cycle_key)

    b_cycles = set()
    c_cycles = set()
    for r in bc_entries:
        cycle_key = (r["symbol"], r["timestamp"].strftime("%Y-%m-%d %H:%M"))
        if r["class"] == "B_TRANSITION":
            b_cycles.add(cycle_key)
        else:
            c_cycles.add(cycle_key)

    print(f"\n  Type B (B_TRANSITION) unique trade opportunities: {len(b_cycles)} (50% sizing)")
    print(f"  Type C (C_ACCUMULATION) unique trade opportunities: {len(c_cycles)} (70% sizing)")
    print(f"  Total BC opportunities: {len(bc_cycles)}")

    # Estimate avg position size impact
    print(f"\n  Sizing impact (assuming 1.0 standard unit per trade):")
    print(f"    Current system: 0 trades from CONFLICT_FLAT")
    print(f"    With classifier: up to {len(bc_cycles)} additional partial-position trades")
    print(f"    Volume-weighted factor: "
          f"{len(b_cycles) * 0.5 + len(c_cycles) * 0.7:.1f} standard units")

    # ── Raw data export for debugging ──────────────────────────────
    output_path = Path(__file__).parent / "conflict_classifier_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("symbol\ttimestamp\tduration\tclass\toss_ev\tshadow_sig\tregime\tp_cont\tecdf\n")
        for rec in enriched:
            f.write(f"{rec['symbol']}\t{rec['timestamp']}\t{rec['duration']}\t{rec['class']}\t"
                    f"{rec['oss_ev']:.4f}\t{rec['shadow_sig']:+d}\t{rec['oss_regime']}\t"
                    f"{rec['oss_p_cont'] or rec['pc']:.4f}\t{rec['oss_ecdf'] or 'N/A'}\n")

    print(f"\n  Raw results exported to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
