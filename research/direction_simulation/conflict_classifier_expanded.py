"""
Conflict Classifier EXPANDED Analysis
======================================
Offline analysis of ALL CONFLICT_FLAT events across ALL dates in proxima_demo.log.
Does NOT modify production code.

Expanded features:
  - Extracts conf fields from PROD_SIGNAL_BREAKDOWN entries
  - Tracks conflict duration per symbol (reset on non-CONFLICT_FLAT)
  - Expanded 4-type classifier (A_NOISE, B_TRANSITION, C_ACCUMULATION, D_INDETERMINACY)
  - Opportunity estimation (sizing, additional volume)
  - Confusion table: what actually happened when classifier would allow
  - PnL / price tracking for actionable events

Usage:
    python research/direction_simulation/conflict_classifier_expanded.py
"""

import re
import sys
import json
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timedelta
from statistics import mean, median, stdev

LOG_PATH = Path(__file__).resolve().parents[2] / "proxima_demo.log"
RESULTS_PATH = Path(__file__).parent / "conflict_classifier_expanded_results.txt"
REPORT_PATH = Path(__file__).parent / "conflict_classifier_expanded_report.txt"

# ── Regex patterns ──────────────────────────────────────────────────

# PROD_SIGNAL_BREAKDOWN with CONFLICT_FLAT (no conf fields)
# "oss=+1(ev=0.5113) shadow=-1 reason=CONFLICT_FLAT final=+0 pc=0.50"
RE_CONFLICT = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? \[PROD_SIGNAL_BREAKDOWN\] "
    r"(\w+) oss=([+-]?\d+)\(ev=([-0-9.]+)\) shadow=([+-]?\d+) "
    r"reason=CONFLICT_FLAT final=([+-]?\d+) pc=([0-9.]+)"
)

# PROD_SIGNAL_BREAKDOWN with conf fields (non-conflict entries)
# "oss=+0(ev=-0.1253,conf=0.40) ev_sig=-1 shadow=-1(conf=1.00) regime=CHAOTIC reason=... final=... pc=..."
# Also matches lines WITHOUT ev_sig and WITHOUT regime (CONFLICT_FLAT has its own regex)
RE_PROD_ALL = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? \[PROD_SIGNAL_BREAKDOWN\] "
    r"(\w+) oss=([+-]?\d+)\(ev=([-0-9.]+)(?:,conf=([0-9.]+))?\) "
    r"(?:ev_sig=([+-]?\d+) )?shadow=([+-]?\d+)\(conf=([0-9.]+)\) "
    r"(?:regime=(\w+) )?reason=(\w+) final=([+-]?\d+) pc=([0-9.]+)"
)

# OSS SURFACE
RE_OSS_SURFACE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? \[OSS SURFACE\] "
    r"(\w+) ecdf=([0-9.]+) exec_drift=([-0-9.]+) .*? "
    r"regime=(\w+) p_cont=([0-9.]+)"
)

# EXPLORATION selected / triggered
RE_EXPLORATION_SELECT = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? \[EXPLORATION\] "
    r"(\w+): selected for forced entry"
)
RE_EXPLORATION_TRIGGER = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? \[EXPLORATION\] "
    r"(\w+): forced triggered=True"
)

# ORDER EXECUTING (actual trade)
RE_ORDER_EXEC = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? \[ORDER EXECUTING\] "
    r"(\w+) (BUY|SELL) price=([0-9.]+) vol=([0-9.]+)"
)

# Trade close / PnL
RE_TRADE_CLOSE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? "
    r"Sync: Closed trade \d+ .*? Exit price: ([0-9.]+), Profit: (\$[0-9.-]+)"
)

RE_THESIS_RESOLVE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? "
    r"\[THESIS_RESOLVE\] id=(\d+) (\w+) profit=([+-]?[0-9.]+)"
)

RE_B4_RESOLVE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? "
    r"\[B4_RESOLVE\] ticket=\d+ pnl=([+-]?[0-9.]+)"
)

RE_TODAY_PNL = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? Today PnL: (\$[0-9.-]+)"
)

# TOP3 direction audit (for cycle reference)
RE_TOP3 = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*? \[TOP3_DIRECTION_AUDIT\] "
    r"symbols=\[(.*?)\]"
)


def parse_ts(ts_str):
    return datetime.strptime(ts_str.split(",")[0], "%Y-%m-%d %H:%M:%S")


def load_log():
    """Load and parse the log file. Returns structured data dict."""
    if not LOG_PATH.exists():
        print(f"FATAL: {LOG_PATH} not found", file=sys.stderr)
        sys.exit(1)

    size_mb = LOG_PATH.stat().st_size / 1_024 / 1_024
    print(f"[1] Reading {LOG_PATH.name} ({size_mb:.0f} MB)...")

    data = {
        "conflicts": [],
        "prod_all": [],
        "oss_surfaces": [],
        "exploration_select": [],
        "exploration_trigger": [],
        "order_exec": [],
        "trade_close": [],
        "thesis_resolve": [],
        "b4_resolve": [],
        "today_pnl": [],
        "top3_audits": [],
    }

    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_CONFLICT.match(line)
            if m:
                data["conflicts"].append({
                    "timestamp": parse_ts(m.group(1)),
                    "symbol": m.group(2),
                    "oss_sig": int(m.group(3)),
                    "oss_ev": float(m.group(4)),
                    "shadow_sig": int(m.group(5)),
                    "final": int(m.group(6)),
                    "pc": float(m.group(7)),
                })
                continue

            m = RE_PROD_ALL.match(line)
            if m:
                data["prod_all"].append({
                    "timestamp": parse_ts(m.group(1)),
                    "symbol": m.group(2),
                    "oss_sig": int(m.group(3)),
                    "oss_ev": float(m.group(4)),
                    "oss_conf": float(m.group(5)) if m.group(5) else 0.0,
                    "ev_sig": int(m.group(6)) if m.group(6) else 0,
                    "shadow_sig": int(m.group(7)),
                    "shadow_conf": float(m.group(8)),
                    "regime": m.group(9) if m.group(9) else "UNKNOWN",
                    "reason": m.group(10),
                    "final": int(m.group(11)),
                    "pc": float(m.group(12)),
                })
                continue

            m = RE_OSS_SURFACE.match(line)
            if m:
                data["oss_surfaces"].append({
                    "timestamp": parse_ts(m.group(1)),
                    "symbol": m.group(2),
                    "ecdf": float(m.group(3)),
                    "exec_drift": float(m.group(4)),
                    "regime": m.group(5),
                    "p_cont": float(m.group(6)),
                })
                continue

            m = RE_EXPLORATION_SELECT.match(line)
            if m:
                data["exploration_select"].append({
                    "timestamp": parse_ts(m.group(1)),
                    "symbol": m.group(2),
                })
                continue

            m = RE_EXPLORATION_TRIGGER.match(line)
            if m:
                data["exploration_trigger"].append({
                    "timestamp": parse_ts(m.group(1)),
                    "symbol": m.group(2),
                })
                continue

            m = RE_ORDER_EXEC.match(line)
            if m:
                data["order_exec"].append({
                    "timestamp": parse_ts(m.group(1)),
                    "symbol": m.group(2),
                    "direction": m.group(3),
                    "price": float(m.group(4)),
                    "volume": float(m.group(5)),
                })
                continue

            m = RE_TRADE_CLOSE.match(line)
            if m:
                profit_str = m.group(3).replace("$", "")
                data["trade_close"].append({
                    "timestamp": parse_ts(m.group(1)),
                    "exit_price": float(m.group(2)),
                    "profit": float(profit_str),
                })
                continue

            m = RE_THESIS_RESOLVE.match(line)
            if m:
                data["thesis_resolve"].append({
                    "timestamp": parse_ts(m.group(1)),
                    "id": int(m.group(2)),
                    "symbol": m.group(3),
                    "profit": float(m.group(4)),
                })
                continue

            m = RE_TODAY_PNL.match(line)
            if m:
                pnl_str = m.group(2).replace("$", "")
                data["today_pnl"].append({
                    "timestamp": parse_ts(m.group(1)),
                    "pnl": float(pnl_str),
                })
                continue

    print(f"  Parsed: {len(data['conflicts'])} CONFLICT_FLAT, "
          f"{len(data['prod_all'])} PROD_ALL, "
          f"{len(data['oss_surfaces'])} OSS SURFACE, "
          f"{len(data['order_exec'])} ORDERS, "
          f"{len(data['exploration_trigger'])} EXPLORATION_TRIGGER, "
          f"{len(data['thesis_resolve'])} THESIS_RESOLVE")

    # Show date range
    all_dates = set()
    for c in data["conflicts"]:
        all_dates.add(c["timestamp"].date())
    print(f"  Date range (CONFLICT_FLAT): {min(all_dates)} to {max(all_dates)} "
          f"({len(all_dates)} unique days)")

    return data


def build_oss_index(oss_surfaces):
    """symbol -> sorted list of (timestamp, record)"""
    idx = defaultdict(list)
    for rec in oss_surfaces:
        idx[rec["symbol"]].append((rec["timestamp"], rec))
    for sym in idx:
        idx[sym].sort(key=lambda x: x[0])
    return idx


def find_nearest_oss(oss_idx, symbol, ts, max_seconds=60):
    """Find nearest OSS SURFACE for symbol within max_seconds."""
    entries = oss_idx.get(symbol, [])
    if not entries:
        return None
    lo, hi = 0, len(entries) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if entries[mid][0] < ts:
            lo = mid + 1
        elif entries[mid][0] > ts:
            hi = mid - 1
        else:
            return entries[mid][1]
    candidates = []
    for ix in (hi, lo):
        if 0 <= ix < len(entries):
            delta = abs((entries[ix][0] - ts).total_seconds())
            if delta <= max_seconds:
                candidates.append((delta, entries[ix][1]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def compute_conflict_durations(conflicts, all_prod):
    """
    Track conflict_duration[sym] that RESETS to 0 when a
    non-CONFLICT_FLAT PROD_SIGNAL_BREAKDOWN entry appears for that symbol.
    Returns list of (event, duration).
    """
    conflict_duration = defaultdict(int)
    by_sym_conflict = defaultdict(list)
    for c in conflicts:
        by_sym_conflict[c["symbol"]].append(c)
    for sym in by_sym_conflict:
        by_sym_conflict[sym].sort(key=lambda x: x["timestamp"])

    # Build per-symbol list of all PROD_SIGNAL_BREAKDOWN timestamps
    all_by_sym = defaultdict(list)
    for rec in all_prod:
        all_by_sym[rec["symbol"]].append(rec)
    for sym in all_by_sym:
        all_by_sym[sym].sort(key=lambda x: x["timestamp"])

    result = []
    for sym, entries in by_sym_conflict.items():
        other_entries = all_by_sym.get(sym, [])
        other_idx = 0

        for i, ev in enumerate(entries):
            # Count how many non-conflict entries have appeared since last conflict
            # This helps reset duration
            if i > 0:
                gap = (ev["timestamp"] - entries[i - 1]["timestamp"]).total_seconds()
                # Check if ANY non-CONFLICT_FLAT entry appeared between
                reset = False
                while other_idx < len(other_entries):
                    oe = other_entries[other_idx]
                    if oe["timestamp"] > entries[i - 1]["timestamp"] and oe["timestamp"] < ev["timestamp"]:
                        if oe["reason"] not in ("CONFLICT_FLAT",):
                            reset = True
                            break
                        other_idx += 1
                    elif oe["timestamp"] >= ev["timestamp"]:
                        break
                    else:
                        other_idx += 1

                if reset or gap >= 120:  # 2 min gap = new cycle
                    conflict_duration[sym] = 1
                else:
                    conflict_duration[sym] += 1
            else:
                conflict_duration[sym] = 1

            result.append((ev, conflict_duration[sym], sym))

    return result


def classify_event(conflict, duration, oss_rec):
    """Expanded 4-type classifier."""
    oss_ev = conflict["oss_ev"]
    abs_ev = abs(oss_ev)
    p_cont = oss_rec["p_cont"] if oss_rec else conflict.get("pc", 0.5)
    regime = oss_rec["regime"] if oss_rec else "UNKNOWN"

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


# ── Analysis ────────────────────────────────────────────────────────

TYPE_DESC = {
    "A_NOISE": "A_NOISE — transient noise, no action",
    "B_TRANSITION": "B_TRANSITION — regime shift, 50% entry allowed",
    "C_ACCUMULATION": "C_ACCUMULATION — smart-money accumulation, 70% entry allowed",
    "D_INDETERMINACY": "D_INDETERMINACY — persistent deadlock, no action",
}
TYPE_ORDER = ["A_NOISE", "B_TRANSITION", "C_ACCUMULATION", "D_INDETERMINACY"]
TYPE_SIZING = {"A_NOISE": 0.0, "B_TRANSITION": 0.5, "C_ACCUMULATION": 0.7, "D_INDETERMINACY": 0.0}


def main():
    data = load_log()
    conflicts = data["conflicts"]
    all_prod = data["prod_all"]
    oss_idx = build_oss_index(data["oss_surfaces"])
    order_execs = data["order_exec"]
    exploration_triggers = data["exploration_trigger"]
    thesis_resolves = data["thesis_resolve"]
    trade_closes = data["trade_close"]

    if not conflicts:
        print("FATAL: No CONFLICT_FLAT events found in log.")
        sys.exit(1)

    print(f"\n[2] Enriching {len(conflicts)} CONFLICT_FLAT events with OSS data...")

    enriched = []
    for event, duration, sym in compute_conflict_durations(conflicts, all_prod):
        oss_rec = find_nearest_oss(oss_idx, sym, event["timestamp"])
        cls = classify_event(event, duration, oss_rec)
        enriched.append({
            **event,
            "duration": duration,
            "class": cls,
            "oss_regime": oss_rec["regime"] if oss_rec else "UNKNOWN",
            "oss_p_cont": oss_rec["p_cont"] if oss_rec else event.get("pc", 0.5),
            "oss_ecdf": oss_rec["ecdf"] if oss_rec else None,
            "oss_exec_drift": oss_rec["exec_drift"] if oss_rec else None,
        })

    # ── Aggregate stats ──────────────────────────────────────────────

    type_counts = defaultdict(int)
    type_durations = defaultdict(list)
    type_regimes = defaultdict(lambda: defaultdict(int))
    type_symbols = defaultdict(lambda: defaultdict(int))
    type_abs_ev = defaultdict(list)
    type_p_cont = defaultdict(list)

    for rec in enriched:
        cls = rec["class"]
        type_counts[cls] += 1
        type_durations[cls].append(rec["duration"])
        type_regimes[cls][rec["oss_regime"]] += 1
        type_symbols[cls][rec["symbol"]] += 1
        type_abs_ev[cls].append(abs(rec["oss_ev"]))
        type_p_cont[cls].append(rec["oss_p_cont"])

    total_conflicts = len(enriched)
    total_type_count = sum(type_counts.values())

    # ── Opportunity estimation (Type B & C) ──────────────────────────

    # Group by unique (symbol, cycle_window) for deduplication
    bc_events = [r for r in enriched if r["class"] in ("B_TRANSITION", "C_ACCUMULATION")]
    bc_unique = set()
    b_unique = set()
    c_unique = set()
    for r in bc_events:
        cycle_key = (r["symbol"], r["timestamp"].strftime("%Y-%m-%d %H:%M"))
        bc_unique.add(cycle_key)
        if r["class"] == "B_TRANSITION":
            b_unique.add(cycle_key)
        else:
            c_unique.add(cycle_key)

    # ── Confusion table: what actually happened for Type B/C events ──

    print(f"\n[3] Building confusion table for Type B/C events...")

    # Map from order exec by symbol to track trades near conflict events
    order_exec_by_sym = defaultdict(list)
    for oe in order_execs:
        order_exec_by_sym[oe["symbol"]].append(oe)

    thesis_by_sym = defaultdict(list)
    for tr in thesis_resolves:
        thesis_by_sym[tr["symbol"]].append(tr)

    trade_close_by_time = sorted(trade_closes, key=lambda x: x["timestamp"])

    confusion_rows = []
    bc_executed = 0
    bc_pnl_total = 0.0
    bc_pnl_trades = []

    for rec in bc_events:
        sym = rec["symbol"]
        t0 = rec["timestamp"]
        t1 = t0 + timedelta(minutes=30)

        # Check for order execution within 30 min
        nearby_orders = [o for o in order_exec_by_sym.get(sym, [])
                         if t0 <= o["timestamp"] <= t1]
        executed = len(nearby_orders) > 0

        # Check for any exploration trigger
        nearby_explore = [e for e in exploration_triggers
                          if e["symbol"] == sym and t0 <= e["timestamp"] <= t1]

        # Check for thesis resolve (trade result) within 60 min
        nearby_thesis = [th for th in thesis_by_sym.get(sym, [])
                         if t0 <= th["timestamp"] <= t1 + timedelta(minutes=30)]

        pnl = 0.0
        for th in nearby_thesis:
            pnl += th["profit"]

        confusion_rows.append({
            "symbol": sym,
            "timestamp": t0,
            "duration": rec["duration"],
            "class": rec["class"],
            "oss_ev": rec["oss_ev"],
            "absorbed_ev": abs(rec["oss_ev"]),
            "p_cont": rec["oss_p_cont"],
            "regime": rec["oss_regime"],
            "executed": executed,
            "exploration_triggered": len(nearby_explore) > 0,
            "num_orders": len(nearby_orders),
            "thesis_resolved": len(nearby_thesis),
            "total_pnl": pnl,
            "oss_direction": "LONG" if rec["oss_ev"] > 0 else "SHORT",
        })

        if executed:
            bc_executed += 1
        if pnl != 0:
            bc_pnl_total += pnl
            bc_pnl_trades.append(pnl)

    # ── Type C high-value events (10+ cycles, high conviction) ──────

    c_high_value = [r for r in enriched
                    if r["class"] == "C_ACCUMULATION" and r["duration"] >= 10]

    # ── REGIME CORRELATION DEEP DIVE ────────────────────────────────

    # Per-regime classifier accuracy/breakdown
    regime_type_matrix = defaultdict(lambda: defaultdict(int))
    for rec in enriched:
        regime_type_matrix[rec["oss_regime"]][rec["class"]] += 1

    # ── CONFUSION TABLE: ALLOWED-by-classifier events outcome ───────

    # For events that would be ALLOWED (Type B/C), what happened?
    allowed_events = [r for r in enriched if r["class"] in ("B_TRANSITION", "C_ACCUMULATION")]
    allowed_executed = sum(1 for r in confusion_rows if r["executed"])
    allowed_not_executed = len(allowed_events) - allowed_executed

    # ── FORWARD ANALYSIS: Price direction validation ─────────────────

    # For each Type B/C event, check if the market subsequently moved
    # in the OSS direction or the Shadow direction by looking at
    # future OSS SURFACE entries for the same symbol.

    def analyze_forward_price(oss_idx, symbol, t0, oss_direction, shadow_sig, max_lookahead=10):
        """Look forward max_lookahead OSS entries for symbol and check
        if p_cont increased (validating OSS) or decreased."""
        entries = oss_idx.get(symbol, [])
        if not entries:
            return None, None, None
        # Find position
        start = None
        for i, (ts, _) in enumerate(entries):
            if ts >= t0:
                start = i
                break
        if start is None:
            return None, None, None

        future = entries[start:start + max_lookahead + 1]
        if len(future) < 2:
            return None, None, None

        initial_p_cont = future[0][1]["p_cont"]
        final_p_cont = future[-1][1]["p_cont"]
        delta_p_cont = final_p_cont - initial_p_cont

        # Determine if price moved in OSS or Shadow direction
        # If oss_direction is LONG (oss_ev > 0) and p_cont increases -> OSS validated
        # If oss_direction is SHORT (oss_ev < 0) and p_cont decreases -> OSS validated
        if oss_direction == "LONG":
            oss_validated = delta_p_cont > 0.05
            shadow_validated = delta_p_cont < -0.05
        else:
            oss_validated = delta_p_cont < -0.05
            shadow_validated = delta_p_cont > 0.05

        return delta_p_cont, oss_validated, shadow_validated

    forward_results = []
    for row in confusion_rows:
        d_p_cont, oss_val, shd_val = analyze_forward_price(
            oss_idx, row["symbol"], row["timestamp"],
            row["oss_direction"], None
        )
        row["forward_delta_p_cont"] = d_p_cont
        row["oss_validated"] = oss_val
        row["shadow_validated"] = shd_val
        forward_results.append(row)

    # Count validation statistics
    oss_validated_count = sum(1 for r in forward_results if r.get("oss_validated"))
    shadow_validated_count = sum(1 for r in forward_results if r.get("shadow_validated"))
    no_clear_direction = len(forward_results) - oss_validated_count - shadow_validated_count

    # ── Build output ────────────────────────────────────────────────

    lines = []
    def w(s=""):
        lines.append(s)

    w("=" * 78)
    w("  CONFLICT CLASSIFIER EXPANDED ANALYSIS — FULL REPORT")
    w(f"  Log: {LOG_PATH.name}")
    w(f"  Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"  Total CONFLICT_FLAT events: {total_conflicts}")
    w("=" * 78)

    # 1. Type distribution
    w()
    w("─" * 78)
    w("  1. TYPE DISTRIBUTION (all dates)")
    w("─" * 78)
    w(f"  {'Type':<30} {'Count':>8} {'%':>7}")
    w(f"  {'─' * 30} {'─' * 8} {'─' * 7}")
    for t in TYPE_ORDER:
        cnt = type_counts.get(t, 0)
        pct = cnt / total_conflicts * 100 if total_conflicts else 0
        w(f"  {TYPE_DESC[t]:<30} {cnt:>8} {pct:>6.1f}%")
    w(f"  {'─' * 30} {'─' * 8} {'─' * 7}")
    w(f"  {'TOTAL':<30} {total_conflicts:>8} 100.0%")

    w()
    bc_allowed = sum(type_counts.get(t, 0) for t in ("B_TRANSITION", "C_ACCUMULATION"))
    bc_pct = bc_allowed / total_conflicts * 100 if total_conflicts else 0
    w(f"  Actionable (B+C): {bc_allowed} / {total_conflicts} ({bc_pct:.1f}%)")

    # 2. Duration distribution per type
    w()
    w("─" * 78)
    w("  2. DURATION DISTRIBUTION PER TYPE")
    w("─" * 78)
    w(f"  {'Type':<20} {'Count':>8} {'Mean':>8} {'Median':>8} {'Max':>6} {'StdDev':>8} {'≥5':>6} {'≥10':>6}")
    w(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 6} {'─' * 8} {'─' * 6} {'─' * 6}")
    for t in TYPE_ORDER:
        durs = type_durations.get(t, [])
        if durs:
            m = mean(durs)
            med = median(durs)
            mx = max(durs)
            sd = stdev(durs) if len(durs) > 1 else 0.0
            gt5 = sum(1 for d in durs if d >= 5)
            gt10 = sum(1 for d in durs if d >= 10)
            w(f"  {t:<20} {len(durs):>8} {m:>8.2f} {med:>8.0f} {mx:>6} {sd:>8.2f} {gt5:>6} {gt10:>6}")
        else:
            w(f"  {t:<20} {0:>8} {'—':>8} {'—':>8} {'—':>6} {'—':>8} {'—':>6} {'—':>6}")

    # 3. Symbols most prone to each type
    w()
    w("─" * 78)
    w("  3. SYMBOLS MOST PRONE TO EACH TYPE (top 7)")
    w("─" * 78)
    for t in TYPE_ORDER:
        sym_counts = type_symbols.get(t, {})
        if not sym_counts:
            continue
        sorted_syms = sorted(sym_counts.items(), key=lambda x: -x[1])[:7]
        top7 = ", ".join(f"{s}({c})" for s, c in sorted_syms)
        t_total = sum(sym_counts.values())
        w(f"  {t:<20} ({t_total:>4} total)  →  {top7}")

    # 4. Regime correlation
    w()
    w("─" * 78)
    w("  4. REGIME CORRELATION PER TYPE")
    w("─" * 78)
    all_regimes = sorted({r for rec in enriched for r in [rec["oss_regime"]]})
    header = f"  {'Type':<20}"
    for r in all_regimes:
        header += f" {r:>22}"
    w(header)
    w(f"  {'─' * 20} {'─' * (22 * len(all_regimes))}")
    for t in TYPE_ORDER:
        reg_counts = type_regimes.get(t, {})
        t_total = sum(reg_counts.values()) or 1
        line = f"  {t:<20}"
        for r in all_regimes:
            cnt = reg_counts.get(r, 0)
            pct = cnt / t_total * 100 if t_total else 0
            if cnt > 0:
                line += f" {cnt:>4}({pct:>4.0f}%)"
            else:
                line += f" {'—':>10}"
        w(line)

    # 5. Type B & C opportunity analysis
    w()
    w("─" * 78)
    w("  5. TYPE B & C — OPPORTUNITY ANALYSIS")
    w("─" * 78)
    w(f"\n  Type B unique trade opportunities: {len(b_unique)}  (sizing: 50%)")
    w(f"  Type C unique trade opportunities: {len(c_unique)}  (sizing: 70%)")
    w(f"  Total BC opportunities: {len(bc_unique)}")
    w()
    b_vol = len(b_unique) * 0.5
    c_vol = len(c_unique) * 0.7
    total_vol = b_vol + c_vol
    w(f"  Additional volume (standard units):")
    w(f"    Type B:  {len(b_unique)} × 0.5 = {b_vol:.1f}")
    w(f"    Type C:  {len(c_unique)} × 0.7 = {c_vol:.1f}")
    w(f"    Total:   {total_vol:.1f} standard units")
    w()
    w(f"  Estimated additional trades: {len(bc_unique)} "
      f"(vs. 0 from current system for CONFLICT_FLAT)")

    # 6. Type C high-value events
    w()
    w("─" * 78)
    w("  6. TYPE C ACCUMULATION — HIGH VALUE EVENTS (10+ cycles)")
    w("─" * 78)
    if c_high_value:
        w(f"\n  Found {len(c_high_value)} high-value accumulation events:")
        w(f"\n  {'Symbol':<10} {'Date':<22} {'Dur':>4} {'OSS_ev':>10} {'p_cont':>7} {'Regime':<20}")
        w(f"  {'─' * 10} {'─' * 22} {'─' * 4} {'─' * 10} {'─' * 7} {'─' * 20}")
        for r in c_high_value[:20]:
            w(f"  {r['symbol']:<10} {r['timestamp'].strftime('%Y-%m-%d %H:%M:%S'):<22} "
              f"{r['duration']:>4} {r['oss_ev']:>+10.4f} {r['oss_p_cont']:>7.2f} "
              f"{r['oss_regime']:<20}")
        if len(c_high_value) > 20:
            w(f"  ... and {len(c_high_value) - 20} more")
    else:
        w(f"\n  No Type C events with duration ≥ 10 cycles found.")
        # Check for duration >= 5
        c_moderate = [r for r in enriched if r["class"] == "C_ACCUMULATION" and r["duration"] >= 5]
        if c_moderate:
            w(f"  However, {len(c_moderate)} events with duration ≥ 5 cycles exist:")
            for r in c_moderate[:10]:
                w(f"    {r['symbol']:<8} dur={r['duration']:>2}  ev={r['oss_ev']:>+8.4f}  "
                  f"p_cont={r['oss_p_cont']:.2f}  {r['oss_regime']}")
        else:
            w(f"  No Type C events with duration ≥ 5 cycles either.")

    # 7. Confusion table
    w()
    w("─" * 78)
    w("  7. CONFUSION TABLE — ALLOWED EVENTS (Type B/C) ACTUAL OUTCOMES")
    w("─" * 78)
    w()
    w(f"  Total Type B/C events: {len(allowed_events)}")
    w(f"  Events with subsequent ORDER EXECUTION: {allowed_executed}")
    w(f"  Events WITHOUT subsequent execution: {allowed_not_executed}")
    w()
    if allowed_executed:
        w(f"  Event details (executed only):")
        w(f"  {'Symbol':<8} {'Date':<20} {'Class':<18} {'Ev':>8} {'Dur':>3} "
          f"{'Orders':>6} {'PnL':>10}")
        w(f"  {'─' * 8} {'─' * 20} {'─' * 18} {'─' * 8} {'─' * 3} "
          f"{'─' * 6} {'─' * 10}")
        for row in confusion_rows:
            if row["executed"]:
                w(f"  {row['symbol']:<8} {row['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{row['class']:<18} {row['oss_ev']:>+8.4f} {row['duration']:>3} "
                  f"{row['num_orders']:>6} {row['total_pnl']:>+10.2f}")
    w()
    w(f"  PnL from BC-allowed, executed events:")
    if bc_pnl_trades:
        w(f"    Total PnL: ${bc_pnl_total:.2f}")
        w(f"    Mean PnL:  ${mean(bc_pnl_trades):.2f}" if len(bc_pnl_trades) > 0 else "    No PnL data")
    else:
        w(f"    No non-zero PnL trades recorded (simulation mode — all PnL = $0.00)")

    w()
    w("── Forward Price Validation (next OSS entries) ──")
    w(f"  Of {len(forward_results)} Type B/C events:")
    w(f"    OSS direction validated by subsequent p_cont movement:  {oss_validated_count} "
      f"({oss_validated_count/len(forward_results)*100:.0f}%)" if forward_results else "    No data")
    w(f"    Shadow direction validated by subsequent p_cont movement: {shadow_validated_count} "
      f"({shadow_validated_count/len(forward_results)*100:.0f}%)" if forward_results else "")
    w(f"    No clear direction: {no_clear_direction}")
    w()

    # Show the validation details for executed events
    executed_forward = [r for r in forward_results if r.get("executed")]
    if executed_forward:
        w(f"  Forward validation — EXECUTED events only:")
        w(f"  {'Symbol':<8} {'Class':<18} {'Exec?':>6} {'Δp_cont':>9} {'OSS_ok?':>8} {'Shad_ok?':>9}")
        w(f"  {'─' * 8} {'─' * 18} {'─' * 6} {'─' * 9} {'─' * 8} {'─' * 9}")
        for r in executed_forward:
            dp = r.get("forward_delta_p_cont")
            dp_str = f"{dp:>+7.3f}" if dp is not None else "   N/A"
            w(f"  {r['symbol']:<8} {r['class']:<18} "
              f"{'Y':>6} {dp_str:>9} "
              f"{'✓' if r.get('oss_validated') else '✗':>8} "
              f"{'✓' if r.get('shadow_validated') else '✗':>9}")

    # 7b. Trade lifecycle — full details for executed events
    w()
    w("── Trade Lifecycle — Executed Events ──")
    for r in executed_forward:
        sym = r["symbol"]
        t0 = r["timestamp"]
        t1 = t0 + timedelta(minutes=30)

        # Find the executed orders
        orders = [o for o in order_execs
                  if o["symbol"] == sym and t0 <= o["timestamp"] <= t1]
        if orders:
            w(f"\n  {sym} at {t0.strftime('%H:%M:%S')} | "
              f"Class={r['class']} | "
              f"OSS={r['oss_direction']}(ev={r['oss_ev']:+.4f}) | "
              f"Shadow={'LONG' if r.get('p_cont', 0) > 0.5 else 'SHORT'}")
            for o in orders:
                # Find matching thesis resolve
                resolve_ts = o["timestamp"] + timedelta(minutes=60)
                matches = [tr for tr in thesis_resolves
                          if tr["symbol"] == sym and tr["timestamp"] <= resolve_ts
                          and tr["timestamp"] >= o["timestamp"]]
                pnl_str = f"PnL=+{matches[0]['profit']:.2f}" if matches else "PnL=N/A"
                w(f"    ORDER: {o['direction']:>4} @ {o['price']:.5f} "
                  f"vol={o['volume']:.2f} {pnl_str}")
                if matches:
                    w(f"    RESOLVE: {matches[0]['timestamp'].strftime('%H:%M:%S')} "
                      f"profit={matches[0]['profit']:+.2f}")

    # 8. Disagreement direction analysis
    w()
    w("─" * 78)
    w("  8. DIRECTION DISAGREEMENT ANALYSIS (OSS vs Shadow)")
    w("─" * 78)
    w()
    oss_long = sum(1 for r in enriched if r["oss_ev"] > 0)
    oss_short = sum(1 for r in enriched if r["oss_ev"] < 0)
    shadow_long = sum(1 for r in enriched if r["shadow_sig"] > 0)
    shadow_short = sum(1 for r in enriched if r["shadow_sig"] < 0)
    oss_strong = sum(1 for r in enriched if abs(r["oss_ev"]) >= 0.8)
    oss_moderate = sum(1 for r in enriched if 0.6 <= abs(r["oss_ev"]) < 0.8)
    oss_weak = sum(1 for r in enriched if 0.3 <= abs(r["oss_ev"]) < 0.6)
    oss_negligible = sum(1 for r in enriched if abs(r["oss_ev"]) < 0.3)

    w(f"  OSS conviction distribution:")
    w(f"    Strong (≥0.80):     {oss_strong:>4} ({oss_strong/total_conflicts*100:>5.1f}%)")
    w(f"    Moderate (0.60-0.79): {oss_moderate:>4} ({oss_moderate/total_conflicts*100:>5.1f}%)")
    w(f"    Weak (0.30-0.59):   {oss_weak:>4} ({oss_weak/total_conflicts*100:>5.1f}%)")
    w(f"    Negligible (<0.30): {oss_negligible:>4} ({oss_negligible/total_conflicts*100:>5.1f}%)")
    w()
    w(f"  OSS direction: LONG={oss_long} SHORT={oss_short}")
    w(f"  Shadow direction: LONG={shadow_long} SHORT={shadow_short}")

    # 9. Regime deep dive
    w()
    w("─" * 78)
    w("  9. REGIME BREAKDOWN — PER-REGIME TYPE DISTRIBUTION")
    w("─" * 78)
    w()
    for regime in sorted(regime_type_matrix.keys()):
        r_counts = dict(regime_type_matrix[regime])
        r_total = sum(r_counts.values())
        w(f"  {regime:<25} (total: {r_total:>4})")
        for t in TYPE_ORDER:
            cnt = r_counts.get(t, 0)
            pct = cnt / r_total * 100 if r_total else 0
            if cnt > 0:
                w(f"    {t:<20} {cnt:>4} ({pct:>5.1f}%)")
        w()

    # 10. Summary / key findings
    w()
    w("=" * 78)
    w("  10. KEY FINDINGS & RECOMMENDATIONS")
    w("=" * 78)
    w()
    a_pct = type_counts.get("A_NOISE", 0) / total_conflicts * 100
    b_pct = type_counts.get("B_TRANSITION", 0) / total_conflicts * 100
    c_pct = type_counts.get("C_ACCUMULATION", 0) / total_conflicts * 100
    d_pct = type_counts.get("D_INDETERMINACY", 0) / total_conflicts * 100

    w(f"  1. {a_pct:.1f}% of conflicts are Type A (noise) — correct to ignore")
    w(f"  2. {b_pct:.1f}% are Type B (transition) — potential 50%-sized entries")
    w(f"  3. {c_pct:.1f}% are Type C (accumulation) — highest-value opportunities")
    w(f"  4. {d_pct:.1f}% are Type D (indeterminacy) — correct to hold")
    w(f"  5. Actionable events (B+C): {bc_allowed} ({bc_pct:.1f}%)")
    w(f"  6. Estimated additional volume: {total_vol:.1f} std units "
      f"({len(bc_unique)} trades)")
    if c_high_value:
        w(f"  7. HIGH-VALUE: {len(c_high_value)} Type C events with 10+ cycle duration")
    if bc_pnl_trades:
        w(f"  8. Realized PnL from BC events: ${bc_pnl_total:.2f} total, "
          f"${mean(bc_pnl_trades):.2f} avg per trade")
    else:
        w(f"  8. PnL data not available (simulation mode — all trades $0.00)")

    # ── Write results ──────────────────────────────────────────────

    # Tabular data
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("symbol\ttimestamp\tduration\tclass\toss_ev\tabsorbed_ev\t"
                "shadow_sig\tregime\tp_cont\tecdf\t"
                "oss_exec_drift\tfinal\n")
        for rec in enriched:
            f.write(f"{rec['symbol']}\t{rec['timestamp']}\t{rec['duration']}\t"
                    f"{rec['class']}\t{rec['oss_ev']:.4f}\t{abs(rec['oss_ev']):.4f}\t"
                    f"{rec['shadow_sig']:+d}\t{rec['oss_regime']}\t"
                    f"{rec['oss_p_cont']:.4f}\t{rec['oss_ecdf'] or 'N/A'}\t"
                    f"{rec['oss_exec_drift'] or 'N/A'}\t{rec['final']:+d}\n")

    # Confusion table export
    confusion_path = RESULTS_PATH.parent / "confusion_table.csv"
    with open(confusion_path, "w", encoding="utf-8") as f:
        f.write("symbol,timestamp,duration,class,oss_ev,absorbed_ev,p_cont,regime,"
                "executed,exploration_triggered,num_orders,thesis_resolved,total_pnl,oss_direction\n")
        for row in confusion_rows:
            f.write(f"{row['symbol']},{row['timestamp']},{row['duration']},"
                    f"{row['class']},{row['oss_ev']:.4f},{row['absorbed_ev']:.4f},"
                    f"{row['p_cont']:.4f},{row['regime']},"
                    f"{row['executed']},{row['exploration_triggered']},{row['num_orders']},"
                    f"{row['thesis_resolved']},{row['total_pnl']:.2f},{row['oss_direction']}\n")

    # Full report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[4] Results written to:")
    print(f"    Tabular:  {RESULTS_PATH}")
    print(f"    Confusion: {confusion_path}")
    print(f"    Report:   {REPORT_PATH}")

    # Print report to stdout
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
