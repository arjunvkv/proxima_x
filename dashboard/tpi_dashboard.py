"""TPI Flow Overlay Dashboard — Layer 7 observability panel.

Shadow mode: tracks TPI signals, alignments, conflicts with no execution impact.
Includes incremental edge attribution and live decay tracking.
"""
import time

from datetime import datetime

# Shadow trackers
_TPI_SHADOW_LOG = []
_TPI_ALIGNMENTS = 0
_TPI_CONFLICTS = 0
_TPI_VETO_OPPORTUNITIES = 0
_TPI_WEAK_ALIGNMENTS = 0
_TPI_CACHED_SIGNALS = {}

# ————————————————————————————————————————
# Incremental Edge Attribution
# ————————————————————————————————————————
_TPI_BASE_WINS = 0
_TPI_BASE_LOSSES = 0
_TPI_ALIGNED_WINS = 0
_TPI_ALIGNED_LOSSES = 0
_TPI_CONFLICT_WINS = 0
_TPI_CONFLICT_LOSSES = 0
_TPI_VETO_AVOIDED_LOSSES = 0
_TPI_TRADE_OUTCOMES = []  # historical log


def record_tpi_shadow(symbol, tpi_signal, alignment, existing_direction_str):
    """Record a TPI shadow observation for metrics."""
    global _TPI_ALIGNMENTS, _TPI_CONFLICTS, _TPI_VETO_OPPORTUNITIES, _TPI_WEAK_ALIGNMENTS
    if alignment == "MATCH":
        _TPI_ALIGNMENTS += 1
    elif alignment == "CONFLICT":
        _TPI_CONFLICTS += 1
    elif alignment == "WEAK":
        _TPI_WEAK_ALIGNMENTS += 1

    if alignment == "CONFLICT" and tpi_signal.get("eligible"):
        _TPI_VETO_OPPORTUNITIES += 1

    _TPI_SHADOW_LOG.append({
        "time": time.time(),
        "symbol": symbol,
        "tpi": tpi_signal.get("tpi"),
        "direction": tpi_signal.get("direction_label"),
        "confidence": tpi_signal.get("confidence"),
        "percentile": tpi_signal.get("percentile"),
        "eligible": tpi_signal.get("eligible"),
        "session_ok": tpi_signal.get("session_ok"),
        "alignment": alignment,
        "existing_direction": existing_direction_str,
        "used_as_outcome": False,
    })
    if len(_TPI_SHADOW_LOG) > 100:
        _TPI_SHADOW_LOG.pop(0)


def record_trade_outcome(symbol, direction_label, alignment, was_profitable):
    """Record a completed trade outcome, grouped by TPI alignment.

    Args:
        symbol: Instrument symbol
        direction_label: TPI direction label at entry time
        alignment: MATCH / CONFLICT / WEAK / NO_SIGNAL
        was_profitable: True if trade closed with positive PnL
    """
    global _TPI_BASE_WINS, _TPI_BASE_LOSSES
    global _TPI_ALIGNED_WINS, _TPI_ALIGNED_LOSSES
    global _TPI_CONFLICT_WINS, _TPI_CONFLICT_LOSSES
    global _TPI_VETO_AVOIDED_LOSSES

    if was_profitable:
        _TPI_BASE_WINS += 1
        if alignment == "MATCH":
            _TPI_ALIGNED_WINS += 1
        elif alignment == "CONFLICT":
            _TPI_CONFLICT_WINS += 1
    else:
        _TPI_BASE_LOSSES += 1
        if alignment == "MATCH":
            _TPI_ALIGNED_LOSSES += 1
        elif alignment == "CONFLICT":
            _TPI_CONFLICT_LOSSES += 1
            # Hypothetical: TPI conflict would have vetoed this losing trade
            _TPI_VETO_AVOIDED_LOSSES += 1

    _TPI_TRADE_OUTCOMES.append({
        "time": time.time(),
        "symbol": symbol,
        "tpi_dir": direction_label,
        "alignment": alignment,
        "outcome": "WIN" if was_profitable else "LOSS",
    })


def cache_tpi_signal(symbol, tpi_signal):
    """Cache latest TPI signal per symbol for dashboard display."""
    _TPI_CACHED_SIGNALS[symbol] = {
        **tpi_signal,
        "cached_at": time.time(),
    }


def generate(eligible_symbols=None, tracker=None, persistence=None, curvature=None):
    """Generate TPI dashboard panel text for terminal display."""
    lines = []
    lines.append("=" * 52)
    lines.append("  LAYER 7 — TPI FLOW OVERLAY (SHADOW MODE)")
    lines.append("=" * 52)
    lines.append("")

    if eligible_symbols is None:
        from layer7.get_tpi_signal import TPI_ELIGIBLE
        eligible_symbols = TPI_ELIGIBLE

    # Per-symbol TPI snapshot
    lines.append("  SYMBOL        TPI      DIR     CONF    PCT   SESSION  ELIG  ALIGN")
    for sym in eligible_symbols:
        sig = _TPI_CACHED_SIGNALS.get(sym, {})
        if not sig:
            tpi_str = "---"
            dir_str = "---"
            conf_str = "---"
            pct_str = "---"
            session_str = "---"
            elig_str = "---"
            align_str = "---"
        else:
            tpi_str = f"{sig.get('tpi', 0):+.5f}"
            dir_str = f"{sig.get('direction_label', '?'):>4s}"
            conf_str = f"{sig.get('confidence', 0):.5f}"
            pct_str = f"{sig.get('percentile', 0):.0f}" if sig.get("percentile") is not None else "N/A"
            session_str = f"{sig.get('session_name', '?')[:4]}" if sig.get("session_name") else "ANY"
            elig_str = "YES" if sig.get("eligible") else "NO"
            align_str = f"{sig.get('alignment', '?'):>6s}" if sig.get("alignment") else "---"
        lines.append(f"  {sym:<12s} {tpi_str:>9s} {dir_str:>6s} {conf_str:>8s} {pct_str:>5s} {session_str:>4s}  {elig_str:>4s} {align_str}")

        # Append persistence + curvature inline if available
        pers = sig.get("persistence")
        curv = sig.get("curvature")
        if pers or curv:
            extra = ""
            if pers:
                extra += f"  P:{pers.get('streak', 0)}p"
            if curv:
                extra += f"  C:{curv.get('state', '?')[:5]}"
            if extra:
                lines.append(f"  {'':12s} {extra}")
    lines.append("")

    # ————————————————————————————————————————
    # Persistence & Curvature summary table
    # ————————————————————————————————————————
    if persistence is not None or curvature is not None:
        lines.append("  PERSISTENCE & CURVATURE")
        lines.append("-" * 52)
        for sym in eligible_symbols:
            p_state = persistence.state(sym) if persistence else {}
            c_state = curvature.state(sym) if curvature else {}
            ps = f"streak={p_state.get('streak', 0):>2d}"
            cs = f"{c_state.get('state', '?')[:10]:>10s}" if isinstance(c_state.get('state'), str) else "NEUTRAL"
            lines.append(f"  {sym:<10s}  {ps}  curv={cs}")
        lines.append("")

    # ... rest of function remains ...

    # ————————————————————————————————————————
    # Incremental Edge Attribution
    # ————————————————————————————————————————
    n_base = _TPI_BASE_WINS + _TPI_BASE_LOSSES
    n_aligned = _TPI_ALIGNED_WINS + _TPI_ALIGNED_LOSSES
    n_conflict = _TPI_CONFLICT_WINS + _TPI_CONFLICT_LOSSES

    lines.append("  INCREMENTAL EDGE ATTRIBUTION")
    lines.append("-" * 52)
    base_wr = _TPI_BASE_WINS / n_base * 100 if n_base > 0 else 0
    aligned_wr = _TPI_ALIGNED_WINS / n_aligned * 100 if n_aligned > 0 else 0
    conflict_wr = _TPI_CONFLICT_WINS / n_conflict * 100 if n_conflict > 0 else 0
    lines.append(f"  Base Win Rate:          {_TPI_BASE_WINS:>3d}/{n_base:<3d} ({base_wr:5.1f}%)")
    lines.append(f"  TPI-Aligned W/R:        {_TPI_ALIGNED_WINS:>3d}/{n_aligned:<3d} ({aligned_wr:5.1f}%)")
    lines.append(f"  TPI-Conflict W/R:       {_TPI_CONFLICT_WINS:>3d}/{n_conflict:<3d} ({conflict_wr:5.1f}%)")
    if n_base > 0 and n_aligned > 0:
        delta = aligned_wr - base_wr
        lines.append(f"  Edge Delta (Aligned):   {delta:+5.1f}%")
    if n_base > 0 and n_conflict > 0:
        delta_c = conflict_wr - base_wr
        lines.append(f"  Edge Delta (Conflict):  {delta_c:+5.1f}%")
    lines.append(f"  Veto Avoided Losses:    {_TPI_VETO_AVOIDED_LOSSES:>3d}")
    lines.append("")

    # ————————————————————————————————————————
    # Promotion Gate Status
    # ————————————————————————————————————————
    lines.append("  PROMOTION GATE STATUS")
    lines.append("-" * 52)
    gates_passed = 0
    n_eligible = len([o for o in _TPI_TRADE_OUTCOMES if o.get("alignment") in ("MATCH", "CONFLICT")])
    if n_eligible >= 50:
        gate_a = aligned_wr >= base_wr + 5.0
        gate_b = conflict_wr <= base_wr - 5.0 or n_conflict == 0
        gate_c = n_eligible >= 50
    else:
        gate_a = gate_b = False
        gate_c = n_eligible >= 50

    if gate_a:
        gates_passed += 1
    if gate_b:
        gates_passed += 1
    if gate_c:
        gates_passed += 1

    lines.append(f"  Gate A (Aligned > Base +5%):  {'PASS' if gate_a else 'PENDING'} ({aligned_wr:.1f}% vs {base_wr:.1f}%)")
    lines.append(f"  Gate B (Conflict < Base -5%): {'PASS' if gate_b else 'PENDING'} ({conflict_wr:.1f}% vs {base_wr:.1f}%)")
    lines.append(f"  Gate C (n >= 50):              {'PASS' if gate_c else 'PENDING'} ({n_eligible} eligible obs)")
    lines.append(f"  Gates Passed: {gates_passed}/3")
    lines.append("")

    # Shadow metrics
    total_shadow = _TPI_ALIGNMENTS + _TPI_CONFLICTS + _TPI_WEAK_ALIGNMENTS
    lines.append("  SHADOW OBSERVATIONS")
    lines.append("-" * 52)
    if total_shadow > 0:
        align_pct = _TPI_ALIGNMENTS / total_shadow * 100
        conflict_pct = _TPI_CONFLICTS / total_shadow * 100
        weak_pct = _TPI_WEAK_ALIGNMENTS / total_shadow * 100
    else:
        align_pct = conflict_pct = weak_pct = 0.0
    lines.append(f"  Alignments:     {_TPI_ALIGNMENTS:>4d} ({align_pct:5.1f}%)")
    lines.append(f"  Conflicts:      {_TPI_CONFLICTS:>4d} ({conflict_pct:5.1f}%)")
    lines.append(f"  Weak Align:     {_TPI_WEAK_ALIGNMENTS:>4d} ({weak_pct:5.1f}%)")
    lines.append(f"  No Signal:      {len(_TPI_SHADOW_LOG) - total_shadow:>4d}")
    lines.append(f"  Eligible Veto:  {_TPI_VETO_OPPORTUNITIES:>4d}")
    lines.append(f"  Total Obs:      {len(_TPI_SHADOW_LOG):>4d}")
    lines.append("")

    # Last observation
    if _TPI_SHADOW_LOG:
        last = _TPI_SHADOW_LOG[-1]
        lines.append("  LAST SHADOW OBSERVATION")
        ts = last.get('time')
        ts_str = time.strftime('%H:%M:%S', time.localtime(ts)) if ts else "N/A"
        tpi_val = last.get('tpi') or 0.0
        pct_val = last.get('percentile') or 0
        lines.append(f"  [{ts_str}] "
                     f"{last.get('symbol', '?')} | TPI={tpi_val:+.5f} "
                     f"| {last.get('direction', '?')} | P{pct_val} "
                     f"| {last.get('alignment', '?')} vs {last.get('existing_direction', '?')}")
    else:
        lines.append("  No shadow observations yet.")
    lines.append("")

    # ————————————————————————————————————————
    # Live Decay Panel
    # ————————————————————————————————————————
    lines.append("  LIVE DECAY (H1 / H3 directional hit)")
    lines.append("-" * 52)
    if tracker is not None and hasattr(tracker, "live_decay_stats"):
        stats = tracker.live_decay_stats()
        h1 = f"{stats['h1_hit_rate']:.1f}%" if stats["h1_hit_rate"] is not None else "pending"
        h3 = f"{stats['h3_hit_rate']:.1f}%" if stats["h3_hit_rate"] is not None else "pending"
        h1_n = stats["h1_resolved"]
        h3_n = stats["h3_resolved"]
        age = f"{stats['avg_signal_age_bars']:.1f} bars" if stats["avg_signal_age_bars"] else "pending"
        hl = f"{stats['half_life_bars']:.2f} bars" if stats["half_life_bars"] is not None else "pending"
        lines.append(f"  H1 Hit Rate:    {h1:>8s}  ({h1_n} resolved)")
        lines.append(f"  H3 Hit Rate:    {h3:>8s}  ({h3_n} resolved)")
        lines.append(f"  Signal Age:     {age}")
        lines.append(f"  Half-Life:      {hl}")
    else:
        if _TPI_TRADE_OUTCOMES:
            n_outcomes = len(_TPI_TRADE_OUTCOMES)
            wins = sum(1 for o in _TPI_TRADE_OUTCOMES if o["outcome"] == "WIN")
            losses = n_outcomes - wins
            h1_hit = wins / n_outcomes * 100 if n_outcomes > 0 else 0
            lines.append(f"  Trade Outcomes: {wins}W / {losses}L ({h1_hit:.1f}%)")
        lines.append("  H1 Hit Rate:    —  (requires tick + bar sync)")
        lines.append("  H3 Hit Rate:    —  (requires tick + bar sync)")
        lines.append("  Signal Age:     —  (requires tick + bar sync)")
        lines.append("  Half-Life:      —  (requires tick + bar sync)")
    lines.append("")
    return "\n".join(lines)
