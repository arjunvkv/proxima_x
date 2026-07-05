"""Dashboard rendering — full dashboard text builder and terminal printer.

All functions take the ProximaDemo instance as their first parameter (named `demo`)
and are stateless: they read from `demo.*` attributes but never mutate system state
except for the spinner index and activity log (which are UI state only).
"""

import os
import sys
import logging
from datetime import datetime

from layer7.get_tpi_signal import TPI_ELIGIBLE
from dashboard.tpi_dashboard import generate as tpi_dashboard_generate
from proxima_ops.governance.system_mode_contract import SystemMode, UIMode

_SYSTEM_MODE = SystemMode()

logger = logging.getLogger("proxima_demo")


def add_activity(demo, msg: str):
    """Append a formatted message to the in-memory activity log (max 6 entries)."""
    ignore_terms = [
        "DAILY REPORT", "WEEKLY REPORT", "====", "----", "DEPLOYMENT",
        "PERFORMANCE", "SIGNALS", "OPEN POSITIONS", "Markets closed",
        "Mapping symbol", "Warming up price buffers", "Initialized buffer",
        "Initializing metadata", "Syncing trade ledger", "Sync: Position"
    ]
    if any(term in msg for term in ignore_terms):
        return

    formatted_msg = msg
    if "Order failed for" in msg:
        try:
            parts = msg.split("Order failed for ")[1].split(": ")
            symbol = parts[0]
            details = parts[1]
            formatted_msg = f"FAILED: {symbol} | {details}"
        except Exception:
            pass
    elif "Sync: Closed trade" in msg:
        try:
            trade_id = msg.split("Closed trade ")[1].split(" ")[0]
            ticket = msg.split("(ticket ")[1].split(")")[0]
            exit_p = msg.split("Exit price: ")[1].split(",")[0]
            profit = msg.split("Profit: ")[1]
            formatted_msg = f"CLOSED: Trade #{trade_id} (Ticket {ticket}) | Exit: {exit_p} | PnL: {profit}"
        except Exception:
            pass
    elif "Spread too high for" in msg:
        try:
            symbol = msg.split("Spread too high for ")[1].split(",")[0]
            formatted_msg = f"BLOCKED: {symbol} | Spread too high"
        except Exception:
            pass
    elif "H20 EXIT: Closing position" in msg:
        try:
            ticket = msg.split("Closing position ")[1].split(" ")[0]
            symbol = msg.split("for ")[1].split(" ")[0]
            formatted_msg = f"H20 EXIT: Closing position {ticket} for {symbol}"
        except Exception:
            pass
    elif "Starting Proxima Ops" in msg:
        formatted_msg = "Engine initialized and active"
    elif "Connected to MT5" in msg:
        try:
            acc = msg.split("Account: ")[1].split(",")[0]
            formatted_msg = f"Connected to MT5 | Account: {acc}"
        except Exception:
            pass
    elif "BUY " in msg and " - ticket=" in msg:
        try:
            parts = msg.split("BUY ")[1].split(" ")
            symbol = parts[0]
            volume = parts[1]
            price = msg.split("@ ")[1].split(" - ")[0]
            ticket = msg.split("ticket=")[1]
            formatted_msg = f"EXECUTED: Buy {volume} {symbol} @ {price} | Ticket: {ticket}"
        except Exception:
            pass
    elif "Closed ticket " in msg:
        try:
            ticket = msg.split("ticket ")[1]
            formatted_msg = f"CLOSED: Ticket {ticket} closed on MT5"
        except Exception:
            pass
    elif "Failed to close ticket " in msg:
        try:
            ticket = msg.split("ticket ")[1].split(":")[0]
            err = msg.split(": ")[1]
            formatted_msg = f"CLOSE FAILED: Ticket {ticket} | {err}"
        except Exception:
            pass

    t = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{t}] {formatted_msg}"

    if demo._activity_log and demo._activity_log[-1][11:] == formatted_msg:
        return

    demo._activity_log.append(formatted)
    if len(demo._activity_log) > 6:
        demo._activity_log.pop(0)


def build_full_dashboard_text(demo, eval_data: dict, open_positions: list, account: dict,
                              score_data: dict, seconds_to_next_eval: int,
                              rotation_events=0, lock_events=0, migration_events=0,
                              avg_hold_bars=0, top3_ranked=None) -> str:
    """Build the full dashboard text with all sections. Returns a plain string."""
    if not hasattr(demo, "_spinner_idx"):
        demo._spinner_idx = 0
    demo._spinner_idx = (demo._spinner_idx + 1) % len(demo._spinners)
    spinner = demo._spinners[demo._spinner_idx]

    perf_summary_inner = demo.perf.summary()
    paper_metrics = {
        'pp': perf_summary_inner.get('pp') if isinstance(perf_summary_inner.get('pp'), (int, float)) else 0,
        'avg_hold': perf_summary_inner.get('avg_hold_bars') if isinstance(perf_summary_inner.get('avg_hold_bars'), (int, float)) else 0,
        'sharpe': perf_summary_inner.get('sharpe') if isinstance(perf_summary_inner.get('sharpe'), (int, float)) else 0,
        'active_assets': len([s for s, d in eval_data.items() if d.get('status') != 'WATCH']),
    }

    out = []
    out.append(demo.dashboard.render(
        eval_data=eval_data, account_info=account, score_data=score_data,
        seconds_to_next_eval=seconds_to_next_eval, spinner=spinner,
        closed_trades=demo.perf.n_trades,
        rotation_events=rotation_events,
        lock_events=lock_events,
        migration_events=migration_events,
        avg_hold_bars=avg_hold_bars,
        top3_ranked=top3_ranked,
        paper_metrics=paper_metrics))

    out.append("OPEN POSITIONS:")
    out.append(f"{'Ticket':<12s} {'Symbol':<10s} {'Side':<6s} {'Volume':<8s} {'Entry Price':<12s} {'Current Price':<14s} {'PnL':<10s} {'Bars':<6s}")
    for pos in open_positions:
        ticket = pos["ticket"]
        meta = demo._active_positions_metadata.get(ticket, {})
        entry_bar = meta.get("entry_bar_time")
        broker_sym = demo.mt5._get_broker_symbol(pos["symbol"])
        elapsed_bars = demo._bars_elapsed(entry_bar, broker_sym)
        elapsed_str = f"{'20+' if elapsed_bars >= 20 else elapsed_bars}/20" if elapsed_bars >= 0 else "N/A"
        out.append(f"{ticket:<12d} {pos['symbol']:<10s} {pos['type']:<6s} {pos['volume']:<8.2f} {pos['price_open']:<12.3f} {pos['price_current']:<14.3f} ${pos['profit']:<8.2f} {elapsed_str:<6s}")
    if not open_positions:
        out.append(" No open positions")
    out.append("")

    if open_positions:
        out.append("OPEN TRADE CONTEXT")
        out.append("=" * 52)
        for pos in open_positions:
            ticket = pos["ticket"]
            meta = demo._active_positions_metadata.get(ticket, {})
            entry_bar = meta.get("entry_bar_time")
            broker_sym = demo.mt5._get_broker_symbol(pos["symbol"])
            elapsed_bars = demo._bars_elapsed(entry_bar, broker_sym)
            elapsed_str = f"{'20+' if elapsed_bars >= 20 else elapsed_bars} bars" if elapsed_bars >= 0 else "N/A"
            es_str = f"{meta.get('entry_es_rank', 0) * 100:.1f}%" if isinstance(meta.get('entry_es_rank'), (int, float)) else "N/A"
            at_str = f"{meta.get('entry_at_rank', 0) * 100:.1f}%" if isinstance(meta.get('entry_at_rank'), (int, float)) else "N/A"
            sym_data = eval_data.get(pos["symbol"], eval_data.get(broker_sym, {}))
            econ_r = sym_data.get("econ_ratio", 0.0)
            exp_m = sym_data.get("expected_move", 0)
            out.append(f" ECON: ratio={econ_r:.4f}x move={exp_m:.6f}")
            sig = meta.get("trigger_count_while_open", 0)
            out.append(f" Ticket {ticket} | Age {elapsed_str} | ES/AT {es_str}/{at_str} | SigOpen {sig} | PnL ${pos['profit']:.2f} | THESIS_ACTIVE")
        out.append("=" * 52)
        out.append("")

    out.append("RECENT ACTIVITY:")
    for log_line in demo._activity_log:
        out.append(f" {log_line}")
    if not demo._activity_log:
        out.append(" No recent activity")
    out.append("")

    full = demo.funnel_dash.generate(
        order_attempts=demo.order_tracker.get_recent(1),
        paper_metrics=paper_metrics)
    out.append(full)

    if _SYSTEM_MODE.ui != UIMode.TRADER_VIEW:
        tpi_panel = tpi_dashboard_generate(
            tracker=demo._tpi_tracker,
            persistence=demo._tpi_persistence,
            curvature=demo._tpi_curvature,
            eligible_symbols=[s for s in demo._observation_universe if s in TPI_ELIGIBLE],
        )
        out.append(tpi_panel)

        n_trades = demo.perf.n_trades
        sig_counts = {}
        for sym in demo._observation_universe:
            sig_counts[sym] = len([x for x in demo.funnel._records if x.get("symbol") == sym]) if hasattr(demo.funnel, "_records") else 0
        guard = demo._sample_guard.guard("DEPLOYMENT_CLASSIFICATION")
        alignment_line = demo._alignment_monitor.dashboard_line(sig_counts)
        out.append(f"V2.2 GUARDS: SampleInspector={guard} | {alignment_line}")
        canonical = set(demo._execution_symbols)
        deployed = set(demo._execution_universe)
        core_present = canonical.issubset(deployed)
        extra = deployed - canonical
        uni_ok = "CORE_OK" if core_present else f"CORE_MISSING diff={canonical - deployed}"
        out.append(f"UNIVERSE: {len(deployed)}-asset ({len(extra)} extra) | {sorted(deployed)} | {uni_ok}")
        if demo._pyramid_log:
            out.append(f"PYRAMID EVENTS ({len(demo._pyramid_log)} total):")
            for pe in demo._pyramid_log[-5:]:
                out.append(f"  {pe['time'][:19]} {pe['symbol']} #{pe['pyramid_number']} ES={pe['es_percentile']:.3f}")
        else:
            out.append("PYRAMID EVENTS: none")
        total_blocked = demo._reinforcement_blocks + demo._flip_blocks
        if total_blocked > 0:
            pct = demo._flip_blocks / total_blocked * 100
            out.append(f"POSITION EXISTS: {total_blocked} total (reinforcement={demo._reinforcement_blocks}, flip_blocked={demo._flip_blocks}, flip_pct={pct:.1f}%)")
        if demo._exception_dashboard.has_active():
            out.append(f"EXCEPTIONS: {demo._exception_dashboard.summary().splitlines()[-1]}")
        prop_syms = [s for s in demo._observation_universe if s in TPI_ELIGIBLE]
        if prop_syms:
            out.append(demo._tpi_propagation.summary(prop_syms))
            dpl_matrix = demo._tpi_propagation.compute(prop_syms)
            demo._impulse_graph.update(dpl_matrix)
            out.append(demo._impulse_graph.summary())
        thermo_syms = [s for s in demo._observation_universe]
        if thermo_syms:
            out.append(demo._tick_thermo.summary(thermo_syms))
        meta_syms = [s for s in demo._observation_universe if s in demo._last_meta_scores]
        if meta_syms:
            out.append(demo._meta_fusion.summary(meta_syms, demo._last_meta_scores))
        out.append(demo._session_cond.summary())
        ent_syms = [s for s in demo._observation_universe]
        if ent_syms:
            out.append(demo._entropy_compression.summary(ent_syms))
        out.append(demo._outcome_ledger.summary())
        out.append(demo._ig_audit.summary(demo._outcome_ledger))
        try:
            fnames, X, Y = demo._outcome_ledger.compute_feature_matrix(horizon="h5")
            if fnames and X:
                demo._redundancy_matrix.compute_pairwise(fnames, X)
        except Exception:
            pass
        out.append(demo._redundancy_matrix.summary(demo._outcome_ledger))
        ig_by_h = demo._ig_audit.compute_by_horizon(demo._outcome_ledger)
        h20_resolved = [r for r in demo._outcome_ledger._resolved if "h20" in r.get("outcomes", {})]
        demo._meta_reweighter.compute_weights(ig_by_h, demo._redundancy_matrix, h20_count=len(h20_resolved))
        out.append(demo._meta_reweighter.summary())
        resolved_n = demo._outcome_ledger.resolved_count()
        demo._layer_pruner.compute_scores(ig_by_h, demo._redundancy_matrix, demo._meta_reweighter, resolved_samples=resolved_n)
        out.append(demo._layer_pruner.summary(demo._outcome_ledger))
        h5_records = [r for r in demo._outcome_ledger._resolved if "h5" in r.get("outcomes", {})]
        h20_records = h20_resolved
        h5_wins = sum(1 for r in h5_records if r["outcomes"]["h5"].get("win"))
        h20_wins = sum(1 for r in h20_records if r["outcomes"]["h20"].get("win"))
        out.append("  RCL DUAL HORIZON")
        out.append("-" * 52)
        out.append(f"  H5 Resolved:        {len(h5_records)}")
        out.append(f"  H20 Resolved:       {len(h20_records)}")
        if h5_records:
            out.append(f"  H5 WR:              {h5_wins}/{len(h5_records)} = {h5_wins/max(len(h5_records),1):.1%}")
        if h20_records:
            out.append(f"  H20 WR:             {h20_wins}/{len(h20_records)} = {h20_wins/max(len(h20_records),1):.1%}")
        if h5_records and h20_records:
            h5_wr = h5_wins / max(len(h5_records), 1)
            h20_wr = h20_wins / max(len(h20_records), 1)
            out.append(f"  Divergence:         {h5_wr - h20_wr:+.1%}")
        out.append("")
        known = {k: v for k, v in demo._session_balance.items() if k != "UNKNOWN"}
        total_signals = sum(known.values())
        out.append("  SESSION BALANCE")
        out.append("-" * 52)
        for sess in ["ASIA", "LONDON", "OVERLAP", "NY", "DEAD"]:
            cnt = known.get(sess, 0)
            bar = "█" * min(cnt, 50) + (" " * max(0, 50 - min(cnt, 50)))
            out.append(f"  {sess:<10s} {cnt:<5d} {bar}")
        if demo._session_balance.get("UNKNOWN", 0):
            out.append(f"  UNKNOWN:            {demo._session_balance['UNKNOWN']}")
        if total_signals > 0:
            max_c = max(known.values())
            min_c = max(min(v for v in known.values() if v > 0), 1)
            imbalance = max_c / min_c
            if max_c >= 20:
                out.append(f"  Imbalance:          {imbalance:.1f}x")
                out.append(f"  Status:             {'BALANCED' if imbalance < 5 else 'SKEWED'}")
            else:
                out.append(f"  Status:             BUILDING (need >=20 signals)")
        out.append("")
        out.append(demo._occupancy_audit.summary())
        out.append(demo._tpi_ab_audit.summary())
        sess_info = demo._spread_normalizer.session_baseline_summary()
        if sess_info:
            out.append(sess_info)
        out.append(demo._funnel_audit.summary())
        out.append("")
        out.append(demo._regime_memory.summary())
        out.append("")
        lines_edge = []
        lines_edge.append("  LIVE TRANSITION EDGE")
        lines_edge.append("-" * 60)
        has_edge = False
        for sym in demo._observation_universe:
            prev_r = demo._regime_snapshot.get(sym) if hasattr(demo, '_regime_snapshot') else None
            curr_r = demo._regime_memory._prev_regime.get(sym)
            if prev_r is not None and curr_r is not None and prev_r != curr_r:
                edge_line = demo._regime_memory.transition_edge_summary(sym, prev_r, curr_r)
                if edge_line:
                    lines_edge.append(edge_line)
                    has_edge = True
        if not has_edge:
            lines_edge.append("  No active transitions to evaluate")
        lines_edge.append("")
        out.extend(lines_edge)
        out.append(demo._signal_decay.summary())
        out.append("")
        out.append(demo._migration.summary())

    ai = demo.mt5.get_account() or {}
    out.append(demo._risk.dashboard_section(ai.get("balance", 0.0), open_positions))

    try:
        report_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "live_observability_report.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# PROXIMA OPS \u2014 LIVE OBSERVABILITY STATS BREAKDOWN\n\n")
            f.write(f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write("```text\n")
            f.write(full)
            f.write("\n```\n")
    except Exception as e:
        logger.error(f"Error writing live_observability_report.md: {e}")

    _sr = getattr(demo, '_last_stre_result', None)
    if _sr:
        out.append("")
        out.append("=" * 52)
        out.append("  SHADOW SYSTEM — TRUTH RECONCILIATION")
        out.append("=" * 52)
        out.append(f"  GT_corr={_sr.get('gt_corr',0):.4f} SY_corr={_sr.get('sy_corr',0):.4f} STAS={_sr.get('stas',0):.4f} Winner={_sr.get('winner','N/A')}")
        sof = _sr.get("SOF")
        if sof is not None:
            out.append(f"  SOF={sof:.6f} EdgePres={_sr.get('edge_preservation',0):.6f} ExecEff={_sr.get('execution_efficiency',0):.6f}")
        p2 = "ENABLED" if getattr(demo, '_stre_coordinator', None) and demo._stre_coordinator.phase2_enabled else "BLOCKED"
        out.append(f"  Phase 2: {p2} | Samples: {_sr.get('samples',0)}")
        out.append("")

    if demo._funnel_failures:
        out.append(f"\n  FUNNEL FAILURE BREAKDOWN ({sum(demo._funnel_failures.values())} total)")
        for reason, count in sorted(demo._funnel_failures.items(), key=lambda x: -x[1]):
            out.append(f"    {reason}: {count}")

    return "\n".join(out)


def print_dashboard(demo, eval_data: dict, open_positions: list, account: dict,
                    score_data: dict, seconds_to_next_eval: int,
                    rotation_events=0, lock_events=0, migration_events=0,
                    avg_hold_bars=0, top3_ranked=None):
    """Build and print the full dashboard to terminal."""
    if not hasattr(demo, "_ansi_initialized"):
        if sys.platform == "win32":
            os.system('')
        demo._ansi_initialized = True

    text = build_full_dashboard_text(
        demo, eval_data, open_positions, account, score_data, seconds_to_next_eval,
        rotation_events=rotation_events, lock_events=lock_events,
        migration_events=migration_events, avg_hold_bars=avg_hold_bars,
        top3_ranked=top3_ranked)

    demo._last_full_dashboard_text = text

    if hasattr(demo, "_clear_seq"):
        sys.stdout.write(demo._clear_seq)
    else:
        demo._clear_seq = '\033c'
        sys.stdout.write(demo._clear_seq)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
