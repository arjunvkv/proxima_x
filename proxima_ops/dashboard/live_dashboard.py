"""
PROXIMA Live Dashboard — Terminal ANSI mode.

Reads ONLY from UnifiedStateBuilder (no direct file reads, no direct MT5 calls).
Displays 6 panels refreshed every 3 seconds.
"""

import os
import sys
import time
from datetime import datetime

from proxima_x.proxima_ops.dashboard.unified_state_builder import UnifiedStateBuilder

# ── ANSI Color Constants ──────────────────────────────────────
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
ORANGE = "\033[38;5;208m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"
WHITE = "\033[97m"

# ── SEGL State → Color Mapping ────────────────────────────────
SEGL_COLORS = {
    "OBSERVE": YELLOW,
    "ARMED": GREEN,
    "EXECUTING": BLUE,
    "COOLDOWN": ORANGE,
    "LOCKED": RED,
}

# ── Helpers ────────────────────────────────────────────────────


def _fmt(val: object, decimals: int = 2) -> str:
    """Format a value, returning '—' for None."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _fmt_price(val: object) -> str:
    """Format a price value (5 decimal places)."""
    if val is None or val == 0.0:
        return "—"
    return f"{val:.5f}"


def _bool_icon(val: bool) -> str:
    """Return ✓ (green) or ✗ (red)."""
    return f"{GREEN}✓{RESET}" if val else f"{RED}✗{RESET}"


def _bool_text(val: bool) -> str:
    """Return colored True/False."""
    return f"{GREEN}True{RESET}" if val else f"{RED}False{RESET}"


def _color_rsi(val: object) -> str:
    """Color-code an RSI value: <30 red, 30–70 white, >70 green."""
    if val is None:
        return f"{GRAY}—{RESET}"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return f"{GRAY}{val}{RESET}"
    if v < 30:
        return f"{RED}{v:.1f}{RESET}"
    if v > 70:
        return f"{GREEN}{v:.1f}{RESET}"
    return f"{WHITE}{v:.1f}{RESET}"


def _color_segl(state: str) -> str:
    """Return colored SEGL state string."""
    color = SEGL_COLORS.get(state.upper(), WHITE)
    return f"{color}{state}{RESET}"


def _color_decision(decision: str) -> str:
    """HOLD=gray, EXECUTE=green."""
    if decision.upper() == "EXECUTE":
        return f"{GREEN}{decision}{RESET}"
    return f"{GRAY}{decision}{RESET}"


def _color_pnl(pnl: float) -> str:
    """Positive=green, negative=red."""
    if pnl >= 0:
        return f"{GREEN}{pnl:+.2f}{RESET}"
    return f"{RED}{pnl:.2f}{RESET}"


def _section_header(title: str, width: int = 70) -> str:
    """Return a centered section header with horizontal rules."""
    side = (width - len(title) - 2) // 2
    lhs = "─" * max(side, 1)
    rhs = "─" * max(width - side - len(title) - 2, 1)
    return f"{BOLD}{lhs} {title} {rhs}{RESET}"


# ── Panel Renderers ────────────────────────────────────────────


def render_market_state(symbols: dict) -> None:
    """Panel 1: Market State — RSI/ATR/Bid/Ask/Spread for ALL symbols from unified state."""
    if not symbols:
        print(f"  {GRAY}No symbol data available{RESET}")
        return

    print(_section_header("Market State"))

    # Header
    header = f"  {'Symbol':12s} {'RSI':8s} {'ATR':10s} {'Bid':12s} {'Ask':12s} {'Spread':8s}"
    print(f"{GRAY}{header}{RESET}")
    print(f"{GRAY}  {'─' * 70}{RESET}")

    # Sort symbols alphabetically for deterministic order
    for sym in sorted(symbols.keys()):
        data = symbols[sym]
        rsi = data.get("rsi")
        atr = data.get("atr")
        bid = data.get("bid")
        ask = data.get("ask")
        spread = data.get("spread")

        # RSI color coding
        rsi_str = f"{rsi:.1f}" if rsi is not None else f"{GRAY}—{RESET}"
        if rsi is not None:
            if rsi < 30:
                rsi_colored = f"{RED}{rsi:.1f}{RESET}"
            elif rsi > 70:
                rsi_colored = f"{GREEN}{rsi:.1f}{RESET}"
            else:
                rsi_colored = f"{WHITE}{rsi:.1f}{RESET}"
        else:
            rsi_colored = f"{GRAY}—{RESET}"

        atr_str = f"{atr:.5f}" if atr is not None else f"{GRAY}—{RESET}"
        bid_str = f"{bid:.5f}" if bid is not None else f"{GRAY}—{RESET}"
        ask_str = f"{ask:.5f}" if ask is not None else f"{GRAY}—{RESET}"
        spread_str = f"{spread}" if spread is not None else f"{GRAY}—{RESET}"

        print(f"  {sym:12s} {rsi_colored:8s} {atr_str:10s} {bid_str:12s} {ask_str:12s} {spread_str:8s}")


def render_execution_state(execution_state: dict) -> None:
    """Panel 2: Execution State — cycle, SEGL, positions, signals, decision, governor."""
    print(_section_header("Execution State"))

    if not execution_state:
        print(f"  {YELLOW}No execution data available{RESET}")
        return

    cycle = execution_state.get("cycle", "—")
    segl = _color_segl(execution_state.get("segl_state", "OBSERVE"))
    open_pos = execution_state.get("open_positions", "—")
    active_sig = execution_state.get("active_signals", "—")
    decision = _color_decision(
        execution_state.get("execution_decision", "HOLD")
    )
    denial = execution_state.get("denial_reason")
    denial_str = f"{YELLOW}{denial}{RESET}" if denial else f"{GRAY}—{RESET}"
    gov_auth = execution_state.get("governor_authorized", False)

    print(f"  Cycle:             {cycle}")
    print(f"  SEGL State:        {segl}")
    print(f"  Open Positions:    {open_pos}")
    print(f"  Active Signals:    {active_sig}")
    print(f"  Execution Decision: {decision}")
    print(f"  Denial Reason:     {denial_str}")
    print(f"  Governor Authorized: {_bool_text(gov_auth)}")


def render_open_positions(execution_state: dict) -> None:
    """Panel 3: Open Positions — table of current positions."""
    print(_section_header("Open Positions"))

    if not execution_state:
        print(f"  {YELLOW}No position data available{RESET}")
        return

    positions = execution_state.get("open_positions_detail", [])
    if not positions:
        print(f"  {GRAY}No open positions{RESET}")
        return

    # Columns: Ticket, Symbol, Type, Volume, Entry, Current, PnL, SL, TP
    header = (
        f"  {'Ticket':<8} {'Symbol':<8} {'Type':<6} {'Volume':<8} "
        f"{'Entry':>10} {'Current':>10} {'PnL':>10} {'SL':>10} {'TP':>10}"
    )
    print(f"{GRAY}{header}{RESET}")
    print(f"{GRAY}  {'─' * 86}{RESET}")

    for pos in positions:
        ticket = pos.get("ticket", "—")
        symbol = pos.get("symbol", "—")
        ptype = pos.get("type", "—")
        volume = pos.get("volume", 0)
        entry = _fmt_price(pos.get("price_open"))
        current = _fmt_price(pos.get("price_current"))
        profit = float(pos.get("profit", 0) or 0)
        swap = float(pos.get("swap", 0) or 0)
        pnl = profit + swap
        pnl_str = _color_pnl(pnl)
        sl = _fmt_price(pos.get("sl"))
        tp = _fmt_price(pos.get("tp"))

        print(
            f"  {str(ticket):<8} {symbol:<8} {str(ptype):<6} {str(volume):<8} "
            f"{entry:>10} {current:>10} {pnl_str:>10} {sl:>10} {tp:>10}"
        )


def render_risk_state(risk_state: dict) -> None:
    """Panel 4: Risk State — account metrics, drawdown, circuit breaker."""
    print(_section_header("Risk State"))

    if not risk_state:
        print(f"  {YELLOW}No risk data available{RESET}")
        return

    balance = _fmt(risk_state.get("balance"))
    equity = _fmt(risk_state.get("equity"))
    free_margin = _fmt(risk_state.get("free_margin"))
    margin_level = _fmt(risk_state.get("margin_level"), 2)
    drawdown = _fmt(risk_state.get("drawdown_pct"), 2)

    cb = risk_state.get("circuit_breaker", "CLOSED")
    if cb == "CLOSED":
        cb_str = f"{GREEN}CLOSED{RESET}"
    elif cb == "TRIGGERED":
        cb_str = f"{RED}TRIGGERED{RESET}"
    else:
        cb_str = str(cb)

    daily_loss = risk_state.get("daily_loss", "—")
    daily_loss_str = (
        f"{RED}{_fmt(daily_loss)}{RESET}"
        if isinstance(daily_loss, (int, float)) and daily_loss < 0
        else _fmt(daily_loss)
    )

    print(f"  Balance:        ${balance}")
    print(f"  Equity:         ${equity}")
    print(f"  Free Margin:    ${free_margin}")
    print(f"  Margin Level:   {margin_level}%")
    print(f"  Drawdown:       {drawdown}%")
    print(f"  Circuit Breaker: {cb_str}")
    print(f"  Daily Loss:     ${daily_loss_str}")


def render_performance_state(performance_state: dict) -> None:
    """Panel 5: Performance — trades, win rate, PnL, extremes."""
    print(_section_header("Performance"))

    if not performance_state:
        print(f"  {YELLOW}No performance data available{RESET}")
        return

    total_trades = performance_state.get("total_trades", "—")
    win_rate = performance_state.get("win_rate", "—")
    total_pnl = performance_state.get("total_pnl", "—")
    largest_winner = performance_state.get("largest_winner", "—")
    largest_loser = performance_state.get("largest_loser", "—")

    # Color win rate
    wr_str = (
        f"{GREEN}{_fmt(win_rate, 1)}%{RESET}"
        if isinstance(win_rate, (int, float)) and win_rate >= 0.5
        else f"{RED}{_fmt(win_rate, 1)}%{RESET}"
        if isinstance(win_rate, (int, float))
        else str(win_rate)
    )

    # Color total PnL
    pnl_str = (
        _color_pnl(total_pnl) if isinstance(total_pnl, (int, float)) else str(total_pnl)
    )

    lw_str = (
        f"{GREEN}${_fmt(largest_winner)}{RESET}"
        if isinstance(largest_winner, (int, float))
        else str(largest_winner)
    )
    ll_str = (
        f"{RED}${_fmt(abs(largest_loser))}{RESET}"
        if isinstance(largest_loser, (int, float))
        else str(largest_loser)
    )

    print(f"  Total Trades:    {total_trades}")
    print(f"  Win Rate:        {wr_str}")
    print(f"  Total PnL:       ${pnl_str}")
    print(f"  Largest Winner:  {lw_str}")
    print(f"  Largest Loser:   {ll_str}")


def render_system_health(system_health: dict) -> None:
    """Panel 6: System Health — connections, data sources, errors."""
    print(_section_header("System Health"))

    if not system_health:
        print(f"  {YELLOW}No health data available{RESET}")
        return

    mt5_ok = system_health.get("mt5_connected", False)
    data_sources = system_health.get("data_sources", {})
    last_error = system_health.get("last_error")
    build_ts = system_health.get("build_timestamp")

    # Connection status
    print(f"  MT5 Connected:       {_bool_icon(mt5_ok)}")

    # Data source traces
    pt_ok = data_sources.get("pipeline_trace", False)
    rt_ok = data_sources.get("regime_tracker", False)
    el_ok = data_sources.get("execution_ledger", False)
    print(f"  Pipeline Trace:      {_bool_icon(pt_ok)}")
    print(f"  Regime Tracker:      {_bool_icon(rt_ok)}")
    print(f"  Execution Ledger:    {_bool_icon(el_ok)}")

    # Build timestamp
    if build_ts and isinstance(build_ts, (int, float)):
        ts_str = datetime.fromtimestamp(build_ts).strftime("%H:%M:%S")
    else:
        ts_str = "—"
    print(f"  Build Timestamp:     {ts_str}")

    # Last error
    if last_error:
        # Truncate long error messages for display
        err = str(last_error)
        if len(err) > 60:
            err = err[:57] + "..."
        print(f"  {RED}Last Error:          {err}{RESET}")
    else:
        print(f"  Last Error:          {GRAY}—{RESET}")


def render_shadow_state(shadow_state: dict) -> None:
    """Panel 7: Shadow Execution Engine — suppression, LKG, conviction deltas."""
    print()
    print(f"{BOLD}╔═══ Shadow Diagnostics ═══╗{RESET}")

    if not shadow_state:
        print(f"  {YELLOW}No shadow data available{RESET}")
        print(f"{BOLD}╚═════════════════════════╝{RESET}")
        return

    # Suppression graph
    sgraph = shadow_state.get("suppression_graph", {})
    edges = sgraph.get("edges", [])
    if edges:
        print(f"  {'Gate Transition':30s} {'Suppression':>12s}")
        print(f"  {'─' * 42}")
        for e in edges:
            u = e.get("source", "?")
            v = e.get("target", "?")
            w = e.get("suppression_magnitude", 0)
            label = f"{u} → {v}"
            color = RED if w > 0.1 else YELLOW if w > 0.05 else GREEN
            print(f"  {label:30s} {color}{w:>12.4f}{RESET}")

    # LKG similarity
    lkg = shadow_state.get("avg_lkg_similarity")
    if lkg is not None:
        color = GREEN if lkg > 0.95 else YELLOW if lkg > 0.85 else RED
        print(f"  {'LKG Similarity':30s} {color}{lkg:>12.4f}{RESET}")

    # Max suppression
    supp = shadow_state.get("max_suppression")
    if supp is not None:
        color = GREEN if supp < 0.05 else YELLOW if supp < 0.15 else RED
        print(f"  {'Max Suppression':30s} {color}{supp:>12.4f}{RESET}")

    # Per-symbol detail
    symbols = shadow_state.get("symbols", {})
    if symbols:
        print(f"  {'─' * 42}")
        print(f"  {'Symbol':12s} {'Supp Δ':>10s} {'LKG Sim':>10s}")
        print(f"  {'─' * 34}")
        for sym, sdata in sorted(symbols.items()):
            sd = sdata.get("suppression_delta", 0)
            lk = sdata.get("lkg_similarity_score", 0)
            print(f"  {sym:12s} {sd:>10.4f} {lk:>10.4f}")

    # Cycle info
    cid = shadow_state.get("cycle_id")
    if cid is not None:
        print(f"  {'Cycle':30s} {cid:>12}")

    print(f"{BOLD}╚═════════════════════════╝{RESET}")


def render_stre_panel(stre_result, gt_suppression) -> None:
    """Panel 8: STR-E Truth Reconciliation + GT suppression visualization."""
    print()
    print(f"{BOLD}╔═══ Truth System Status ═══╗{RESET}")

    if not stre_result:
        print(f"  {YELLOW}Collecting data (need 10+ samples){RESET}")
        print(f"{BOLD}╚══════════════════════════╝{RESET}")
        return

    s = stre_result.get("samples", 0)
    gt_c = stre_result.get("gt_corr", 0)
    sy_c = stre_result.get("sy_corr", 0)
    stas = stre_result.get("stas", 0)
    winner = stre_result.get("winner", "N/A")

    gt_color = GREEN if gt_c > sy_c else RED
    sy_color = GREEN if sy_c > gt_c else RED
    stas_color = GREEN if stas > 0 else YELLOW if stas == 0 else RED

    print(f"  {'Samples':20s} {s:>6}")
    print(f"  {'GT Correlation':20s} {gt_color}{gt_c:>8.4f}{RESET}")
    print(f"  {'SY Correlation':20s} {sy_color}{sy_c:>8.4f}{RESET}")
    print(f"  {'STAS Score':20s} {stas_color}{stas:>8.4f}{RESET}")
    print(f"  {'Winner':20s} {winner:>8}")

    sof = stre_result.get("SOF")
    if sof is not None:
        ee = stre_result.get("execution_efficiency", 0)
        ep = stre_result.get("edge_preservation", 0)
        print(f"  {'SOF Score':20s} {sof:>10.6f}")
        print(f"  {'Edge Preservation':20s} {ep:>10.6f}")
        print(f"  {'Exec Efficiency':20s} {ee:>10.6f}")

    if stre_result.get("phase2_blocked"):
        print(f"  {'Phase 2':20s} {RED}BLOCKED{RESET}")
    else:
        print(f"  {'Phase 2':20s} {GREEN}ENABLED{RESET}")

    if gt_suppression:
        flow = gt_suppression.suppression_flow()
        if flow:
            print(f"  {'─' * 36}")
            print(f"  {'Conviction Drop Between Layers':36s}")
            for f in flow:
                label = f"{f['source']} → {f['target']}"
                drop = f.get("drop", 0)
                color = RED if drop > 0.1 else YELLOW if drop > 0.05 else GREEN
                print(f"  {label:28s} {color}{drop:>8.4f}{RESET}")

    print(f"{BOLD}╚══════════════════════════╝{RESET}")


def render_footer() -> None:
    """Print a closing separator line."""
    print(f"{GRAY}{'─' * 70}{RESET}")


def render_all(state: dict) -> None:
    """Render all 6 dashboard panels from the unified state dict."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{BOLD}╔═══ PROXIMA Live Dashboard ═══╗{RESET}")
    print(f"{BOLD}║   {now_str}{' ' * (29 - len(now_str))}║{RESET}")
    print(f"{BOLD}╚══════════════════════════════╝{RESET}")
    print()

    if not state:
        print(f"  {YELLOW}Waiting for data...{RESET}")
        return

    render_market_state(state.get("market_state", {}).get("symbols", {}))
    print()
    render_execution_state(state.get("execution_state", {}))
    print()
    render_open_positions(state.get("execution_state", {}))
    print()
    render_risk_state(state.get("risk_state", {}))
    print()
    render_performance_state(state.get("performance_state", {}))
    print()
    _shadow_state = state.get("system_health", {}).get("shadow", {})
    render_system_health(state.get("system_health", {}))
    render_shadow_state(_shadow_state)
    render_stre_panel(_shadow_state.get("stre"), None)
    render_footer()


# ── Main Loop ─────────────────────────────────────────────────


def main() -> None:
    """Entry point: create builder, refresh every 3 seconds."""
    builder = UnifiedStateBuilder()

    while True:
        try:
            state = builder.build()
        except Exception:
            state = {}

        # Clear terminal
        os.system("cls" if os.name == "nt" else "clear")

        try:
            render_all(state)
        except Exception as exc:
            print(f"{RED}Dashboard render error: {exc}{RESET}")

        time.sleep(3)


if __name__ == "__main__":
    main()
