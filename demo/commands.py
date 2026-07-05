"""Command handler implementations for the ProximaDemo CLI router.

Each handler takes (demo, args, update) and returns a response string.
Stateless — they read from `demo.*` but mutate only through method calls.
"""

import logging

logger = logging.getLogger("proxima_demo")


def setup_commands(demo):
    """Register all command handlers with the command router."""
    demo._router.register("status", lambda a, u: cmd_status(demo, a, u))
    demo._router.register("portfolio", lambda a, u: cmd_portfolio(demo, a, u))
    demo._router.register("trades", lambda a, u: cmd_trades(demo, a, u))
    demo._router.register("signal", lambda a, u: cmd_signal(demo, a, u))
    demo._router.register("pause", lambda a, u: cmd_pause(demo, a, u))
    demo._router.register("resume", lambda a, u: cmd_resume(demo, a, u))
    demo._router.register("closeall", lambda a, u: cmd_closeall(demo, a, u))
    demo._router.register("health", lambda a, u: cmd_health(demo, a, u))
    demo._router.register("report", lambda a, u: cmd_report(demo, a, u))
    demo._router.register("alpha", lambda a, u: cmd_alpha(demo, a, u))
    demo._router.register("tpi_mode", lambda a, u: cmd_tpi_mode(demo, a, u))


def cmd_alpha(demo, args, update) -> str:
    return demo.dashboard.generate_alpha_snapshot()


def cmd_tpi_mode(demo, args, update) -> str:
    if args:
        mode = args[0].upper()
        try:
            demo._tpi_calibration.set_mode(mode)
            return f"TPI_MODE set to {mode}"
        except ValueError as e:
            return str(e)
    stats = demo._tpi_calibration.gate_stats()
    return (f"TPI_MODE: {stats['mode']}\n"
            f"Gate blocks: {stats['total_triggers_blocked']}\n"
            f"By gate: {stats['by_gate']}\n"
            f"Shadow opps: {stats['shadow_opportunities']}\n"
            f"Usage: /tpi_mode [HARD_GATE|SOFT_SCORE]")


def cmd_status(demo, args, update) -> str:
    ds = demo.score.summary()
    perf = demo.perf.summary()
    mt5_h = demo.mt5_monitor.health_summary
    positions = demo.positions.positions
    return (f"Proxima Ops — Status\n"
            f"Score: {ds['current_score']:.3f} ({ds['classification']})\n"
            f"Positions: {len(positions)}\n"
            f"Today PnL: ${perf['today_pnl']:.2f}\n"
            f"Sharpe: {perf['sharpe']:.3f}\n"
            f"PP: {perf['pp']:.3f}\n"
            f"MT5: {mt5_h['mt5_status']}\n"
            f"Paused: {demo._paused}")


def cmd_portfolio(demo, args, update) -> str:
    positions = demo.positions.positions
    if not positions:
        return "No open positions"
    lines = ["Portfolio:"]
    for p in positions:
        lines.append(f"{p['symbol']} {p['type']} | {p['volume']} | "
                     f"Entry: {p['price_open']:.3f} | PnL: ${p['profit']:.2f}")
    return "\n".join(lines)


def cmd_trades(demo, args, update) -> str:
    trades = demo.trade_ledger.get_recent(10)
    if not trades:
        return "No trades recorded"
    lines = ["Last 10 Trades:"]
    for t in trades:
        lines.append(f"{t['symbol']} {t['signal_type']} | "
                     f"Entry: {t['entry_price']} | "
                     f"PnL: ${t['profit_money']:.2f} | {t['status']}")
    return "\n".join(lines)


def cmd_signal(demo, args, update) -> str:
    if not args:
        return "Usage: /signal EURJPY"
    symbol = args[0].upper()
    tick = demo.tick_source.next_tick(symbol) if demo.tick_source else (demo._tick_cache.get_tick(symbol) if demo._tick_cache else demo.mt5.get_tick(symbol))
    if tick is None:
        return f"Could not get tick for {symbol}"
    return (f"{symbol} — Live Tick\n"
            f"Bid: {tick['bid']:.5f}\n"
            f"Ask: {tick['ask']:.5f}\n"
            f"Spread: {tick['spread']}")


def cmd_pause(demo, args, update) -> str:
    demo._paused = True
    return "Trading PAUSED. No new entries. Monitoring continues."


def cmd_resume(demo, args, update) -> str:
    demo._paused = False
    return "Trading RESUMED."


def cmd_closeall(demo, args, update) -> str:
    demo.positions.refresh()
    for pos_ca in demo.positions.positions:
        m_ca = demo._active_positions_metadata.get(pos_ca["ticket"])
        if m_ca:
            m_ca["expected_exit_reason"] = "MANUAL"
    demo._save_active_positions_metadata()
    demo._sdl.reset()
    results = demo.orders.close_all()
    closed = sum(1 for r in results if r["closed"])
    failed = sum(1 for r in results if not r["closed"])
    return f"Close All: {closed} closed, {failed} failed"


def cmd_health(demo, args, update) -> str:
    mt5_h = demo.mt5_monitor.health_summary
    ds = demo.score.summary()
    return (f"Health Check:\n"
            f"MT5: {mt5_h['mt5_status']}\n"
            f"Uptime: {mt5_h['uptime_minutes']}m\n"
            f"Deployment Score: {ds['current_score']:.3f} ({ds['classification']})")


def cmd_report(demo, args, update) -> str:
    return demo._daily_report.generate()
