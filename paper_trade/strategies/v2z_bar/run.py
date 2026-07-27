"""Run V2+z Bar paper trading — bar-level stop checks matching hfdf_m1 backtest."""
import sys, os, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from paper_trade.core.feed import Feed
from paper_trade.core.executor import Executor
from paper_trade.core.logger import Logger
from paper_trade.core.risk import Risk
from paper_trade.core.dashboard import Dashboard
from paper_trade.core.stats import SessionStats
from paper_trade.core.config import STRATEGIES
from paper_trade.core import registry as acct_registry
from strategy import STRATEGY_NAME, CONFIG, generate_signal, seed_history, BarStopManager

HEARTBEAT_INTERVAL = 30
LOOP_SLEEP = 0.5


def _mt5_healthy(feed):
    try:
        return feed.mt5.terminal_info() is not None
    except Exception:
        return False


def _heartbeat_loop(login, name, stop_event):
    while not stop_event.is_set():
        time.sleep(HEARTBEAT_INTERVAL)
        acct_registry.heartbeat(login, name)


def main():
    cfg = STRATEGIES[STRATEGY_NAME]
    mt5_path = cfg.get("mt5_path")

    feed = Feed(mode="live", pairs=cfg["pairs"], mt5_path=mt5_path).connect()
    mt5_account = feed.mt5.account_info().login
    print(f"Using MT5 account: {mt5_account}", file=sys.stderr)

    ok, reason = acct_registry.claim(mt5_account, STRATEGY_NAME)
    if not ok:
        print(f"ACCOUNT CONFLICT: {reason}", file=sys.stderr)
        sys.exit(1)
    print(f"Account {mt5_account} claimed for {STRATEGY_NAME}", file=sys.stderr)

    stop_heartbeat = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(mt5_account, STRATEGY_NAME, stop_heartbeat), daemon=True
    )
    hb_thread.start()

    stats = SessionStats()
    logger = Logger(STRATEGY_NAME)
    print("Seeding initial M1 bar history from MT5...", file=sys.stderr)
    seed_history(feed)
    print("Seeding complete. Signal generator ready.", file=sys.stderr)

    stop_mgr = BarStopManager(cfg)
    risk = Risk(cfg)
    exec_ = Executor(feed, logger, magic=cfg.get("magic", 202410))
    dash = Dashboard()
    dash.set_status("starting")
    dash.start()

    log_path = logger.start()
    print(f"Logging to {log_path}", file=sys.stderr)

    entry_queue = []

    try:
        while True:
            now = time.gmtime()
            current_ts = int(time.time())

            ok, reason = risk.check_market_hours(now)
            if not ok:
                dash.set_status(f"idle:{reason}")
                time.sleep(60)
                continue

            if feed.mode == "live" and not _mt5_healthy(feed):
                dash.set_status("mt5_disconnected")
                time.sleep(5)
                continue

            dash.set_status("running")

            data = feed.current_bar()
            if not data:
                time.sleep(LOOP_SLEEP)
                continue

            for pair, p in data.items():
                if "spread" in p:
                    risk.update_spread_baseline(pair, p["spread"])

            settle_sec = CONFIG.get("settle_seconds", 1)
            if current_ts % 60 < settle_sec:
                signals = []
                bar_data = {}
            else:
                signals, bar_data = generate_signal(data, current_time=current_ts)

            # Process entry queue (delayed entries if needed)
            remaining = []
            for eq in entry_queue:
                if current_ts >= eq["fire_at"]:
                    pair = eq["pair"]
                    direction = eq["direction"]
                    p = data.get(pair, {})
                    cur_spread = p.get("spread", 0)
                    norm_spread = risk.normal_spread(pair)

                    if stop_mgr.pair_count(pair) >= 1:
                        logger.log("BLOCK", pair, "already_open")
                        stats.record_reject(pair, "already_open")
                        continue
                    ok_s, reason_s = risk.check_all(now, cur_spread, norm_spread,
                                                    stop_mgr.total_count(), pair=pair)
                    if not ok_s:
                        logger.log("BLOCK", pair, reason_s)
                        stats.record_reject(pair, reason_s)
                    else:
                        lot = cfg["lot_size"]
                        meta = {"z_score": eq["signal"].get("z_score"),
                                "atr": eq["signal"].get("atr")}
                        fill = exec_.submit_market(pair, direction, lot,
                                                   timestamp=current_ts, signal_meta=meta)
                        if fill:
                            atr_v = eq["signal"].get("atr", 1)
                            entry_bar_time = eq["signal"].get("bar_time", current_ts)
                            stop_mgr.add(pair, direction, fill["entry_price"], atr_v,
                                         entry_time=entry_bar_time)
                            stats.record_fill(fill)
                            dash.set_last_signal(
                                f"{pair}{'+' if direction > 0 else '-'} @{fill['entry_price']:.5f}")
                else:
                    remaining.append(eq)
            entry_queue = remaining

            for signal in signals:
                if signal.get("confidence", 0) < 0.30:
                    continue
                if stop_mgr.pair_count(signal["pair"]) >= 1:
                    logger.log("SKIP", signal["pair"], "already_open")
                    stats.record_reject(signal["pair"], "already_open")
                elif any(eq["pair"] == signal["pair"] for eq in entry_queue):
                    logger.log("SKIP", signal["pair"], "queued_already")
                    stats.record_reject(signal["pair"], "queued_already")
                else:
                    entry_queue.append({
                        "pair": signal["pair"], "direction": signal["direction"],
                        "fire_at": current_ts, "signal": signal,
                    })

            # Bar-level stop checks (using same bar_data as signal generation)
            closed = stop_mgr.check_stops(bar_data, current_ts)
            for ct in closed:
                exit_price = ct["exit"]
                pnl_info = exec_.close_position(
                    {"pair": ct["pair"], "direction": ct["direction"],
                     "entry_price": ct["entry"], "lot_size": cfg["lot_size"]},
                    exit_price=exit_price, exit_time=current_ts)
                if pnl_info:
                    stats.record_close({**ct, **pnl_info})
                    dash.set_last_signal(
                        f"{ct['pair']} X ${pnl_info.get('gross_pnl', 0):.2f}")

            dash.push_snapshot(stats.snapshot())
            time.sleep(LOOP_SLEEP)
    except KeyboardInterrupt:
        pass
    finally:
        stop_heartbeat.set()
        dash.stop()
        feed.close()
        logger.close()
        acct_registry.release(mt5_account, STRATEGY_NAME)
        print(f"Account {mt5_account} released", file=sys.stderr)


if __name__ == "__main__":
    main()
