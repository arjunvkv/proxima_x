"""Run paper trading for Tokyo H0 strategy."""
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
from strategy import STRATEGY_NAME, CONFIG, generate_signal

MIN_CONFIDENCE = 0.40
HEARTBEAT_INTERVAL = 30


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
    mt5_account = cfg.get("mt5_account")
    mt5_path = cfg.get("mt5_path")

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
    feed = Feed(mode="live", pairs=cfg["pairs"], mt5_path=mt5_path).connect()
    risk = Risk(cfg)
    exec_ = Executor(feed, logger)
    dash = Dashboard()
    dash.set_status("starting")
    dash.start()

    log_path = logger.start()
    print(f"Logging to {log_path}", file=sys.stderr)

    try:
        while True:
            now = time.gmtime()

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
                time.sleep(0.5)
                continue

            for pair, p in data.items():
                if "spread" in p:
                    risk.update_spread_baseline(pair, p["spread"])

            signals = generate_signal(data)
            if signals is not None:
                if not isinstance(signals, list):
                    signals = [signals]
                for signal in signals:
                    if signal.get("confidence", 0) >= MIN_CONFIDENCE:
                        pair = signal["pair"]
                        direction = signal["direction"]
                        p = data.get(pair, {})
                        cur_spread = p.get("spread", 0)
                        norm_spread = risk.normal_spread(pair)

                        ok_s, reason = risk.check_all(now, cur_spread, norm_spread, len(exec_.positions))
                        if not ok_s:
                            logger.log("BLOCK", pair, reason)
                            stats.record_reject(pair, reason)
                        else:
                            fill = exec_.submit_market(pair, direction, cfg["lot_size"])
                            if fill:
                                stats.record_fill(fill)
                                dash.set_last_signal(f"{pair}{'+' if direction>0 else '-'}")

            closed = exec_.check_open_positions(int(time.time()), hold_seconds=cfg["hold_bars"] * 60)
            for c in closed:
                stats.record_close(c)

            dash.push_snapshot(stats.snapshot())
            time.sleep(0.5)
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
