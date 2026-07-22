"""Run M1 Z-Reversal paper trading with trailing stop management."""
import sys, os, time, threading, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from paper_trade.core.feed import Feed
from paper_trade.core.executor import Executor
from paper_trade.core.logger import Logger
from paper_trade.core.risk import Risk
from paper_trade.core.dashboard import Dashboard
from paper_trade.core.stats import SessionStats
from paper_trade.core.config import STRATEGIES
from paper_trade.core import registry as acct_registry
from strategy import STRATEGY_NAME, CONFIG, generate_signal, seed_history, TrailingStopManager

MIN_CONFIDENCE = 0.30
HEARTBEAT_INTERVAL = 30
LOOP_INTERVAL = 0.1


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
    seed_history(feed)
    risk = Risk(cfg)
    exec_ = Executor(feed, logger, magic=cfg.get("magic", 202407))
    trail_mgr = TrailingStopManager(cfg)
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
                time.sleep(LOOP_INTERVAL)
                continue

            # --- Process pending delayed entries ---
            remaining = []
            for eq in entry_queue:
                if current_ts >= eq["fire_at"]:
                    pair = eq["pair"]
                    direction = eq["direction"]
                    signal = eq["signal"]
                    p = data.get(pair, {})
                    cur_spread = p.get("spread", 0)
                    norm_spread = risk.normal_spread(pair)

                    if trail_mgr.pair_count(pair) >= 1:
                        logger.log("BLOCK", pair, "already_open")
                        stats.record_reject(pair, "already_open")
                        continue
                    ok_s, reason_s = risk.check_all(now, cur_spread, norm_spread,
                                                     trail_mgr.total_count(), pair=pair)
                    if not ok_s:
                        logger.log("BLOCK", pair, reason_s)
                        stats.record_reject(pair, reason_s)
                    else:
                        lot = cfg["lot_size"] * random.uniform(0.8, 1.2)
                        act_lot = round(lot, 2)
                        meta = {"z_score": signal.get("z_score"), "currency": pair}
                        fill = exec_.submit_market(pair, direction, act_lot,
                                                   timestamp=current_ts, signal_meta=meta)
                        if fill:
                            atr_v = signal.get("atr", 1)
                            ticket = trail_mgr.add(pair, direction, fill["entry_price"], atr_v, lot_size=act_lot, spread=cur_spread, timestamp=current_ts)
                            stats.record_fill(fill)
                            dash.set_last_signal(
                                f"{pair}{'+' if direction > 0 else '-'} @{fill['entry_price']:.5f}")
                else:
                    remaining.append(eq)
            entry_queue = remaining

            # --- Update spread baselines ---
            for p in data.values():
                if "spread" in p:
                    for pair in cfg["pairs"]:
                        risk.update_spread_baseline(pair, p["spread"])

            # --- Heartbeat ---
            if current_ts % 60 == 0:
                pass  # heartbeat placeholder

            # --- Generate signals from tick data ---
            signal = generate_signal(data, current_time=current_ts)
            if signal is not None and signal.get("confidence", 0) >= MIN_CONFIDENCE:
                if trail_mgr.pair_count(signal["pair"]) >= 1:
                    logger.log("SKIP", signal["pair"], f"already_open")
                    stats.record_reject(signal["pair"], "already_open")
                elif any(eq["pair"] == signal["pair"] for eq in entry_queue):
                    logger.log("SKIP", signal["pair"], f"queued_already")
                    stats.record_reject(signal["pair"], "queued_already")
                else:
                    delay = signal.get("delay_s", 3)
                    entry_queue.append({
                        "pair": signal["pair"],
                        "direction": signal["direction"],
                        "fire_at": current_ts + delay,
                        "signal": signal,
                    })

            # --- Trailing stop checks ---
            for pair, p in data.items():
                bid = p.get("bid", 0)
                ask = p.get("ask", 0)
                if bid > 0 and ask > 0:
                    closed_trades = trail_mgr.update(bid, ask, timestamp=current_ts)
                    for ct in closed_trades:
                        exit_price = bid if ct["direction"] == 1 else ask
                        pnl_info = exec_.close_position(
                            {"pair": ct["pair"], "direction": ct["direction"],
                             "entry_price": ct["entry"], "lot_size": ct.get("lot_size", cfg["lot_size"])},
                            exit_price=exit_price, exit_time=current_ts)
                        if pnl_info:
                            stats.record_close({**ct, **pnl_info})
                            dash.set_last_signal(
                                f"{ct['pair']} X ${pnl_info.get('gross_pnl', 0):.2f}")

            # --- Expiry checks ---
            expired = trail_mgr.check_expiry(current_ts)
            for et in expired:
                bid = data.get(et["pair"], {}).get("bid", 0)
                ask = data.get(et["pair"], {}).get("ask", 0)
                exit_price = bid if et["direction"] == 1 else ask
                pnl_info = exec_.close_position(
                    {"pair": et["pair"], "direction": et["direction"],
                     "entry_price": et["entry"], "lot_size": et.get("lot_size", cfg["lot_size"])},
                    exit_price=exit_price, exit_time=current_ts)
                if pnl_info:
                    stats.record_close({**et, **pnl_info})

            dash.push_snapshot(stats.snapshot())
            time.sleep(LOOP_INTERVAL)
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
