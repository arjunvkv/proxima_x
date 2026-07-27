"""Run V2+z Paper trading with trailing stop management."""
import gc, sys, os, time, threading, json
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from paper_trade.core.feed import Feed
from paper_trade.core.executor import Executor
from paper_trade.core.logger import Logger
from paper_trade.core.risk import Risk
from paper_trade.components import pip_value_usd
from paper_trade.core.dashboard import Dashboard
from paper_trade.core.stats import SessionStats
from paper_trade.core.config import STRATEGIES
from paper_trade.core import registry as acct_registry
from strategy import STRATEGY_NAME, CONFIG, generate_signal, seed_history, TrailingStopManager, _states

MIN_CONFIDENCE = 0.30
HEARTBEAT_INTERVAL = 30
LOOP_INTERVAL = 0.1
GC_INTERVAL = 300


class TradeTracer:
    """Logs every detail of each trade lifecycle for backtest verification."""

    def __init__(self, trail_mgr, stats, cfg):
        self.trail_mgr = trail_mgr
        self.stats = stats
        self.cfg = cfg
        self._trace_file = None
        self._active_trades = {}

    def start(self):
        path = f"paper_trade/trade_logs/trace_{int(time.time())}.jsonl"
        self._trace_file = open(path, "w")
        print(f"Trade trace: {path}", file=sys.stderr)
        return self

    def close(self):
        if self._trace_file:
            self._trace_file.close()

    def _log(self, event, data):
        line = json.dumps({"ts": int(time.time()), "event": event, **data})
        if self._trace_file:
            self._trace_file.write(line + "\n")
            self._trace_file.flush()

    def on_signal(self, pair, signal, completed_bar):
        ps = _states.get(pair)
        z_buf = list(ps.z_buf.buf) if ps and ps.z_buf.buf else []
        z_mean = sum(z_buf) / len(z_buf) if len(z_buf) > 5 else 0
        z_std = (sum((x - z_mean) ** 2 for x in z_buf) / (len(z_buf) - 1)) ** 0.5 if len(z_buf) > 5 else 0
        ret = completed_bar["close"] - ps.last_close if ps else 0
        self._log("signal", {
            "pair": pair,
            "direction": signal["direction"],
            "z_score": round(signal["z_score"], 4),
            "z_mean": round(z_mean, 8),
            "z_std": round(z_std, 8),
            "return": round(ret, 8),
            "atr": round(signal["atr"], 8),
            "confidence": round(signal["confidence"], 4),
            "bar_time": completed_bar["time"],
            "bar_open": round(completed_bar["open"], 5),
            "bar_high": round(completed_bar["high"], 5),
            "bar_low": round(completed_bar["low"], 5),
            "bar_close": round(completed_bar["close"], 5),
            "stop_a": self.cfg.get("stop_a"),
            "trig_a": self.cfg.get("trig_a"),
            "gap_a": self.cfg.get("gap_a"),
            "tp_mult": self.cfg.get("tp_mult"),
            "min_tp_pips": self.cfg.get("min_tp_pips"),
            "z_window": self.cfg.get("z_window"),
            "z_buf_size": len(z_buf),
        })

    def on_fill(self, fill, signal):
        pair = fill["pair"]
        self._active_trades[pair] = {
            "entry_time": fill["entry_time"],
            "entry_price": fill["entry_price"],
            "direction": fill["direction"],
            "z_score": signal.get("z_score"),
            "tp_price": signal.get("tp_price"),
        }
        self._log("fill", {
            "pair": pair,
            "ticket": fill.get("ticket"),
            "direction": fill["direction"],
            "entry_price": round(fill["entry_price"], 5),
            "spread": round(fill.get("spread", 0), 8),
            "slip": round(fill.get("slip", 0), 8),
            "z_score": round(signal.get("z_score", 0), 4),
            "atr": round(signal.get("atr", 0), 8),
            "tp_price": round(signal.get("tp_price", 0), 5) if signal.get("tp_price") else None,
        })

    def on_bar(self, pair, bar, trail_pos=None):
        tinfo = None
        if trail_pos:
            stop_pips = abs(trail_pos["stop"] - trail_pos["entry"]) / 0.0001
            best_pips = abs(trail_pos["best"] - trail_pos["entry"]) / 0.0001
            entry = trail_pos["entry"]
            tinfo = {
                "best": round(trail_pos["best"], 5),
                "stop": round(trail_pos["stop"], 5),
                "best_diff_pips": round(abs(trail_pos["best"] - entry) / 0.0001, 2),
                "stop_diff_pips": round(abs(trail_pos["stop"] - entry) / 0.0001, 2),
                "trailing_active": trail_pos["best"] != entry,
            }
        self._log("bar", {
            "pair": pair,
            "time": bar["time"],
            "open": round(bar["open"], 5),
            "high": round(bar["high"], 5),
            "low": round(bar["low"], 5),
            "close": round(bar["close"], 5),
            "trail": tinfo,
        })

    def on_close(self, pair, exit_price, gross_pnl, reason, trail_exit=None):
        trade = self._active_trades.pop(pair, {})
        hold = int(time.time()) - trade.get("entry_time", int(time.time()))
        entry = trade.get("entry_price", 0)
        pips = (exit_price - entry) / 0.0001 * trade.get("direction", 0)
        self._log("close", {
            "pair": pair,
            "direction": trade.get("direction"),
            "entry_price": round(entry, 5),
            "exit_price": round(exit_price, 5),
            "gross_pnl": round(gross_pnl, 2),
            "pips": round(pips, 2),
            "hold_seconds": hold,
            "reason": reason,
            "z_score": trade.get("z_score"),
            "trail_exit": round(trail_exit, 5) if trail_exit else None,
        })


def _mt5_healthy(feed):
    try:
        return feed.mt5.terminal_info() is not None
    except Exception:
        return False


def _heartbeat_loop(login, name, stop_event):
    while not stop_event.is_set():
        time.sleep(HEARTBEAT_INTERVAL)
        acct_registry.heartbeat(login, name)


def _get_mt5_close_info(mt5, mt5_ticket):
    """Get actual exit price and profit from MT5 history for a closed position.
    Uses 0 as from_dt to avoid server-timezone (UTC+3) vs local-time mismatch."""
    try:
        deals = mt5.history_deals_get(0, datetime.now() + timedelta(days=1), position=mt5_ticket)
        if deals:
            for d in deals:
                if d.profit != 0:
                    return {"exit_price": d.price, "gross_pnl": d.profit}
    except Exception:
        pass
    return None


def main():
    cfg = STRATEGIES[STRATEGY_NAME]
    mt5_path = cfg.get("mt5_path")
    run_id = int(time.time())

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

    stats = SessionStats(run_id=run_id)
    logger = Logger(STRATEGY_NAME)
    seed_history(feed)

    trail_mgr = TrailingStopManager(cfg)
    risk = Risk(cfg)
    exec_ = Executor(feed, logger, magic=cfg.get("magic", 202410), run_id=run_id)
    dash = Dashboard()
    dash.set_status("starting")
    dash.start()

    log_path = logger.start()
    print(f"Logging to {log_path}", file=sys.stderr)
    print(f"RUN_ID={run_id}  (trade log: paper_trade/trade_logs/run_{run_id}.jsonl)", file=sys.stderr)

    tracer = TradeTracer(trail_mgr, stats, cfg).start()
    _completed_bars_prev = {}

    entry_queue = []
    _last_gc = time.time()

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

            # Process pending delayed entries
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
                        lot = cfg["lot_size"]
                        meta = {"z_score": signal.get("z_score"), "atr": signal.get("atr"),
                                "tp_price": signal.get("tp_price")}
                        fill = exec_.submit_market(pair, direction, lot,
                                                   timestamp=current_ts, signal_meta=meta)
                        if fill:
                            atr_v = signal.get("atr", 1)
                            trail_mgr.add(pair, direction, fill["entry_price"], atr_v,
                                           lot_size=lot, spread=cur_spread, timestamp=current_ts,
                                           tp_price=meta.get("tp_price"),
                                           mt5_ticket=fill.get("ticket"))
                            stats.record_fill(fill)
                            tracer.on_fill(fill, meta)
                            dash.set_last_signal(
                                f"{pair}{'+' if direction > 0 else '-'} @{fill['entry_price']:.5f}")
                else:
                    remaining.append(eq)
            entry_queue = remaining

            # Update spread baselines
            for pair in cfg["pairs"]:
                p = data.get(pair, {})
                if "spread" in p:
                    risk.update_spread_baseline(pair, p["spread"])

            # Generate signals from tick data + collect bar completions
            signals, completed_bars = generate_signal(data, current_time=current_ts)
            for signal in signals:
                bar_for_signal = completed_bars.get(signal["pair"])
                if bar_for_signal:
                    tracer.on_signal(signal["pair"], signal, bar_for_signal)
                if signal.get("confidence", 0) < MIN_CONFIDENCE:
                    continue
                if trail_mgr.pair_count(signal["pair"]) >= 1:
                    logger.log("SKIP", signal["pair"], "already_open")
                    stats.record_reject(signal["pair"], "already_open")
                elif any(eq["pair"] == signal["pair"] for eq in entry_queue):
                    logger.log("SKIP", signal["pair"], "queued_already")
                    stats.record_reject(signal["pair"], "queued_already")
                else:
                    delay = signal.get("delay_s", 0)
                    entry_queue.append({
                        "pair": signal["pair"], "direction": signal["direction"],
                        "fire_at": current_ts + delay, "signal": signal,
                    })

            # Sync trail_mgr with MT5 — remove positions already closed by TP or manual
            if feed.mode == "live":
                for t in list(trail_mgr.positions.keys()):
                    p_ = trail_mgr.positions[t]
                    mt5_ticket = p_.get("mt5_ticket")
                    if mt5_ticket is None:
                        continue
                    mt5_positions = feed.mt5.positions_get(ticket=mt5_ticket)
                    if mt5_positions is None:
                        # MT5 error/glitch (e.g. weekend/low-volume) — don't assume closed
                        continue
                    if len(mt5_positions) > 0:
                        # Position still open at MT5
                        continue
                    # Position was truly closed (TP hit or manual close from MT5)
                    trail_mgr.positions.pop(t)
                    close_info = _get_mt5_close_info(feed.mt5, mt5_ticket)
                    if close_info:
                        stats.record_close({**p_, **close_info, "exit_time": current_ts})
                        tracer.on_close(p_["pair"], close_info["exit_price"], close_info["gross_pnl"], "tp_manual")
                    else:
                        stats.record_close({**p_, "exit_price": p_["entry"], "gross_pnl": 0, "exit_time": current_ts})

            # Bar-level trailing stop checks (matches backtest OHLC logic)
            for pair, bar in completed_bars.items():
                trail_positions = [p for p in trail_mgr.positions.values() if p["pair"] == pair]
                trail_pos = trail_positions[0] if trail_positions else None
                tracer.on_bar(pair, bar, trail_pos=trail_pos)
                closed_trades = trail_mgr.check_bars(pair, bar, timestamp=current_ts)
                for ct in closed_trades:
                    exit_price = ct.get("exit", bar["close"])
                    exit_reason = ct.get("reason", "stop")
                    pnl_info = exec_.close_position(
                        {"pair": ct["pair"], "direction": ct["direction"],
                         "entry_price": ct["entry"], "lot_size": ct.get("lot_size", cfg["lot_size"]),
                         "ticket": ct.get("mt5_ticket")},
                        exit_price=exit_price, exit_time=current_ts)
                    if pnl_info:
                        stats.record_close({**ct, **pnl_info})
                        tracer.on_close(ct["pair"], exit_price, pnl_info.get("gross_pnl", 0), exit_reason, trail_exit=ct.get("exit"))
                        dash.set_last_signal(
                            f"{ct['pair']} X ${pnl_info.get('gross_pnl', 0):.2f}")

            # Expiry checks
            expired = trail_mgr.check_expiry(current_ts)
            for et in expired:
                bid = data.get(et["pair"], {}).get("bid", 0)
                ask = data.get(et["pair"], {}).get("ask", 0)
                exit_price = bid if et["direction"] == 1 else ask
                pnl_info = exec_.close_position(
                    {"pair": et["pair"], "direction": et["direction"],
                     "entry_price": et["entry"], "lot_size": et.get("lot_size", cfg["lot_size"]),
                     "ticket": et.get("mt5_ticket")},
                    exit_price=exit_price, exit_time=current_ts)
                if pnl_info:
                    stats.record_close({**et, **pnl_info})
                    tracer.on_close(et["pair"], exit_price, pnl_info.get("gross_pnl", 0), "expiry")

            snap = stats.snapshot()
            snap["run_id"] = run_id
            dash.push_snapshot(snap)
            if time.time() - _last_gc > GC_INTERVAL:
                gc.collect()
                _last_gc = time.time()
            time.sleep(LOOP_INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        stop_heartbeat.set()
        dash.stop()
        stats.close()
        tracer.close()
        feed.close()
        logger.close()
        acct_registry.release(mt5_account, STRATEGY_NAME)
        print(f"Account {mt5_account} released", file=sys.stderr)


if __name__ == "__main__":
    main()
