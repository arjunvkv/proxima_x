"""Trace one live trade from entry to exit, comparing against backtest simulation."""
import sys, os, time, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from paper_trade.core.feed import Feed
from strategy import CONFIG, PairState, seed_history, TrailingStopManager

LOG_FILE = f"trace_trade_{int(time.time())}.jsonl"
PAIRS = CONFIG["pairs"]

def log_event(event, data):
    line = json.dumps({"ts": int(time.time()), "event": event, **data})
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def main():
    feed = Feed(mode="live", pairs=PAIRS).connect()
    seed_history(feed)

    # Reconstruct PairState logic for shadow tracking
    states = {}
    for pair in PAIRS:
        ps = PairState(pair, CONFIG)
        bars = feed.copy_m1_history(pair, count=100)
        if bars:
            for b in bars:
                ps.seed_bar(b)
        states[pair] = ps

    shadow_mgr = TrailingStopManager(CONFIG)

    print("=== TRACE: Waiting for next trade entry ===", flush=True)

    while True:
        data = feed.current_bar()
        now = int(time.time())

        # Update shadow bar builders and detect bar completions
        for pair, p in data.items():
            bid = p.get("bid", 0)
            if bid <= 0:
                continue
            tick_time = p.get("time", now)
            bar, signal = states[pair].update(bid, tick_time)

            if bar is not None:
                # Log every completed bar
                log_event("bar", {
                    "pair": pair, "time": bar["time"],
                    "open": round(bar["open"], 5),
                    "high": round(bar["high"], 5),
                    "low": round(bar["low"], 5),
                    "close": round(bar["close"], 5),
                    "atr": round(states[pair].atr_buf.value() or 0, 8),
                })

                # Run shadow trailing stop checks
                closed = shadow_mgr.check_bars(pair, bar, timestamp=now)
                for ct in closed:
                    exit_price = ct.get("exit", bar["close"])
                    gross = (ct["exit"] - ct["entry"]) * ct["direction"]
                    log_event("shadow_close", {
                        "pair": ct["pair"], "direction": ct["direction"],
                        "entry": ct["entry"], "exit": exit_price,
                        "exit_reason": "stop",
                        "gross_pnl": round(gross, 8),
                        "hold_seconds": now - ct.get("entry_time", now),
                    })

            if signal is not None:
                log_event("signal", {
                    "pair": signal["pair"], "direction": signal["direction"],
                    "z_score": round(signal["z_score"], 4),
                    "atr": round(signal["atr"], 8),
                    "tp_price": round(signal["tp_price"], 5) if signal["tp_price"] else None,
                    "bar_time": signal["bar_time"],
                })

        # Check if MT5 has positions — compare with shadow
        try:
            import MetaTrader5 as mt5
            mt5_positions = mt5.positions_get()
            if mt5_positions:
                for mp in mt5_positions:
                    # Check if this is a new entry we haven't shadowed yet
                    if not shadow_mgr.pair_count(mp.symbol):
                        log_event("live_entry", {
                            "pair": mp.symbol,
                            "ticket": mp.ticket,
                            "direction": -1 if mp.type == 1 else 1,
                            "entry_price": round(mp.price_open, 5),
                            "volume": mp.volume,
                            "sl": round(mp.sl, 5),
                            "tp": round(mp.tp, 5),
                            "profit": round(mp.profit, 2),
                        })
                        # Add to shadow with estimated atr
                        direction = -1 if mp.type == 1 else 1
                        atr_v = states[mp.symbol].atr_buf.value() or 0.001
                        shadow_mgr.add(mp.symbol, direction, mp.price_open, atr_v,
                                        lot_size=mp.volume, timestamp=now,
                                        mt5_ticket=mp.ticket)

                    # Log ongoing status for tracked positions
                    elif shadow_mgr.pair_count(mp.symbol):
                        atr_v = states[mp.symbol].atr_buf.value() or 0.001
                        s = CONFIG["stop_a"] * atr_v
                        tg = CONFIG["trig_a"] * atr_v
                        gp = CONFIG["gap_a"] * atr_v
                        shadow_p = None
                        for t in shadow_mgr.positions.values():
                            if t["pair"] == mp.symbol:
                                shadow_p = t
                                break
                        log_event("live_status", {
                            "pair": mp.symbol,
                            "ticket": mp.ticket,
                            "profit": round(mp.profit, 2),
                            "price": round(mp.price_current, 5),
                            "direction": -1 if mp.type == 1 else 1,
                            "spread": round((mp.ask or mp.price_current) - (mp.bid or mp.price_current), 5),
                            "shadow_stop": round(shadow_p["stop"], 5) if shadow_p else None,
                            "shadow_s": round(s, 8),
                            "shadow_tg": round(tg, 8),
                            "shadow_gp": round(gp, 8),
                        })
        except Exception as e:
            log_event("mt5_error", {"error": str(e)})

        time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTrace stopped.", flush=True)
