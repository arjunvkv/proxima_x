"""Live execution probe: fire random market orders, measure fill quality.
Fires ~12 trades (2 per pair) across the 6 cross pairs, holds ~60s, closes.
Records: latency, slippage, spread at entry/close, fill price vs requested.
Prints summary table + saves raw data to paper_trade/trade_logs/probe_<ts>.jsonl
"""

import json, os, sys, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from paper_trade.components import pip_value_usd

PAIRS = ["GBPNZD", "EURNZD", "GBPAUD", "EURAUD", "GBPCAD", "AUDNZD"]
_PIP_SIZE = {"JPY": 0.01}
_PIP_DEFAULT = 0.0001
HOLD_SEC = 60
TRADES_PER_PAIR = 2
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "trade_logs")


def pip_size(pair):
    return _PIP_SIZE.get(pair[-3:], _PIP_DEFAULT)


def calc_pnl(pair, entry, exit, lot, direction):
    ps = pip_size(pair)
    pips = (exit - entry) / ps
    pv = pip_value_usd(pair, exit)
    return round(pips * pv * lot * direction, 2)


def log_result(results, pair, side, metrics):
    results.append({"pair": pair, "side": side, **metrics})


def write_log(results, run_ts):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"probe_{run_ts}.jsonl")
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nRaw data saved to {path}", file=sys.stderr)
    return path


def print_summary(results):
    """Print a clean summary table of all fills."""
    print("\n" + "=" * 95, file=sys.stderr)
    print("  EXECUTION PROBE — ENTRY FILLS", file=sys.stderr)
    print("=" * 95, file=sys.stderr)
    header = f"  {'Pair':>8} {'Side':>5} {'Request':>10} {'Fill':>10} {'Slippage':>10} {'Spr B4':>9} {'Spr Af':>9} {'Lat(μs)':>8} {'Dev':>5}"
    print(header, file=sys.stderr)
    print("  " + "-" * 95, file=sys.stderr)

    entries = [r for r in results if r.get("event") == "entry"]
    all_slip_pips = []
    all_lat_us = []
    all_spread_before = []
    all_spread_after = []

    for r in entries:
        slip_pips = r["slippage_pips"]
        lat_us = r["latency_us"]
        sp_b = r["spread_before"]
        sp_a = r["spread_after"]
        all_slip_pips.append(slip_pips)
        all_lat_us.append(lat_us)
        all_spread_before.append(sp_b)
        all_spread_after.append(sp_a)

        line = (f"  {r['pair']:>8} {r['direction']:>5} "
                f"{r['request_price']:>10.5f} {r['fill_price']:>10.5f} "
                f"{slip_pips:>10.3f} {sp_b:>9.5f} {sp_a:>9.5f} "
                f"{lat_us:>8.0f} {r.get('deviation', 10):>5}")
        print(line, file=sys.stderr)

    print("  " + "-" * 95, file=sys.stderr)
    if entries:
        print(f"  AVG:     {'':>13} {'':>10} {sum(all_slip_pips)/len(all_slip_pips):>10.3f} "
              f"{sum(all_spread_before)/len(all_spread_before):>9.5f} {sum(all_spread_after)/len(all_spread_after):>9.5f} "
              f"{sum(all_lat_us)/len(all_lat_us):>8.0f}", file=sys.stderr)
        print(f"  MAX:     {'':>13} {'':>10} {max(all_slip_pips):>10.3f}", file=sys.stderr)
        print(f"  MIN:     {'':>13} {'':>10} {min(all_slip_pips):>10.3f}", file=sys.stderr)

    # Close summary
    closes = [r for r in results if r.get("event") == "close"]
    if closes:
        print("\n" + "=" * 95, file=sys.stderr)
        print("  EXECUTION PROBE — CLOSE FILLS", file=sys.stderr)
        print("=" * 95, file=sys.stderr)
        print(header, file=sys.stderr)
        print("  " + "-" * 95, file=sys.stderr)
        for r in closes:
            line = (f"  {r['pair']:>8} {r['direction']:>5} "
                    f"{r['request_price']:>10.5f} {r['fill_price']:>10.5f} "
                    f"{r['slippage_pips']:>10.3f} {r['spread_before']:>9.5f} {r['spread_after']:>9.5f} "
                    f"{r['latency_us']:>8.0f}")
            print(line, file=sys.stderr)

    # PnL per pair
    print("\n" + "=" * 95, file=sys.stderr)
    print("  PNL PER TRADE", file=sys.stderr)
    print("=" * 95, file=sys.stderr)
    print(f"  {'Pair':>8} {'Dir':>5} {'Entry':>10} {'Exit':>10} {'PnL($)':>8} {'Hold(s)':>8}", file=sys.stderr)
    print("  " + "-" * 95, file=sys.stderr)
    total_pnl = 0
    wins = 0
    total_trades = 0
    for entry in entries:
        pair = entry["pair"]
        dir_ = entry["direction"]
        entry_price = entry["fill_price"]
        entry_time = entry["ts"]
        close = next((c for c in closes if c.get("fill_ticket") == entry.get("fill_ticket") and c["pair"] == pair), None)
        if close:
            exit_price = close["fill_price"]
            hold = close["ts"] - entry_time
            direction = 1 if dir_ == "BUY" else -1
            pnl = calc_pnl(pair, entry_price, exit_price, 1.0, direction)
            total_pnl += pnl
            total_trades += 1
            if pnl > 0:
                wins += 1
            print(f"  {pair:>8} {dir_:>5} {entry_price:>10.5f} {exit_price:>10.5f} {pnl:>8.2f} {hold:>8}", file=sys.stderr)
    print("  " + "-" * 95, file=sys.stderr)
    if total_trades > 0:
        wr = wins / total_trades * 100
        print(f"  TOTAL PNL: ${total_pnl:.2f}  |  WR: {wr:.0f}%  ({wins}/{total_trades})", file=sys.stderr)
    print("=" * 95, file=sys.stderr)


def main():
    import MetaTrader5 as mt5
    run_ts = int(time.time())

    # Connect
    init = mt5.initialize()
    if not init:
        print(f"MT5 init failed: {mt5.last_error()}", file=sys.stderr)
        sys.exit(1)
    info = mt5.account_info()
    if info is None:
        logged = mt5.login(5053225887, password="9mgfii383z", server="MetaQuotes-Demo")
        if not logged:
            print(f"MT5 login failed: {mt5.last_error()}", file=sys.stderr)
            mt5.shutdown()
            sys.exit(1)
        info = mt5.account_info()
    print(f"Connected: login={info.login} balance=${info.balance:.2f}", file=sys.stderr)

    # Generate trade plan: 2 per pair, mix of buys/sells
    random.seed(run_ts)
    trade_plan = []
    for pair in PAIRS:
        for _ in range(TRADES_PER_PAIR):
            direction = random.choice(["BUY", "SELL"])
            trade_plan.append((pair, direction))
    random.shuffle(trade_plan)

    results = []
    open_positions = []  # (pair, direction, ticket, entry_price, entry_ts)

    print(f"\nFiring {len(trade_plan)} probe trades...\n", file=sys.stderr)

    try:
        for i, (pair, direction) in enumerate(trade_plan):
            symbol = mt5.symbol_info(pair)
            if symbol is None:
                print(f"  SKIP {pair}: symbol not found", file=sys.stderr)
                continue

            tick = mt5.symbol_info_tick(pair)
            if tick is None:
                print(f"  SKIP {pair}: no tick", file=sys.stderr)
                continue

            mt5_order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
            price = tick.ask if direction == "BUY" else tick.bid

            spread_before = tick.ask - tick.bid

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pair,
                "volume": 1.0,
                "type": mt5_order_type,
                "price": price,
                "deviation": 10,
                "magic": 999999,
                "comment": f"probe_{run_ts}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC if (symbol.filling_mode & 2) else (mt5.ORDER_FILLING_FOK if (symbol.filling_mode & 1) else mt5.ORDER_FILLING_RETURN),
            }

            t_before = time.time_ns()
            result = mt5.order_send(request)
            t_after = time.time_ns()

            latency_ns = t_after - t_before
            latency_us = latency_ns / 1000.0

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"  {i+1:>2}. {pair:>8} {direction:>5} — RETCODE {result.retcode}", file=sys.stderr)
                continue

            # Get tick after fill
            tick2 = mt5.symbol_info_tick(pair)
            spread_after = (tick2.ask - tick2.bid) if tick2 else spread_before

            fill_price = result.price
            slip = abs(fill_price - price)
            slip_pips = slip / pip_size(pair)
            slip_usd = slip_pips * pip_value_usd(pair, fill_price)

            log_result(results, pair, direction, {
                "event": "entry",
                "ts": int(time.time()),
                "direction": direction,
                "request_price": round(price, 5),
                "fill_price": round(fill_price, 5),
                "slippage_pips": round(slip_pips, 4),
                "slippage_usd": round(slip_usd, 2),
                "latency_ns": latency_ns,
                "latency_us": round(latency_us, 1),
                "spread_before": round(spread_before, 5),
                "spread_after": round(spread_after, 5),
                "deviation": 10,
                "retcode": result.retcode,
                "fill_ticket": result.order,
                "deal_ticket": result.deal,
                "comment": result.comment,
            })

            open_positions.append((pair, direction, result.order, fill_price, int(time.time())))
            print(f"  {i+1:>2}. {pair:>8} {direction:>5} @ {fill_price:.5f}  slip={slip_pips:.3f}pips  lat={latency_us:.0f}μs  spread={spread_before:.1f}", file=sys.stderr)

            # Small delay between entries
            if i < len(trade_plan) - 1:
                time.sleep(2)

        # Wait for hold time
        if open_positions:
            print(f"\nHolding {len(open_positions)} positions for {HOLD_SEC}s...", file=sys.stderr)
            held = 0
            while held < HOLD_SEC:
                time.sleep(1)
                held += 1
                if held % 15 == 0:
                    print(f"  hold {held}s", file=sys.stderr)

            # Close all
            print(f"\nClosing {len(open_positions)} positions...", file=sys.stderr)
            for pair, direction, ticket, entry_price, entry_ts in open_positions:
                tick = mt5.symbol_info_tick(pair)
                if tick is None:
                    continue

                close_order_type = mt5.ORDER_TYPE_SELL if direction == "BUY" else mt5.ORDER_TYPE_BUY
                close_price = tick.bid if direction == "BUY" else tick.ask

                pos_type = 0 if direction == "BUY" else 1
                positions = mt5.positions_get(symbol=pair)
                pos = None
                if positions:
                    for p in positions:
                        if p.magic == 999999 and p.type == pos_type:
                            pos = p
                            break

                if pos is None:
                    print(f"  Position not found for {pair} {direction} ticket={ticket}", file=sys.stderr)
                    continue

                spread_before_close = tick.ask - tick.bid

                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pair,
                    "volume": 1.0,
                    "type": close_order_type,
                    "position": pos.ticket,
                    "price": close_price,
                "deviation": 100,
                    "magic": 999999,
                    "comment": f"probe_c_{run_ts}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC if (mt5.symbol_info(pair).filling_mode & 2) else (mt5.ORDER_FILLING_FOK if (mt5.symbol_info(pair).filling_mode & 1) else mt5.ORDER_FILLING_RETURN),
                }

                t_before = time.time_ns()
                result = mt5.order_send(close_request)
                t_after = time.time_ns()

                lat_us = (t_after - t_before) / 1000.0

                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"  CLOSE FAIL {pair} {direction}: retcode={result.retcode}", file=sys.stderr)
                    continue

                tick3 = mt5.symbol_info_tick(pair)
                spread_after_close = (tick3.ask - tick3.bid) if tick3 else spread_before_close
                close_fill = result.price
                slip = abs(close_fill - close_price)
                slip_pips = slip / pip_size(pair)

                log_result(results, pair, direction, {
                    "event": "close",
                    "ts": int(time.time()),
                    "direction": direction,
                    "request_price": round(close_price, 5),
                    "fill_price": round(close_fill, 5),
                    "slippage_pips": round(slip_pips, 4),
                    "latency_us": round(lat_us, 1),
                    "spread_before": round(spread_before_close, 5),
                    "spread_after": round(spread_after_close, 5),
                    "fill_ticket": ticket,
                    "retcode": result.retcode,
                })

                print(f"  CLOSE {pair:>8} {direction:>5} @ {close_fill:.5f}  slip={slip_pips:.3f}pips  lat={lat_us:.0f}μs", file=sys.stderr)
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted, closing all...", file=sys.stderr)
    finally:
        # Emergency close any remaining positions
        for pair, dir_, _, _, _ in open_positions:
            pos_type = 0 if dir_ == "BUY" else 1
            positions = mt5.positions_get(symbol=pair)
            if positions:
                for p in positions:
                    if p.magic == 999999:
                        close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
                        tick = mt5.symbol_info_tick(pair)
                        if tick is None:
                            continue
                        cp = tick.bid if p.type == 0 else tick.ask
                        req = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": pair, "volume": p.volume,
                            "type": close_type, "position": p.ticket,
                            "price": cp, "deviation": 10,
                            "magic": 999999, "comment": "probe_cleanup",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC if (mt5.symbol_info(pair).filling_mode & 2) else (mt5.ORDER_FILLING_FOK if (mt5.symbol_info(pair).filling_mode & 1) else mt5.ORDER_FILLING_RETURN),
                        }
                        mt5.order_send(req)

    # Write raw data
    write_log(results, run_ts)

    # Print summary
    print_summary(results)

    mt5.shutdown()


if __name__ == "__main__":
    main()
