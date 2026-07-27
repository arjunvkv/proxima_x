"""Verify M1 Z-Reversal live trades against backtest signals.

Fetches today's M1 data from MT5, runs the exact strategy pipeline,
compares signal timestamps/prices with actual trades, and simulates
trailing stop exits to check profitability.
"""
import sys, os, time, random
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(__file__))
import MetaTrader5 as mt5

from paper_trade.strategies.m1_z_reversal.strategy import (
    ZBuffer, ATRBuffer, RollingStats, TrailingStopManager, CONFIG
)
from paper_trade.strategies.m1_z_reversal.strategy import PairState

# ─── Actual live trade data from MT5 ─────────────────────────────

LIVE_TRADES = {
    "EURUSD": [
        {"dir": -1, "entry": 1.14178, "vol": 0.02, "time": None},
        {"dir": -1, "entry": 1.14213, "vol": 0.10, "time": None},
        {"dir": -1, "entry": 1.14214, "vol": 0.09, "time": None},
    ],
    "EURJPY": [
        {"dir": 1, "entry": 186.161, "vol": 0.01, "time": None},
        {"dir": 1, "entry": 186.138, "vol": 0.10, "time": None},
        {"dir": 1, "entry": 186.138, "vol": 0.11, "time": None},
        {"dir": 1, "entry": 186.124, "vol": 0.11, "time": None},
        {"dir": 1, "entry": 186.123, "vol": 0.09, "time": None},
        {"dir": 1, "entry": 186.122, "vol": 0.11, "time": None},
        {"dir": 1, "entry": 186.122, "vol": 0.08, "time": None},
    ],
}

# ─── Fetch trade times from MT5 positions history ────────────────

def fetch_trade_times():
    """Populate LIVE_TRADES times by matching entries in position or deal history."""
    mt5.initialize()
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    since = now - timedelta(hours=6)
    # Try positions first
    positions = mt5.positions_get()
    if positions:
        for pos in positions:
            for pair, trades in LIVE_TRADES.items():
                if pos.symbol != pair:
                    continue
                for t in trades:
                    if t["time"] is not None:
                        continue
                    if abs(pos.price_open - t["entry"]) < 0.0005:
                        t["time"] = int(pos.time)
    # Fall back to history deals
    history = mt5.history_deals_get(since, now)
    if history:
        for deal in history:
            for pair, trades in LIVE_TRADES.items():
                if deal.symbol != pair:
                    continue
                for t in trades:
                    if t["time"] is not None:
                        continue
                    if abs(deal.price - t["entry"]) < 0.0005:
                        t["time"] = int(deal.time)
    return LIVE_TRADES

# ─── Signal detection on M1 bars ─────────────────────────────────

def eval_signal(pair, z_thresh=2.0, atr_pctl=0.25):
    """Run PairState-like signal detection over M1 bars.
    Returns list of signal dicts.
    """
    n_bars = 500
    rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M1, 0, n_bars)
    if rates is None or len(rates) < 120:
        print(f"  Not enough M1 data for {pair}: {len(rates) if rates else 0}")
        return [], []

    zb = ZBuffer(window=50)
    atrb = ATRBuffer(window=20)
    gate = RollingStats(window=100)
    last_close = None

    signals = []
    bar_records = []

    for i, r in enumerate(rates):
        bar = {
            "time": int(r[0]), "open": float(r[1]),
            "high": float(r[2]), "low": float(r[3]), "close": float(r[4]),
        }
        bar_records.append(bar)

        if last_close is None:
            last_close = bar["close"]
            rng = bar["high"] - bar["low"]
            atrb.add(rng)
            continue

        ret = bar["close"] - last_close
        z_v = zb.z_score(ret)
        atr_v = atrb.value()
        gate_v = gate.quantile(atr_pctl)

        sig = None
        if z_v is not None and atr_v is not None and gate_v is not None:
            if abs(z_v) > z_thresh and atr_v > gate_v:
                direction = -1 if z_v > 0 else 1
                sig = {
                    "pair": pair,
                    "direction": direction,
                    "bar_time": bar["time"],
                    "close": bar["close"],
                    "z_score": round(z_v, 3),
                    "atr": round(atr_v, 6),
                    "gate": round(gate_v, 6) if gate_v else None,
                }
                signals.append(sig)

        zb.add(ret)
        if atr_v is not None:
            gate.add(atr_v)
        rng = bar["high"] - bar["low"]
        atrb.add(rng)
        last_close = bar["close"]

    return signals, bar_records


# ─── Trailing stop simulation on M1 bars ─────────────────────────

def simulate_trail(signals, bar_records, pair):
    """Simulate TrailingStopManager on subsequent M1 bars.
    Returns trades with exit info.
    """
    cfg = CONFIG.copy()
    tsm = TrailingStopManager(cfg)
    results = []

    # Build lookup: bar_time -> bar
    bar_map = {b["time"]: b for b in bar_records}

    for sig in signals:
        dir_m = sig["direction"]
        entry_bar = bar_map.get(sig["bar_time"])
        if not entry_bar:
            continue

        entry_price = entry_bar["close"]
        atr_v = sig["atr"]
        ticket = tsm.add(pair, dir_m, entry_price, atr_v, timestamp=sig["bar_time"])

        # Track through subsequent bars
        closed_by_stop = False
        exit_price = None
        exit_time = None

        bar_times = sorted([b["time"] for b in bar_records if b["time"] > sig["bar_time"]])
        for bt in bar_times:
            b = bar_map[bt]
            bid = b["close"]
            # For trailing stop check, use close as both bid/ask approximation
            closed = tsm.update(bid, bid, bt)
            for cp in closed:
                if cp["ticket"] == ticket:
                    exit_price = bid
                    exit_time = bt
                    closed_by_stop = True
                    break
            if closed_by_stop:
                break

        # Expiry check
        if not closed_by_stop and bar_times:
            last_bt = bar_times[-1]
            expired = tsm.check_expiry(last_bt)
            for cp in expired:
                if cp["ticket"] == ticket:
                    exit_price = bar_map[last_bt]["close"]
                    exit_time = last_bt

        # If still open, close at last bar
        if exit_price is None and bar_times:
            exit_price = bar_map[bar_times[-1]]["close"]
            exit_time = bar_times[-1]

        raw_pnl = (exit_price - entry_price) * dir_m if exit_price else 0
        pip_size = 0.01 if "JPY" in pair else 0.0001
        pnl_pips = raw_pnl / pip_size
        dur_min = (exit_time - sig["bar_time"]) / 60 if exit_time else 0

        results.append({
            "bar_time": sig["bar_time"],
            "dir": "BUY" if dir_m > 0 else "SELL",
            "entry": entry_price,
            "exit": exit_price or 0,
            "z": sig["z_score"],
            "atr": sig["atr"],
            "pnl_pips": round(pnl_pips, 2),
            "dur_min": round(dur_min, 1),
            "reason": "stop" if closed_by_stop else "expiry",
        })

    return results


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("M1 Z-REVERSAL BACKTEST vs LIVE TRADE VERIFICATION")
    print("=" * 70)

    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return

    account = mt5.account_info()
    print(f"Account: {account.login}  Balance: ${account.balance:.2f}")
    print()

    for pair in ["EURUSD", "EURJPY"]:
        print(f"\n{'─'*70}")
        print(f"  PAIR: {pair}")
        print(f"{'─'*70}")

        signals, bar_records = eval_signal(pair)
        print(f"  M1 bars loaded: {len(bar_records)}")
        print(f"  Signals detected: {len(signals)}")

        # Convert bar times to readable
        from datetime import datetime, timezone
        for s in signals:
            t = datetime.fromtimestamp(s["bar_time"], tz=timezone.utc)
            s["time_str"] = t.strftime("%H:%M:%S")
            dir_s = "BUY" if s["direction"] > 0 else "SELL"
            print(f"    {t.strftime('%H:%M:%S')} {dir_s}  "
                  f"z={s['z_score']:+.3f}  close={s['close']:.5f}  "
                  f"atr={s['atr']:.6f}")

        # Find live trades for this pair
        live = LIVE_TRADES.get(pair, [])
        live_entry_times = {}
        for lt in live:
            t_str = "??:??:??"
            if lt["time"]:
                t_str = datetime.fromtimestamp(lt["time"], tz=timezone.utc).strftime("%H:%M:%S")
            live_entry_times[lt["entry"]] = lt["time"]
            print(f"    >>> LIVE trade: {'BUY' if lt['dir']>0 else 'SELL'} @ {lt['entry']:.5f} "
                  f"vol={lt['vol']} time={t_str}")

        # Match live trades to signals (pip-relative threshold)
        pip_size = 0.01 if "JPY" in pair else 0.0001
        max_pip_diff = 1.5  # allow up to 1.5 pips price diff
        max_price_diff = max_pip_diff * pip_size
        print(f"\n  ── Signal vs Live Match (max {max_pip_diff}p diff) ──")
        matched_live = set()
        for s in signals:
            for li, lt in enumerate(live):
                if li in matched_live:
                    continue
                price_diff = abs(s["close"] - lt["entry"])
                dir_match = s["direction"] == lt["dir"]
                time_ok = True
                if lt["time"] and s["bar_time"]:
                    time_diff = abs(s["bar_time"] - lt["time"])
                    time_ok = time_diff < 300

                if price_diff < max_price_diff and dir_match and time_ok:
                    matched_live.add(li)
                    lt_dir = "BUY" if lt["dir"] > 0 else "SELL"
                    sig_dir = "BUY" if s["direction"] > 0 else "SELL"
                    print(f"    ✓ Signal {s['time_str']} {sig_dir} "
                          f"z={s['z_score']:+.3f} @ {s['close']:.5f}")
                    print(f"      → Live  {lt_dir} @ {lt['entry']:.5f} vol={lt['vol']}  "
                          f"(Δ={price_diff/pip_size:.2f}p)")
                    break

        for li, lt in enumerate(live):
            lt_t = "??:??:??"
            if lt["time"]:
                lt_t = datetime.fromtimestamp(lt["time"], tz=timezone.utc).strftime("%H:%M:%S")
            if li in matched_live:
                continue
            print(f"    ✗ UNMATCHED live: {'BUY' if lt['dir']>0 else 'SELL'} "
                  f"@ {lt['entry']:.5f} vol={lt['vol']} time={lt_t}")

        print(f"  Matched {len(matched_live)}/{len(live)} live trades to signals")

        # Trailing stop simulation
        print(f"\n  ── Trailing Stop Simulation ──")
        results = simulate_trail(signals, bar_records, pair)
        wins = sum(1 for r in results if r["pnl_pips"] > 0)
        total = len(results)
        wr = wins / total * 100 if total > 0 else 0
        total_pnl = sum(r["pnl_pips"] for r in results)

        print(f"  Trades: {total}  Wins: {wins}  WR: {wr:.1f}%  Total: {total_pnl:+.2f}p")
        for r in results:
            t_str = datetime.fromtimestamp(r["bar_time"], tz=timezone.utc).strftime("%H:%M:%S")
            print(f"    {t_str} {r['dir']}  "
                  f"entry={r['entry']:.5f} exit={r['exit']:.5f}  "
                  f"z={r['z']:+.3f}  PnL={r['pnl_pips']:+.2f}p  "
                  f"dur={r['dur_min']}m  [{r['reason']}]")

    mt5.shutdown()
    print(f"\n{'='*70}")
    print("DONE")

if __name__ == "__main__":
    main()
