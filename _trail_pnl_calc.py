"""Trailing stop PnL + expiry PnL for the 10 orphaned trades."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import MetaTrader5 as mt5
from paper_trade.strategies.m1_z_reversal.strategy import TrailingStopManager, CONFIG

mt5.initialize()

# Live trade data from MT5 check
trades = [
    ("EURUSD", -1, 1.14178, 0.02),
    ("EURUSD", -1, 1.14213, 0.10),
    ("EURUSD", -1, 1.14214, 0.09),
    ("EURJPY",  1, 186.16100, 0.01),
    ("EURJPY",  1, 186.13800, 0.10),
    ("EURJPY",  1, 186.13800, 0.11),
    ("EURJPY",  1, 186.12400, 0.11),
    ("EURJPY",  1, 186.12300, 0.09),
    ("EURJPY",  1, 186.12200, 0.11),
    ("EURJPY",  1, 186.12200, 0.08),
]

def fetch_bars(pair, count=500):
    rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M1, 0, count)
    if rates is None:
        return []
    return [{"time": int(r[0]), "open": float(r[1]), "high": float(r[2]),
             "low": float(r[3]), "close": float(r[4])} for r in rates]

eurusd_bars = fetch_bars("EURUSD")
eurjpy_bars = fetch_bars("EURJPY")
bars_map = {"EURUSD": eurusd_bars, "EURJPY": eurjpy_bars}

def pip_value_usd(pair, price):
    if "JPY" in pair:
        return 0.01 / price * 100000
    else:
        return 10.0

def run_simulation(mode="trailing_stop"):
    """mode: 'trailing_stop' or 'expiry_54m'"""
    total_pnl = 0
    results = []

    for pair, direction, entry, volume in trades:
        bars = bars_map[pair]
        pip_size = 0.01 if "JPY" in pair else 0.0001

        # Find entry time from bars
        entry_time = None
        for b in bars:
            if abs(b["close"] - entry) / pip_size < 3:
                entry_time = b["time"]
                break
        if entry_time is None:
            for b in bars:
                if b["low"] <= entry <= b["high"]:
                    entry_time = b["time"]
                    break
        if entry_time is None:
            continue

        cfg = CONFIG.copy()
        tsm = TrailingStopManager(cfg)
        atr_v = 0.0015 if "JPY" not in pair else 0.020
        ticket = tsm.add(pair, direction, entry, atr_v, lot_size=volume,
                         spread=0, timestamp=entry_time)

        exit_price = None
        exit_reason = "open"

        for j, b in enumerate(bars):
            if b["time"] <= entry_time:
                continue
            bid = b["close"]

            if mode == "trailing_stop":
                closed = tsm.update(bid, bid, b["time"])
                for cp in closed:
                    if cp["ticket"] == ticket:
                        exit_price = bid
                        exit_reason = "trailing_stop"
                        break
                if exit_price is not None:
                    break
            elif mode == "expiry_54m":
                max_hold = cfg.get("max_hold_min", 54) * 60
                if b["time"] - entry_time >= max_hold:
                    expired = tsm.check_expiry(b["time"])
                    for cp in expired:
                        if cp["ticket"] == ticket:
                            exit_price = b["close"]
                            exit_reason = "expiry_54m"
                            break
                    if exit_price is not None:
                        break

        if exit_price is None:
            exit_price = bars[-1]["close"]
            exit_reason = "end_of_data"

        raw_pnl = (exit_price - entry) * direction
        pnl_pips = raw_pnl / pip_size
        pv = pip_value_usd(pair, (entry + exit_price) / 2)
        pnl_usd = round(pnl_pips * pv * volume, 2)
        total_pnl += pnl_usd

        dir_s = "BUY" if direction == 1 else "SELL"
        results.append({
            "pair": pair, "dir": dir_s, "entry": entry, "exit": exit_price,
            "vol": volume, "pips": round(pnl_pips, 2), "usd": pnl_usd,
            "reason": exit_reason
        })

    return results, total_pnl


def print_results(results, total_pnl, label):
    print(f"\n  {label}")
    print(f"  {'=' * 55}")
    for pair_name in ["EURUSD", "EURJPY"]:
        pr = [r for r in results if r["pair"] == pair_name]
        if not pr:
            continue
        pair_pnl = sum(r["usd"] for r in pr)
        print(f"  {pair_name} ({len(pr)} trades, total: ${pair_pnl:+.2f})")
        for r in pr:
            print(f"    {r['dir']} {r['entry']:.5f} -> {r['exit']:.5f}  "
                  f"vol={r['vol']}  {r['pips']:+.2f}p  ${r['usd']:+.2f}  [{r['reason']}]")
    print(f"  {'─' * 55}")
    print(f"  TOTAL: ${total_pnl:+.2f}")

# Run trailing stop simulation
ts_results, ts_pnl = run_simulation("trailing_stop")
print_results(ts_results, ts_pnl, "TRAILING STOP (bar-level simulation)")

# Run expiry simulation
ex_results, ex_pnl = run_simulation("expiry_54m")
print_results(ex_results, ex_pnl, "EXPIRY AT 54 MIN (should-have-been behavior)")

# Compare with actual outcome
print(f"\n  ACTUAL OUTCOME (from MT5 balance change): +$1.61")
print(f"  (positions held ~2.6h, closed by hydration expiry at ~18:30 UTC)")

print()
print(f"  COMPARISON SUMMARY:")
print(f"    Trailing stop:  ${ts_pnl:+.2f}")
print(f"    Expiry @ 54m:   ${ex_pnl:+.2f}")
print(f"    Actual (held):   +$1.61")
print(f"    Best:            expiry @ 54m (if trailing stops are too tight for EURJPY)")

mt5.shutdown()
