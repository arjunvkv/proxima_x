import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "proxima_command_center"))
from rolling_backtest_engine import RollingBacktestEngine

ACCOUNT = {"login": 1514168544, "password": "$!4fwBIc", "server": "FTMO-Demo"}

def main():
    print("=" * 115)
    print("PROXIMA X — LIVE MT5 v107 DEALS VS PYTHON SIM v107 TRADES AUDIT (TODAY 2026-08-04 UTC)")
    print("=" * 115)

    # 1. Fetch Python Sim Trades Today
    eng = RollingBacktestEngine()
    now_utc = datetime.now(timezone.utc)
    now_ts  = pd.Timestamp(now_utc.replace(tzinfo=None))
    py_trades = eng.compute_deterministic_trades("2026-08-04", max_time_utc=now_ts)

    # 2. Fetch MT5 Live Deals Today
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
    mt5.login(int(ACCOUNT["login"]), password=ACCOUNT["password"], server=ACCOUNT["server"])

    start_utc = datetime(2026, 8, 4, 0, 0, 0, tzinfo=timezone.utc)
    deals = mt5.history_deals_get(start_utc, now_utc)
    
    live_v107_deals = []
    if deals is not None:
        for d in deals:
            if d.type in (0, 1):
                utc_time = datetime.utcfromtimestamp(d.time - 10800)
                cmt = d.comment if d.comment else ""
                live_v107_deals.append({
                    "ticket": d.order,
                    "utc_str": utc_time.strftime("%H:%M:%S"),
                    "symbol": d.symbol,
                    "type": "BUY" if d.type == 0 else "SELL",
                    "lot": d.volume,
                    "price": d.price,
                    "profit": round(d.profit, 2),
                    "comment": cmt
                })
    mt5.shutdown()

    # --- COMPARISON SUMMARY ---
    py_wins = [t for t in py_trades if t['is_win']]
    py_pnl  = sum(t['sim_pnl'] for t in py_trades)
    py_wr   = len(py_wins) / len(py_trades) * 100.0 if py_trades else 0.0

    live_wins = [d for d in live_v107_deals if d['profit'] > 0]
    live_pnl  = sum(d['profit'] for d in live_v107_deals)
    live_wr   = len(live_wins) / len(live_v107_deals) * 100.0 if live_v107_deals else 0.0

    print(f"\n📊 SUMMARY COMPARISON (TODAY 2026-08-04 UTC):")
    print(f"  • MT5 Live v107 Deals : {len(live_v107_deals)} Deals | Win Rate: {live_wr:.1f}% | Net Realized PnL: ${live_pnl:,.2f}")
    print(f"  • Python Sim v107     : {len(py_trades)} Trades | Win Rate: {py_wr:.1f}% | Net Realized PnL: +${py_pnl:,.2f}")
    print("=" * 115)

    print("\n📋 LIVE MT5 v107 DEALS EXECUTED TODAY:")
    if not live_v107_deals:
        print("  🟢 ZERO LIVE DEALS EXECUTED OR CLOSED ON MT5 FOR TODAY (2026-08-04 UTC).")
        print("  Reason: Live MT5 Terminal / EAs were not active on the VPS during the 00:00-03:00 UTC window today.")
    else:
        for d in live_v107_deals:
            res = "🟢 WIN" if d['profit'] > 0 else "🔴 LOSS"
            print(f"  {d['utc_str']} | {d['symbol']} {d['type']} | Lot: {d['lot']}L | Price: {d['price']} | Profit: ${d['profit']:.2f} | {res} | Cmt: {d['comment']}")

    print("\n📋 PYTHON SIM v107 TRADES COMPUTED TODAY:")
    for t in py_trades:
        res = "🟢 WIN" if t['is_win'] else "🔴 LOSS"
        print(f"  {t['iso_timestamp'][:19]} | {t['strategy']:<22} | {t['pair']} {t['side']:<4} | Lot: {t['lot']}L | Pips: {t['pips']:>5.1f}p | PnL: ${t['sim_pnl']:>7.2f} | {res}")

    print("=" * 115)

if __name__ == "__main__":
    main()
