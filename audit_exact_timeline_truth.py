import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import json

ACCOUNT = {"login": 1514168544, "password": "$!4fwBIc", "server": "FTMO-Demo"}

def fetch_live_deals(from_dt, to_dt):
    if not mt5.initialize():
        return []
    mt5.login(int(ACCOUNT["login"]), password=ACCOUNT["password"], server=ACCOUNT["server"])
    
    deals = mt5.history_deals_get(from_dt, to_dt)
    if deals is None or len(deals) == 0:
        mt5.shutdown()
        return []

    rows = []
    for d in deals:
        if d.type in (0, 1): # BUY=0, SELL=1
            # EET to UTC correction (-10800 seconds / -3 hours)
            utc_time = datetime.utcfromtimestamp(d.time - 10800)
            rows.append({
                "ticket": d.order,
                "utc_time": utc_time,
                "utc_str": utc_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "date_str": utc_time.strftime("%Y-%m-%d"),
                "symbol": d.symbol,
                "type": "BUY" if d.type == 0 else "SELL",
                "volume": d.volume,
                "price": d.price,
                "profit": round(d.profit, 2),
                "comment": d.comment
            })
    mt5.shutdown()
    return rows

def compute_proven_python_sim(date_str, max_time_utc=None):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "proxima_command_center"))
    from rolling_backtest_engine import RollingBacktestEngine
    eng = RollingBacktestEngine()
    trades = eng.compute_deterministic_trades(date_str, max_time_utc=max_time_utc)
    return trades

def main():
    print("=" * 100)
    print("PROXIMA X — TRUTH RECONCILIATION AUDIT (YESTERDAY 2026-08-03 & TODAY 2026-08-04)")
    print("=" * 100)

    # 1. Fetch MT5 Live Deals
    start_utc = datetime(2026, 8, 3, 0, 0, 0, tzinfo=timezone.utc)
    end_utc   = datetime.now(timezone.utc)

    live_deals = fetch_live_deals(start_utc, end_utc)
    
    # 2. Compute Python Sim for 2026-08-03 and 2026-08-04
    now_ts = pd.Timestamp(end_utc.replace(tzinfo=None))
    
    sim_yesterday = compute_proven_python_sim("2026-08-03")
    sim_today     = compute_proven_python_sim("2026-08-04", max_time_utc=now_ts)

    # --- METRICS CALCULATION ---
    # Live Deals Yesterday & Today
    live_y = [d for d in live_deals if d["date_str"] == "2026-08-03"]
    live_t = [d for d in live_deals if d["date_str"] == "2026-08-04"]

    def calc_stats(deals_list, pnl_key="profit"):
        if not deals_list:
            return {"trades": 0, "wins": 0, "losses": 0, "wr": 0.0, "pnl": 0.0, "pf": 0.0}
        wins = [d for d in deals_list if d[pnl_key] > 0]
        losses = [d for d in deals_list if d[pnl_key] <= 0]
        net_pnl = sum(d[pnl_key] for d in deals_list)
        gw = sum(d[pnl_key] for d in wins)
        gl = abs(sum(d[pnl_key] for d in losses))
        pf = round(gw / gl, 2) if gl > 0 else (99.0 if gw > 0 else 1.0)
        wr = round(len(wins) / len(deals_list) * 100.0, 1)
        return {"trades": len(deals_list), "wins": len(wins), "losses": len(losses), "wr": wr, "pnl": round(net_pnl, 2), "pf": pf}

    s_live_y = calc_stats(live_y, "profit")
    s_live_t = calc_stats(live_t, "profit")
    s_sim_y  = calc_stats(sim_yesterday, "sim_pnl")
    s_sim_t  = calc_stats(sim_today, "sim_pnl")

    print("\n📊 1. YESTERDAY (2026-08-03) TRUTH COMPARISON:")
    print(f"  • MT5 Live Executed : {s_live_y['trades']} Trades | {s_live_y['wr']}% WR ({s_live_y['wins']}W / {s_live_y['losses']}L) | Net PnL: ${s_live_y['pnl']:,.2f} | PF: {s_live_y['pf']}")
    print(f"  • Proven Python Sim : {s_sim_y['trades']} Trades | {s_sim_y['wr']}% WR ({s_sim_y['wins']}W / {s_sim_y['losses']}L) | Net PnL: +${s_sim_y['pnl']:,.2f} | PF: {s_sim_y['pf']}")

    print("\n📊 2. TODAY (2026-08-04) TRUTH COMPARISON:")
    print(f"  • MT5 Live Executed : {s_live_t['trades']} Trades | {s_live_t['wr']}% WR ({s_live_t['wins']}W / {s_live_t['losses']}L) | Net PnL: ${s_live_t['pnl']:,.2f} | PF: {s_live_t['pf']}")
    print(f"  • Proven Python Sim : {s_sim_t['trades']} Trades | {s_sim_t['wr']}% WR ({s_sim_t['wins']}W / {s_sim_t['losses']}L) | Net PnL: +${s_sim_t['pnl']:,.2f} | PF: {s_sim_t['pf']}")

    # Total Overall
    all_live = live_deals
    all_sim  = sim_yesterday + sim_today
    s_live_all = calc_stats(all_live, "profit")
    s_sim_all  = calc_stats(all_sim, "sim_pnl")

    print("\n🏆 OVERALL COMBINED TIMELINE (2026-08-03 to 2026-08-04):")
    print(f"  🔴 ACTUAL MT5 LIVE REALITY : {s_live_all['trades']} Trades | {s_live_all['wr']}% WR | Net PnL: ${s_live_all['pnl']:,.2f} | PF: {s_live_all['pf']}")
    print(f"  🟢 PROVEN PYTHON TRUTH     : {s_sim_all['trades']} Trades | {s_sim_all['wr']}% WR | Net PnL: +${s_sim_all['pnl']:,.2f} | PF: {s_sim_all['pf']}")

    print("\n" + "=" * 100)
    print("DETAILED BREAKDOWN OF TODAY'S (2026-08-04) PROVEN PYTHON SIM TRADES:")
    print("=" * 100)
    for t in sim_today:
        print(f"  {t['iso_timestamp']} | {t['strategy']:<20} | {t['pair']} {t['side']} | Pips: {t['pips']:>5.1f}p | PnL: ${t['sim_pnl']:>8.2f} | Result: {'🟢 WIN' if t['is_win'] else '🔴 LOSS'}")

if __name__ == "__main__":
    main()
