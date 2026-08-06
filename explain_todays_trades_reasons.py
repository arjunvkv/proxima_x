import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "proxima_command_center"))
from rolling_backtest_engine import RollingBacktestEngine

ACCOUNT = {"login": 1514168544, "password": "$!4fwBIc", "server": "FTMO-Demo"}

def main():
    print("=" * 110)
    print("PROXIMA X — TODAY'S (2026-08-04 UTC) PROVEN PYTHON SIM TRADES: PER-TRADE DETAILED REASONING")
    print("=" * 110)

    eng = RollingBacktestEngine()
    now_utc = datetime.now(timezone.utc)
    now_ts = pd.Timestamp(now_utc.replace(tzinfo=None))
    
    trades = eng.compute_deterministic_trades("2026-08-04", max_time_utc=now_ts)

    if not trades:
        print("No Python sim trades generated for today so far.")
        return

    wins = [t for t in trades if t['is_win']]
    losses = [t for t in trades if not t['is_win']]
    net_pnl = sum(t['sim_pnl'] for t in trades)
    wr = len(wins) / len(trades) * 100.0

    print(f"\n📊 TODAY'S SUMMARY (2026-08-04 UTC):")
    print(f"  • Total Trades   : {len(trades)}")
    print(f"  • Win Rate       : {wr:.1f}% ({len(wins)} Wins / {len(losses)} Losses)")
    print(f"  • Net Realized PnL: +${net_pnl:,.2f}")
    print("=" * 110)

    reasons = {
        "Tokyo H0 (v107)": (
            "Fires at exactly 00:00 UTC session open. Calculates return over past 30 min (00:00 vs 23:30) "
            "across 18 pairs, ranks all pairs from worst to best, and buys top 3 most-declined pairs. "
            "Exits after 12 M5 bars (60 min hold). Fades Asian open over-reaction."
        ),
        "MSV Asian (v107)": (
            "Fires at 00:30 UTC when Asian FX exhaustion is detected across JPY pairs (dispersion > 95%). "
            "Enters USDJPY BUY for a 60 min mean-reversion hold (12 M5 bars)."
        ),
        "Ultra Monster (v107)": (
            "Rolling Opening Range Breakout (ORB) evaluated every 30 min (:00 and :30). Looks at past 12 M5 bars (60 min). "
            "Filters pairs with range >= 6.0 pips. Confirms entry when current bar close breaks ABOVE 60-min High (BUY) "
            "or BELOW 60-min Low (SELL). Selects the single pair with largest range. Holds for 3 M5 bars (15 min exit)."
        ),
        "CPPF Z (v107)": (
            "Cross-Pair Volatility Dislocation gate. Fires when rolling 200-bar z-score <= -6.0 shock on EURAUD or GBPAUD. "
            "Enters BUY for 90 min hold (18 M5 bars)."
        )
    }

    for i, t in enumerate(trades, 1):
        strat_name = t['strategy']
        pair = t['pair']
        side = t['side']
        pnl = t['sim_pnl']
        pips = t['pips']
        res_str = "🟢 WIN" if t['is_win'] else "🔴 LOSS"
        ts = t['iso_timestamp']
        
        strat_explain = reasons.get(strat_name, "Standard deterministic strategy rule.")

        print(f"\nTrade #{i:02d} | {ts} | {strat_name} | {pair} {side}")
        print(f"  • Outcome       : {res_str} | Pips: {pips:+.1f}p | Net PnL: ${pnl:+,.2f} | Lot: {t['lot']}L")
        print(f"  • Trigger Reason: {strat_explain}")
        
        if strat_name.startswith("Tokyo H0"):
            print(f"  • Trade Logic   : {pair} was identified as one of the 3 most-declined pairs over 23:30->00:00 UTC. Entered BUY at 00:00 UTC open.")
        elif strat_name.startswith("MSV Asian"):
            print(f"  • Trade Logic   : Asian session JPY exhaustion threshold met at 00:30 UTC. Entered USDJPY BUY.")
        elif strat_name.startswith("Ultra Monster"):
            print(f"  • Trade Logic   : At {ts[:16]}, {pair} showed 60-min range >= 6.0p with a confirmed close breakout in direction {side}. Entered at candle open, held 15 min.")

    print("\n" + "=" * 110)

if __name__ == "__main__":
    main()
