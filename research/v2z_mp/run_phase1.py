"""Phase 1: Cross-pair ranking backtest runner.
Pulls M1 data from MT5 and runs the V2z-MP backtest engine."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from research.v2z_mp.backtest import CrossPairBacktest
from datetime import datetime, date
import MetaTrader5 as mt5


def pull_m1_data(pairs, start_date, end_date):
    """Pull M1 OHLC data for all pairs from MT5."""
    if not mt5.initialize():
        print("MT5 init failed")
        return None

    data = {}
    for pair in pairs:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1,
                                      datetime.combine(start_date, datetime.min.time()),
                                      datetime.combine(end_date, datetime.min.time()))
        if rates is None or len(rates) == 0:
            print(f"  {pair}: NO DATA")
            continue
        print(f"  {pair}: {len(rates)} bars")
        data[pair] = rates
    mt5.shutdown()
    return data


# Define liquid pairs (Tokyo H0 set + additional majors)
PAIRS_TOKYO_H0 = [
    'EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD',
    'GBPUSD', 'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD',
    'USDJPY', 'USDCHF', 'USDCAD',
    'AUDCAD', 'AUDNZD',
]

PAIRS_EXTENDED = PAIRS_TOKYO_H0 + [
    'AUDUSD', 'NZDUSD', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY',
    'USDNOK', 'USDSEK', 'USDMXN', 'EURHUF', 'EURPLN', 'GBPSEK',
]

# Test windows
WINDOWS = {
    'forward': (date(2026, 6, 8), date(2026, 7, 26)),
    'feb_mar': (date(2026, 2, 1), date(2026, 4, 1)),
    'oct_dec25': (date(2025, 10, 1), date(2026, 1, 1)),
}


def run_phase1a(pairs=None, window='forward', verbose=True):
    """Phase 1A: Replicate Tokyo H0 at 00:00 UTC.
    Uses: 18 pairs, lookback=15min, hold=15min, pick bottom 3, LONG only.
    """
    if pairs is None:
        pairs = PAIRS_TOKYO_H0

    start, end = WINDOWS[window]
    print(f"\n{'='*60}")
    print(f"Phase 1A: Tokyo H0 replication at 00:00 UTC")
    print(f"Window: {window} ({start} to {end})")
    print(f"Pairs: {len(pairs)}")
    print(f"{'='*60}")

    data = pull_m1_data(pairs, start, end)
    if data is None or len(data) == 0:
        print("No data available")
        return None

    bt = CrossPairBacktest(
        pairs=list(data.keys()),
        m1_data=data,
        start_date=start,
        end_date=end,
        scan_freq_min=1440,  # Once per day
        lookback_min=15,
        hold_min=15,
        picks_per_side=3,
        time_filter=[0],  # Only 00:00 UTC
        spread_cost_pips=0.7,
        commission_per_lot=7.0,
        lot_size=0.25,
        max_concurrent=3,
        verbose=verbose,
    )
    bt.direction = 1  # LONG only (like Tokyo H0)
    results = bt.run()
    bt.print_results(results)
    return results


def run_phase1b(pairs=None, window='forward', scan_freq=60, hold_min=15, lookback_min=15):
    """Phase 1B: Hour sweep — test each hour independently."""
    if pairs is None:
        pairs = PAIRS_EXTENDED

    start, end = WINDOWS[window]
    print(f"\n{'='*60}")
    print(f"Phase 1B: Hour sweep — every {scan_freq}min")
    print(f"Window: {window} ({start} to {end})")
    print(f"{'='*60}")

    data = pull_m1_data(pairs, start, end)
    if data is None or len(data) == 0:
        print("No data available")
        return None

    print("\n=== Hour-by-hour results ===")
    all_results = []
    for hour in range(24):
        bt = CrossPairBacktest(
            pairs=list(data.keys()),
            m1_data=data,
            start_date=start,
            end_date=end,
            scan_freq_min=scan_freq,
            lookback_min=lookback_min,
            hold_min=hold_min,
            picks_per_side=3,
            time_filter=[hour],
            spread_cost_pips=0.7,
            commission_per_lot=7.0,
            lot_size=0.25,
            max_concurrent=6,
            verbose=False,
        )
        results = bt.run()
        all_results.append(results)

        t = results['total_trades']
        wr = results['win_rate']
        pnl = results['total_pnl_usd']
        tpday = results['trades_per_day']
        if t > 0:
            mark = " *** " if wr > 60 and pnl > 0 else ""
            print(f"  Hour {hour:02d}: {t:4d} trades {wr:5.1f}% WR ${pnl:+8.2f} PnL {tpday:.1f}/day{mark}")

    # Summary
    print(f"\n{'='*60}")
    best_wr_hour = max(range(24), key=lambda h: all_results[h]['win_rate'] if all_results[h]['total_trades'] > 0 else 0)
    best_pnl_hour = max(range(24), key=lambda h: all_results[h]['total_pnl_usd'])
    total_trades = sum(r['total_trades'] for r in all_results)
    total_pnl = sum(r['total_pnl_usd'] for r in all_results)
    weighted_wr = sum(r['total_trades'] * r['win_rate'] for r in all_results) / total_trades if total_trades > 0 else 0

    print(f"Total: {total_trades} trades, {weighted_wr:.1f}% WR, ${total_pnl:.2f} PnL")
    print(f"Best WR hour: {best_wr_hour}:00 ({all_results[best_wr_hour]['win_rate']:.1f}%)")
    print(f"Best PnL hour: {best_pnl_hour}:00 (${all_results[best_pnl_hour]['total_pnl_usd']:.2f})")
    print(f"{'='*60}\n")

    return all_results


def run_phase1c(pairs=None, window='forward'):
    """Phase 1C: Test different scan frequencies on winning hours."""
    if pairs is None:
        pairs = PAIRS_EXTENDED

    start, end = WINDOWS[window]
    print(f"\n{'='*60}")
    print(f"Phase 1C: Frequency sweep")
    print(f"Window: {window} ({start} to {end})")
    print(f"{'='*60}")

    data = pull_m1_data(pairs, start, end)
    if data is None or len(data) == 0:
        print("No data available")
        return None

    freqs = [15, 30, 60, 120, 240, 480]
    holds = [15, 30, 60, 60, 120, 240]
    lookbacks = [15, 15, 15, 30, 30, 60]

    print(f"\n{'Freq':>6s} {'Hold':>6s} {'Look':>6s} {'Trades':>8s} {'WR':>6s} {'PnL':>10s} {'/day':>6s}")
    print('-' * 50)

    for freq, hold, lb in zip(freqs, holds, lookbacks):
        bt = CrossPairBacktest(
            pairs=list(data.keys()),
            m1_data=data,
            start_date=start,
            end_date=end,
            scan_freq_min=freq,
            lookback_min=lb,
            hold_min=hold,
            picks_per_side=3,
            time_filter=None,  # All hours
            spread_cost_pips=0.7,
            commission_per_lot=7.0,
            lot_size=0.25,
            max_concurrent=12,
            verbose=False,
        )
        r = bt.run()
        print(f"{freq:6d} {hold:6d} {lb:6d} {r['total_trades']:8d} {r['win_rate']:5.1f}% ${r['total_pnl_usd']:>8.2f} {r['trades_per_day']:6.1f}")

    print()


if __name__ == '__main__':
    # Phase 1A: Tokyo H0 replication
    r = run_phase1a()
    if r:
        print("\nPhase 1A complete. Press Enter to continue to Phase 1B...")
        input()

        # Phase 1B: Hour sweep
        r1b = run_phase1b()

        if r1b:
            print("\nPhase 1B complete. Press Enter to continue to Phase 1C...")
            input()

            # Phase 1C: Frequency sweep
            run_phase1c()
