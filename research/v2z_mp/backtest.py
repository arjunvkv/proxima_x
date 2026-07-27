"""V2z-MP Cross-Pair Ranking Backtest Engine.
Point-in-time simulation using MT5 M1 OHLC data.
No look-ahead bias: all signals computed from past data only."""
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict


class CrossPairBacktest:
    """Cross-pair ranking mean reversion backtest engine."""

    def __init__(self, pairs, m1_data, start_date, end_date,
                 scan_freq_min=60, lookback_min=15, hold_min=15,
                 picks_per_side=3, spread_cost_pips=1.4,
                 commission_per_lot=7.0, lot_size=0.25,
                 max_concurrent=12, max_spread_pips=5.0,
                 time_filter=None,
                 verbose=True):
        self.pairs = pairs
        self.m1_data = m1_data  # dict: pair -> np.array of [time, open, high, low, close, tick_volume, spread]
        self.start_date = start_date
        self.end_date = end_date
        self.scan_freq_min = scan_freq_min
        self.lookback_min = lookback_min
        self.hold_min = hold_min
        self.picks_per_side = picks_per_side
        self.spread_cost_pips = spread_cost_pips
        self.commission_per_lot = commission_per_lot
        self.lot_size = lot_size
        self.max_concurrent = max_concurrent
        self.max_spread_mult = max_spread_mult
        self.time_filter = time_filter
        self.verbose = verbose

        self.direction = 0  # 0=both, 1=long only, -1=short only

    @staticmethod
    def _get_hour(t):
        """Get hour from np.datetime64 or datetime."""
        if isinstance(t, np.datetime64):
            return int(str(t)[11:13])
        return t.hour

    def _get_price(self, pair, t):
        """Get the price at or after time t for a pair."""
        arr = self.m1_data[pair]
        if arr is None or len(arr) == 0:
            return None
        t_s = np.datetime64(t, 's') if not isinstance(t, np.datetime64) else t.astype('datetime64[s]')
        idx = np.searchsorted(arr['time'].astype('datetime64[s]'), t_s, side='left')
        if idx >= len(arr):
            return None
        return arr[idx]

    def _compute_return(self, pair, t, lookback_min):
        """Compute return from t-lookback to t for a pair."""
        arr = self.m1_data[pair]
        if arr is None or len(arr) == 0:
            return None
        t_s = np.datetime64(t, 's') if not isinstance(t, np.datetime64) else t.astype('datetime64[s]')
        t_start = t_s - np.timedelta64(lookback_min, 'm')

        times_s = arr['time'].astype('datetime64[s]')
        idx_now = np.searchsorted(times_s, t_s, side='left') - 1
        idx_past = np.searchsorted(times_s, t_start, side='left')

        if idx_now < 0 or idx_past < 0 or idx_past >= idx_now:
            return None
        open_now = float(arr['open'][idx_now])
        open_past = float(arr['open'][idx_past])
        if open_past <= 0:
            return None
        return (open_now - open_past) / open_past

    def _get_median_spread(self, pair):
        """Get median spread for a pair from its data."""
        arr = self.m1_data[pair]
        if arr is None or len(arr) == 0:
            return 100
        spreads = arr['spread']
        valid = spreads > 0
        if np.sum(valid) < 10:
            return 100
        return np.median(spreads[valid])

    def run(self):
        """Run the backtest."""
        trades = []
        scan_count = defaultdict(int)

        # Build scan times as np.datetime64[s]
        start_dt64 = np.datetime64(datetime.combine(self.start_date, datetime.min.time()), 's')
        end_dt64 = np.datetime64(datetime.combine(self.end_date, datetime.min.time()) + timedelta(days=1), 's')
        freq_delta = np.timedelta64(self.scan_freq_min, 'm')

        scan_times = []
        t = start_dt64
        while t < end_dt64:
            if self.time_filter is None or self._get_hour(t) in self.time_filter:
                scan_times.append(t)
            t += freq_delta

        if self.verbose:
            days = (end_dt64 - start_dt64).astype('timedelta64[D]').astype(int)
            print(f"Total scans: {len(scan_times)} over {days} days")

        open_positions = {}  # pair -> {direction, entry_time, entry_price, hold_until}

        for scan_t in scan_times:
            # Close expired positions
            expired = [p for p, pos in open_positions.items()
                       if scan_t >= pos['hold_until']]
            for p in expired:
                pos = open_positions.pop(p)
                bar = self._get_price(p, scan_t)
                if bar is None:
                    continue
                exit_price = float(bar['open'])
                if pos['direction'] == 1:
                    pnl_pips = (exit_price - pos['entry_price']) / 0.0001
                else:
                    pnl_pips = (pos['entry_price'] - exit_price) / 0.0001

                spread_cost = self.spread_cost_pips * 2
                commission = self.commission_per_lot * self.lot_size
                pnl_usd = pnl_pips * 10 * self.lot_size - spread_cost * 10 * self.lot_size - commission
                win = pnl_pips > spread_cost + commission / (10 * self.lot_size)

                trade = {
                    'pair': p,
                    'direction': pos['direction'],
                    'entry_time': pos['entry_time'],
                    'exit_time': scan_t,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pips': round(pnl_pips, 2),
                    'pnl_usd': round(pnl_usd, 2),
                    'win': win,
                    'hold_min': (scan_t - pos['entry_time']).astype('timedelta64[m]').astype(float),
                    'exit_reason': 'hold_expiry',
                }
                trades.append(trade)

            if len(open_positions) >= self.max_concurrent:
                continue

            # Rank pairs by return
            pair_returns = []
            for pair in self.pairs:
                ret = self._compute_return(pair, scan_t, self.lookback_min)
                if ret is None:
                    continue
                bar_now = self._get_price(pair, scan_t)
                if bar_now is None:
                    continue
                median_spread = self._get_median_spread(pair)
                if median_spread > 0 and bar_now['spread'] > median_spread * self.max_spread_mult:
                    continue
                if pair in open_positions:
                    continue
                pair_returns.append((pair, ret, float(bar_now['open']), float(bar_now['spread'])))

            if len(pair_returns) < self.picks_per_side * 2:
                continue

            pair_returns.sort(key=lambda x: x[1])

            long_picks = pair_returns[:self.picks_per_side]
            short_picks = pair_returns[-self.picks_per_side:]

            for pair, ret, entry_price, spread in long_picks:
                if len(open_positions) >= self.max_concurrent:
                    break
                if self.direction >= 0:
                    hold_until = scan_t + np.timedelta64(self.hold_min, 'm')
                    open_positions[pair] = {
                        'direction': 1,
                        'entry_time': scan_t,
                        'entry_price': entry_price,
                        'hold_until': hold_until,
                    }
                    scan_count[self._get_hour(scan_t)] += 1

            for pair, ret, entry_price, spread in reversed(short_picks):
                if len(open_positions) >= self.max_concurrent:
                    break
                if self.direction <= 0:
                    hold_until = scan_t + np.timedelta64(self.hold_min, 'm')
                    open_positions[pair] = {
                        'direction': -1,
                        'entry_time': scan_t,
                        'entry_price': entry_price,
                        'hold_until': hold_until,
                    }
                    scan_count[self._get_hour(scan_t)] += 1

        # Close remaining positions
        for pair, pos in list(open_positions.items()):
            bar = self._get_price(pair, end_dt64)
            if bar is None:
                continue
            exit_price = float(bar['open'])
            if pos['direction'] == 1:
                pnl_pips = (exit_price - pos['entry_price']) / 0.0001
            else:
                pnl_pips = (pos['entry_price'] - exit_price) / 0.0001
            spread_cost = self.spread_cost_pips * 2
            commission = self.commission_per_lot * self.lot_size
            pnl_usd = pnl_pips * 10 * self.lot_size - spread_cost * 10 * self.lot_size - commission
            win = pnl_pips > spread_cost + commission / (10 * self.lot_size)
            trades.append({
                'pair': pair, 'direction': pos['direction'],
                'entry_time': pos['entry_time'], 'exit_time': end_dt64,
                'entry_price': pos['entry_price'], 'exit_price': exit_price,
                'pnl_pips': round(pnl_pips, 2), 'pnl_usd': round(pnl_usd, 2),
                'win': win,
                'hold_min': (end_dt64 - pos['entry_time']).astype('timedelta64[m]').astype(float),
                'exit_reason': 'end_of_test',
            })

        # Compute stats
        total = len(trades)
        wins = sum(1 for t in trades if t['win'])
        total_pnl = sum(t['pnl_usd'] for t in trades)
        winning_trades = [t for t in trades if t['win']]
        losing_trades = [t for t in trades if not t['win']]
        avg_win = float(np.mean([t['pnl_usd'] for t in winning_trades])) if winning_trades else 0
        avg_loss = float(np.mean([t['pnl_usd'] for t in losing_trades])) if losing_trades else 0
        # Compute days from date range
        days = (self.end_date - self.start_date).days
        trades_per_day = total / days if days > 0 else 0

        # Compute daily stats from trades
        daily_from_trades = {}
        for t in trades:
            et = t['entry_time']
            if isinstance(et, np.datetime64):
                dk = str(et)[:10]
            else:
                dk = et.strftime('%Y-%m-%d') if hasattr(et, 'strftime') else str(et)[:10]
            if dk not in daily_from_trades:
                daily_from_trades[dk] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
            daily_from_trades[dk]['trades'] += 1
            if t['win']:
                daily_from_trades[dk]['wins'] += 1
            daily_from_trades[dk]['pnl'] += t['pnl_usd']

        results = {
            'total_trades': total,
            'win_rate': wins / total * 100 if total > 0 else 0,
            'total_pnl_usd': round(total_pnl, 2),
            'avg_win_usd': round(avg_win, 2),
            'avg_loss_usd': round(avg_loss, 2),
            'trades_per_day': round(trades_per_day, 1),
            'days': days,
            'start_date': str(self.start_date),
            'end_date': str(self.end_date),
            'scan_freq_min': self.scan_freq_min,
            'lookback_min': self.lookback_min,
            'hold_min': self.hold_min,
            'picks_per_side': self.picks_per_side,
            'pairs_count': len(self.pairs),
            'hour_scan_count': dict(scan_count),
            'daily_stats': daily_from_trades,
            'trades': trades,
        }
        return results

    def print_results(self, results):
        """Print formatted results."""
        if results['total_trades'] == 0:
            print("NO TRADES")
            return
        print(f"\n{'='*60}")
        print(f"V2z-MP Cross-Pair Ranking Backtest")
        print(f"{'='*60}")
        print(f"Period: {results['start_date']} to {results['end_date']}")
        print(f"Config: scan={results['scan_freq_min']}min lookback={results['lookback_min']}min "
              f"hold={results['hold_min']}min picks={results['picks_per_side']}/side")
        print(f"Pairs: {results['pairs_count']}")
        print(f"{'='*60}")
        print(f"Total trades:    {results['total_trades']}")
        print(f"Win rate:        {results['win_rate']:.1f}%")
        print(f"Total PnL:       ${results['total_pnl_usd']:.2f}")
        print(f"Avg win:         ${results['avg_win_usd']:.2f}")
        print(f"Avg loss:        ${results['avg_loss_usd']:.2f}")
        print(f"Trades/day:      {results['trades_per_day']}")
        print(f"Days:            {results['days']}")
        print(f"{'='*60}")

        # Hour breakdown
        print("\nHour breakdown (scans -> trades):")
        for h in range(24):
            c = results['hour_scan_count'].get(h, 0)
            if c > 0:
                print(f"  Hour {h:02d}: {c} scans")

        # Pair breakdown
        pair_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
        for t in results['trades']:
            p = t['pair']
            pair_stats[p]['trades'] += 1
            if t['win']:
                pair_stats[p]['wins'] += 1
            pair_stats[p]['pnl'] += t['pnl_usd']
        print("\nPair breakdown:")
        for pair in sorted(pair_stats.keys()):
            s = pair_stats[pair]
            wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
            print(f"  {pair:8s}: {s['trades']:4d} trades {wr:5.1f}% WR ${s['pnl']:8.2f}")

        return results
