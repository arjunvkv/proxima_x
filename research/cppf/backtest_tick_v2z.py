"""V2+z on Exness tick data — uses the proven tick-level TrailingStopManager
with correct per-pair spread costs and optional z-score entry filter.

Usage:
    python research/cppf/backtest_tick_v2z.py EURUSD
    python research/cppf/backtest_tick_v2z.py EURUSD --z 2.0
    python research/cppf/backtest_tick_v2z.py EURJPY --scan
"""
import argparse, time
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_trade.strategies.m1_z_reversal.strategy import PairState, TrailingStopManager, CONFIG

# Per-pair spread cost (raw price)
SPREAD_COST = {
    "EURUSD": 0.000015,  # 0.15 pips
    "EURJPY": 0.0050,    # 50 MP
    "GBPJPY": 0.0060,    # 60 MP
}

TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")


def load_ticks(pair="EURJPY", months=None):
    if months is None:
        months = [(2025, 10), (2025, 11), (2025, 12)]
    dfs = []
    for y, m in months:
        p = TICK_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        if not p.exists():
            continue
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts": str, "B": np.float64, "A": np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
            format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    if not dfs:
        raise FileNotFoundError(f"No tick data found for {pair}")
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t = t.set_index("Ts")
    return t


def backtest_ticks(ticks_df, cost_raw, config=None, seed_count=60, pair="EURUSD", z_thresh=0.0):
    """Run tick-level backtest with optional z-threshold filter."""
    cfg = {**CONFIG, **(config or {})}
    cfg.setdefault("min_stop_pips", 1.5)

    # Build M1 bars from ticks for seeding
    mp_series = ((ticks_df['B'] + ticks_df['A']) / 2) * 10000
    bars = mp_series.resample('1min').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    ).dropna()

    ps = PairState(pair, cfg)
    seed_end_idx = min(seed_count, len(bars) - 10)
    for i in range(seed_end_idx):
        bar = bars.iloc[i]
        ps.seed_bar({
            'open': bar['open'] / 10000,
            'high': bar['high'] / 10000,
            'low': bar['low'] / 10000,
            'close': bar['close'] / 10000,
            'time': int(bar.name.timestamp()),
        })

    seed_end_ts = int(bars.index[seed_end_idx - 1].timestamp()) if seed_end_idx > 0 else 0

    tsm = TrailingStopManager(cfg)
    times = ticks_df.index.astype(np.int64) // 10 ** 9
    bids = ticks_df['B'].values
    asks = ticks_df['A'].values

    trades = {}
    prev_bar_min = -1
    total_ticks = len(times)
    skipped_by_filter = 0

    for i in range(total_ticks):
        ts = int(times[i])
        if ts <= seed_end_ts:
            continue
        bid = float(bids[i])
        ask = float(asks[i])
        mid = (bid + ask) / 2.0

        # Trailing stop checks
        closed = tsm.update(bid, ask, ts)
        for cp in closed:
            tr = trades.get(cp['ticket'])
            if tr:
                tr['exit'] = bid if tr['dir'] == 1 else ask
                tr['exit_time'] = ts
                tr['pnl'] = (tr['exit'] - tr['entry']) * tr['dir'] - cost_raw
                tr['dur_bars'] = (ts - tr['entry_time']) / 60
                tr['exit_reason'] = 'stop'

        # Generate signals
        sig = ps.update(mid, ts)
        if sig:
            bar_min = ts // 60
            if bar_min == prev_bar_min:
                continue
            prev_bar_min = bar_min

            # Apply z-threshold filter
            if z_thresh > 0 and abs(sig['z_score']) < z_thresh:
                skipped_by_filter += 1
                continue

            dir_mult = cfg.get('_direction_mult', 1)
            trade_dir = sig['direction'] * dir_mult
            entry = ask if trade_dir == 1 else bid
            ticket = tsm.add(pair, trade_dir, entry, sig['atr'], timestamp=ts,
                             spread=abs(ask - bid))
            trades[ticket] = {
                'bar_time': sig['bar_time'],
                'dir': trade_dir,
                'entry': entry,
                'entry_time': ts,
                'z': sig['z_score'],
                'atr': sig['atr'],
                'exit': None, 'exit_time': None, 'pnl': None,
                'dur_bars': None, 'exit_reason': None,
            }

    # Close remaining on expiry
    final_ts = int(times[-1])
    expired = tsm.check_expiry(final_ts)
    for cp in expired:
        tr = trades.get(cp['ticket'])
        if tr and tr['exit'] is None:
            tr['exit'] = (float(bids[-1]) + float(asks[-1])) / 2.0
            tr['exit_time'] = final_ts
            tr['pnl'] = (tr['exit'] - tr['entry']) * tr['dir'] - cost_raw
            tr['dur_bars'] = (tr['exit_time'] - tr['entry_time']) / 60
            tr['exit_reason'] = 'expiry'

    for tr in trades.values():
        if tr['exit'] is None:
            tr['exit'] = (float(bids[-1]) + float(asks[-1])) / 2.0
            tr['exit_time'] = final_ts
            tr['pnl'] = (tr['exit'] - tr['entry']) * tr['dir'] - cost_raw
            tr['dur_bars'] = (tr['exit_time'] - tr['entry_time']) / 60
            tr['exit_reason'] = 'expiry'

    trades_list = list(trades.values())
    meta = {"skipped": skipped_by_filter}
    return trades_list, meta


def summary(trades, meta=None):
    if not trades:
        print("  No trades")
        return

    pnls = [t['pnl'] for t in trades if t['pnl'] is not None]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    total = len(pnls)
    wr = wins / total if total > 0 else 0
    gross_pnl = sum(pnls)
    avg_pnl = gross_pnl / total if total > 0 else 0

    pnl_no_cost = [t['pnl'] + SPREAD_COST.get("EURUSD", 0.000015) for t in trades if t['pnl'] is not None]
    gross_no_cost = sum(pnl_no_cost)

    win_sub = [p for p in pnls if p > 0]
    lose_sub = [p for p in pnls if p <= 0]
    avg_win = np.mean(win_sub) if win_sub else 0
    avg_loss = np.mean(lose_sub) if lose_sub else 0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Time span
    times = [t['entry_time'] for t in trades if t['entry_time']]
    n_days = (max(times) - min(times)) / 86400 if times else 1

    print(f"  Trades:      {total:,}")
    print(f"  Trades/day:  {total / n_days:.0f}")
    print(f"  Win rate:    {wins}/{total} = {wr:.1%}")
    print(f"  Gross PnL:   {gross_pnl:.6f}")
    print(f"  Avg trade:   {avg_pnl:.6f}")
    print(f"  Avg win:     {avg_win:.6f}  Avg loss: {avg_loss:.6f}  Payoff: {payoff:.2f}")
    print(f"  Max win:     {max(pnls):.6f}  Max loss: {min(pnls):.6f}")
    pf = sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)) if losses > 0 and sum(p for p in pnls if p < 0) != 0 else float('inf')
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Raw PnL (no cost): {gross_no_cost:.6f}")

    zs = [t['z'] for t in trades if t['z'] is not None]
    if zs:
        print(f"  Avg |z|:     {sum(abs(z) for z in zs) / len(zs):.2f}")

    if meta:
        print(f"  Skipped (z-filter): {meta.get('skipped', 0)}")


def scan(config_grid, ticks, cost_raw, pair="EURUSD"):
    """Scan over z-threshold values."""
    pair_upper = pair.upper()
    cost = SPREAD_COST.get(pair_upper, cost_raw)

    print(f"\n{'='*55}")
    print(f"V2+z TICK SCAN — {pair_upper}")
    print(f"  {len(ticks):,} ticks loaded")
    print(f"  Spread cost: {cost}")
    print(f"{'='*55}")

    best = {"pnl": -999, "z": 0, "wr": 0, "n": 0}
    for z in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        trades, meta = backtest_ticks(ticks, cost, pair=pair_upper, z_thresh=z)
        if not trades:
            continue
        pnls = [t['pnl'] for t in trades if t['pnl'] is not None]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        tp = sum(pnls)
        n = len(pnls)
        n_days = 92  # ~3 months
        print(f"  z>={z:.1f}: n={n:>5d} ({n/n_days:.0f}/d)  WR={wr:>5.1%}  PnL={tp:>+.6f}")
        if tp > best["pnl"]:
            best = {"pnl": tp, "z": z, "wr": wr, "n": n}

    print(f"\n  BEST: z>={best['z']}  n={best['n']}  WR={best['wr']:.1%}  PnL={best['pnl']:.6f}")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", nargs="?", default="EURUSD")
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--months", nargs="+", default=None,
                    help="months as YYYY-MM (e.g. 2025-10 2025-11)")
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()

    pair = args.pair.upper()
    cost = SPREAD_COST.get(pair, 0.005)

    months = args.months
    if months:
        months = [(int(m.split("-")[0]), int(m.split("-")[1])) for m in months]
    else:
        months = [(2025, 10), (2025, 11), (2025, 12)]

    print(f"Loading {pair} tick data...")
    t0 = time.time()
    ticks = load_ticks(pair, months=months)
    print(f"  {len(ticks):,} ticks loaded in {time.time()-t0:.1f}s")
    print(f"  Spread cost: {cost:.6f}")

    if args.scan:
        scan(None, ticks, cost, pair)
    else:
        trades, meta = backtest_ticks(ticks, cost, pair=pair, z_thresh=args.z)
        print(f"\nResults (z>={args.z}):")
        summary(trades, meta)
