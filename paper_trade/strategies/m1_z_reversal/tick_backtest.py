"""Tick-level backtester using BarBuilder+PairState+TrailingStopManager.

Processes raw tick data through the exact same pipeline as the live strategy.
Use this to validate strategy changes before deploying live.

Usage:
    python tick_backtest.py                    # Default run on EURJPY
    python tick_backtest.py EURUSD             # Specific pair
    python tick_backtest.py EURJPY --config z_thresh=2.5,min_stop_pips=3.0
    python tick_backtest.py EURJPY --invert    # Flip direction (momentum)
"""
import argparse, time, random
import numpy as np
import pandas as pd
from pathlib import Path
from paper_trade.strategies.m1_z_reversal.strategy import PairState, TrailingStopManager, CONFIG

COST_RAW = 0.005  # 50 MP in raw price units (0.5 pips for EURJPY)
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")


def load_ticks(pair="EURJPY", months=None):
    """Load Exness tick data. months is list of (year, month) tuples."""
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
        raise FileNotFoundError(f"No tick data found for {pair} in {TICK_DIR}")
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t = t.set_index("Ts")
    return t


def backtest_ticks(ticks_df, config=None, seed_count=60, pair="EURJPY"):
    """Run tick-level backtest.

    Args:
        ticks_df: DataFrame with columns 'B' (bid), 'A' (ask), index is datetime
        config: dict overriding CONFIG parameters
        seed_count: number of M1 bars to seed from the beginning of tick data
        pair: currency pair name

    Returns:
        list of trade dicts: {bar_time, dir, entry, exit, pnl, z, atr,
                              entry_time, exit_time, dur_bars, exit_reason}
    """
    cfg = {**CONFIG, **(config or {})}
    cfg.setdefault("min_stop_pips", 1.5)

    # Build M1 bars from ticks for seeding
    mp_series = ((ticks_df['B'] + ticks_df['A']) / 2) * 10000
    bars = mp_series.resample('1min').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    ).dropna()

    # Seed PairState
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

    # Run through all ticks
    tsm = TrailingStopManager(cfg)
    times = ticks_df.index.astype(np.int64) // 10 ** 9
    bids = ticks_df['B'].values
    asks = ticks_df['A'].values

    trades = {}
    prev_bar_min = -1

    for i in range(len(times)):
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
                tr['pnl'] = (tr['exit'] - tr['entry']) * tr['dir'] - COST_RAW
                tr['dur_bars'] = (ts - tr['entry_time']) / 60
                tr['exit_reason'] = 'stop'

        # Generate signals
        sig = ps.update(mid, ts)
        if sig:
            bar_min = ts // 60
            if bar_min == prev_bar_min:
                continue
            prev_bar_min = bar_min

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
            tr['pnl'] = (tr['exit'] - tr['entry']) * tr['dir'] - COST_RAW
            tr['dur_bars'] = (tr['exit_time'] - tr['entry_time']) / 60
            tr['exit_reason'] = 'expiry'

    for tr in trades.values():
        if tr['exit'] is None:
            tr['exit'] = (float(bids[-1]) + float(asks[-1])) / 2.0
            tr['exit_time'] = final_ts
            tr['pnl'] = (tr['exit'] - tr['entry']) * tr['dir'] - COST_RAW
            tr['dur_bars'] = (tr['exit_time'] - tr['entry_time']) / 60
            tr['exit_reason'] = 'expiry'

    return list(trades.values())


def summary(trades):
    """Print summary stats for a list of trades."""
    if not trades:
        print("  No trades")
        return

    pnls = [t['pnl'] for t in trades if t['pnl'] is not None]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    total = len(pnls)
    wr = wins / total if total > 0 else 0
    gross_pnl = sum(pnls)

    print(f"  Trades: {total}")
    print(f"  Win rate: {wins}/{total} = {wr:.1%}")
    print(f"  Gross PnL: {gross_pnl:.4f} raw ({gross_pnl * 10000:.1f} MP, {gross_pnl / 0.01:.2f} pips)")
    if total > 0:
        print(f"  Avg trade: {gross_pnl/total:.4f} raw ({(gross_pnl/total) * 10000:.1f} MP)")
    print(f"  Best: {max(pnls):.4f}  Worst: {min(pnls):.4f}")
    pf = sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)) if losses > 0 and sum(p for p in pnls if p < 0) != 0 else float('inf')
    print(f"  Profit factor: {pf:.2f}")

    zs = [t['z'] for t in trades if t['z'] is not None]
    if zs:
        print(f"  Avg |z|: {sum(abs(z) for z in zs) / len(zs):.2f}")


def scan(config_grid, ticks=None, pair="EURJPY"):
    """Scan over a grid of config overrides, printing results."""
    if ticks is None:
        ticks = load_ticks(pair)
    print(f"Loaded {len(ticks):,} ticks for {pair}")
    print(f"\n{'='*60}")
    print("SCAN")
    print('='*60)
    best = {"pnl": -999, "cfg": None, "wr": 0, "n": 0}
    for cfg in config_grid:
        trades = backtest_ticks(ticks, config=cfg, pair=pair)
        pnls = [tr["pnl"] for tr in trades if tr["pnl"] is not None]
        if not pnls:
            continue
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        tp = sum(pnls)
        if tp > best["pnl"]:
            best = {"pnl": tp, "cfg": cfg, "wr": wr, "n": len(pnls)}
        print(f"  {cfg}: n={len(pnls):4d} WR={wr:.1%} PnL={tp:+.2f}p")
    print(f"\nBEST: n={best['n']} WR={best['wr']:.1%} PnL={best['pnl']:.2f}p cfg={best['cfg']}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tick-level backtester")
    parser.add_argument("pair", nargs="?", default="EURJPY", help="Currency pair")
    parser.add_argument("--invert", action="store_true", help="Flip direction (momentum)")
    parser.add_argument("--config", default=None, help="Comma-separated key=value overrides")
    args = parser.parse_args()

    cfg = {}
    if args.config:
        for kv in args.config.split(","):
            k, v = kv.split("=")
            cfg[k] = float(v) if "." in v else int(v)
    if args.invert:
        cfg["_direction_mult"] = -1

    print(f"Loading {args.pair} tick data...")
    ticks = load_ticks(args.pair)

    print(f"Running backtest (invert={args.invert}, config={cfg})...")
    trades = backtest_ticks(ticks, config=cfg, pair=args.pair)
    summary(trades)
