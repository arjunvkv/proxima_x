"""Phase Dislocation — backtest harness.

Usage:
    python research/phase_dislocation/backtest.py --data <dir>
    python research/phase_dislocation/backtest.py --generate-synthetic

The backtest replays 1-minute bar data through the strategy and reports:
    - Total trades
    - Win rate
    - Sharpe ratio
    - Profit factor
    - Max drawdown
    - Per-pair PnL
    - Trade journal
"""
import os, sys, glob, json, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from research.phase_dislocation.strategy import PhaseDislocation, PAIRS, TRIANGLES


def load_dukascopy_csv(filepath):
    """Load a Dukascopy M1 CSV. Expected columns: Timestamp,Open,High,Low,Close,Volume."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    if "Timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["Timestamp"])
    elif "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"])
    elif "Date" in df.columns:
        df["time"] = pd.to_datetime(df["Date"])
    else:
        date_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        if date_cols:
            df["time"] = pd.to_datetime(df[date_cols[0]])
        else:
            raise ValueError(f"Cannot find time column in {filepath}: columns={list(df.columns)}")
    df["time"] = df["time"].astype(np.int64) // 10**9
    return df[["time", "Open", "High", "Low", "Close", "Volume"]].copy()


def pair_from_filename(filepath):
    """Extract pair name from Dukascopy filename like 'eurjpy-m1-bid-2024-01-01-2024-09-30.csv'."""
    base = os.path.basename(filepath)
    parts = base.split("-")
    return parts[0].upper()


def load_market_parquet(market_dir="data/market"):
    """Load all pairs from data/market/ parquet files."""
    import pandas as pd
    all_data = {}
    for pair in PAIRS:
        path = os.path.join(market_dir, f"{pair}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            all_data[pair] = df
            print(f"  {pair}: {len(df)} rows")
        else:
            print(f"  {pair}: MISSING")

    pairs_present = [p for p in PAIRS if p in all_data]
    missing = [p for p in PAIRS if p not in all_data]
    if missing:
        print(f"Warning: missing data for {missing}")
    if len(pairs_present) < 6:
        print(f"Need at least 6 pairs. Only have {len(pairs_present)}. Aborting.")
        return None

    # Find common timestamps across all present pairs
    common_times = None
    for p in pairs_present:
        t = set(all_data[p]["time"].values)
        if common_times is None:
            common_times = t
        else:
            common_times = common_times & t
    common_times = sorted(common_times)
    print(f"Common timestamps: {len(common_times)}")

    start_time = int(common_times[0])
    n_bars = len(common_times)

    data = {}
    for pair in pairs_present:
        df = all_data[pair]
        time_map = dict(zip(df["time"].values, range(len(df))))
        rows = []
        for t in common_times:
            idx = time_map[t]
            row = df.iloc[idx]
            mid = (row["open"] + row["close"]) / 2
            spread = row["high"] - row["low"]
            rows.append({
                "bid": round(mid - spread / 2, 5),
                "ask": round(mid + spread / 2, 5),
                "time": int(t),
            })
        data[pair] = rows

    return data, start_time, n_bars, pairs_present


def generate_synthetic_data(n_bars=10000, seed=42):
    """Generate synthetic 1-min bar data for testing."""
    rng = np.random.RandomState(seed)
    start_time = int(time.time()) - n_bars * 60

    data = {}
    base_prices = {
        "EURUSD": 1.10, "USDJPY": 150.0, "EURJPY": 165.0,
        "EURGBP": 0.85, "GBPUSD": 1.29, "GBPJPY": 193.5,
        "AUDUSD": 0.65, "AUDJPY": 97.5,
    }

    for pair in PAIRS:
        base = base_prices[pair]
        prices = base * np.exp(np.cumsum(rng.normal(0, 0.0003, n_bars)))
        bid = prices * (1 - 0.0001 * rng.uniform(0.5, 1.5))
        ask = prices * (1 + 0.0001 * rng.uniform(0.5, 1.5))

        rows = []
        for i in range(n_bars):
            t = start_time + i * 60
            rows.append({"bid": round(bid[i], 5), "ask": round(ask[i], 5), "time": t})
        data[pair] = rows

    return data, start_time, n_bars


def run_backtest(data, start_time, n_bars, hold_minutes=5, lot_size=0.5, active_pairs=None):
    """Run the strategy over the data stream.

    Returns list of trade dicts and list of bar-level snapshots.
    """
    if active_pairs is None:
        active_pairs = PAIRS
    strategy = PhaseDislocation()
    trades = []
    open_positions = {}  # pair -> {entry_time, entry_price, direction, lot, signal_meta}

    for i in range(n_bars):
        bar_data = {}
        for pair in active_pairs:
            row = data[pair][i]
            bar_data[pair] = {"bid": row["bid"], "ask": row["ask"], "time": row["time"]}

        signals = strategy.generate_signal(bar_data)

        current_time = start_time + i * 60

        closed_this_bar = []
        for pair, pos in list(open_positions.items()):
            age = current_time - pos["entry_time"]
            if age >= hold_minutes * 60:
                exit_price = (bar_data[pair]["bid"] + bar_data[pair]["ask"]) / 2
                entry = pos["entry_price"]
                gross_pnl = lot_size * (exit_price - entry) * pos["direction"]
                trades.append({
                    "pair": pair,
                    "entry_time": pos["entry_time"],
                    "exit_time": current_time,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "direction": pos["direction"],
                    "gross_pnl": round(gross_pnl, 2),
                    "hold_minutes": round(age / 60, 1),
                    "triangle": pos.get("triangle", "?"),
                    "dislocation_z": pos.get("dislocation_z", 0),
                })
                closed_this_bar.append(pair)

        for pair in closed_this_bar:
            del open_positions[pair]

        if signals and len(open_positions) == 0:
            sig = signals[0]  # take first signal only (max 1 position)
            pair = sig["pair"]
            direction = sig["direction"]
            entry_price = (bar_data[pair]["bid"] + bar_data[pair]["ask"]) / 2
            open_positions[pair] = {
                "entry_time": current_time,
                "entry_price": entry_price,
                "direction": direction,
                "lot": lot_size,
                "triangle": sig.get("triangle", "?"),
                "dislocation_z": sig.get("dislocation_z", 0),
            }

    return trades


def compute_metrics(trades):
    """Compute validation metrics from trade list."""
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "sharpe": 0, "profit_factor": 0, "max_dd": 0}

    pnls = np.array([t["gross_pnl"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate = len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    profit_factor = np.sum(wins) / abs(np.sum(losses)) if np.sum(losses) != 0 else float("inf")

    sharpe = 0
    if len(pnls) >= 2 and np.std(pnls) > 1e-10:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(1440 / 5)

    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    max_dd = abs(np.min(dd))

    pair_pnl = {}
    for t in trades:
        p = t["pair"]
        pair_pnl[p] = pair_pnl.get(p, 0) + t["gross_pnl"]

    tri_pnl = {}
    for t in trades:
        tri = t.get("triangle", "?")
        tri_pnl[tri] = tri_pnl.get(tri, 0) + t["gross_pnl"]

    return {
        "total_trades": len(trades),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": round(win_rate, 1),
        "avg_pnl": round(float(np.mean(pnls)), 4),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 2),
        "total_pnl": round(float(np.sum(pnls)), 2),
        "pair_pnl": {k: round(v, 2) for k, v in sorted(pair_pnl.items(), key=lambda x: abs(x[1]), reverse=True)},
        "triangle_pnl": {k: round(v, 2) for k, v in sorted(tri_pnl.items(), key=lambda x: abs(x[1]), reverse=True)},
    }


def main():
    parser = argparse.ArgumentParser(description="Phase Dislocation Backtest")
    parser.add_argument("--data", help="Directory containing Dukascopy M1 CSV files")
    parser.add_argument("--market", action="store_true", help="Load from data/market/ parquet files")
    parser.add_argument("--temp", action="store_true", help="Load from data/temp/ mt5_m1_9day parquet")
    parser.add_argument("--generate-synthetic", action="store_true", help="Run on synthetic data")
    parser.add_argument("--hold-minutes", type=int, default=5, help="Position hold time in minutes")
    parser.add_argument("--lot-size", type=float, default=0.5, help="Lot size")
    parser.add_argument("--bars", type=int, default=10000, help="Number of bars for synthetic data")
    args = parser.parse_args()

    if args.market:
        print("Loading market parquet data...")
        result = load_market_parquet()
        if result is None:
            return
        data, start_time, n_bars, pairs_present = result

    elif args.temp:
        print("Loading temp multi-pair parquet data...")
        import pandas as pd
        path = "data/temp/mt5_m1_9day.parquet"
        df = pd.read_parquet(path)
        print(f"Loaded {len(df):,} rows, pairs: {sorted(df['pair'].unique())}")

        # Only use pairs we need
        avail_pairs = [p for p in PAIRS if p in df["pair"].unique()]
        print(f"Available strategy pairs: {avail_pairs}")

        # Pivot to pair -> list of rows
        data = {}
        all_times = set()
        for pair in avail_pairs:
            sub = df[df["pair"] == pair].sort_values("time")
            rows = []
            for _, row in sub.iterrows():
                t = int(row["time"].timestamp()) if hasattr(row["time"], "timestamp") else int(row["time"])
                mid = (row["open"] + row["close"]) / 2
                spread = row["high"] - row["low"]
                rows.append({
                    "bid": round(mid - spread / 2, 5),
                    "ask": round(mid + spread / 2, 5),
                    "time": t,
                })
                all_times.add(t)
            data[pair] = rows

        common_times = sorted(all_times)
        # Filter to only times that exist in all available pairs
        for pair in avail_pairs:
            times_in_pair = set(r["time"] for r in data[pair])
            common_times = [t for t in common_times if t in times_in_pair]

        # Align all pairs to common times
        aligned = {}
        for pair in avail_pairs:
            time_map = {r["time"]: r for r in data[pair]}
            aligned[pair] = [time_map[t] for t in common_times]
        data = aligned
        pairs_present = avail_pairs
        start_time = int(common_times[0])
        n_bars = len(common_times)
        print(f"Common timestamps: {n_bars}")

    elif args.data:
        csv_files = glob.glob(os.path.join(args.data, "*.csv"))
        if not csv_files:
            print(f"No CSV files found in {args.data}")
            return

        print(f"Loading {len(csv_files)} CSV files from {args.data}...")
        all_data = {}
        for f in csv_files:
            pair = pair_from_filename(f)
            df = load_dukascopy_csv(f)
            all_data[pair] = df
            print(f"  {pair}: {len(df)} rows")

        pairs_present = [p for p in PAIRS if p in all_data]
        missing = [p for p in PAIRS if p not in all_data]
        if missing:
            print(f"Warning: missing data for {missing}")
            if len(pairs_present) < 3:
                print("Need at least 3 pairs for a triangle. Aborting.")
                return

        common_times = None
        for p in pairs_present:
            t = set(all_data[p]["time"].values)
            if common_times is None:
                common_times = t
            else:
                common_times = common_times & t
        common_times = sorted(common_times)
        print(f"Common timestamps: {len(common_times)}")

        start_time = int(common_times[0])
        n_bars = len(common_times)

        data = {}
        for pair in pairs_present:
            df = all_data[pair]
            time_map = dict(zip(df["time"].values, range(len(df))))
            rows = []
            for t in common_times:
                idx = time_map[t]
                row = df.iloc[idx]
                mid = (row["Open"] + row["Close"]) / 2
                spread = row["High"] - row["Low"]
                rows.append({
                    "bid": round(mid - spread / 2, 5),
                    "ask": round(mid + spread / 2, 5),
                    "time": int(t),
                })
            data[pair] = rows

    elif args.generate_synthetic:
        print(f"Generating {args.bars} bars of synthetic data...")
        data, start_time, n_bars = generate_synthetic_data(n_bars=args.bars)
        pairs_present = PAIRS
    else:
        print("Specify --market, --temp, --data <dir>, or --generate-synthetic")
        return

    print(f"\nRunning backtest: {len(pairs_present)} pairs, {n_bars} bars, hold {args.hold_minutes}min...")
    trades = run_backtest(data, start_time, n_bars, hold_minutes=args.hold_minutes, lot_size=args.lot_size, active_pairs=pairs_present)
    metrics = compute_metrics(trades)

    print("\n" + "=" * 60)
    print("  PHASE DISLOCATION — BACKTEST RESULTS")
    print("=" * 60)

    if trades:
        print(f"  Total Trades:  {metrics['total_trades']}")
        print(f"  Wins / Losses: {metrics['wins']} / {metrics['losses']}")
        print(f"  Win Rate:      {metrics['win_rate']}%")
        print(f"  Avg PnL:       ${metrics['avg_pnl']:.2f}")
        print(f"  Avg Win:       ${metrics['avg_win']:.2f}")
        print(f"  Avg Loss:      ${metrics['avg_loss']:.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"  Sharpe (5min): {metrics['sharpe']}")
        print(f"  Max DD:        ${metrics['max_dd']:.2f}")
        print(f"  Total PnL:     ${metrics['total_pnl']:.2f}")

        print("\n  Per-Pair PnL:")
        for pair, pnl in metrics["pair_pnl"].items():
            print(f"    {pair:>7s}: ${pnl:>8.2f}")

        print("\n  Per-Triangle PnL:")
        for tri, pnl in metrics["triangle_pnl"].items():
            print(f"    {tri:>20s}: ${pnl:>8.2f}")

        worst = min(trades, key=lambda t: t["gross_pnl"])
        best = max(trades, key=lambda t: t["gross_pnl"])
        print(f"\n  Best Trade:  {best['pair']} ${best['gross_pnl']:.2f} (z={best['dislocation_z']:.1f})")
        print(f"  Worst Trade: {worst['pair']} ${worst['gross_pnl']:.2f} (z={worst['dislocation_z']:.1f})")
    else:
        print("  No trades generated.")

    print("=" * 60)


if __name__ == "__main__":
    main()
