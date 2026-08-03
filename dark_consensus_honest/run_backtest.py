from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proxima_honest_backtest.engine.reconciliation import reconcile
from proxima_honest_backtest.engine.types import Trade
from proxima_honest_backtest.execution.execution_simulator import (
    ExecutionSimulator,
    list_broker_profiles,
)

from strategy import DarkConsensusStrategy

DATA_DIR = Path(__file__).resolve().parent / "data"
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

PIP_VALUES: Dict[str, float] = {
    "EURUSD": 10.0,
    "EURJPY": None,
    "GBPJPY": None,
}


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _pip_to_usd(pips: float, symbol: str, usdjpy_rate: Optional[float] = None) -> float:
    if symbol.upper().endswith("JPY"):
        if usdjpy_rate is None or usdjpy_rate <= 0:
            return 0.0
        return pips * 1000.0 / usdjpy_rate
    return pips * 10.0


def _calc_pnl_usd(
    entry_price: float,
    exit_price: float,
    quantity: float,
    side: str,
    symbol: str,
    usdjpy_rate: Optional[float] = None,
) -> float:
    if side.upper() == "BUY":
        price_diff = exit_price - entry_price
    else:
        price_diff = entry_price - exit_price

    if symbol.upper().endswith("JPY"):
        pip_move = price_diff / 0.01
    else:
        pip_move = price_diff / 0.0001

    lot_size = quantity / 100000.0
    return pip_move * _pip_to_usd(1.0, symbol, usdjpy_rate) * lot_size


def _download_monthly(
    sym: str, year: int, month: int, mt5_tf: int
) -> Optional[Any]:
    import MetaTrader5 as mt5
    import pandas as pd
    from_dt = datetime(year, month, 1)
    if month == 12:
        to_dt = datetime(year + 1, 1, 1)
    else:
        to_dt = datetime(year, month + 1, 1)
    rates = mt5.copy_rates_range(sym, mt5_tf, from_dt, to_dt)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def load_data_from_mt5(
    symbols: List[str],
    from_date: datetime,
    to_date: datetime,
    timeframe: str = "m1",
) -> Dict[str, Any]:
    try:
        import MetaTrader5 as mt5
        mt5_terminal = "C:\\Program Files\\FundedNext MT5 Terminal\\terminal64.exe"
        init_ok = mt5.initialize(path=mt5_terminal)
        print(f"  MT5 init: {init_ok}")
        if not init_ok:
            err = mt5.last_error()
            print(f"  MT5 error: {err}")
            return {}

        import pandas as pd
        tf_map = {"m1": mt5.TIMEFRAME_M1, "m5": mt5.TIMEFRAME_M5, "m15": mt5.TIMEFRAME_M15}
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)

        months = []
        cur = datetime(from_date.year, from_date.month, 1)
        end = datetime(to_date.year, to_date.month, 1)
        while cur <= end:
            months.append((cur.year, cur.month))
            if cur.month == 12:
                cur = datetime(cur.year + 1, 1, 1)
            else:
                cur = datetime(cur.year, cur.month + 1, 1)

        result: Dict[str, Any] = {}
        for sym in symbols:
            all_dfs = []
            for year, month in months:
                df = _download_monthly(sym, year, month, mt5_tf)
                count = len(df) if df is not None else 0
                if count > 3:
                    all_dfs.append(df)
                    print(f"    {sym} {year}-{month:02d}: {count} bars")
            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True).sort_values("time")
                combined = combined.drop_duplicates(subset=["time"])
                result[sym] = combined
                print(f"  {sym}: {len(combined)} bars loaded ({len(months)} months)")
            else:
                print(f"  WARNING: No data for {sym}")
                result[sym] = None

        mt5.shutdown()
        return result
    except Exception as e:
        print(f"ERROR loading from MT5: {e}")
        return {}


def _load_parquet_from_dirs(sym: str, search_dirs: List[Path]) -> Optional[Any]:
    import pandas as pd
    for base_dir in search_dirs:
        sym_dir = base_dir / sym
        if not sym_dir.exists():
            continue
        parquet_files = sorted(sym_dir.glob("*.parquet"))
        if not parquet_files:
            continue
        dfs = [pd.read_parquet(pf) for pf in parquet_files]
        combined = pd.concat(dfs, ignore_index=True).sort_values("time")
        combined = combined.drop_duplicates(subset=["time"])
        return combined
    return None


def load_data_from_parquet(symbols: List[str], timeframe: str = "m1") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    bt_data_dir = Path(__file__).resolve().parent.parent / "proxima_honest_backtest" / "data" / timeframe
    search_dirs = [DATA_DIR, bt_data_dir]
    for sym in symbols:
        df = _load_parquet_from_dirs(sym, search_dirs)
        if df is not None:
            result[sym] = df
        else:
            result[sym] = None
    return result


def save_data_to_parquet(data: Dict[str, Any]) -> None:
    for sym, df in data.items():
        if df is None or df.empty:
            continue
        sym_dir = DATA_DIR / sym
        sym_dir.mkdir(parents=True, exist_ok=True)
        path = sym_dir / f"{sym}.parquet"
        df.to_parquet(path, index=False)
        print(f"  Saved {len(df)} bars to {path}")


def align_bars(
    data: Dict[str, Any],
    seed_bars: int = 60,
) -> Tuple[List[datetime], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    valid_symbols = [s for s, d in data.items() if d is not None and not d.empty]
    if not valid_symbols:
        return [], {}, {}

    import pandas as pd

    common_times = None
    aligned_closes: Dict[str, np.ndarray] = {}
    aligned_volumes: Dict[str, np.ndarray] = {}

    for sym in valid_symbols:
        df = data[sym]
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        df = df.drop_duplicates(subset=["time"]).set_index("time").sort_index()
        aligned_closes[sym] = df["close"].values
        aligned_volumes[sym] = df.get("volume", df.get("tick_volume", np.ones(len(df)))).values
        if common_times is None:
            common_times = df.index.values
        else:
            common_times = np.intersect1d(common_times, df.index.values)

    if common_times is None or len(common_times) == 0:
        return [], {}, {}

    common_times = np.sort(common_times)
    filtered_closes: Dict[str, np.ndarray] = {}
    filtered_volumes: Dict[str, np.ndarray] = {}
    time_set = set(common_times)
    for sym in valid_symbols:
        df = data[sym].copy()
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        mask = df.index.isin(time_set)
        filtered = df.loc[mask]
        filtered_closes[sym] = filtered["close"].values
        filtered_volumes[sym] = filtered.get("volume", filtered.get("tick_volume", np.ones(len(filtered)))).values

    return list(common_times), filtered_closes, filtered_volumes


def run_backtest(
    strategy: DarkConsensusStrategy,
    timestamps: List[datetime],
    closes: Dict[str, np.ndarray],
    volumes: Dict[str, np.ndarray],
    broker_profile: str = "exness",
    lot_size: float = 1.0,
    seed_bars: int = 60,
    initial_equity: float = 10000.0,
    verbose: bool = False,
    usdjpy_close: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    pairs = strategy.parameters["pairs"]
    valid_pairs = [p for p in pairs if p in closes]
    if not valid_pairs:
        return {"trades": [], "equity_curve": [], "returns": [], "error": "no data"}

    n = len(timestamps)

    close_arr = np.column_stack([closes[p] for p in valid_pairs])
    ret_arr = np.full_like(close_arr, np.nan, dtype=np.float64)
    ret_arr[1:] = np.log(close_arr[1:] / close_arr[:-1])

    simulators: Dict[str, ExecutionSimulator] = {}
    for p in valid_pairs:
        simulators[p] = ExecutionSimulator(profile_name=broker_profile, seed=42)

    trades: List[Trade] = []
    first_ts = _ts_to_dt(timestamps[0])
    equity_curve: List[Tuple[datetime, float]] = [(first_ts, initial_equity)]
    returns_list: List[float] = []
    open_positions: Dict[str, Dict[str, Any]] = {}
    report_interval = max(1, n // 100)

    pair_to_idx = {p: i for i, p in enumerate(valid_pairs)}

    for i in range(n):
        if i < 2:
            continue

        ts_dt = _ts_to_dt(timestamps[i])
        usdjpy_rate = float(usdjpy_close[i]) if usdjpy_close is not None and i < len(usdjpy_close) else None

        equity = initial_equity + sum(t.pnl for t in trades)
        equity_curve.append((ts_dt, equity))

        if i < seed_bars:
            continue

        for pos_key in list(open_positions.keys()):
            pos = open_positions[pos_key]
            pos["bars_held"] += 1
            if pos["bars_held"] >= strategy.parameters["hold_bars"]:
                idx = pair_to_idx[pos["pair"]]
                exit_price = float(close_arr[i, idx])
                pnl = _calc_pnl_usd(pos["entry_price"], exit_price, lot_size * 100000.0,
                                    "BUY" if pos["direction"] > 0 else "SELL", pos["pair"], usdjpy_rate)
                trade = Trade(
                    timestamp=ts_dt, symbol=pos["pair"],
                    side="SELL" if pos["direction"] > 0 else "BUY",
                    quantity=lot_size * 100000.0, price=exit_price,
                    commission=pos.get("commission", 0.0), pnl=pnl,
                )
                trades.append(trade)
                returns_list.append(pnl / initial_equity)
                if verbose:
                    print(f"  EXIT {pos['pair']} @ {ts_dt.time():%H:%M:%S} | P&L: ${pnl:+.2f}")
                del open_positions[pos_key]

        current_equity = initial_equity + sum(t.pnl for t in trades)
        equity_curve[-1] = (ts_dt, current_equity)

        bars_dict = {p: float(close_arr[i, pair_to_idx[p]]) for p in pairs if p in pair_to_idx}
        prev_returns = {p: float(ret_arr[i, pair_to_idx[p]]) for p in pairs if p in pair_to_idx}

        missing = [p for p in pairs if p not in prev_returns]
        if missing:
            continue
        if any(np.isnan(list(prev_returns.values()))):
            continue

        signal = strategy.on_bars(bars_dict, prev_returns, ts_dt)
        if signal is not None and signal.signal != 0.0:
            pair = signal.metadata["pair"]
            direction = signal.metadata["direction"]
            if pair not in open_positions:
                simulator = simulators[pair]
                report = simulator.execute_order(
                    side=direction.upper(),
                    quantity=lot_size * 100000.0,
                    symbol=pair,
                    price=bars_dict[pair],
                    volatility=0.2,
                    hour_utc=ts_dt.hour,
                )
                if not report.filled:
                    if verbose:
                        print(f"  REJECT {pair} @ {ts_dt.time():%H:%M:%S} | {report.reject_reason}")
                    continue

                open_positions[pair] = {
                    "pair": pair,
                    "direction": 1.0 if direction == "LONG" else -1.0,
                    "entry_price": report.fill_price,
                    "entry_time": ts_dt,
                    "bars_held": 0,
                    "commission": report.trade.commission,
                }
                if verbose:
                    print(f"  ENTER {pair} {direction} @ {report.fill_price:.5f} | conf={signal.confidence:.2f}")

        total_bars_processed = i - seed_bars + 1
        if not verbose and total_bars_processed > 0 and total_bars_processed % report_interval == 0:
            pct = min(100, total_bars_processed * 100 // (n - seed_bars))
            print(f"  Progress: {pct}% ({total_bars_processed}/{n - seed_bars} bars, {len(trades)} trades)", end="\r")

    if not verbose:
        print()

    passed, trade_pnl, equity_delta, diff = reconcile(trades, equity_curve)
    metrics = _compute_metrics(trades, returns_list, equity_curve, initial_equity)
    metrics["reconciliation_passed"] = passed
    metrics["reconciliation_diff"] = diff

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "returns": returns_list,
        "metrics": metrics,
        "n_bars": n - seed_bars,
        "n_trades": len(trades),
    }


def _ts_to_dt(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts
    return pd.Timestamp(ts).to_pydatetime()


def _compute_metrics(
    trades: List[Trade],
    returns: List[float],
    equity_curve: List[Tuple[datetime, float]],
    initial_equity: float,
) -> Dict[str, Any]:
    if not trades:
        return {
            "total_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "commission_total": 0.0,
            "avg_bars_held": 0.0,
        }

    pnls = np.array([t.pnl for t in trades], dtype=np.float64)
    total_pnl = float(np.sum(pnls))
    gross_profit = float(np.sum(pnls[pnls > 0]))
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0])))
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    n_trades = len(trades)
    n_wins = int(np.sum(pnls > 0))
    win_rate = float(n_wins / n_trades) if n_trades > 0 else 0.0
    total_commission = float(np.sum([t.commission for t in trades]))
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    eq_values = [e for _, e in equity_curve]
    max_dd = _calc_max_drawdown(eq_values)

    ret_arr = np.array(returns, dtype=np.float64) if returns else np.array([0.0])
    sharpe = _calc_sharpe(ret_arr)

    return {
        "total_pnl": total_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "commission_total": total_commission,
    }


def _calc_max_drawdown(equity_curve: List[float]) -> float:
    arr = np.array(equity_curve, dtype=np.float64)
    peaks = np.maximum.accumulate(arr)
    drawdowns = (peaks - arr) / peaks
    max_dd = float(np.max(drawdowns))
    return max_dd if np.isfinite(max_dd) else 0.0


def _calc_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return 0.0
    mean_ret = float(np.mean(returns))
    daily_sharpe = (mean_ret - risk_free_rate) / std
    return daily_sharpe * math.sqrt(252 * 24 * 60)


def print_metrics(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("  DARK CONSENSUS — BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Trades:          {metrics['n_trades']}")
    print(f"  Win Rate:        {metrics['win_rate']:.1%}")
    print(f"  Net P&L:         ${metrics['total_pnl']:+.2f}")
    print(f"  Gross Profit:    ${metrics['gross_profit']:.2f}")
    print(f"  Gross Loss:      ${metrics['gross_loss']:.2f}")
    print(f"  Profit Factor:   {metrics['profit_factor']:.2f}")
    print(f"  Sharpe (annual): {metrics['sharpe']:.2f}")
    print(f"  Max DD:          {metrics['max_dd']:.2%}")
    print(f"  Avg Win:         ${metrics['avg_win']:.2f}")
    print(f"  Avg Loss:        ${metrics['avg_loss']:.2f}")
    print(f"  Commission:      ${metrics['commission_total']:.2f}")
    print(f"  Reconciled:      {'YES' if metrics.get('reconciliation_passed', True) else 'FAILED'}")
    print("=" * 60)


def main():
    config = load_config()
    strat_cfg = config["strategy"]
    exec_cfg = config["execution"]
    bt_cfg = config["backtest"]

    strategy = DarkConsensusStrategy(parameters=strat_cfg)

    pairs = strat_cfg["pairs"]
    seed_bars = int(bt_cfg["seed_history_bars"])
    initial_equity = float(bt_cfg["initial_equity"])
    lot_size = float(exec_cfg["lot_size"])
    broker_profile = exec_cfg["broker_profile"]

    from_date = datetime(2025, 1, 1)
    to_date = datetime(2026, 7, 28)

    print(f"\nDark Consensus Honest Backtest")
    print(f"  Pairs:           {pairs}")
    print(f"  Threshold:       {strat_cfg['mag_threshold']}")
    print(f"  Hold:            {strat_cfg['hold_bars']} bars")
    print(f"  Session:         {strat_cfg['session_start']}-{strat_cfg['session_end']} UTC")
    print(f"  Broker:          {broker_profile}")
    print(f"  Lot:             {lot_size}")
    print(f"  Period:          {from_date.date()} to {to_date.date()}")
    print(f"  Seed bars:       {seed_bars}")

    print("\nLoading data...")
    data = load_data_from_parquet(pairs)
    has_data = any(d is not None and not d.empty for d in data.values())

    if not has_data:
        print("No cached data found. Attempting MT5 download...")
        data = load_data_from_mt5(pairs, from_date, to_date)
        if any(d is not None and not d.empty for d in data.values()):
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            save_data_to_parquet(data)
        else:
            print("ERROR: No data available. Ensure MT5 terminal is running.")
            return

    print("\nAligning bars...")
    timestamps, closes, volumes = align_bars(data, seed_bars=seed_bars)
    if not timestamps:
        print("ERROR: No common timestamps across pairs.")
        return
    print(f"  Aligned bars: {len(timestamps)}")

    if "EURJPY" in closes and "EURUSD" in closes:
        usdjpy_close = closes["EURJPY"] / closes["EURUSD"]
        print(f"  USDJPY: derived from EURJPY/EURUSD ({len(usdjpy_close)} bars)")
    else:
        usdjpy_close = None
        print("  WARNING: Cannot derive USDJPY — JPY pair PnL will be $0")

    t0 = time.perf_counter()
    result = run_backtest(
        strategy=strategy,
        timestamps=timestamps,
        closes=closes,
        volumes=volumes,
        broker_profile=broker_profile,
        lot_size=lot_size,
        seed_bars=seed_bars,
        initial_equity=initial_equity,
        verbose=True,
        usdjpy_close=usdjpy_close,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nElapsed: {elapsed:.2f}s")
    print_metrics(result["metrics"])

    report_path = Path(__file__).resolve().parent / "reports"
    report_path.mkdir(exist_ok=True)

    with open(report_path / "backtest_result.json", "w") as f:
        json.dump(result["metrics"], f, indent=2, default=str)
    print(f"\nReport saved to {report_path / 'backtest_result.json'}")

    return result


if __name__ == "__main__":
    main()
