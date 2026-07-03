from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from research.adaptive_alpha_engine.aae_validator import AAEValidator, AAEResult


class LiveSystem:
    def __init__(self, validator: AAEValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> AAEResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)
        energy_storage: NDArray[np.float64] = signals["energy_storage"]
        adaptive_time: NDArray[np.float64] = signals["adaptive_time"]
        price: NDArray[np.float64] = signals["price"]
        high: NDArray[np.float64] = signals.get("high", price)
        low: NDArray[np.float64] = signals.get("low", price)

        n = len(price)

        # Rolling percentiles
        rolling_90th = self.validator.rolling_percentile(energy_storage, 504, 90)
        rolling_50th = self.validator.rolling_percentile(energy_storage, 504, 50)

        # AT quintile buckets
        at_clean = adaptive_time[~np.isnan(adaptive_time)]
        if len(at_clean) > 0:
            quintile_edges = np.nanpercentile(at_clean, [20, 40, 60, 80])
        else:
            quintile_edges = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        at_size_map = [0.10, 0.25, 0.50, 0.75, 1.0]

        def _at_quintile(at_val: float) -> int:
            if np.isnan(at_val):
                return 0
            if at_val <= quintile_edges[0]:
                return 0
            elif at_val <= quintile_edges[1]:
                return 1
            elif at_val <= quintile_edges[2]:
                return 2
            elif at_val <= quintile_edges[3]:
                return 3
            return 4

        # True Range and ATR(20)
        prev_close = np.roll(price, 1)
        prev_close[0] = price[0]
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - prev_close),
                                   np.abs(low - prev_close)))
        atr20 = np.full(n, np.nan, dtype=np.float64)
        for i in range(20, n):
            atr20[i] = np.mean(tr[i - 19:i + 1])

        position = 0
        entry_price = 0.0
        current_size = 0.0
        trade_returns: list[float] = []
        equity_values: list[float] = [0.0]

        for i in range(504, n):
            if position == 0:
                if not np.isnan(rolling_90th[i]) and not np.isnan(energy_storage[i]):
                    window = energy_storage[max(0, i - 504):i]
                    window_median = float(np.nanmedian(window)) if len(window) > 0 else 0.0
                    if energy_storage[i] > rolling_90th[i] and energy_storage[i] > window_median:
                        at_bucket = _at_quintile(adaptive_time[i])
                        current_size = at_size_map[at_bucket]
                        entry_price = price[i]
                        position = 1
            elif position == 1:
                stop_loss = entry_price - 2.0 * atr20[i]
                exit_signal = False
                if not np.isnan(rolling_50th[i]) and energy_storage[i] < rolling_50th[i]:
                    exit_signal = True
                elif price[i] <= stop_loss:
                    exit_signal = True

                if exit_signal:
                    ret = float(np.log(price[i] / entry_price))
                    trade_returns.append(ret)
                    equity_values.append(equity_values[-1] + ret)
                    position = 0
                    entry_price = 0.0

        total_return = float(np.sum(trade_returns)) if trade_returns else 0.0
        n_trades = len(trade_returns)
        win_rate = float(np.mean([r > 0 for r in trade_returns])) if trade_returns else 0.0
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r < 0]
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        sharpe = float(np.nanmean(trade_returns) / max(np.nanstd(trade_returns), 1e-12)) if trade_returns else 0.0

        eq = np.array(equity_values, dtype=np.float64)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.maximum(peak, 1e-12)
        max_dd = float(np.min(dd))

        print(f"\n  [Live System — {self.asset}]")
        print(f"  Total Return: {total_return:.4f}")
        print(f"  Trades: {n_trades}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg Win: {avg_win:.4f}  Avg Loss: {avg_loss:.4f}")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Max DD: {max_dd:.2%}")

        return AAEResult(
            rq_name="RQ9: Live System",
            status="COMPLETE",
            metrics={
                "total_return": total_return,
                "n_trades": n_trades,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
            },
        )
