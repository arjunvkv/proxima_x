from dataclasses import dataclass, field
import numpy as np
from typing import Optional
import json

from research.adaptive_alpha_engine.aae_validator import AAEValidator
from proxima_v1.core.signal_engine import SignalEngine
from proxima_v1.core.risk_engine import RiskEngine, RiskConfig
from proxima_v1.core.portfolio_engine import PortfolioEngine
from proxima_v1.core.execution_engine import ExecutionEngine
from proxima_v1.core.position_manager import PositionManager
from proxima_v1.core.trade_lifecycle import TradeLifecycle
from proxima_v1.core.metrics import MetricsTracker, TradeRecord


class CapitalEfficiency:
    SIZING_METHODS = ["FIXED", "AT_SIZE", "VOL_SIZE", "HYBRID"]

    def __init__(self, assets: Optional[list[str]] = None, capital: float = 100000.0):
        self.assets = assets if assets is not None else ["EURJPY", "USDJPY", "GBPJPY"]
        self.capital = capital
        self._np_rng = np.random.RandomState(42)

    def _generate_synthetic_data(self, n_bars: int = 2000) -> dict[str, dict]:
        data = {}
        for asset in self.assets:
            base_price = 100.0 + hash(asset) % 50
            close = base_price + np.cumsum(self._np_rng.randn(n_bars) * 0.3)
            close = np.abs(close) + 50.0
            high = close + np.abs(self._np_rng.randn(n_bars)) * 0.5
            low = close - np.abs(self._np_rng.randn(n_bars)) * 0.5
            low = np.minimum(low, close)
            high = np.maximum(high, close)
            returns = np.diff(np.log(close), prepend=np.log(close[0]))
            data[asset] = {
                "price": close.astype(np.float64),
                "high": high.astype(np.float64),
                "low": low.astype(np.float64),
                "returns": returns.astype(np.float64),
                "volume": np.ones(n_bars, dtype=np.float64),
            }
        return data

    def _compute_atr(self, data: dict, period: int = 20) -> np.ndarray:
        high = data["high"]
        low = data["low"]
        close = data["price"]
        n = len(close)
        tr = np.zeros(n, dtype=np.float64)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)
        atr = np.full(n, np.nan, dtype=np.float64)
        for i in range(period - 1, n):
            atr[i] = float(np.mean(tr[i - period + 1 : i + 1]))
        atr = np.nan_to_num(atr, nan=float(np.mean(tr[:period])))
        return atr

    def _compute_sizing_pcts(
        self, method: str, scores: np.ndarray, at_buckets: np.ndarray,
        atr_values: np.ndarray, atr_median: float, base_pct: float = 0.02,
    ) -> np.ndarray:
        n = len(scores)
        if method == "FIXED":
            pcts = np.full(n, base_pct, dtype=np.float64)
        elif method == "AT_SIZE":
            pcts = np.zeros(n, dtype=np.float64)
            for i in range(n):
                pcts[i] = base_pct * (1.0 + 0.25 * float(at_buckets[i])) * float(scores[i])
            pcts = np.clip(pcts, 0.005, 0.05)
        elif method == "VOL_SIZE":
            ratio = atr_median / np.maximum(atr_values, 1e-12)
            pcts = base_pct * ratio
            pcts = np.clip(pcts, 0.005, 0.05)
        elif method == "HYBRID":
            at_pcts = np.zeros(n, dtype=np.float64)
            for i in range(n):
                at_pcts[i] = base_pct * (1.0 + 0.25 * float(at_buckets[i])) * float(scores[i])
            ratio = atr_median / np.maximum(atr_values, 1e-12)
            pcts = at_pcts * ratio
            pcts = np.clip(pcts, 0.005, 0.05)
        else:
            pcts = np.full(n, base_pct, dtype=np.float64)
        return pcts

    def _run_backtest(self, sizing_method: str, max_positions: int = 3) -> dict:
        np.random.seed(42)
        n_bars = 2000
        warmup = 504

        data = self._generate_synthetic_data(n_bars)
        atr_data = {}
        atr_medians = {}
        for asset in self.assets:
            atr_data[asset] = self._compute_atr(data[asset], 20)
            atr_valid = atr_data[asset][~np.isnan(atr_data[asset])]
            atr_medians[asset] = float(np.median(atr_valid)) if len(atr_valid) > 0 else 1.0

        sig_engines = {}
        risk_engines = {}
        for asset in self.assets:
            re_config = RiskConfig(
                base_position_pct=0.02,
                max_position_pct=0.05,
                min_position_pct=0.005,
            )
            risk_engines[asset] = RiskEngine(asset, re_config)
            sig_engines[asset] = SignalEngine(asset)

        pm = PositionManager(max_positions=max_positions)
        exec_eng = ExecutionEngine()
        metrics = MetricsTracker()

        active_positions = {}
        holding_periods = {}
        open_times = {}
        total_exposure_records = []

        for i in range(warmup, n_bars):
            bar_prices = {}
            for asset in self.assets:
                bar_prices[asset] = float(data[asset]["price"][i])

            bar_signals = {}
            for asset in self.assets:
                ret = float(data[asset]["returns"][i])
                noise = float(self._np_rng.randn() * 0.1)
                score = float(np.clip(0.5 + ret * 5.0 + noise, 0.0, 1.0))
                bar_signals[asset] = score

            adaptive_time_signals = {}
            if i > warmup + 100:
                for asset in self.assets:
                    at_chunk = data[asset]["returns"][i - 100 : i]
                    at_clean = np.nan_to_num(at_chunk, nan=0.0)
                    if len(at_clean) < 2:
                        at_sig = np.array([0.0], dtype=np.float64)
                    else:
                        flipped = -at_clean
                        flipped_cum = np.cumsum(np.abs(np.diff(flipped, prepend=flipped[0])))
                        at_sig = flipped_cum / max(flipped_cum[-1], 1e-12)
                    adaptive_time_signals[asset] = at_sig
            else:
                for asset in self.assets:
                    adaptive_time_signals[asset] = np.array([0.0])

            for asset in self.assets:
                score = bar_signals[asset]
                at_bucket = risk_engines[asset].get_at_bucket(
                    adaptive_time_signals.get(asset, np.array([0.0])), 5
                )

                if sizing_method in ("AT_SIZE", "HYBRID"):
                    size_pct = self._compute_sizing_pcts(
                        sizing_method,
                        np.array([score]),
                        np.array([at_bucket]),
                        np.array([atr_data[asset][i]]),
                        atr_medians[asset],
                    )[0]
                    size_val = size_pct * self.capital
                elif sizing_method == "VOL_SIZE":
                    size_pct = self._compute_sizing_pcts(
                        sizing_method,
                        np.array([score]),
                        np.array([at_bucket]),
                        np.array([atr_data[asset][i]]),
                        atr_medians[asset],
                    )[0]
                    size_val = size_pct * self.capital
                else:
                    size_val = 0.02 * self.capital

                pos = active_positions.get(asset)
                should_exit = False
                if pos is not None:
                    hp = holding_periods.get(asset, 20)
                    entry_t = open_times.get(asset, 0)
                    bars_held = i - entry_t
                    if bars_held >= hp or score < 0.3:
                        should_exit = True

                if should_exit and pos is not None:
                    exit_price = float(data[asset]["price"][i])
                    pnl = pos["size"] * (exit_price - pos["entry_price"])
                    pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
                    record = TradeRecord(
                        asset=asset, side="LONG",
                        entry_time=pos["entry_time"], exit_time=i,
                        entry_price=pos["entry_price"], exit_price=exit_price,
                        size=pos["size"], pnl=pnl, pnl_pct=pnl_pct,
                        holding_period=i - pos["entry_time"],
                        signal_score=pos["signal_score"],
                    )
                    metrics.record_trade(record)
                    del active_positions[asset]
                    if asset in holding_periods:
                        del holding_periods[asset]
                    if asset in open_times:
                        del open_times[asset]

                if score > 0.6 and asset not in active_positions:
                    if len(active_positions) < max_positions:
                        entry_price = float(data[asset]["price"][i])
                        stop_width = risk_engines[asset].compute_stop_width(
                            at_bucket, entry_price * 0.01
                        )
                        hp = risk_engines[asset].compute_holding_period(at_bucket)
                        active_positions[asset] = {
                            "entry_price": entry_price,
                            "size": size_val,
                            "entry_time": i,
                            "signal_score": score,
                        }
                        holding_periods[asset] = hp
                        open_times[asset] = i

            n_open = len(active_positions)
            total_exposure = sum(p["size"] for p in active_positions.values())
            total_exposure_records.append(total_exposure)

            daily_pnl = 0.0
            for asset, pos in active_positions.items():
                current_price = float(data[asset]["price"][i])
                pnl = pos["size"] * (current_price - pos["entry_price"])
                daily_pnl += pnl

            total_equity = self.capital + sum(
                t.pnl for t in metrics.trades
            ) + sum(
                active_positions[a]["size"] * (
                    float(data[a]["price"][i]) - active_positions[a]["entry_price"]
                ) for a in active_positions
            )
            metrics.record_daily_pnl(daily_pnl, total_equity)

        all_metrics = metrics.compute_all_metrics()
        avg_pos = float(np.mean(total_exposure_records)) / max(self.capital, 1.0)
        all_metrics["avg_positions"] = float(np.clip(
            float(np.mean([len(active_positions) for _ in range(warmup, n_bars)])), 0.0, 10.0
        ))

        n_tracked = n_bars - warmup
        pos_counts = np.zeros(n_tracked, dtype=np.float64)
        for j in range(n_tracked):
            idx = warmup + j
            count = 0
            for asset in self.assets:
                for ap in active_positions:
                    if asset == ap:
                        count += 1
            pos_counts[j] = float(count)
        all_metrics["avg_positions"] = float(np.mean(pos_counts))
        return all_metrics

    def _compute_efficiency(self, metrics: dict) -> dict:
        sharpe = float(metrics.get("sharpe", 0.0))
        pp = float(metrics.get("pp", 0.5))
        max_dd = float(metrics.get("max_dd", 0.0))
        total_return = float(metrics.get("total_return", 0.0))
        n_trades = int(metrics.get("n_trades", 0))
        avg_pos = float(metrics.get("avg_positions", 1.0))

        return_per_risk = sharpe
        return_per_dd = total_return / max(abs(max_dd), 1e-12)
        return_per_position = total_return / max(avg_pos, 1.0)
        return_per_trade = total_return / max(n_trades, 1)
        return {
            "return_per_risk": float(np.nan_to_num(return_per_risk, nan=0.0)),
            "return_per_dd": float(np.nan_to_num(return_per_dd, nan=0.0)),
            "return_per_position": float(np.nan_to_num(return_per_position, nan=0.0)),
            "return_per_trade": float(np.nan_to_num(return_per_trade, nan=0.0)),
        }

    def run(self) -> dict:
        np.random.seed(42)
        sizing_comparison = {}
        efficiency = {}

        for method in self.SIZING_METHODS:
            m = self._run_backtest(method)
            sizing_comparison[method] = {
                "sharpe": m["sharpe"],
                "pp": m["pp"],
                "max_dd": m["max_dd"],
                "total_return": m["total_return"],
                "n_trades": m["n_trades"],
                "avg_positions": m["avg_positions"],
            }
            efficiency[method] = self._compute_efficiency(m)

        def rank_key(method):
            s = float(sizing_comparison[method]["sharpe"])
            dd = float(sizing_comparison[method]["max_dd"])
            dd_penalty = 1.0 / max(abs(dd) * 10.0 + 0.1, 0.1)
            return float(s * dd_penalty)

        method_rankings = sorted(
            self.SIZING_METHODS, key=rank_key, reverse=True
        )

        best_method = method_rankings[0]
        optimal_concentration = {"best_n_positions": 3, "sharpe_at_best": 0.0}

        n_positions_tests = [1, 2, 3]
        sharpe_at_n = {}
        for n in n_positions_tests:
            m = self._run_backtest(best_method, max_positions=n)
            sharpe_at_n[n] = float(m["sharpe"])

        best_n = max(n_positions_tests, key=lambda x: sharpe_at_n[x])
        optimal_concentration = {
            "best_n_positions": best_n,
            "sharpe_at_best": sharpe_at_n[best_n],
            "sharpe_by_n": sharpe_at_n,
        }

        return {
            "sizing_comparison": sizing_comparison,
            "method_rankings": method_rankings,
            "efficiency": efficiency,
            "optimal_concentration": optimal_concentration,
            "best_method": best_method,
        }

    def save(self, path: str):
        result = self.run()
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
