from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from proxima_honest_backtest.engine.types import PointInTime, Trade
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator


@dataclass
class ComparisonReport:
    strategy_name: str
    symbol: str
    broker_results: dict[str, dict] = field(default_factory=dict)
    ranking: list[str] = field(default_factory=list)
    spread_sensitivity: float = 0.0
    commission_impact: float = 0.0
    details: dict = field(default_factory=dict)


class BrokerComparer:
    DEFAULT_BROKERS = [
        "exness",
        "dukascopy",
        "fundednext",
        "fusionmarkets",
        "ftmo",
    ]

    def __init__(self, broker_names: Optional[list[str]] = None) -> None:
        self.broker_names = broker_names or self.DEFAULT_BROKERS

    def compare(
        self,
        strategy_func: Callable,
        symbol: str,
        ticks: list[PointInTime],
        n_runs: int = 10,
    ) -> ComparisonReport:
        broker_results: dict[str, dict] = {}
        seeds = list(range(n_runs))

        for broker in self.broker_names:
            metrics_list: list[dict] = []
            for seed in seeds:
                metrics = self._run_single_backtest(strategy_func, ticks, broker, seed)
                metrics_list.append(metrics)

            avg_metrics = self._average_metrics(metrics_list)
            broker_results[broker] = avg_metrics

        ranking = sorted(
            self.broker_names,
            key=lambda b: broker_results[b].get("total_pnl", 0.0),
            reverse=True,
        )

        spreads = []
        pnls = []
        for broker in self.broker_names:
            br = broker_results[broker]
            spread = br.get("avg_spread", 0.0)
            pnl = br.get("total_pnl", 0.0)
            spreads.append(spread)
            pnls.append(pnl)

        if len(spreads) >= 2:
            corr_matrix = np.corrcoef(spreads, pnls)
            spread_sensitivity = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        else:
            spread_sensitivity = 0.0

        total_gross_profit = sum(
            broker_results[b].get("gross_profit", 0.0) for b in self.broker_names
        )
        total_commission = sum(
            broker_results[b].get("commission_total", 0.0) for b in self.broker_names
        )
        commission_impact = (
            total_commission / total_gross_profit if total_gross_profit > 0 else 1.0
        )

        return ComparisonReport(
            strategy_name=strategy_func.__name__ if hasattr(strategy_func, "__name__") else "strategy",
            symbol=symbol,
            broker_results=broker_results,
            ranking=ranking,
            spread_sensitivity=spread_sensitivity,
            commission_impact=commission_impact,
            details={
                "n_brokers": len(self.broker_names),
                "n_runs": n_runs,
                "n_ticks": len(ticks),
            },
        )

    def compare_multiple_symbols(
        self,
        strategy_func: Callable,
        symbol_tick_data: dict[str, list[PointInTime]],
        n_runs: int = 5,
    ) -> list[ComparisonReport]:
        reports: list[ComparisonReport] = []
        for symbol, ticks in symbol_tick_data.items():
            report = self.compare(strategy_func, symbol, ticks, n_runs=n_runs)
            reports.append(report)
        return reports

    def _run_single_backtest(
        self,
        strategy_func: Callable,
        ticks: list[PointInTime],
        broker_name: str,
        seed: int,
    ) -> dict:
        simulator = ExecutionSimulator(broker_name=broker_name)
        trades: list[Trade] = strategy_func(ticks, simulator)

        metrics = self._compute_metrics(trades)

        broker_profile = simulator.broker_profile if hasattr(simulator, "broker_profile") else {}
        metrics["avg_spread"] = broker_profile.get("typical_spread", 0.0)
        metrics["seed"] = seed

        return metrics

    def _compute_metrics(
        self,
        trades: list[Trade],
        initial_equity: float = 10000.0,
    ) -> dict:
        if not trades:
            return {
                "total_pnl": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "sharpe": 0.0,
                "max_dd": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "commission_total": 0.0,
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

        equity_curve = self._build_equity_curve(trades, initial_equity)
        max_dd = self._calc_max_drawdown(equity_curve)

        returns = pnls / initial_equity
        sharpe = self._calc_sharpe(returns.tolist())

        return {
            "total_pnl": total_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "win_rate": win_rate,
            "total_trades": n_trades,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "commission_total": total_commission,
        }

    def _average_metrics(self, metrics_list: list[dict]) -> dict:
        if not metrics_list:
            return {}

        keys = metrics_list[0].keys()
        averaged: dict[str, Any] = {}
        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            if all(isinstance(v, (int, float)) for v in values):
                averaged[key] = float(np.mean(values))
            elif all(isinstance(v, str) for v in values):
                averaged[key] = values[0]
            else:
                averaged[key] = values[0] if values else None
        return averaged

    def _build_equity_curve(
        self,
        trades: list[Trade],
        initial_equity: float,
    ) -> list[float]:
        eq = initial_equity
        curve: list[float] = [eq]
        for t in trades:
            eq += t.pnl
            curve.append(eq)
        return curve

    def _calc_max_drawdown(self, equity_curve: list[float]) -> float:
        arr = np.array(equity_curve, dtype=np.float64)
        peaks = np.maximum.accumulate(arr)
        drawdowns = (peaks - arr) / peaks
        max_dd = float(np.max(drawdowns))
        return max_dd if np.isfinite(max_dd) else 0.0

    def _calc_sharpe(self, returns: list[float], risk_free_rate: float = 0.0) -> float:
        arr = np.array(returns, dtype=np.float64)
        if len(arr) < 2:
            return 0.0
        std = float(np.std(arr, ddof=1))
        if std == 0.0:
            return 0.0
        mean_ret = float(np.mean(arr))
        daily_sharpe = (mean_ret - risk_free_rate) / std
        return daily_sharpe * np.sqrt(252)

    def generate_report_markdown(self, report: ComparisonReport) -> str:
        lines: list[str] = []
        lines.append(f"# Broker Comparison: {report.strategy_name} on {report.symbol}")
        lines.append("")
        lines.append("## Ranking (by Net PnL)")
        lines.append("")
        lines.append("| Rank | Broker | Net PnL | Win Rate | Sharpe | Max DD | Profit Factor | Trades |")
        lines.append("|------|--------|---------|----------|--------|--------|---------------|--------|")

        for rank, broker in enumerate(report.ranking, 1):
            r = report.broker_results.get(broker, {})
            pnl = r.get("total_pnl", 0.0)
            wr = r.get("win_rate", 0.0)
            sharpe = r.get("sharpe", 0.0)
            mdd = r.get("max_dd", 0.0)
            pf = r.get("profit_factor", 0.0)
            nt = r.get("total_trades", 0)
            lines.append(
                f"| {rank} | {broker} | ${pnl:+.2f} | {wr:.1%} | {sharpe:.2f} | "
                f"{mdd:.1%} | {pf:.2f} | {nt} |"
            )

        lines.append("")
        lines.append(f"**Spread Sensitivity (correlation):** {report.spread_sensitivity:.4f}")
        lines.append(f"**Commission Impact:** {report.commission_impact:.2%} of gross profit")
        lines.append("")

        lines.append("## Detailed Broker Results")
        lines.append("")
        for broker in report.ranking:
            r = report.broker_results.get(broker, {})
            lines.append(f"### {broker}")
            lines.append(f"- Net PnL: ${r.get('total_pnl', 0.0):+.2f}")
            lines.append(f"- Gross Profit: ${r.get('gross_profit', 0.0):.2f}")
            lines.append(f"- Gross Loss: ${r.get('gross_loss', 0.0):.2f}")
            lines.append(f"- Win Rate: {r.get('win_rate', 0.0):.1%}")
            lines.append(f"- Sharpe Ratio: {r.get('sharpe', 0.0):.3f}")
            lines.append(f"- Max Drawdown: {r.get('max_dd', 0.0):.1%}")
            lines.append(f"- Profit Factor: {r.get('profit_factor', 0.0):.3f}")
            lines.append(f"- Avg Win: ${r.get('avg_win', 0.0):.2f}")
            lines.append(f"- Avg Loss: ${r.get('avg_loss', 0.0):.2f}")
            lines.append(f"- Commission Total: ${r.get('commission_total', 0.0):.2f}")
            lines.append(f"- Total Trades: {r.get('total_trades', 0)}")
            lines.append("")

        return "\n".join(lines)

    def plot_comparison(
        self,
        report: ComparisonReport,
        save_path: Optional[str] = None,
    ) -> None:
        import matplotlib.pyplot as plt

        brokers = report.ranking
        n_brokers = len(brokers)

        metrics_to_plot = [
            ("total_pnl", "Net PnL ($)", "green"),
            ("sharpe", "Sharpe Ratio", "blue"),
            ("win_rate", "Win Rate", "purple"),
            ("max_dd", "Max Drawdown", "red"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Broker Comparison: {report.strategy_name} on {report.symbol}",
            fontsize=14,
            fontweight="bold",
        )

        for ax, (metric_key, ylabel, color) in zip(axes.flatten(), metrics_to_plot):
            values: list[float] = []
            for broker in brokers:
                r = report.broker_results.get(broker, {})
                val = r.get(metric_key, 0.0)
                if metric_key == "win_rate":
                    val = val * 100.0
                elif metric_key == "max_dd":
                    val = val * 100.0
                values.append(val)

            x = np.arange(n_brokers)
            bars = ax.bar(x, values, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(brokers, rotation=30, ha="right")
            ax.set_ylabel(ylabel)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{val:.1f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8,
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
