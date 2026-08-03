from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from proxima_honest_backtest.engine.reconciliation import reconcile
from proxima_honest_backtest.engine.rolling_buffer import RollingBuffer
from proxima_honest_backtest.engine.types import PointInTime, SignalResult, Trade
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from proxima_honest_backtest.strategies.base import BaseStrategy


@dataclass
class BacktestResult:
    symbol: str
    strategy_name: str
    broker_profile: str
    n_bars: int
    n_trades: int
    total_pnl: float
    total_commission: float
    net_pnl: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_consecutive_losses: int
    reconciliation_pass: bool
    equity_curve: List[Tuple[datetime, float]]
    trades: List[Trade]
    details: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """Simple event-driven backtester connecting strategy → execution → reconciliation."""

    def __init__(
        self,
        strategy: BaseStrategy,
        execution_simulator: Optional[ExecutionSimulator] = None,
        initial_equity: float = 10000.0,
        volatility_lookback: int = 20,
    ) -> None:
        self.strategy = strategy
        self.simulator = execution_simulator or ExecutionSimulator("exness")
        self.initial_equity = initial_equity
        self.volatility_lookback = volatility_lookback

    def run(self, symbol: str, data: pd.DataFrame) -> BacktestResult:
        strategy = self.strategy
        strategy.reset()

        history = RollingBuffer(maxlen=200, columns=["close", "high", "low", "volume"])
        trades: List[Trade] = []
        equity_curve: List[Tuple[datetime, float]] = []

        position = 0
        entry_price = 0.0
        entry_qty = 0.0
        equity = self.initial_equity
        cash = self.initial_equity
        position_value = 0.0

        pnl_list: List[float] = []
        win_streak = 0
        max_consec_losses = 0

        bars = data.iterrows()
        for idx, (_, row) in enumerate(bars):
            ts = row.get("time") or pd.Timestamp.now()
            history.append({
                "close": float(row["close"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "volume": float(row.get("tick_volume", 0)),
            })

            bar_dict = {
                "time": ts,
                "open": float(row.get("open", row["close"])),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("tick_volume", 0)),
            }

            signal = strategy.on_bar(bar_dict, history)

            if signal is not None and position == 0 and abs(signal.signal) > 0.5:
                direction = "LONG" if signal.signal > 0 else "SHORT"
                quantity = 10000.0
                volatility = self._estimate_volatility(history)
                hour = ts.hour if hasattr(ts, "hour") else 0
                report = self.simulator.execute_order(
                    side=direction,
                    quantity=quantity,
                    symbol=symbol,
                    price=float(row["close"]),
                    volatility=volatility,
                    hour_utc=hour,
                )
                if report.filled:
                    position = 1 if direction == "LONG" else -1
                    entry_price = report.fill_price
                    entry_qty = report.trade.quantity
                    trades.append(report.trade)
                    equity_curve.append((ts, equity))

            elif signal is not None and position != 0:
                close_side = "SELL" if position == 1 else "BUY"
                exit_price = float(row["close"])
                pnl = self.simulator.calculate_pnl(
                    entry_price, exit_price, entry_qty,
                    "BUY" if position == 1 else "SELL", symbol,
                )
                commission = entry_qty / 100000.0 * self.simulator.profile.commission_per_lot
                net_pnl = pnl - commission

                trade = Trade(
                    timestamp=ts,
                    symbol=symbol,
                    side="SELL" if position == 1 else "BUY",
                    quantity=entry_qty,
                    price=exit_price,
                    commission=commission,
                    pnl=net_pnl,
                )
                trades.append(trade)

                equity += net_pnl
                cash += net_pnl
                position_value = 0.0
                pnl_list.append(net_pnl)

                if net_pnl > 0:
                    win_streak = max(win_streak + 1, 1)
                else:
                    win_streak = 0
                    max_consec_losses = max(max_consec_losses, win_streak + 1)

                equity_curve.append((ts, equity))
                position = 0

            equity_curve.append((ts, equity))

        return self._build_result(symbol, data, trades, equity_curve, pnl_list, max_consec_losses)

    def _estimate_volatility(self, history: RollingBuffer) -> float:
        closes = list(history.get_column("close"))
        if len(closes) < 10:
            return 0.001
        recent = closes[-min(self.volatility_lookback, len(closes)):]
        returns = [(recent[i] - recent[i - 1]) / recent[i - 1] for i in range(1, len(recent))]
        if not returns:
            return 0.001
        return float(np.std(returns))

    def _build_result(
        self,
        symbol: str,
        data: pd.DataFrame,
        trades: List[Trade],
        equity_curve: List[Tuple[datetime, float]],
        pnl_list: List[float],
        max_consec_losses: int,
    ) -> BacktestResult:
        if not pnl_list:
            return BacktestResult(
                symbol=symbol,
                strategy_name=self.strategy.name,
                broker_profile=self.simulator.profile_name,
                n_bars=len(data),
                n_trades=0,
                total_pnl=0.0,
                total_commission=0.0,
                net_pnl=0.0,
                sharpe=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_consecutive_losses=0,
                reconciliation_pass=True,
                equity_curve=equity_curve,
                trades=trades,
            )

        total_pnl = sum(t.pnl for t in trades)
        total_commission = sum(t.commission for t in trades)
        net_pnl = total_pnl - total_commission
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]
        win_rate = len(wins) / len(pnl_list) if pnl_list else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0.0

        eq_values = [e for _, e in equity_curve]
        peak = eq_values[0]
        max_dd = 0.0
        for v in eq_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        returns = [pnl / self.initial_equity for pnl in pnl_list]
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 288))

        eq_series = [(ts, eq) for ts, eq in equity_curve if len(equity_curve) > 0]
        rec_pass, _, _, _ = reconcile(
            trades, eq_series, tick_size=1e-8
        ) if trades and len(eq_series) >= 2 else (True, 0.0, 0.0, 0.0)

        return BacktestResult(
            symbol=symbol,
            strategy_name=self.strategy.name,
            broker_profile=self.simulator.profile_name,
            n_bars=len(data),
            n_trades=len(pnl_list),
            total_pnl=total_pnl,
            total_commission=total_commission,
            net_pnl=net_pnl,
            sharpe=sharpe,
            max_drawdown_pct=max_dd * 100,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_losses=max_consec_losses,
            reconciliation_pass=rec_pass,
            equity_curve=equity_curve,
            trades=trades,
            details={
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "n_wins": len(wins),
                "n_losses": len(losses),
            },
        )
