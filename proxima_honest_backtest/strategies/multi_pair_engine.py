from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from proxima_honest_backtest.engine.reconciliation import reconcile
from proxima_honest_backtest.engine.types import SignalResult, Trade
from proxima_honest_backtest.examples.backtest_engine import BacktestResult
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy


@dataclass
class MultiBacktestResult:
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


class MultiPairBacktestEngine:
    """Backtest engine for strategies that need multiple pairs simultaneously."""

    def __init__(
        self,
        strategy: MultiPairStrategy,
        execution_simulator: Optional[ExecutionSimulator] = None,
        initial_equity: float = 10000.0,
        volatility_lookback: int = 20,
    ) -> None:
        self.strategy = strategy
        self.simulator = execution_simulator or ExecutionSimulator("exness")
        self.initial_equity = initial_equity
        self.volatility_lookback = volatility_lookback

    def run(self, pairs_data: Dict[str, pd.DataFrame],
            pre_aligned: Optional[List[Dict]] = None) -> MultiBacktestResult:
        strategy = self.strategy
        strategy.reset()

        history: Dict[str, List[float]] = defaultdict(list)
        trades: List[Trade] = []
        equity_curve: List[Tuple[datetime, float]] = []

        positions: Dict[str, Dict[str, Any]] = {}
        equity = self.initial_equity
        pnl_list: List[float] = []
        win_streak = 0
        max_consec_losses = 0

        aligned = pre_aligned if pre_aligned is not None else self._align_bars(pairs_data)

        for ts_row in aligned:
            ts = ts_row["time"]
            bars: Dict[str, Dict] = {}
            for pair in pairs_data:
                val = ts_row.get(pair)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    bars[pair] = {
                        "time": ts,
                        "open": ts_row.get(f"{pair}_open", val),
                        "high": ts_row.get(f"{pair}_high", val),
                        "low": ts_row.get(f"{pair}_low", val),
                        "close": val,
                        "volume": ts_row.get(f"{pair}_volume", 0),
                        "spread": ts_row.get(f"{pair}_spread", 0),
                    }
                    history[pair].append(val)

            if len(bars) < 2:
                continue

            signals = strategy.on_bars(bars, dict(history))

            if signals:
                for signal in signals:
                    pair = signal.metadata.get("pair", "")
                    action = signal.metadata.get("action", "")

                    if "ENTER" in action and pair not in positions:
                        direction = "LONG" if signal.signal > 0 else "SHORT"
                        quantity = 10000.0
                        price = float(signal.metadata.get("entry_price", bars[pair]["close"]))
                        volatility = self._estimate_volatility(history.get(pair, []))
                        hour = ts.hour if hasattr(ts, "hour") else 0

                        report = self.simulator.execute_order(
                            side=direction,
                            quantity=quantity,
                            symbol=pair,
                            price=price,
                            volatility=volatility,
                            hour_utc=hour,
                            timestamp=ts,
                        )
                        if report.filled:
                            positions[pair] = {
                                "side": direction,
                                "entry_price": report.fill_price,
                                "quantity": report.trade.quantity,
                            }
                            trades.append(report.trade)

                    elif "EXIT" in action and pair in positions:
                        pos = positions.pop(pair)
                        exit_price = bars[pair]["close"]
                        pnl = self.simulator.calculate_pnl(
                            pos["entry_price"],
                            exit_price,
                            pos["quantity"],
                            "BUY" if pos["side"] == "LONG" else "SELL",
                            pair,
                        )
                        commission = pos["quantity"] / 100000.0 * self.simulator.profile.commission_per_lot
                        net_pnl = pnl - commission

                        trades.append(Trade(
                            timestamp=ts,
                            symbol=pair,
                            side="SELL" if pos["side"] == "LONG" else "BUY",
                            quantity=pos["quantity"],
                            price=exit_price,
                            commission=commission,
                            pnl=net_pnl,
                        ))

                        equity += net_pnl
                        pnl_list.append(net_pnl)
                        if net_pnl > 0:
                            win_streak = max(win_streak + 1, 1)
                        else:
                            win_streak = 0
                            max_consec_losses = max(max_consec_losses, abs(win_streak) + 1)

            equity_curve.append((ts, equity))

        pair_summary = self._summarize(pairs_data, trades, equity_curve, pnl_list, max_consec_losses)
        pair_summary.trades = trades
        return pair_summary

    def _align_bars(self, pairs_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Align bars across all pairs by timestamp — O(1) concat instead of N merges."""
        pieces = []
        for pair, df in pairs_data.items():
            sub = df.set_index("time")[["close", "open", "high", "low", "tick_volume", "spread"]]
            sub.columns = [
                pair, f"{pair}_open", f"{pair}_high",
                f"{pair}_low", f"{pair}_volume", f"{pair}_spread",
            ]
            pieces.append(sub)
        if not pieces:
            return []
        combined = pd.concat(pieces, axis=1, sort=True)
        combined.sort_index(inplace=True)
        combined.ffill(inplace=True)
        combined.reset_index(inplace=True)
        combined.rename(columns={"index": "time"}, inplace=True)
        return combined.to_dict("records")

    def _estimate_volatility(self, closes: List[float]) -> float:
        if len(closes) < 10:
            return 0.001
        recent = closes[-min(self.volatility_lookback, len(closes)):]
        returns = [(recent[i] - recent[i - 1]) / recent[i - 1] for i in range(1, len(recent))]
        if not returns:
            return 0.001
        return float(np.std(returns))

    def _summarize(
        self,
        pairs_data: Dict[str, pd.DataFrame],
        trades: List[Trade],
        equity_curve: List[Tuple[datetime, float]],
        pnl_list: List[float],
        max_consec_losses: int,
    ) -> MultiBacktestResult:
        n_bars = max(len(df) for df in pairs_data.values()) if pairs_data else 0
        n_trades = len(pnl_list)

        if not pnl_list:
            return MultiBacktestResult(
                symbol=", ".join(sorted(pairs_data.keys())),
                strategy_name=self.strategy.name,
                broker_profile=self.simulator.profile_name,
                n_bars=n_bars,
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

        rec_pass, _, _, _ = reconcile(
            trades, equity_curve, tick_size=1e-8
        ) if trades and len(equity_curve) >= 2 else (True, 0.0, 0.0, 0.0)

        return MultiBacktestResult(
            symbol=", ".join(sorted(pairs_data.keys())),
            strategy_name=self.strategy.name,
            broker_profile=self.simulator.profile_name,
            n_bars=n_bars,
            n_trades=n_trades,
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
                "total_commission": total_commission,
            },
        )
