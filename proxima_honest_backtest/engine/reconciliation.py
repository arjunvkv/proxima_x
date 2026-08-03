from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Iterator, List, Tuple

from proxima_honest_backtest.engine.types import Trade


def reconcile(
    trades: List[Trade], equity_series: List[Tuple[datetime, float]], tick_size: float = 0.00001
) -> Tuple[bool, float, float, float]:
    if len(equity_series) < 2:
        return True, sum(t.pnl for t in trades), 0.0, 0.0

    trade_pnl = sum(t.pnl for t in trades)
    equity_delta = equity_series[-1][1] - equity_series[0][1]
    diff = abs(trade_pnl - equity_delta)

    if diff < tick_size * 10:
        return True, trade_pnl, equity_delta, diff

    return False, trade_pnl, equity_delta, diff


def reconcile_streaming(
    trades: deque[Trade], equity_series: deque[Tuple[datetime, float]]
) -> Iterator[Tuple[datetime, bool, float]]:
    trade_list = list(trades)
    equity_list = list(equity_series)

    for i in range(1, len(equity_list)):
        ts = equity_list[i][0]
        cum_trade_pnl = sum(t.pnl for t in trade_list[:i])
        equity_delta = equity_list[i][1] - equity_list[0][1]
        running_diff = abs(cum_trade_pnl - equity_delta)
        match = running_diff < 0.0001
        yield (ts, match, running_diff)
