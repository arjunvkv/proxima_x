import time
import uuid
import numpy as np
from typing import Optional
from config.settings import LOT_SIZE
from data.models import DirectionHypothesis, PaperPosition, ExecutionResult, Tick

class PaperExecutor:
    def __init__(self):
        self.positions: list[PaperPosition] = []
        self._closed_positions: list[PaperPosition] = []
        self._last_prices: dict[str, float] = {}
    
    def sync(self) -> None:
        pass

    def position_count(self) -> int:
        return len(self.positions)

    def update_prices(self, ticks: list[Tick]) -> None:
        for tick in ticks:
            self._last_prices[tick.symbol] = tick.mid
    
    def execute(self, hypothesis: DirectionHypothesis, tick: Optional[Tick] = None, sl: Optional[float] = None, tp: Optional[float] = None) -> ExecutionResult:
        if tick is not None:
            fill_price = tick.ask if hypothesis.direction > 0 else tick.bid
        else:
            price = self._last_prices.get(hypothesis.symbol, 0.0)
            if price == 0.0:
                return ExecutionResult(success=False, reason="NO_PRICE")
            fill_price = price * 1.0001 if hypothesis.direction > 0 else price * 0.9999
        
        direction = "BUY" if hypothesis.direction > 0 else "SELL"
        spread = 0.0002
        slippage = spread * 0.1
        dec = 3 if "JPY" in hypothesis.symbol else 5
        if direction == "BUY":
            fill_price = fill_price + slippage
            stop = round(sl if sl is not None else fill_price * (1.0 - 0.0030), dec)
            target = round(tp if tp is not None else fill_price * (1.0 + 0.0060), dec)
        else:
            fill_price = fill_price - slippage
            stop = round(sl if sl is not None else fill_price * (1.0 + 0.0030), dec)
            target = round(tp if tp is not None else fill_price * (1.0 - 0.0060), dec)
        fill_price = round(fill_price, dec)
        
        position = PaperPosition(
            id=str(uuid.uuid4())[:8],
            symbol=hypothesis.symbol,
            direction=direction,
            entry_price=fill_price,
            current_price=fill_price,
            entry_time=hypothesis.timestamp or time.time(),
            lots=LOT_SIZE,
            stop_loss=stop,
            take_profit=target,
            drs_entry=hypothesis.drs_score,
            currency_strengths_entry={
                "base": hypothesis.base_strength,
                "quote": hypothesis.quote_strength
            }
        )
        
        self.positions.append(position)
        return ExecutionResult(success=True, position_id=position.id, price=fill_price)
    
    def close_position(self, position_id: str, exit_price: float, reason: str = "") -> ExecutionResult:
        for i, pos in enumerate(self.positions):
            if pos.id == position_id:
                pos.current_price = exit_price
                pos.pnl = self._calculate_pnl(pos, exit_price)
                self._closed_positions.append(pos)
                self.positions.pop(i)
                return ExecutionResult(success=True, position_id=position_id, price=exit_price, reason=reason)
        return ExecutionResult(success=False, reason="POSITION_NOT_FOUND")
    
    def close_all(self, prices: dict[str, float], reason: str = "MANUAL") -> list[ExecutionResult]:
        results = []
        for pos in list(self.positions):
            price = prices.get(pos.symbol, pos.entry_price)
            r = self.close_position(pos.id, price, reason)
            results.append(r)
        return results
    
    def get_position(self, position_id: str) -> Optional[PaperPosition]:
        for pos in self.positions:
            if pos.id == position_id:
                return pos
        return None
    
    def positions_summary(self) -> list[dict]:
        return [
            {
                "id": p.id, "symbol": p.symbol, "direction": p.direction,
                "entry": p.entry_price, "current": p.current_price,
                "pnl": self._calculate_pnl(p, p.current_price),
                "age_s": time.time() - p.entry_time
            }
            for p in self.positions
        ]
    
    def _calculate_pnl(self, position: PaperPosition, exit_price: float) -> float:
        quote = position.symbol[3:6]
        usd_quote_rate = 1.0
        if quote != "USD":
            pair1 = f"USD{quote}"
            if pair1 in self._last_prices:
                usd_quote_rate = self._last_prices[pair1]
            else:
                pair2 = f"{quote}USD"
                if pair2 in self._last_prices:
                    rate = self._last_prices[pair2]
                    usd_quote_rate = 1.0 / rate if rate > 0 else 1.0
                    
        if position.direction == "BUY":
            diff = exit_price - position.entry_price
        else:
            diff = position.entry_price - exit_price
            
        return (diff * position.lots * 100000) / usd_quote_rate

