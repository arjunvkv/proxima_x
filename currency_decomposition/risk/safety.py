import time
from config.settings import (
    MAX_POSITIONS, MAX_EXPOSURE_PER_CURRENCY, MAX_CURRENCY_POSITIONS,
    STOP_LOSS_PIPS, TAKE_PROFIT_PIPS, MAX_HOLD_HOURS,
    MAX_DAILY_LOSS, MAX_DRAWDOWN, INITIAL_CAPITAL, LOT_SIZE,
    BASE_CURRENCY_MAP, PROFIT_TARGET, STOP_LOSS_AMOUNT, PROFIT_COOLDOWN
)
from data.models import DirectionHypothesis, PaperPosition, TradeRecord, HealthStatus

CLOSE = "CLOSE"
HOLD = "HOLD"

class RiskEngine:
    def __init__(self):
        self.positions: list[PaperPosition] = []
        self.trades: list[TradeRecord] = []
        self.health = HealthStatus()
        self._daily_pnl = 0.0
        self._peak_capital = INITIAL_CAPITAL
        self._halted = False
        self._halt_reason = ""
        self._profit_target_triggered = False
        self._profit_cooldown_until = 0.0
        self.profit_target_hit_time = 0.0
        self._stop_loss_triggered = False
        self._stop_loss_cooldown_until = 0.0
        self.stop_loss_hit_time = 0.0
    


    def set_positions(self, positions: list[PaperPosition]) -> None:
        self.positions = positions
    
    def approve(self, hypothesis: DirectionHypothesis) -> bool:
        if self._halted:
            return False
        if len(self.positions) >= MAX_POSITIONS:
            return False
        if not self._check_currency_exposure(hypothesis):
            return False
        if self._daily_pnl <= -MAX_DAILY_LOSS:
            self._halt("MAX_DAILY_LOSS")
            return False
        return True
    
    def size(self, hypothesis: DirectionHypothesis) -> float:
        return LOT_SIZE
    
    def _check_currency_exposure(self, hypothesis: DirectionHypothesis) -> bool:
        base, quote = self._get_currencies(hypothesis.symbol)
        base_count = sum(1 for p in self.positions if self._get_currencies(p.symbol)[0] == base)
        quote_count = sum(1 for p in self.positions if self._get_currencies(p.symbol)[1] == quote)
        if base_count >= MAX_CURRENCY_POSITIONS or quote_count >= MAX_CURRENCY_POSITIONS:
            return False
        return True
    
    def check_stops(self, prices: dict[str, float]) -> list[PaperPosition]:
        to_close = []
        for pos in self.positions:
            price = prices.get(pos.symbol)
            if price is None:
                continue
            if pos.direction == "BUY":
                if price <= pos.stop_loss:
                    to_close.append(pos)
                elif price >= pos.take_profit:
                    to_close.append(pos)
            elif pos.direction == "SELL":
                if price >= pos.stop_loss:
                    to_close.append(pos)
                elif price <= pos.take_profit:
                    to_close.append(pos)
            
            age_hours = (time.time() - pos.entry_time) / 3600
            if age_hours >= MAX_HOLD_HOURS:
                to_close.append(pos)
        
        return to_close
    
    def update_pnl(self, pnl_change: float) -> None:
        self._daily_pnl += pnl_change
        current_capital = INITIAL_CAPITAL + sum(t.pnl or 0 for t in self.trades)
        if current_capital > self._peak_capital:
            self._peak_capital = current_capital
        drawdown = self._peak_capital - current_capital
        if drawdown >= MAX_DRAWDOWN:
            self._halt("MAX_DRAWDOWN")
    
    def reset_daily(self) -> None:
        self._daily_pnl = 0.0

    def reset_state(self) -> None:
        self._daily_pnl = 0.0
        self._profit_target_triggered = False
        self._profit_cooldown_until = 0.0
        self.profit_target_hit_time = 0.0
        self._stop_loss_triggered = False
        self._stop_loss_cooldown_until = 0.0
        self.stop_loss_hit_time = 0.0
        self._stop_loss_triggered = False
        self._stop_loss_cooldown_until = 0.0
        self.stop_loss_hit_time = 0.0
        self._halted = False
        self._halt_reason = ""
    
    def _halt(self, reason: str) -> None:
        self._halted = True
        self._halt_reason = reason
    
    def is_halted(self) -> bool:
        return self._halted
    
    def halt_reason(self) -> str:
        return self._halt_reason
    
    def check_profit_target(self, positions: list[PaperPosition]) -> bool:
        now = time.time()
        if self._profit_target_triggered:
            if now >= self._profit_cooldown_until and not positions:
                self._profit_target_triggered = False
            return True
        if now < self._profit_cooldown_until:
            return False
        total = sum(p.pnl or 0 for p in positions)
        if total >= PROFIT_TARGET:
            self._profit_target_triggered = True
            self.profit_target_hit_time = now
            self._profit_cooldown_until = now + PROFIT_COOLDOWN
            import sys
            print(f"[PROFIT TARGET] total_pnl={total:.2f} >= {PROFIT_TARGET} — closing all", file=sys.stderr)
            return True
        return False

    def cooldown_active(self) -> bool:
        return time.time() < self._profit_cooldown_until

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._stop_loss_triggered = getattr(self, '_stop_loss_triggered', False)
        self._stop_loss_cooldown_until = getattr(self, '_stop_loss_cooldown_until', 0.0)
        self.stop_loss_hit_time = getattr(self, 'stop_loss_hit_time', 0.0)

    def check_stop_loss(self, positions: list[PaperPosition]) -> bool:
        now = time.time()
        if self._stop_loss_triggered:
            if now >= self._stop_loss_cooldown_until and not positions:
                self._stop_loss_triggered = False
            return True
        if now < self._stop_loss_cooldown_until:
            return False
        total = sum(p.pnl or 0 for p in positions)
        if total <= STOP_LOSS_AMOUNT:
            self._stop_loss_triggered = True
            self.stop_loss_hit_time = now
            self._stop_loss_cooldown_until = now + PROFIT_COOLDOWN
            import sys
            print(f"[STOP LOSS] total_pnl={total:.2f} <= {STOP_LOSS_AMOUNT} — closing all", file=sys.stderr)
            return True
        return False

    def reset_profit_target(self) -> None:
        self._profit_target_triggered = False
        self._profit_cooldown_until = time.time() + PROFIT_COOLDOWN

    def _get_currencies(self, symbol: str):
        if len(symbol) == 6:
            return symbol[:3], symbol[3:]
        return symbol[:3], symbol[3:6]

