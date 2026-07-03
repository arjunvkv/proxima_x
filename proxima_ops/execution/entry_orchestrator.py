import hashlib, logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from proxima_ops.execution.mt5_connector import MT5Connector
from proxima_ops.execution.order_manager import OrderManager
from proxima_ops.risk.market_observability_filter import MarketObservabilityFilter

logger = logging.getLogger("proxima_ops.entry_orchestrator")


class SLIDRegistry:
    """Signal Lifecycle ID registry — guarantees one entry per symbol per cycle."""

    def __init__(self):
        self._used_slids: Set[str] = set()

    def _compute_slid(self, symbol: str, cycle_id: str) -> str:
        raw = f"{symbol}|{cycle_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def is_used(self, symbol: str, cycle_id: str) -> bool:
        slid = self._compute_slid(symbol, cycle_id)
        return slid in self._used_slids

    def register(self, symbol: str, cycle_id: str) -> str:
        slid = self._compute_slid(symbol, cycle_id)
        if slid in self._used_slids:
            logger.warning("[SLID_DUP] %s already registered for %s cycle=%s", slid, symbol, cycle_id)
        self._used_slids.add(slid)
        return slid

    def reset(self):
        self._used_slids.clear()


class VolumeScaler:
    """Deterministic volume scaling engine — computed once per SLID at entry time.
    Supports per-symbol overrides for volatility-normalized sizing."""

    def __init__(self, base_volume: float = 0.01, max_volume: float = 0.05,
                 step: float = 0.01,
                 symbol_config: Optional[Dict[str, Dict[str, float]]] = None):
        self.base_volume = base_volume
        self.max_volume = max_volume
        self.step = step
        self.symbol_config = symbol_config or {}

    def compute(self, confidence: float, regime_stability: float = 1.0,
                previous_pnl_factor: float = 1.0,
                symbol: str = "") -> float:
        base = self.base_volume
        mx = self.max_volume
        if symbol and symbol in self.symbol_config:
            cfg = self.symbol_config[symbol]
            base = cfg.get("base", base)
            mx = cfg.get("max", mx)
        scaling = confidence * regime_stability * previous_pnl_factor
        effective = base + self.step * int(scaling * 10)
        final_vol = min(max(effective, base), mx)
        logger.info(
            "[VOL_SCALE] sym=%s base=%.2f max=%.2f conf=%.3f reg=%.3f pnl=%.3f → scaled=%.2f final=%.2f",
            symbol, base, mx, confidence, regime_stability, previous_pnl_factor,
            effective, final_vol
        )
        return final_vol


class EntryOrchestrator:
    """Entry Orchestrator (EO) — deterministic entry gate between SSG and OrderManager.

    Flow:
        SSG → EO.validate() → EO.execute() → OrderManager → PositionRegistry
    """

    def __init__(self, connector: MT5Connector, order_manager: OrderManager,
                 max_trades: int = 1,
                 symbol_volume_config: Optional[Dict[str, Dict[str, float]]] = None):
        self.connector = connector
        self.order_manager = order_manager
        self.max_trades = max_trades
        self.slid_registry = SLIDRegistry()
        self.volume_scaler = VolumeScaler(symbol_config=symbol_volume_config)
        self._entry_log: List[dict] = []

    def validate(self, signal: dict, cycle_id: str, positions: list,
                 position_symbols: set, mof_blocks_new_entries: bool = False,
                 mof_last_state: str = "UNKNOWN") -> dict:
        """Validate entry eligibility. Returns {'approved': bool, 'reason': str}."""
        symbol = signal.get("symbol", "?")
        confidence = signal.get("confidence", 0.0)

        # 1. MOF gating
        if mof_blocks_new_entries:
            return {"approved": False, "reason": f"MOF blocks new entries (INFORMATION_DEGRADED)", "symbol": symbol}

        # 2. MOF state check
        if mof_last_state == "INFORMATION_DEGRADED":
            return {"approved": False, "reason": f"MOF={mof_state} below STRUCTURE_LIMITED", "symbol": symbol}

        # 3. Symbol already in positions
        if symbol in position_symbols:
            return {"approved": False, "reason": f"{symbol} already has open position", "symbol": symbol}

        # 4. Max trades check
        open_count = len(positions) if positions else 0
        if open_count >= self.max_trades:
            return {"approved": False, "reason": f"max_trades={self.max_trades} reached ({open_count} open)", "symbol": symbol}

        # 5. SLID deduplication
        if self.slid_registry.is_used(symbol, cycle_id):
            return {"approved": False, "reason": f"SLID already used for {symbol} cycle={cycle_id}", "symbol": symbol}

        # 6. Minimum confidence
        if confidence < 0.1:
            return {"approved": False, "reason": f"confidence={confidence:.3f} < 0.1 minimum", "symbol": symbol}

        return {"approved": True, "reason": "entry validated", "symbol": symbol}

    def execute(self, signal: dict, cycle_id: str, live_mode: bool,
                previous_pnl_factor: float = 1.0) -> dict:
        """Execute a validated entry. Returns execution result."""
        symbol = signal.get("symbol", "?")
        direction = signal.get("direction", 0)
        confidence = signal.get("confidence", 0.0)
        regime_stability = signal.get("regime_stability", 1.0)
        ecdf = signal.get("ecdf", 0.5)

        order_type = "BUY" if direction >= 0 else "SELL"

        # 1. Register SLID
        slid = self.slid_registry.register(symbol, cycle_id)

        # 2. Compute volume (scaling once per SLID, symbol-aware for volatility)
        volume = self.volume_scaler.compute(confidence, regime_stability, previous_pnl_factor, symbol=symbol)

        # 3. Get current price
        tick = None
        try:
            import MetaTrader5 as mt5
            self.connector.ensure_connection()
            mt5.symbol_select(symbol, True)
            tick = mt5.symbol_info_tick(symbol)
        except Exception:
            pass

        if tick is None:
            price = 0.0
        else:
            price = tick.ask if order_type == "BUY" else tick.bid

        # 4. Compute SL/TP at entry time (from OrderManager)
        sl, tp = self.order_manager._resolve_sl_tp(symbol, price, order_type, 0.0, 0.0)

        if live_mode:
            result = self.order_manager.place_order(
                symbol=symbol, order_type=order_type,
                volume=volume, price=price,
                sl=sl, tp=tp,
                comment=f"SEL_V2_{slid}",
            )
            executed = result is not None
            ticket = result.get("ticket") if result else None
        else:
            result = None
            executed = False
            ticket = None

        entry_record = {
            "slid": slid,
            "symbol": symbol,
            "direction": order_type,
            "volume": volume,
            "price": price,
            "sl": sl,
            "tp": tp,
            "confidence": confidence,
            "cycle_id": cycle_id,
            "live": live_mode,
            "executed": executed,
            "ticket": ticket,
        }
        self._entry_log.append(entry_record)

        logger.info(
            "[EO_ENTRY] %s slid=%s %s vol=%.2f price=%s SL=%s TP=%s live=%s executed=%s",
            symbol, slid, order_type, volume, price, sl, tp, live_mode, executed
        )

        return entry_record

    def get_entry_log(self) -> List[dict]:
        return self._entry_log.copy()

    def reset(self):
        self.slid_registry.reset()
        self._entry_log.clear()
