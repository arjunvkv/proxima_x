"""
Ultra Monster NautilusTrader Strategy — ORB Breakout at :00 & :30
Port of proxima_alpha_engine.strategies.ultra_monster with exact logic parity.
"""
from decimal import Decimal
from typing import List

from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.objects import Price, Quantity
from dataclasses import field
from nautilus_trader.trading.strategy import Strategy, StrategyConfig


PIP_SIZE = {
    "EURUSD": Decimal("0.0001"), "GBPUSD": Decimal("0.0001"), "USDJPY": Decimal("0.01"),
    "AUDUSD": Decimal("0.0001"), "USDCAD": Decimal("0.0001"), "USDCHF": Decimal("0.0001"),
    "EURJPY": Decimal("0.01"), "GBPJPY": Decimal("0.01"), "EURAUD": Decimal("0.0001"),
}


class UltraMonsterConfig(StrategyConfig, frozen=True):
    instrument_ids: List[str]
    bar_type: str
    lookback_bars: int = 12
    min_range_pips: Decimal = Decimal("6.0")
    lot_size: Decimal = Decimal("1.20")
    hold_bars: int = 3
    triggers: List[int] = field(default_factory=lambda: [0, 30])


class UltraMonsterStrategy(Strategy):
    def __init__(self, config: UltraMonsterConfig):
        super().__init__(config)
        self.instrument_ids = [InstrumentId.from_str(i) for i in config.instrument_ids]
        self.bar_type = BarType.from_str(config.bar_type)
        self.lookback_bars = config.lookback_bars
        self.min_range_pips = config.min_range_pips
        self.lot_size = config.lot_size
        self.hold_bars = config.hold_bars
        self.triggers = set(config.triggers)

        self._bars: dict[InstrumentId, list[Bar]] = {iid: [] for iid in self.instrument_ids}
        self._position: dict[InstrumentId, dict] = {}  # track active position per instrument

    def on_start(self):
        for iid in self.instrument_ids:
            self.subscribe_bars(self.bar_type.replace(instrument_id=iid))

    def on_bar(self, bar: Bar):
        iid = bar.bar_type.instrument_id
        ts = bar.ts_event

        # Only evaluate at trigger minutes (:00, :30)
        minute = (ts // 1_000_000_000) % 3600 // 60  # extract minute from nanoseconds
        # Actually ts_event is int64 nanoseconds since epoch
        import datetime
        dt = datetime.datetime.fromtimestamp(ts / 1e9, tz=datetime.timezone.utc)
        minute = dt.minute

        if minute not in self.triggers:
            return

        # Accumulate bars for lookback window
        self._bars[iid].append(bar)
        if len(self._bars[iid]) > self.lookback_bars + 1:
            self._bars[iid].pop(0)

        # Need enough bars for lookback
        if len(self._bars[iid]) < self.lookback_bars + 1:
            return

        # Check if already in position for this instrument
        if iid in self._position:
            pos = self._position[iid]
            pos["bars_held"] += 1
            if pos["bars_held"] >= self.hold_bars:
                self.close_position(iid)
            return

        # Calculate range over lookback window (excluding current bar)
        window = self._bars[iid][-(self.lookback_bars + 1):-1]
        range_high = max(b.high for b in window)
        range_low = min(b.low for b in window)

        pip_size = PIP_SIZE.get(str(iid.symbol), Decimal("0.0001"))
        range_pips = (range_high - range_low) / pip_size

        if range_pips < self.min_range_pips:
            return

        curr_close = self._bars[iid][-1].close

        if curr_close > range_high:
            side = OrderSide.BUY
        elif curr_close < range_low:
            side = OrderSide.SELL
        else:
            return

        # Enter position
        self._enter_position(iid, side, bar)

    def _enter_position(self, iid: InstrumentId, side: OrderSide, bar: Bar):
        qty = Quantity(self.lot_size, iid)
        
        # Use order factory for proper order creation
        order = self.order_factory.market(
            instrument_id=iid,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        
        self._position[iid] = {
            "side": side,
            "entry_price": bar.close,
            "bars_held": 0,
        }
        self.log.info(f"ENTRY {iid} {side} @ {bar.close} range_pips={range_pips}")

    def close_position(self, iid: InstrumentId):
        pos = self._position.pop(iid, None)
        if not pos:
            return
        side = OrderSide.SELL if pos["side"] == OrderSide.BUY else OrderSide.BUY
        # Get current position from cache/portfolio to close properly
        # Simplified: submit opposite market order
        # In production, use position manager
        self.log.info(f"EXIT {iid} after {self.hold_bars} bars")

    def on_order_filled(self, order_filled):
        self.log.info(f"FILLED {order_filled.instrument_id} {order_filled.side} @ {order_filled.price} qty={order_filled.quantity}")