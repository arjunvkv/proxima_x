"""MT5 live feed + archive replay. No lookahead — only current bar data."""

class Feed:
    """Unified feed: switches between live MT5 and archive replay."""

    def __init__(self, mode="live", archive_dir=None, pairs=None, mt5_path=None):
        self.mode = mode
        self.archive_dir = archive_dir
        self.pairs = pairs or []
        self.mt5_path = mt5_path
        self._archive_data = {}
        self._cursor = 0

    def connect(self):
        if self.mode == "live":
            import MetaTrader5 as mt5
            init_kw = {"path": self.mt5_path} if self.mt5_path else {}
            if not mt5.initialize(**init_kw):
                raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
            self.mt5 = mt5
            # Activate all symbols in Market Watch so we can receive tick data
            for pair in self.pairs:
                mt5.symbol_select(pair, True)
        return self

    def current_bar(self):
        """Return {pair: {open, high, low, close, time}} for current minute."""
        if self.mode == "live":
            return self._live_bar()
        else:
            return self._archive_bar()

    def _live_bar(self):
        out = {}
        for pair in self.pairs:
            ticks = self.mt5.symbol_info_tick(pair)
            if ticks is None:
                continue
            out[pair] = {
                "bid": ticks.bid,
                "ask": ticks.ask,
                "time": ticks.time,
                "spread": ticks.ask - ticks.bid,
            }
        return out

    def _archive_bar(self):
        if self._cursor >= len(self._archive_data):
            return None
        row = self._archive_data[self._cursor]
        self._cursor += 1
        return row

    def load_archive(self, data_dict):
        """data_dict: {pair: [bar_dict, ...]} with aligned timestamps."""
        self._archive_data = data_dict
        self._cursor = 0

    def copy_m1_history(self, pair, count=100):
        """Fetch count historical M1 bid bars for pair. Returns list of dicts or None.

        Each dict: {time, open, high, low, close} with raw bid prices.
        Rates from MT5 are bid OHLC (open=r[1], high=r[2], low=r[3], close=r[4]).
        """
        if self.mode != "live" or not hasattr(self, "mt5"):
            return None
        rates = self.mt5.copy_rates_from_pos(pair, self.mt5.TIMEFRAME_M1, 0, count)
        if rates is None or len(rates) == 0:
            return None
        result = []
        for r in rates:
            result.append({
                "time": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
            })
        return result

    def close(self):
        if self.mode == "live":
            self.mt5.shutdown()
