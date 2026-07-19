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

    def close(self):
        if self.mode == "live":
            self.mt5.shutdown()
