import csv, os, random, time
from typing import Any


class MT5TickFeed:
    def __init__(self, fallback_csv: str | None = None) -> None:
        self._ticks: list[dict[str, Any]] = []
        self._cursor = 0
        self._fallback_csv = fallback_csv
        self._mode = "synthetic_fallback"

    def load_csv(self, path: str) -> None:
        self._ticks.clear()
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._ticks.append({
                    "timestamp": float(row.get("timestamp", 0)),
                    "symbol": row.get("symbol", "EURUSD"),
                    "bid": float(row.get("bid", 0)),
                    "ask": float(row.get("ask", 0)),
                    "volume": float(row.get("volume", 0)),
                })
        self._mode = "csv"
        print(f"[MT5Feed] loaded {len(self._ticks)} ticks from {path}")

    def load_live_batch(self, symbols: list[str], count: int = 5000) -> None:
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                raise RuntimeError("MT5 init failed")
            self._ticks.clear()
            for sym in symbols:
                ticks = mt5.copy_ticks_from(sym, time.time() - 86400, count, mt5.COPY_TICKS_ALL)
                if ticks is not None and len(ticks) > 0:
                    for t in ticks[:count // len(symbols)]:
                        self._ticks.append({
                            "timestamp": t[0],
                            "symbol": sym,
                            "bid": t[1],
                            "ask": t[2],
                            "volume": t[3],
                        })
            if not self._ticks:
                raise RuntimeError("No live ticks returned")
            self._mode = "live"
            print(f"[MT5Feed] loaded {len(self._ticks)} live ticks for {symbols}")
        except Exception as e:
            print(f"[MT5Feed] live failed ({e}), falling back to synthetic")
            self._generate_fallback(symbols, count)

    def _generate_fallback(self, symbols: list[str], count: int) -> None:
        base_prices = {"EURUSD": 1.10, "AUDUSD": 0.72, "GBPUSD": 1.25,
                       "USDJPY": 110.0, "USDCHF": 0.92, "USDCAD": 1.35, "NZDUSD": 0.68}
        now = time.time()
        self._ticks.clear()
        for i in range(count):
            sym = random.choice(symbols)
            base = base_prices.get(sym, 1.10)
            spread = base * random.uniform(0.0001, 0.0005)
            self._ticks.append({
                "timestamp": now - (count - i) * 0.1,
                "symbol": sym,
                "bid": base - spread / 2 + random.uniform(-0.001, 0.001),
                "ask": base + spread / 2 + random.uniform(-0.001, 0.001),
                "volume": random.randint(1, 100),
            })
        self._mode = "synthetic_fallback"
        print(f"[MT5Feed] generated {len(self._ticks)} synthetic fallback ticks")

    def next_batch(self, batch_size: int = 100) -> list[dict[str, Any]]:
        if self._cursor >= len(self._ticks):
            return []
        batch = self._ticks[self._cursor:self._cursor + batch_size]
        self._cursor += batch_size
        return batch

    def reset(self) -> None:
        self._cursor = 0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def total_ticks(self) -> int:
        return len(self._ticks)
