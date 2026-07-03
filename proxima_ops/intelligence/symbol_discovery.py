import time
import logging
import MetaTrader5 as mt5
from typing import List, Optional

logger = logging.getLogger("proxima_ops.intelligence.symbol_discovery")

EXCLUDED_PATTERNS = [
    "SWAP", "SPOT", "WDO", "WIN", "IND", "IDX",
    "DIX", "UDI", "BRL", "COFFEE", "SUGAR", "WHEAT",
    "CORN", "SOYBEAN", "GAS", "OIL", "NGAS", "CL",
]


class SymbolDiscovery:
    """Fetches all MT5 symbols and filters to a tradeable candidate list."""

    def __init__(self, cache_ttl: int = 300, max_spread: float = 50.0):
        self.cache_ttl = cache_ttl  # seconds between cache refresh
        self.max_spread = max_spread
        self._cache = None
        self._cache_time = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self) -> List[dict]:
        """Return filtered candidate symbols with metadata.

        Steps:
          1. Fetch all symbols via mt5.symbols_get()
          2. Filter by basic tradeability criteria
          3. For survivors, compute spread from tick data
          4. Exclude wide-spread symbols
          5. Return enriched dict list
        """
        now = time.time()

        # ----- cache check -----
        if self._cache is not None and (now - self._cache_time) < self.cache_ttl:
            logger.info(
                "[DISC] cache hit — returning %d symbols", len(self._cache)
            )
            return self._cache

        logger.info("[DISC] cache miss — re-fetching all symbols")

        # ----- fetch & filter -----
        raw = self._fetch_all_symbols()
        candidates = []

        for sym in raw:
            if not self._is_tradeable(sym):
                continue
            spread = self._get_spread_points(sym.name)
            # Let SIL scoring handle quality — high spreads get low SIL scores
            if spread > self.max_spread * 5:  # extreme outlier filter only
                continue

            candidates.append(
                {
                    "symbol": sym.name,
                    "description": sym.description or "",
                    "spread_points": spread,
                    "trade_mode": self._trade_mode_str(sym.trade_mode),
                    "volume_min": sym.volume_min,
                    "volume_max": sym.volume_max,
                    "digits": sym.digits,
                    "point": sym.point,
                    "visible": bool(sym.visible),
                    "currency_base": sym.currency_base or "",
                    "currency_profit": sym.currency_profit or "",
                    "path": sym.path or "",
                }
            )

        # ----- update cache -----
        self._cache = candidates
        self._cache_time = now

        logger.info(
            "[DISC] discovery complete — %d tradeable symbols", len(candidates)
        )
        return candidates

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_all_symbols(self) -> list:
        """Raw MT5 symbols_get()."""
        symbols = mt5.symbols_get()
        if symbols is None:
            logger.error(
                "[DISC] mt5.symbols_get() returned None — MT5 not initialized?"
            )
            return []
        return symbols

    def _is_tradeable(self, symbol) -> bool:
        """Basic filter pass — visibility, trade mode, min volume, exclusions."""
        if not symbol.visible:
            return False
        if symbol.trade_mode == 0:  # SYMBOL_TRADE_MODE_DISABLED
            return False
        if symbol.volume_min <= 0:
            return False
        if self._should_exclude(symbol.name):
            return False
        return True

    def _get_spread_points(self, symbol_name: str) -> float:
        """Fetch tick and compute spread in points.

        Returns float spread in points. If tick data is not available
        (e.g. market closed, symbol idle), returns a high default so the
        symbol is not disqualified outright — the SIL scoring layer will
        rank it appropriately.
        """
        try:
            tick = mt5.symbol_info_tick(symbol_name)
            if tick is None:
                return self.max_spread * 2  # high default, not disqualifying

            info = mt5.symbol_info(symbol_name)
            if info is None or info.point <= 0:
                return self.max_spread * 2

            spread_points = (tick.ask - tick.bid) / info.point
            return max(0.0, spread_points)
        except Exception:
            return self.max_spread * 2

    def _should_exclude(self, symbol_name: str) -> bool:
        """Check symbol name against exclusion patterns (case-insensitive)."""
        upper = symbol_name.upper()
        for pat in EXCLUDED_PATTERNS:
            if pat in upper:
                return True
        return False

    @staticmethod
    def _trade_mode_str(mode: int) -> str:
        """Map numeric trade_mode to human-readable string."""
        mapping = {
            0: "DISABLED",
            1: "ENABLED",
            2: "CLOSE_ONLY",
            3: "LONG_ONLY",
            4: "SHORT_ONLY",
        }
        return mapping.get(mode, f"UNKNOWN({mode})")


if __name__ == "__main__":
    mt5.initialize()
    disc = SymbolDiscovery()
    symbols = disc.discover()
    print(f"Discovered {len(symbols)} tradeable symbols:")
    for s in symbols:
        print(
            f"  {s['symbol']:15s} spread={s['spread_points']:5.1f} path={s.get('path','')}"
        )
    mt5.shutdown()
