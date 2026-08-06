"""Warmup bootstrap — build byte-identical causal history BEFORE the first decision.

The backtest builds `history[p]` as closes of bars strictly before ts from a
static series. Live must reconstruct that exact history from the SAME broker's
M5 candles (FTMO) before LiveRunner starts, else the first live decision differs
from the backtest's first decision on identical data.

Invariant (apples-to-apples, gate HALT if broken):
    bootstrap_hash(backtest_history) == bootstrap_hash(live_history)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from proxima_honest_backtest.live.market_state import MarketStateBuilder

WARMUP_BARS = 100  # >= Tokyo max lookback(6)+confirm(3)+2 with margin


def copy_m5_history(mt5: Any, symbol: str, n_bars: int = WARMUP_BARS) -> List[Dict[str, Any]]:
    """Pull the last n_bars M5 candles from the broker (chronological)."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_bars)
    if rates is None or len(rates) == 0:
        return []
    out = []
    for r in rates:
        out.append({
            "time": datetime.fromtimestamp(int(r["time"]), tz=timezone.utc),
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]),
        })
    return out


def history_hash(history: Dict[str, List[float]]) -> str:
    """Deterministic hash over per-pair close series (order-preserving)."""
    h = hashlib.sha256()
    for pair in sorted(history.keys()):
        h.update(pair.encode("utf-8"))
        h.update(b"\x00")
        for c in history[pair]:
            h.update(f"{c:.8f}".encode("utf-8"))
            h.update(b"\x00")
    return h.hexdigest()


class BootstrapHistory:
    """Builds MarketStateBuilder history from broker M5 closes; hashable."""

    def __init__(self, pairs: List[str], warmup_bars: int = WARMUP_BARS) -> None:
        self.pairs = pairs
        self.warmup_bars = warmup_bars
        self.state = MarketStateBuilder()
        self._warmup_complete = False

    def bootstrap(self, mt5: Any) -> Dict[str, Any]:
        """Load last warmup_bars closes per pair into state; no decisions."""
        got: Dict[str, int] = {}
        for p in self.pairs:
            bars = copy_m5_history(mt5, p, self.warmup_bars)
            for b in bars:
                close = b["close"]
                if close is None or close != close:
                    continue
                # inject directly as pre-history (before any decision bar)
                self.state._closes[p].append(float(close))
            got[p] = len(bars)
        self._warmup_complete = all(got[p] >= 2 for p in self.pairs)
        return {"bars_per_pair": got, "complete": self._warmup_complete}

    def seed_from_data(self, data: Dict[str, Any]) -> Dict[str, int]:
        """Seed live history from the SAME parquet the backtest consumed.

        This is the strongest apples-to-apples form of the warmup: both backtest
        and live start from byte-identical closes, immune to pull-timing drift
        between two separate MT5 calls. run_proxima_live --hash should be given
        the hash computed over this same data via expected_hash_from_data().
        """
        for p, df in data.items():
            for close in df["close"].tolist():
                if close is None or close != close:
                    continue
                self.state._closes[p].append(float(close))
        self._warmup_complete = all(len(self.state._closes[p]) >= 2 for p in self.pairs)
        return {"seeded": {p: len(self.state._closes[p]) for p in self.pairs},
                "complete": self._warmup_complete}

    @property
    def warmup_complete(self) -> bool:
        return self._warmup_complete

    @property
    def warmup_complete(self) -> bool:
        return self._warmup_complete

    def history(self) -> Dict[str, List[float]]:
        return self.state.history()

    def closes_hash(self) -> str:
        return history_hash(self.state.closes)


def warmup_hash_json(history: Dict[str, List[float]]) -> str:
    """JSON-stable representation for cross-run comparison."""
    return json.dumps(
        {p: [round(c, 8) for c in closes] for p, closes in sorted(history.items())},
        sort_keys=True,
    )


def history_from_data(data: Dict[str, Any], warmup_bars: Optional[int] = None) -> Dict[str, List[float]]:
    """Reconstruct the causal history the backtest would see, from a parquet
    data dict (the same M5 series the engine consumes).

    Mirrors MultiPairBacktestEngine.run(): `_closes[p]` accumulates every bar's
    close after it is processed. The bootstrap seeds `_closes` with the full
    exported closes so that at the first live decision bar, `history[p]`
    (closed[-1]) contains exactly the closes strictly before it — identical to
    what the backtest engine had at that bar.
    """
    out: Dict[str, List[float]] = {}
    for pair, df in data.items():
        closes = df["close"].tolist()
        if warmup_bars is not None:
            closes = closes[-warmup_bars:]
        out[pair] = [float(c) for c in closes if c == c]
    return out


def expected_hash_from_data(data: Dict[str, Any], warmup_bars: Optional[int] = None) -> str:
    """The hash run_proxima_live --hash must equal for the gate to pass."""
    return history_hash(history_from_data(data, warmup_bars))