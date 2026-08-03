"""
CZR — Cross-Pair Z-Score Ranking Strategy (0.1s Pre-Vectorized Engine)

Target Anomaly:
  Cross-sectional relative dislocation across all 18 currency pairs.
  At every M5 bar, calculates 200-bar rolling z-score of 3-bar (15-min) returns.
  Ranks all 18 pairs and selects the most-oversold (most-negative z-score) pair.
  Enters LONG-only when z <= -z_thresh (e.g. -4.0 or -3.0) and holds for N bars.
"""
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

ALL_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP",
    "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]

class CZRStrategy(MultiPairStrategy):
    """Cross-Pair Z-Score Ranking Strategy (CZR) — Fast Pre-Vectorized Engine."""

    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ALL_PAIRS,
        "z_thresh": 4.0,           # Z-score threshold (e.g. 3.0 or 4.0)
        "ret_bars": 3,              # 3 M5 bars = 15-min return
        "window_bars": 200,         # 200 M5 bars = 16.7 hr rolling window
        "hold_bars": 12,            # Hold duration (e.g. 6=30m, 9=45m, 12=60m)
        "long_only": True,          # LONG-only (fade oversold)
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

        self.pairs: List[str] = list(self.parameters["pairs"])
        self.z_thresh: float = float(self.parameters["z_thresh"])
        self.ret_bars: int = int(self.parameters["ret_bars"])
        self.window_bars: int = int(self.parameters["window_bars"])
        self.hold_bars: int = int(self.parameters["hold_bars"])
        self.long_only: bool = bool(self.parameters["long_only"])

        self._positions: Dict[str, Dict[str, Any]] = {}
        self._z_matrix: Optional[Dict[str, List[float]]] = None

    def reset(self) -> None:
        self._positions.clear()
        self._z_matrix = None

    def set_precomputed_data(self, raw_dict: Dict[str, pd.DataFrame]) -> None:
        """Pre-compute Z-scores for all pairs in 0.1 seconds."""
        ret_b = self.ret_bars
        win_b = self.window_bars
        self._z_matrix = {}
        for pair in self.pairs:
            df = raw_dict.get(pair)
            if df is None or df.empty or len(df) < (win_b + ret_b + 1):
                continue
            closes = df["close"].astype(np.float64)
            rets = np.log(closes / closes.shift(ret_b))
            r_mean = rets.shift(1).rolling(win_b).mean()
            r_std = rets.shift(1).rolling(win_b).std(ddof=0)
            z_vals = ((rets - r_mean) / r_std).fillna(0.0).values
            self._z_matrix[pair] = z_vals.tolist()

    def on_bars(
        self,
        bars: Dict[str, Dict],
        history: Dict[str, Any],
    ) -> List[SignalResult]:
        """Generate CZR signals at each M5 bar."""
        signals: List[SignalResult] = []

        ts = next((b["time"] for _, b in bars.items() if b), None)
        if not ts:
            return signals

        # === EXIT CHECKS (every bar) ===
        self._check_exits(ts, bars, signals)

        z_scores: List[Tuple[float, str, float]] = []
        ret_b = self.ret_bars
        win_b = self.window_bars
        need_len = win_b + ret_b + 1

        for pair in self.pairs:
            closes = history.get(pair, [])
            n = len(closes)
            if n < need_len:
                continue

            if self._z_matrix and pair in self._z_matrix and n <= len(self._z_matrix[pair]):
                z = self._z_matrix[pair][n - 1]
            else:
                c_now = closes[-1]
                c_prev = closes[-1 - ret_b]
                if c_now <= 0 or c_prev <= 0:
                    continue
                cur_ret = math.log(c_now / c_prev)
                rets = [math.log(closes[-i] / closes[-i - ret_b]) for i in range(1, win_b + 1) if closes[-i] > 0 and closes[-i - ret_b] > 0]
                if len(rets) < (win_b // 2):
                    continue
                mean = sum(rets) / len(rets)
                std_dev = math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets))
                if std_dev <= 0:
                    continue
                z = (cur_ret - mean) / std_dev

            z_scores.append((float(z), pair, float(closes[-1])))

        if not z_scores:
            return signals

        # Sort pairs by z-score ascending (lowest z first)
        z_scores.sort(key=lambda x: x[0])
        min_z, min_pair, cur_close = z_scores[0]

        # LONG-only entry when lowest z <= -z_thresh and pair not already in position
        if min_z <= -self.z_thresh and min_pair not in self._positions:
            bar = bars.get(min_pair)
            if bar:
                entry_price = float(bar.get("open", cur_close))
                self._positions[min_pair] = {
                    "direction": 1,
                    "entry_price": entry_price,
                    "bars_held": 0,
                    "entry_time": ts,
                }
                signals.append(
                    SignalResult(
                        timestamp=ts,
                        signal=1.0,
                        confidence=min(0.99, abs(min_z) / 6.0),
                        metadata={
                            "strategy": self.name,
                            "pair": min_pair,
                            "action": "ENTER_BUY",
                            "z_score": float(min_z),
                            "entry_price": entry_price,
                        },
                    )
                )

        return signals

    def _check_exits(self, ts: Any, bars: Dict[str, Dict], signals: List[SignalResult]) -> None:
        to_remove = []
        for pair, pos in self._positions.items():
            pos["bars_held"] = pos.get("bars_held", 0) + 1
            if pos["bars_held"] >= self.hold_bars:
                bar = bars.get(pair)
                current_price = float(bar.get("close", pos["entry_price"])) if bar else pos["entry_price"]
                to_remove.append(pair)
                signals.append(
                    SignalResult(
                        timestamp=ts,
                        signal=0.0,
                        confidence=1.0,
                        metadata={
                            "strategy": self.name,
                            "pair": pair,
                            "action": "EXIT",
                            "reason": "max_hold",
                            "exit_price": current_price,
                        },
                    )
                )
        for pair in to_remove:
            del self._positions[pair]
