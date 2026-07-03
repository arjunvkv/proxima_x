"""Wave 12 — Regime-Aware Execution Calibration Layer (RAECL).

Provides:
- Market regime classification per cycle
- Edge-class to regime compatibility mapping
- Regime-gated execution filtering
- Regime stability tracking
"""

from __future__ import annotations

import json, os, math
from typing import Any, Dict, Optional
import numpy as np

_MANIFEST_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "deployment_manifest.json")
)

# ---- helpers ----

def _compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.full_like(closes, np.nan)
    avg_loss = np.full_like(closes, np.nan)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0
    return rsi

def _compute_atr(highs, lows, closes, period=14):
    n = min(len(highs), len(lows), len(closes))
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr = np.full(n, np.nan)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def _ema(values: np.ndarray, span: int) -> np.ndarray:
    out = np.full_like(values, np.nan)
    if len(values) == 0: return out
    out[0] = values[0]
    alpha = 2.0 / (span + 1)
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out

REGIME_LABELS = [
    "LOW_VOL_CONSOLIDATION",
    "MEAN_REVERSION_ACTIVE",
    "TREND_EXPANSION",
    "VOLATILITY_SPIKE",
    "HIGH_VOL_RANGE",
]

# Per-regime, per-strategy activation probability (from empirical observation)
REGIME_EDGE_COMPATIBILITY = {
    "LOW_VOL_CONSOLIDATION": {
        "mean_reversion": 0.65,   # moderate — works in narrow ranges
        "pullback": 0.85,         # high — EURJPY pullback works here
        "vol_expansion": 0.10,    # low — needs volatility
    },
    "MEAN_REVERSION_ACTIVE": {
        "mean_reversion": 0.90,
        "pullback": 0.40,
        "vol_expansion": 0.20,
    },
    "TREND_EXPANSION": {
        "mean_reversion": 0.15,
        "pullback": 0.70,
        "vol_expansion": 0.85,
    },
    "VOLATILITY_SPIKE": {
        "mean_reversion": 0.10,
        "pullback": 0.30,
        "vol_expansion": 0.95,
    },
    "HIGH_VOL_RANGE": {
        "mean_reversion": 0.50,
        "pullback": 0.60,
        "vol_expansion": 0.60,
    },
}


class MarketRegimeClassifier:
    """Classifies market regime per cycle using deterministic rules."""

    def __init__(self):
        self._history: list[dict] = []
        self._current_regime: str = "LOW_VOL_CONSOLIDATION"
        self._regime_start_cycle: int = 0

    def classify(
        self,
        closes_by_symbol: dict[str, np.ndarray],
        highs_by_symbol: Optional[dict[str, np.ndarray]] = None,
        lows_by_symbol: Optional[dict[str, np.ndarray]] = None,
        cycle: int = 0,
    ) -> str:
        """Classify current market regime.

        Uses aggregate statistics across all available symbols.
        Returns regime label string.
        """
        atr_ratios = []
        rsi_values = []
        trend_strengths = []

        for sym, closes in closes_by_symbol.items():
            if len(closes) < 30:
                continue
            highs = (highs_by_symbol or {}).get(sym)
            lows = (lows_by_symbol or {}).get(sym)
            if highs is None: highs = closes * 1.002
            if lows is None: lows = closes * 0.998

            # ATR ratio (current vs 50-bar median)
            atr = _compute_atr(highs, lows, closes)
            valid_atr = atr[~np.isnan(atr)]
            if len(valid_atr) > 20:
                current_atr = float(valid_atr[-1])
                baseline_atr = float(np.nanmedian(valid_atr[-50:])) if len(valid_atr) >= 50 else float(np.nanmean(valid_atr))
                if baseline_atr > 0:
                    atr_ratios.append(current_atr / baseline_atr)

            # RSI position
            rsi = _compute_rsi(closes)
            if len(rsi) > 2:
                rsi_values.append(float(rsi[-1]))

            # Trend strength: EMA(20) slope normalized by ATR
            if len(closes) > 25:
                ema20 = _ema(closes, 20)
                if not np.isnan(ema20[-1]) and not np.isnan(ema20[-5]) and len(valid_atr) > 2:
                    slope = (ema20[-1] - ema20[-5]) / max(float(valid_atr[-1]) if not np.isnan(valid_atr[-1]) else 1e-12, 1e-12)
                    trend_strengths.append(abs(slope))

        # Aggregate
        avg_atr_ratio = np.mean(atr_ratios) if atr_ratios else 1.0
        avg_rsi = np.mean(rsi_values) if rsi_values else 50.0
        avg_trend = np.mean(trend_strengths) if trend_strengths else 0.0

        # RSI dispersion (how spread across symbols)
        rsi_dispersion = np.std(rsi_values) if len(rsi_values) >= 2 else 0.0

        # ---- classification logic ----

        # Volatility Spike: ATR > 1.5x baseline
        if avg_atr_ratio > 1.5:
            regime = "VOLATILITY_SPIKE"
        # High Vol Range: ATR > 1.2x baseline
        elif avg_atr_ratio > 1.2:
            regime = "HIGH_VOL_RANGE"
        # Trend Expansion: strong trend + moderate vol
        elif avg_trend > 2.0 and avg_atr_ratio > 0.8:
            regime = "TREND_EXPANSION"
        # Mean Reversion Active: RSI near extremes (below 35 or above 65)
        elif avg_rsi < 35 or avg_rsi > 65:
            regime = "MEAN_REVERSION_ACTIVE"
        else:
            regime = "LOW_VOL_CONSOLIDATION"

        # Track stability
        if regime != self._current_regime:
            self._regime_start_cycle = cycle
            self._current_regime = regime

        entry = {
            "cycle": cycle,
            "regime": regime,
            "features": {
                "avg_atr_ratio": round(float(avg_atr_ratio), 4),
                "avg_rsi": round(float(avg_rsi), 2),
                "avg_trend_strength": round(float(avg_trend), 4),
                "rsi_dispersion": round(float(rsi_dispersion), 4),
            },
        }
        self._history.append(entry)
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        return regime

    @property
    def current_regime(self) -> str:
        return self._current_regime

    @property
    def regime_duration(self) -> int:
        """Cycles since last regime transition."""
        return len(self._history) - self._regime_start_cycle if self._history else 0

    def get_edge_compatibility(self, strategy: str) -> float:
        """Return activation probability for a strategy in current regime (0-1)."""
        regime_map = REGIME_EDGE_COMPATIBILITY.get(self._current_regime, {})
        return regime_map.get(strategy, 0.5)

    def get_active_strategies(self, min_prob: float = 0.3) -> list[str]:
        """Return strategies with compatibility above min_prob in current regime."""
        regime_map = REGIME_EDGE_COMPATIBILITY.get(self._current_regime, {})
        return [s for s, p in regime_map.items() if p >= min_prob]

    def filter_edges_by_regime(self, edges: list[dict], min_compat: float = 0.3) -> list[dict]:
        """Filter edges to only those compatible with current regime."""
        active_strats = self.get_active_strategies(min_compat)
        return [e for e in edges if e.get("strategy", "") in active_strats]

    def get_summary(self) -> dict:
        """Return regime classification summary."""
        if not self._history:
            return {"regime": "UNKNOWN", "duration": 0, "transition_count": 0}

        transitions = sum(
            1 for i in range(1, len(self._history))
            if self._history[i]["regime"] != self._history[i - 1]["regime"]
        )
        return {
            "current_regime": self._current_regime,
            "duration": self.regime_duration,
            "transition_count": transitions,
            "history_length": len(self._history),
        }


class RegimeGatedFilter:
    """Filters edges and signals based on current market regime."""

    def __init__(self, classifier: MarketRegimeClassifier):
        self.classifier = classifier

    def filter_signals(self, all_signals: list[dict], min_compat: float = 0.3) -> list[dict]:
        """Keep only signals from strategies compatible with current regime."""
        active_strats = self.classifier.get_active_strategies(min_compat)
        return [s for s in all_signals if s.get("strategy", "") in active_strats]

    def regime_label(self) -> str:
        return self.classifier.current_regime

    def regime_info(self, signal: dict) -> dict:
        """Return regime compatibility info for a single signal."""
        strategy = signal.get("strategy", "")
        compat = self.classifier.get_edge_compatibility(strategy)
        return {
            "regime": self.classifier.current_regime,
            "strategy": strategy,
            "compatibility": compat,
            "eligible": compat >= 0.3,
        }


# ---------------------------------------------------------------------------
# Compatibility layer — legacy API for governance_pipeline.py and bridge
# ---------------------------------------------------------------------------


class RegimeMetaType:
    """Regime meta-types expected by governance_pipeline and bridge.

    Maps 5 new regime labels to 3 meta-types:
        LOW_VOL_CONSOLIDATION  → STABLE_FLOW
        MEAN_REVERSION_ACTIVE  → STABLE_FLOW
        TREND_EXPANSION        → FAST_TRANSITION
        VOLATILITY_SPIKE       → FAST_TRANSITION
        HIGH_VOL_RANGE         → SLOW_DISSOLUTION
    """

    STABLE_FLOW = "STABLE_FLOW"
    FAST_TRANSITION = "FAST_TRANSITION"
    SLOW_DISSOLUTION = "SLOW_DISSOLUTION"

    _REGIME_TO_META = {
        "LOW_VOL_CONSOLIDATION": STABLE_FLOW,
        "MEAN_REVERSION_ACTIVE": STABLE_FLOW,
        "TREND_EXPANSION": FAST_TRANSITION,
        "VOLATILITY_SPIKE": FAST_TRANSITION,
        "HIGH_VOL_RANGE": SLOW_DISSOLUTION,
    }

    @classmethod
    def from_regime(cls, regime: str) -> str:
        """Map a regime label to a meta-type."""
        return cls._REGIME_TO_META.get(regime, cls.STABLE_FLOW)

    @classmethod
    def meta_regime(cls, meta_type: str) -> str:
        """Reverse map: meta-type → most likely regime label."""
        reverse = {
            cls.STABLE_FLOW: "LOW_VOL_CONSOLIDATION",
            cls.FAST_TRANSITION: "TREND_EXPANSION",
            cls.SLOW_DISSOLUTION: "HIGH_VOL_RANGE",
        }
        return reverse.get(meta_type, "LOW_VOL_CONSOLIDATION")


class GovernorParameterMapper:
    """Maps regime meta-types to governor tuning parameters."""

    _PARAMS: dict = {
        RegimeMetaType.STABLE_FLOW: {
            "temporal_persistence_window": 3,
            "reversal_spike_window": 5,
            "reversal_drop_threshold": 0.20,
            "price_lookback_period": 20,
            "max_amplification": 1.5,
            "recovery_discount": 0.3,
        },
        RegimeMetaType.FAST_TRANSITION: {
            "temporal_persistence_window": 5,
            "reversal_spike_window": 3,
            "reversal_drop_threshold": 0.15,
            "price_lookback_period": 10,
            "max_amplification": 1.3,
            "recovery_discount": 0.5,
        },
        RegimeMetaType.SLOW_DISSOLUTION: {
            "temporal_persistence_window": 4,
            "reversal_spike_window": 4,
            "reversal_drop_threshold": 0.18,
            "price_lookback_period": 15,
            "max_amplification": 1.4,
            "recovery_discount": 0.4,
        },
    }

    @classmethod
    def get_params(cls, meta_type: str) -> dict:
        """Get governor tuning parameters for a meta-type."""
        return cls._PARAMS.get(meta_type, cls._PARAMS[RegimeMetaType.STABLE_FLOW]).copy()

    @classmethod
    def apply_to_governor(cls, governor, adjustments: dict) -> None:
        """Apply parameter adjustments to a governor instance.

        Uses the formal ExecutionGovernor.apply_regime_params() interface.
        Legacy fallback: if the method does not exist, uses direct attribute
        mutation (for backward compatibility with pre-contract governors).
        """
        if not adjustments:
            return
        if hasattr(governor, "apply_regime_params"):
            governor.apply_regime_params(adjustments)
            return
        # Legacy fallback — remove once all governors have apply_regime_params
        pg = getattr(governor, "persistence_gate", None)
        if pg and "temporal_persistence_window" in adjustments:
            win = adjustments["temporal_persistence_window"]
            pg.min_persist_cycles = pg.min_persist_cycles or {}
            for state in pg.min_persist_cycles:
                pg.min_persist_cycles[state] = max(1, win // 2)

        rf = getattr(governor, "reversal_filter", None)
        if rf:
            if "reversal_spike_window" in adjustments:
                rf.spike_window = adjustments["reversal_spike_window"]
            if "reversal_drop_threshold" in adjustments:
                rf.drop_threshold = adjustments["reversal_drop_threshold"]

        pw = getattr(governor, "price_weighting", None)
        if pw:
            if "price_lookback_period" in adjustments:
                pw.lookback_period = adjustments["price_lookback_period"]
            if "max_amplification" in adjustments:
                pw.max_amplification = adjustments["max_amplification"]
            if "recovery_discount" in adjustments:
                pw.recovery_discount = adjustments["recovery_discount"]


class RegimeTimeScaleClassifier:
    """Legacy regime classifier — wraps MarketRegimeClassifier for compatibility.

    Provides the evaluate(rfe_output, price_history) → dict interface
    expected by GovernancePipeline.
    """

    def __init__(self, classifier: Optional[MarketRegimeClassifier] = None):
        self._classifier = classifier or MarketRegimeClassifier()
        self.param_mapper = GovernorParameterMapper
        self._regime_history: dict = {}
        self._governors: Dict[str, Any] = {}
        self._initialized: bool = False

    def initialize_governors(self, symbols: list) -> None:
        """Explicitly create governors for all known symbols.

        Separates governor creation from evaluate(), making evaluate()
        read-only with respect to governor lifecycle.
        """
        from .execution_governor import ExecutionGovernor, TemporalPersistenceGate, PriceContextWeighting, ReversalFilter

        adjustments = self.param_mapper.get_params(RegimeMetaType.STABLE_FLOW)
        for sym in symbols:
            if sym not in self._governors:
                gov = ExecutionGovernor(
                    persistence_gate=TemporalPersistenceGate(min_persist_cycles={
                        "WARNING": adjustments.get("temporal_persistence_window", 3),
                        "EXIT_PREP": adjustments.get("temporal_persistence_window", 3),
                        "EXIT": 1,
                    }),
                    price_weighting=PriceContextWeighting(
                        lookback_period=adjustments.get("price_lookback_period", 20),
                        max_amplification=adjustments.get("max_amplification", 1.5),
                        recovery_discount=adjustments.get("recovery_discount", 0.3),
                    ),
                    reversal_filter=ReversalFilter(
                        spike_window=adjustments.get("reversal_spike_window", 3),
                        drop_threshold=adjustments.get("reversal_drop_threshold", 0.15),
                    ),
                )
                self._governors[sym] = gov
        self._initialized = True

    def ensure_governor(self, sym: str) -> bool:
        """Lazy governor creation for a single symbol.

        Falls back to initialization if not yet created.
        Returns True if governor was created, False if already existed.
        """
        if sym not in self._governors:
            self.initialize_governors([sym])
            return True
        return False

    def get_governor(self, sym: str):
        """Public accessor for per-symbol governor.

        Returns None if no governor exists for this symbol.
        Use ensure_governor() first to create one if needed.
        """
        return self._governors.get(sym)

    def evaluate(
        self,
        rfe_output: dict,
        price_history: Optional[dict] = None,
    ) -> dict:
        """Evaluate regime classifications for all symbols in rfe_output.

        Returns dict with decisions and regime_classifications matching
        the legacy API expected by GovernancePipeline.
        """
        from .execution_governor import ExecutionGovernor

        evaluations = rfe_output.get("evaluations", {})
        decisions: dict = {}
        regime_classifications: dict = {}

        # Build closes dict from price_history
        closes_by_symbol: dict = {}
        if price_history:
            for sym, prices in price_history.items():
                if prices and isinstance(prices[0], dict):
                    closes = [p.get("close", p.get("price", 0.0)) for p in prices if isinstance(p, dict)]
                else:
                    closes = prices
                arr = np.array(closes, dtype=float)
                if len(arr) > 0:
                    closes_by_symbol[sym] = arr

        # Build synthetic price history if none provided
        if not closes_by_symbol and evaluations:
            for sym, ev in evaluations.items():
                price = ev.get("current_price", 1.0)
                base = ev.get("entry_price", price)
                closes_by_symbol[sym] = np.array([base * 0.99, base, price])

        # Classify regime once
        cycle = len(getattr(self._classifier, '_history', []))
        regime = self._classifier.classify(closes_by_symbol, cycle=cycle)
        meta_type = RegimeMetaType.from_regime(regime)
        adjustments = self.param_mapper.get_params(meta_type)

        # Build per-symbol decisions — use existing governors only
        # (create missing ones via ensure_governor for backward compatibility)
        for sym, ev in evaluations.items():
            rfe_pressure = ev.get("score", 0.0)
            rfe_state = ev.get("state", "INFO")

            self.ensure_governor(sym)

            decisions[sym] = {
                "governor_state": "HOLD",
                "governor_cycles": 0,
                "rfe_state": rfe_state,
                "rfe_pressure": rfe_pressure,
                "action": {
                    "type": "NONE",
                    "fraction": 0.0,
                    "reason": f"regime={regime} meta={meta_type}",
                },
            }

            regime_classifications[sym] = {
                "meta_type": meta_type,
                "confidence": 0.6,
                "source": "regime_classifier",
                "override_active": False,
                "classifier_meta": meta_type,
                "cluster": "N/A",
                "lead_estimate": 0,
                "adjustments": adjustments,
            }

        return {
            "decisions": decisions,
            "regime_classifications": regime_classifications,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }

    def reset(self) -> None:
        """Clear internal state."""
        self._classifier = MarketRegimeClassifier()
        self._regime_history.clear()
        self._governors.clear()
