import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class RealityVector:
    """4D decomposed RealityScore — shadow instrumentation (P0.11 Phase 1)."""
    E_exec: float = 0.0
    E_pred: float = 0.0
    E_regime: float = 0.0
    E_contam: float = 0.0
    dfad_exec: float = 0.0


class ExecutionStatistics:
    def __init__(self):
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.slippage_records = []
        self.latency_records = []
        # E5: DFAD — execution delta ground truth
        self.dfad_records = []  # list of (symbol, side, dfad_pts, reality_score)
        self.dfad_by_symbol = defaultdict(list)  # symbol -> list of dfad_pts
        self.dfad_by_side = defaultdict(list)  # "BUY"/"SELL" -> list of dfad_pts
        # P0.24: Reconciliation event tracking for causal analysis
        self._reconciliation_records: list[dict] = []
        # P0.11 Phase 1: Shadow RealityVector + reversal tracking
        self._cf_records: list[dict] = []
        self._reversal_stats: dict[str, dict] = {}
        # Phase 2: Timestamped RealityVector history for drift detection
        self._rv_history: Dict[str, List[Tuple[float, RealityVector]]] = {}

    def record_order_attempt(self):
        self.total_orders += 1

    def record_order_result(self, success: bool, latency_ms: float = 0.0, slippage_pts: float = 0.0):
        if success:
            self.successful_orders += 1
            self.latency_records.append(latency_ms)
            self.slippage_records.append(slippage_pts)
        else:
            self.failed_orders += 1

    def record_dfad(self, symbol: str, side: str, dfad_pts: float, reality_score: float):
        """E5: Record execution delta — fill_price - signal_price in normalized units."""
        self.dfad_records.append((symbol, side, dfad_pts, reality_score))
        self.dfad_by_symbol[symbol].append(dfad_pts)
        self.dfad_by_side[side].append(dfad_pts)

    def record_reconciliation(self, event_type: str, symbol: str, ticket: int, details: dict = None):
        """P0.24: Track orphan position reconciliation events for causal analysis."""
        rec = {
            "event_type": event_type,
            "symbol": symbol,
            "ticket": ticket,
            "details": details or {},
        }
        if not hasattr(self, "_reconciliation_records"):
            self._reconciliation_records = []
        self._reconciliation_records.append(rec)

    def reality_vector(self, symbol: str = "") -> RealityVector:
        """P0.11 Phase 1: Return RealityVector for a symbol.
        Phase 1: shadow computation using available data.
        Returns default values if insufficient data.
        """
        # E_exec from DFAD (asymmetric exponential transform)
        dfad_vals = self.dfad_by_symbol.get(symbol, []) if symbol else []
        if not dfad_vals:
            # Aggregate across all symbols
            for vals in self.dfad_by_symbol.values():
                dfad_vals.extend(vals)
        if dfad_vals:
            dfad_mean = sum(dfad_vals) / len(dfad_vals)
            # Asymmetric exponential: penalize positive DFAD (bad slippage), reward negative (price improvement)
            dfad_pos = max(0, dfad_mean)
            dfad_neg = max(0, -dfad_mean)
            E_exec = math.exp(-dfad_pos) * (1.0 + 0.5 * dfad_neg)
            dfad_exec = dfad_mean
        else:
            E_exec = 0.5
            dfad_exec = 0.0

        # E_pred from reversal events (Jeffreys Beta-Binomial posterior)
        rev = self._reversal_stats.get(symbol, {"total": 0, "correct": 0})
        if rev["total"] > 0:
            E_pred = (rev["correct"] + 0.5) / (rev["total"] + 1.0)
        else:
            E_pred = 0.5

        # E_regime placeholder (Phase 1)
        E_regime = 0.5

        # E_contam Phase 2 activation: structural mismatch between execution and prediction
        E_contam = abs(E_exec - E_pred)

        return RealityVector(
            E_exec=E_exec,
            E_pred=E_pred,
            E_regime=E_regime,
            E_contam=E_contam,
            dfad_exec=dfad_exec,
        )

    def log_reality_vector(self, symbol: str, rv: RealityVector) -> None:
        """Phase 2: Log timestamped RealityVector for drift detection."""
        if symbol not in self._rv_history:
            self._rv_history[symbol] = []
        self._rv_history[symbol].append((time.time(), rv))

    def record_reversal_event(self, symbol: str, success: bool) -> None:
        """P0.11 Phase 1: Record a reversal event outcome."""
        if symbol not in self._reversal_stats:
            self._reversal_stats[symbol] = {"total": 0, "correct": 0}
        self._reversal_stats[symbol]["total"] += 1
        if success:
            self._reversal_stats[symbol]["correct"] += 1

    def log_cf_record(self, record: dict) -> None:
        """P0.12 Phase 1: Log a counterfactual evaluation record."""
        self._cf_records.append(record)

    def get_cf_records(self, n: int = 10) -> list[dict]:
        """P0.12 Phase 1: Get recent counterfactual records."""
        return self._cf_records[-n:]

    def get_reversal_stats(self, symbol: str = "") -> dict:
        """P0.11 Phase 1: Get reversal statistics."""
        if symbol:
            return self._reversal_stats.get(symbol, {"total": 0, "correct": 0})
        return dict(self._reversal_stats)

    def get_summary(self) -> dict:
        avg_latency = sum(self.latency_records) / max(len(self.latency_records), 1)
        avg_slippage = sum(self.slippage_records) / max(len(self.slippage_records), 1)
        success_rate = (self.successful_orders / max(self.total_orders, 1)) * 100.0
        avg_dfad = sum(self.dfad_records) / max(len(self.dfad_records), 1) if self.dfad_records else 0.0
        # Per-symbol DFAD stats
        symbol_dfad = {}
        for sym, vals in self.dfad_by_symbol.items():
            symbol_dfad[sym] = {
                "mean": sum(vals) / max(len(vals), 1),
                "count": len(vals),
                "max": max(vals) if vals else 0.0,
                "min": min(vals) if vals else 0.0,
            }
        # Per-side DFAD stats
        side_dfad = {}
        for side, vals in self.dfad_by_side.items():
            side_dfad[side] = {
                "mean": sum(vals) / max(len(vals), 1),
                "count": len(vals),
            }
        return {
            "total": self.total_orders,
            "success": self.successful_orders,
            "failed": self.failed_orders,
            "success_rate_pct": success_rate,
            "avg_latency_ms": avg_latency,
            "avg_slippage_pts": avg_slippage,
            "avg_dfad_pts": avg_dfad,
            "symbol_dfad": symbol_dfad,
            "side_dfad": side_dfad,
            # P0.24
            "reconciliation_events": self._reconciliation_records,
            "reconciliation_count": len(self._reconciliation_records),
            "reversal_stats": dict(self._reversal_stats),
            "reversal_count": sum(s["total"] for s in self._reversal_stats.values()),
            "cf_records_count": len(self._cf_records),
        }


def classify_execution_mode(rv: RealityVector) -> str:
    """
    Phase 2: Classify execution mode based on RealityVector dimensions.
    Returns one of: AGGRESSIVE, MODERATE, DEFENSIVE, PASSIVE.

    E_exec > 0.7 AND E_pred > 0.6 → AGGRESSIVE (full send)
    E_exec > 0.4 AND E_pred > 0.3 → MODERATE (conditional entry)
    E_contam > 0.6                → DEFENSIVE (tight stops, reduced size)
    Otherwise                      → PASSIVE (skip)
    """
    if rv.E_exec > 0.7 and rv.E_pred > 0.6:
        return "AGGRESSIVE"
    if rv.E_exec > 0.4 and rv.E_pred > 0.3:
        return "MODERATE"
    if rv.E_contam > 0.6:
        return "DEFENSIVE"
    return "PASSIVE"


def survivorship_bias_correction(symbol_history: list) -> float:
    """A10: Survivorship bias correction factor.
    Returns factor in [0.5, 1.0] based on active ratio.
    """
    if len(symbol_history) < 10:
        return 1.0
    active_ratio = sum(1 for s in symbol_history if s.get("active", False)) / len(symbol_history)
    return min(1.0, max(0.5, active_ratio))