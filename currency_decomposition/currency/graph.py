import enum
import numpy as np
from collections import defaultdict, deque
from typing import Optional
from config.settings import CURRENCY_LIST, BASE_CURRENCY_MAP, WLS_REGULARIZATION, MIN_SOLVE_PAIRS, MIN_GRAPH_CONNECTIVITY, DIRECTION_PERSISTENCE_CYCLES, WLS_SMOOTH_ALPHA
from .wls_solver import WLSSolver
from .observability import CurrencyObservability
from .topology import GraphTopology
from data.models import CurrencyState


class GraphState(enum.Enum):
    BOOTSTRAP = 1
    PARTIAL = 2
    READY = 3


class CurrencyGraph:
    def __init__(self):
        self.solver = WLSSolver()
        self.observability = CurrencyObservability()
        self.topology = GraphTopology()
        self.state = CurrencyState()
        self._init_state()
        self._residuals_history = deque(maxlen=100)
        self._strength_history = deque(maxlen=100)
        self._solve_times = deque(maxlen=20)
        self._active_pair_count = 0
        self._spread_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self._strength_prev_sign: dict[str, int] = {}
        self._strength_streak: dict[str, int] = {}
        self._strength_peak: dict[str, float] = {}
        self._strength_trough: dict[str, float] = {}
        self._smoothed_strengths: dict[str, float] = {}

    def _init_state(self):
        self.state.strengths = {c: 0.0 for c in CURRENCY_LIST}
        self.state.prior = {c: 0.0 for c in CURRENCY_LIST}
        self.state.covariance = np.eye(len(CURRENCY_LIST)).tolist()
        self.state.quality = 0.5

    def update(self, returns: dict[str, float], weights: Optional[dict[str, float]] = None,
               timestamp: float = 0.0, available_count: int = None) -> None:
        weights = weights or {}
        active = sum(1 for v in returns.values() if v != 0.0)
        self._active_pair_count = active
        self._known_pairs = max(available_count if available_count else len(returns), 1)
        self.state.coverage = min(1.0, active / self._known_pairs)
        if active < MIN_SOLVE_PAIRS:
            self._update_quality_no_solve(active)
            return
        try:
            strengths = self.solver.solve(returns, weights, self.state.prior)
            residuals = self.solver.compute_residuals(returns, strengths)
            self.state.strengths = strengths
            self.state.prior = strengths
            self.state.last_solve_timestamp = timestamp
            for ccy in CURRENCY_LIST:
                val = strengths.get(ccy, 0.0)
                prev = self._smoothed_strengths.get(ccy, val)
                self._smoothed_strengths[ccy] = WLS_SMOOTH_ALPHA * val + (1 - WLS_SMOOTH_ALPHA) * prev
            self._update_strength_persistence(strengths)
            self.state.solve_count += 1
            active_symbols = [s for s, v in returns.items() if v != 0.0]
            self.state.observability = self.observability.calculate(active_symbols)
            self._strength_history.append(strengths.copy())
            self._residuals_history.append(residuals)
            for sym, (base, quote) in BASE_CURRENCY_MAP.items():
                sp = strengths.get(base, 0.0) - strengths.get(quote, 0.0)
                self._spread_history[sym].append(sp)
            self._update_quality(returns, residuals)
        except np.linalg.LinAlgError:
            self.state.quality *= 0.95

    def _update_quality_no_solve(self, active: int) -> None:
        graph_state = self._compute_graph_state()
        if graph_state == GraphState.BOOTSTRAP:
            decay = 1.0
        elif graph_state == GraphState.PARTIAL:
            decay = 0.98
        else:
            decay = 0.95
        self.state.quality *= decay

    def _update_quality(self, returns: dict, residuals: dict) -> None:
        active_pairs = {k for k, v in returns.items() if v != 0.0}
        r_vals = [returns[k] for k in active_pairs]
        res_vals = [residuals.get(k, 0.0) for k in active_pairs]
        residual_norm = float(np.mean(np.abs(res_vals)))
        return_norm = float(np.mean(np.abs(r_vals))) + 1e-10
        fit_quality = 1.0 - min(residual_norm / return_norm, 1.0)
        total_known = getattr(self, '_known_pairs', max(len(BASE_CURRENCY_MAP), 1))
        active_ratio = len(active_pairs) / total_known
        new_quality = fit_quality * active_ratio
        self.state.quality = 0.7 * self.state.quality + 0.3 * new_quality

    def _update_strength_persistence(self, strengths: dict[str, float]) -> None:
        for ccy, val in strengths.items():
            sign = 1 if val > 0 else (-1 if val < 0 else 0)
            prev = self._strength_prev_sign.get(ccy, 0)
            if sign == 0:
                continue
            if sign == prev:
                self._strength_streak[ccy] = self._strength_streak.get(ccy, 0) + 1
                if val > self._strength_peak.get(ccy, float("-inf")):
                    self._strength_peak[ccy] = val
                if val < self._strength_trough.get(ccy, float("inf")):
                    self._strength_trough[ccy] = val
            else:
                self._strength_prev_sign[ccy] = sign
                self._strength_streak[ccy] = 1
                self._strength_peak[ccy] = val
                self._strength_trough[ccy] = val

    def get_strength_persistence(self) -> dict[str, dict]:
        return {
            c: {
                "direction": self._strength_prev_sign.get(c, 0),
                "streak": self._strength_streak.get(c, 0),
                "peak": self._strength_peak.get(c, 0.0),
                "trough": self._strength_trough.get(c, 0.0),
            }
            for c in CURRENCY_LIST
        }

    def _compute_graph_state(self) -> GraphState:
        if self._active_pair_count < MIN_SOLVE_PAIRS:
            return GraphState.BOOTSTRAP
        if self._active_pair_count < 20:
            return GraphState.PARTIAL
        return GraphState.READY

    def graph_state(self) -> GraphState:
        return self._compute_graph_state()

    def execution_allowed(self, returns: dict[str, float] = None) -> bool:
        base = self._active_pair_count >= MIN_SOLVE_PAIRS and self.state.quality >= 0.3
        if not base:
            return False
        if returns is not None:
            conn = self.connectivity_score(returns)
            if conn < MIN_GRAPH_CONNECTIVITY:
                return False
        return True

    def spread_is_persistent(self, symbol: str, n: int = None) -> bool:
        if n is None:
            n = DIRECTION_PERSISTENCE_CYCLES
        hist = list(self._spread_history.get(symbol, []))
        if len(hist) < n:
            return False
        signs = [1 if x > 0 else -1 for x in hist[-n:]]
        return len(set(signs)) == 1

    def connectivity_score(self, returns: dict[str, float]) -> float:
        active = [s for s, v in returns.items() if v != 0]
        if not active:
            return 0.0
        degrees: dict[str, int] = {c: 0 for c in CURRENCY_LIST}
        for symbol in active:
            base = symbol[:3]
            quote = symbol[3:6] if len(symbol) >= 6 else ""
            degrees[base] = degrees.get(base, 0) + 1
            if quote:
                degrees[quote] = degrees.get(quote, 0) + 1
        degree_score = sum(min(v, 4) for v in degrees.values()) / (len(CURRENCY_LIST) * 4)
        coverage_score = len(active) / max(len(BASE_CURRENCY_MAP), 1)
        return degree_score * 0.5 + coverage_score * 0.5

    def strength_zscore(self) -> dict[str, float]:
        values = list(self.state.strengths.values())
        if not values:
            return {c: 0.0 for c in CURRENCY_LIST}
        mean = float(np.mean(values))
        std = float(np.std(values)) + 1e-10
        return {c: (self.state.strengths[c] - mean) / std for c in CURRENCY_LIST}

    def strength_stability(self) -> dict[str, float]:
        if len(self._strength_history) < 10:
            return {}
        result = {}
        for c in CURRENCY_LIST:
            values = [x[c] for x in self._strength_history]
            result[c] = 1.0 / (1.0 + float(np.std(values)))
        return result

    def missing_symbols(self, returns: dict[str, float] = None) -> list[str]:
        from config.settings import SYMBOLS
        if returns is None:
            return list(SYMBOLS)
        active = {s for s, v in returns.items() if v != 0.0}
        return [s for s in SYMBOLS if s not in active]

    def missing_currency_impact(self, returns: dict[str, float] = None) -> dict[str, int]:
        missing = self.missing_symbols(returns)
        impact = {c: 0 for c in CURRENCY_LIST}
        for s in missing:
            base = s[:3]
            quote = s[3:6]
            if base in impact:
                impact[base] += 1
            if quote in impact:
                impact[quote] += 1
        return impact

    def health_report(self, returns: dict[str, float] = None) -> dict:
        active = self._active_pair_count
        total = len(BASE_CURRENCY_MAP)
        missing = self.missing_symbols()
        conn = self.connectivity_score(returns) if returns else 0.0
        stability = self.strength_stability()
        obs = self.state.observability

        worst_currency = min(CURRENCY_LIST, key=lambda c: obs.get(c, 0))
        best_currency = max(CURRENCY_LIST, key=lambda c: obs.get(c, 0))

        avg_stability = float(np.mean(list(stability.values()))) if stability else 0.0

        if conn >= 0.7 and avg_stability >= 0.5:
            confidence_level = "HIGH"
        elif conn >= 0.45 and avg_stability >= 0.3:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        return {
            "active_pairs": active,
            "total_pairs": total,
            "missing_pairs": len(missing),
            "connectivity": round(conn, 3),
            "worst_currency": worst_currency,
            "best_currency": best_currency,
            "confidence_level": confidence_level,
            "avg_stability": round(avg_stability, 3),
        }

    def currency_stress_test(self, returns: dict[str, float]) -> dict:
        results = {}
        for removed in CURRENCY_LIST:
            filtered = {s: v for s, v in returns.items() if removed not in s}
            if sum(1 for v in filtered.values() if v != 0) < MIN_SOLVE_PAIRS:
                results[removed] = None
                continue
            try:
                strengths = self.solver.solve(filtered, {}, {})
                results[removed] = strengths
            except Exception:
                results[removed] = None
        return results

    def strength(self, currency: str, raw: bool = False) -> float:
        if raw:
            return self.state.strengths.get(currency, 0.0)
        return self._smoothed_strengths.get(currency, self.state.strengths.get(currency, 0.0))

    def strengths_raw(self) -> dict[str, float]:
        return dict(self.state.strengths)

    def strengths(self) -> dict[str, float]:
        if self._smoothed_strengths:
            return dict(self._smoothed_strengths)
        return dict(self.state.strengths)

    def residual(self, symbol: str) -> float:
        if self._residuals_history:
            return self._residuals_history[-1].get(symbol, 0.0)
        return 0.0

    def quality(self) -> float:
        return self.state.quality

    def coverage(self) -> float:
        return self.state.coverage

    def reset(self) -> None:
        self._init_state()
        self._smoothed_strengths.clear()
        self._strength_history.clear()
        self._residuals_history.clear()
        self._strength_prev_sign.clear()
        self._strength_streak.clear()
        self._strength_peak.clear()
        self._strength_trough.clear()
        self._spread_history.clear()
        self.state.solve_count = 0

    def get_state_snapshot(self) -> CurrencyState:
        return self.state

    def restore_state(self, state: CurrencyState) -> None:
        self.state = state

