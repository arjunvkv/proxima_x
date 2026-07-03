"""Shadow Execution Feedback Loop Validation.

This is the final pre-deployment validation. It tests whether upstream
intelligence (Geometry Forecaster, Regime Classifier, Execution Governor)
survives downstream execution noise (slippage, latency, partial fills).

In real markets:
- Slippage alters regime timing
- Partial fills distort coherence maps
- Execution latency shifts classification boundaries
- Spread spikes mimic PRE_COLLAPSE

This module wraps the GovernancePipeline and injects execution reality
effects into the upstream layers to measure feedback loop stability.
"""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .cluster_geometry_forecaster import PreRegimeType
from .execution_governor import GovernorState
from .governance_pipeline import GovernancePipeline
from .regime_classifier import RegimeMetaType

# ---------------------------------------------------------------------------
# 1. Perturbation Models
# ---------------------------------------------------------------------------


class SlippageInjector:
    """Simulates slippage by adding noise to RFE pressure and cluster states.

    Parameters
    ----------
    slippage_std : float, default=0.03
        Standard deviation of Gaussian noise (2-5% pressure distortion).
    bias : float, default=0.0
        Directional bias (symmetric by default).
    """

    def __init__(self, slippage_std: float = 0.03, bias: float = 0.0) -> None:
        self.slippage_std = slippage_std
        self.bias = bias

    def apply_to_pressure(self, pressure: float) -> float:
        """Add Gaussian noise to pressure, clipped to [0, 1]."""
        noise = random.gauss(0.0, self.slippage_std) + self.bias
        return max(0.0, min(1.0, pressure + noise))

    def apply_to_cluster_state(self, state: dict) -> dict:
        """Perturb coherence and divergence by small amounts."""
        result = dict(state)
        if "coherence" in result:
            noise = random.gauss(0.0, self.slippage_std * 0.5)
            result["coherence"] = max(0.0, min(1.0, result["coherence"] + noise))
        if "divergence" in result:
            noise = random.gauss(0.0, self.slippage_std * 0.5)
            result["divergence"] = max(0.0, min(1.0, result["divergence"] + noise))
        if "net_direction" in result:
            noise = random.gauss(0.0, self.slippage_std * 0.3)
            result["net_direction"] = max(-1.0, min(1.0, result["net_direction"] + noise))
        return result


class LatencyInjector:
    """Simulates execution latency by delaying state updates.

    Parameters
    ----------
    delay_cycles : int, default=2
        Number of cycles of delay (1-3).
    """

    def __init__(self, delay_cycles: int = 2) -> None:
        self.delay_cycles = delay_cycles
        self._buffer: List[dict] = []

    def apply_to_pipeline(self, rfe_output: dict) -> dict:
        """Delay RFE output by ``delay_cycles`` cycles.

        Returns an empty evaluation dict until enough cycles have elapsed,
        then returns the RFE output from ``delay_cycles`` cycles ago.
        """
        self._buffer.append(rfe_output)
        idx = len(self._buffer) - 1 - self.delay_cycles
        if idx < 0:
            # No data available yet — return empty evaluation structure
            return {
                "evaluations": {},
                "summary": {},
                "transitions": {},
                "temporal": {},
                "breaches": [],
                "timestamp": "",
            }
        return self._buffer[idx]

    def reset(self) -> None:
        """Clear internal buffer."""
        self._buffer.clear()


class PartialFillInjector:
    """Simulates partial fills by distorting trade state.

    Parameters
    ----------
    fill_ratio : float, default=0.7
        Fraction of position actually filled (0.5-1.0).
    """

    def __init__(self, fill_ratio: float = 0.7) -> None:
        self.fill_ratio = fill_ratio

    def apply_to_trade(self, trade: dict) -> dict:
        """Adjust trade state to reflect partial fill.

        Reduces score proportionally and adjusts PnL.
        """
        result = dict(trade)
        if "current_pnl" in result:
            result["current_pnl"] = round(result["current_pnl"] * self.fill_ratio, 4)
        # Reduce the fill-affected score to simulate incomplete execution
        if "score" in result:
            result["score"] = round(result["score"] * (0.5 + 0.5 * self.fill_ratio), 4)
        return result


# ---------------------------------------------------------------------------
# 2. Stability Metrics
# ---------------------------------------------------------------------------


class FeedbackStabilityMetrics:
    """Measures how much upstream layers drift under execution perturbation.

    Compares clean (unperturbed) pipeline results against perturbed results
    cycle-by-cycle and per-symbol.
    """

    @staticmethod
    def compare(clean_results: List[dict], perturbed_results: List[dict]) -> dict:
        """Compare clean vs perturbed results across all five metrics.

        Parameters
        ----------
        clean_results : list of dict
            Pipeline results from clean baseline run (one per cycle).
        perturbed_results : list of dict
            Pipeline results from perturbed run (one per cycle).

        Returns
        -------
        dict with keys:
            regime_classification_stability  : float [0,1]
            governor_decision_stability      : float [0,1]
            geometry_detection_stability     : dict
            false_exit_rate_change           : float
            missed_exit_rate_change          : float
        """
        n = min(len(clean_results), len(perturbed_results))
        if n == 0:
            return {
                "regime_classification_stability": 1.0,
                "governor_decision_stability": 1.0,
                "geometry_detection_stability": {
                    "lead_time_delta": 0,
                    "status": "no_data",
                    "clean_first_cycle": None,
                    "perturbed_first_cycle": None,
                },
                "false_exit_rate_change": 0.0,
                "missed_exit_rate_change": 0.0,
            }

        # -- 1. Regime classification stability --
        total_regime = 0
        matched_regime = 0
        for i in range(n):
            clean_regs = clean_results[i].get("regime_classifications", {})
            pert_regs = perturbed_results[i].get("regime_classifications", {})
            symbols = set(list(clean_regs.keys()) + list(pert_regs.keys()))
            for sym in symbols:
                cr = clean_regs.get(sym, {}).get("meta_type", "")
                pr = pert_regs.get(sym, {}).get("meta_type", "")
                if cr and pr:
                    total_regime += 1
                    if cr == pr:
                        matched_regime += 1

        regime_stab = matched_regime / max(1, total_regime)

        # -- 2. Governor decision stability --
        total_gov = 0
        matched_gov = 0
        for i in range(n):
            clean_dec = clean_results[i].get("decisions", {})
            pert_dec = perturbed_results[i].get("decisions", {})
            symbols = set(list(clean_dec.keys()) + list(pert_dec.keys()))
            for sym in symbols:
                cgs = clean_dec.get(sym, {}).get("governor_state", "")
                pgs = pert_dec.get(sym, {}).get("governor_state", "")
                if cgs and pgs:
                    total_gov += 1
                    if cgs == pgs:
                        matched_gov += 1

        gov_stab = matched_gov / max(1, total_gov)

        # -- 3. Geometry detection stability --
        def _first_precollapse(results: List[dict]):
            for i, res in enumerate(results):
                geo = res.get("geometry_forecasts", {})
                for cluster, forecast in geo.items():
                    if isinstance(forecast, dict):
                        if forecast.get("pre_regime") == PreRegimeType.PRE_COLLAPSE:
                            return i
            return None

        clean_first = _first_precollapse(clean_results)
        pert_first = _first_precollapse(perturbed_results)

        if clean_first is not None and pert_first is not None:
            lead_delta = abs(pert_first - clean_first)
        else:
            lead_delta = 0

        if lead_delta < 2:
            geo_status = "excellent"
        elif lead_delta < 5:
            geo_status = "acceptable"
        else:
            geo_status = "warning"

        # -- 4 & 5. Exit rate changes --
        clean_exits = 0
        pert_exits = 0
        false_exits = 0  # exits in perturbed but not clean
        missed_exits = 0  # exits in clean but not perturbed

        for i in range(n):
            clean_dec = clean_results[i].get("decisions", {})
            pert_dec = perturbed_results[i].get("decisions", {})
            symbols = set(list(clean_dec.keys()) + list(pert_dec.keys()))
            for sym in symbols:
                ca = clean_dec.get(sym, {}).get("action", {}).get("type", "NONE")
                pa = pert_dec.get(sym, {}).get("action", {}).get("type", "NONE")
                clean_is_exit = ca in ("CLOSE", "CLOSE_PARTIAL")
                pert_is_exit = pa in ("CLOSE", "CLOSE_PARTIAL")
                if clean_is_exit:
                    clean_exits += 1
                if pert_is_exit:
                    pert_exits += 1
                if pert_is_exit and not clean_is_exit:
                    false_exits += 1
                if clean_is_exit and not pert_is_exit:
                    missed_exits += 1

        false_rate = false_exits / max(1, n)
        missed_rate = missed_exits / max(1, n)

        return {
            "regime_classification_stability": round(regime_stab, 4),
            "governor_decision_stability": round(gov_stab, 4),
            "geometry_detection_stability": {
                "lead_time_delta": lead_delta,
                "status": geo_status,
                "clean_first_cycle": clean_first,
                "perturbed_first_cycle": pert_first,
            },
            "false_exit_rate_change": round(false_rate, 4),
            "missed_exit_rate_change": round(missed_rate, 4),
            "clean_exits": clean_exits,
            "perturbed_exits": pert_exits,
            "false_exits": false_exits,
            "missed_exits": missed_exits,
        }


# ---------------------------------------------------------------------------
# 3. Main Shadow Execution Validator
# ---------------------------------------------------------------------------


class ShadowExecutionLoop:
    """Simulates closed-loop execution feedback to validate system stability.

    Wraps the GovernancePipeline and injects execution reality effects
    into the upstream layers to measure feedback loop stability.

    Three perturbation types:
    1. SLIPPAGE: random price offset on execution (noise injection)
    2. LATENCY: delayed state updates (shifted timing)
    3. PARTIAL_FILL: incomplete execution (state distortion)

    Scenarios
    ---------
    1. CLEAN BASELINE: no perturbations
    2. SLIPPAGE ONLY: random price noise
    3. LATENCY ONLY: 2-cycle delay
    4. PARTIAL FILL ONLY: 70% fill ratio
    5. FULL COMBINED: all perturbations together
    """

    def __init__(self, pipeline: Optional[GovernancePipeline] = None) -> None:
        self.pipeline = pipeline or GovernancePipeline()
        self.slippage = SlippageInjector()
        self.latency = LatencyInjector()
        self.partial_fill = PartialFillInjector()
        self.metrics = FeedbackStabilityMetrics()

    # ------------------------------------------------------------------
    # Scenario runners
    # ------------------------------------------------------------------

    def run_clean_baseline(self, test_data: dict) -> List[dict]:
        """Run pipeline without perturbations -> reference results."""
        return self._run_cycles(
            test_data,
            apply_slippage=False,
            apply_latency=False,
            apply_partial_fill=False,
        )

    def run_slippage_scenario(self, test_data: dict) -> List[dict]:
        """Run with slippage noise injected into cluster states and RFE."""
        return self._run_cycles(
            test_data,
            apply_slippage=True,
            apply_latency=False,
            apply_partial_fill=False,
        )

    def run_latency_scenario(self, test_data: dict) -> List[dict]:
        """Run with 2-cycle execution delay."""
        return self._run_cycles(
            test_data,
            apply_slippage=False,
            apply_latency=True,
            apply_partial_fill=False,
        )

    def run_partial_fill_scenario(self, test_data: dict) -> List[dict]:
        """Run with partial fill simulation."""
        return self._run_cycles(
            test_data,
            apply_slippage=False,
            apply_latency=False,
            apply_partial_fill=True,
        )

    def run_combined_scenario(self, test_data: dict) -> List[dict]:
        """Run with ALL perturbations simultaneously."""
        return self._run_cycles(
            test_data,
            apply_slippage=True,
            apply_latency=True,
            apply_partial_fill=True,
        )

    def validate_all(self, test_data: dict) -> dict:
        """Run all scenarios and produce comprehensive validation report.

        Returns a dict containing per-scenario metrics, layer stability
        analysis, invariance analysis, and deployment readiness verdict.
        """
        # Run all five scenarios
        clean = self.run_clean_baseline(test_data)
        slippage = self.run_slippage_scenario(test_data)
        latency = self.run_latency_scenario(test_data)
        partial = self.run_partial_fill_scenario(test_data)
        combined = self.run_combined_scenario(test_data)

        scenario_data = {
            "CLEAN BASELINE": clean,
            "SLIPPAGE ONLY": slippage,
            "LATENCY ONLY": latency,
            "PARTIAL FILL": partial,
            "FULL COMBINED": combined,
        }

        # Compute metrics per scenario vs clean
        scenario_results: Dict[str, dict] = {}
        for name, perturbed in scenario_data.items():
            if name == "CLEAN BASELINE":
                scenario_results[name] = self.metrics.compare(clean, clean)
            else:
                scenario_results[name] = self.metrics.compare(clean, perturbed)

        # Layer stability: worst-case delta across all scenarios
        geo_deltas: List[float] = []
        regime_deltas: List[float] = []
        gov_deltas: List[float] = []
        exit_deltas: List[float] = []

        for name, m in scenario_results.items():
            if name == "CLEAN BASELINE":
                continue
            gd = m.get("geometry_detection_stability", {})
            geo_deltas.append(float(gd.get("lead_time_delta", 0)))
            regime_deltas.append(1.0 - m.get("regime_classification_stability", 1.0))
            gov_deltas.append(1.0 - m.get("governor_decision_stability", 1.0))
            exit_deltas.append(
                m.get("false_exit_rate_change", 0.0)
                + m.get("missed_exit_rate_change", 0.0)
            )

        max_geo = max(geo_deltas) if geo_deltas else 0.0
        max_regime = max(regime_deltas) if regime_deltas else 0.0
        max_gov = max(gov_deltas) if gov_deltas else 0.0
        max_exit = max(exit_deltas) if exit_deltas else 0.0

        layer_stability = {
            "geometry_detection": {
                "name": "Geometry Detection",
                "clean": 1.0,
                "max_delta": round(max_geo, 3),
                "status": (
                    "✅ STABLE"
                    if max_geo < 2
                    else "⚠️ WARNING" if max_geo < 5 else "❌ DEGRADED"
                ),
            },
            "regime_classifier": {
                "name": "Regime Classifier",
                "clean": 1.0,
                "max_delta": round(max_regime, 3),
                "status": (
                    "✅ STABLE"
                    if max_regime <= 0.10
                    else "⚠️ WARNING" if max_regime <= 0.30 else "❌ DEGRADED"
                ),
            },
            "governor": {
                "name": "Governor Decisions",
                "clean": 1.0,
                "max_delta": round(max_gov, 3),
                "status": (
                    "✅ STABLE"
                    if max_gov <= 0.10
                    else "⚠️ WARNING" if max_gov <= 0.30 else "❌ DEGRADED"
                ),
            },
            "exit_rate": {
                "name": "Exit Rate",
                "clean": 0.0,
                "max_delta": round(max_exit, 3),
                "status": (
                    "✅ STABLE"
                    if max_exit <= 0.10
                    else "⚠️ WARNING" if max_exit <= 0.30 else "❌ DEGRADED"
                ),
            },
        }

        # Identify most and least affected layers
        sorted_layers = sorted(
            layer_stability.items(),
            key=lambda x: x[1]["max_delta"],
            reverse=True,
        )
        most_affected = sorted_layers[0]
        least_affected = sorted_layers[-1]

        surviving = sum(
            1 for v in layer_stability.values() if "DEGRADED" not in v["status"]
        )
        degraded = sum(
            1 for v in layer_stability.values() if "DEGRADED" in v["status"]
        )

        # Root cause analysis
        root_cause = self._analyze_root_cause(scenario_results)

        return {
            "scenario_results": scenario_results,
            "layer_stability": layer_stability,
            "most_affected": most_affected,
            "least_affected": least_affected,
            "surviving_layers": surviving,
            "degraded_layers": degraded,
            "total_layers": len(layer_stability),
            "root_cause": root_cause,
            "recommendation": self._generate_recommendation(layer_stability),
            "verdict": self._generate_verdict(layer_stability),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_fresh_pipeline(self) -> GovernancePipeline:
        """Create a fresh pipeline instance (no shared state)."""
        return GovernancePipeline()

    def _run_cycles(
        self,
        test_data: dict,
        apply_slippage: bool = False,
        apply_latency: bool = False,
        apply_partial_fill: bool = False,
    ) -> List[dict]:
        """Run pipeline across all cycles with optional perturbations.

        Returns list of pipeline results, one per cycle.
        """
        pipeline = self._create_fresh_pipeline()
        cluster_states_list = test_data.get("cluster_states", [])
        rfe_outputs_list = test_data.get("rfe_outputs", [])
        price_history = test_data.get("price_history", {})

        cumulative_prices: Dict[str, List[float]] = defaultdict(list)
        self.latency.reset()

        results: List[dict] = []
        n_cycles = len(cluster_states_list)

        for cycle in range(n_cycles):
            # Base data for this cycle
            cs = dict(cluster_states_list[cycle]) if cycle < len(cluster_states_list) else {}
            rfe = (
                copy.deepcopy(rfe_outputs_list[cycle])
                if cycle < len(rfe_outputs_list)
                else {}
            )

            # ---- Apply perturbations ----
            if apply_slippage:
                cs = self._perturb_cluster_states(cs)
                rfe = self._perturb_rfe_slippage(rfe)

            if apply_latency:
                rfe = self.latency.apply_to_pipeline(rfe)

            if apply_partial_fill:
                rfe = self._perturb_rfe_partial_fill(rfe)

            # Build cumulative price history
            if price_history:
                for sym, prices in price_history.items():
                    if cycle < len(prices):
                        cumulative_prices[sym].append(prices[cycle])

            # Run the pipeline
            result = pipeline.evaluate(cs, rfe, dict(cumulative_prices))
            results.append(result)

        return results

    def _perturb_cluster_states(self, cs: dict) -> dict:
        """Apply slippage to all cluster states."""
        perturbed = {}
        for cluster, state in cs.items():
            perturbed[cluster] = self.slippage.apply_to_cluster_state(state)
        return perturbed

    def _perturb_rfe_slippage(self, rfe: dict) -> dict:
        """Apply slippage noise to RFE evaluations."""
        if "evaluations" not in rfe:
            return rfe
        result = dict(rfe)
        evals = {}
        for sym, ev in rfe["evaluations"].items():
            ev_copy = dict(ev)
            if "score" in ev_copy:
                ev_copy["score"] = self.slippage.apply_to_pressure(ev_copy["score"])
            evals[sym] = ev_copy
        result["evaluations"] = evals
        return result

    def _perturb_rfe_partial_fill(self, rfe: dict) -> dict:
        """Apply partial fill distortion to RFE evaluations."""
        if "evaluations" not in rfe:
            return rfe
        result = dict(rfe)
        evals = {}
        for sym, ev in rfe["evaluations"].items():
            evals[sym] = self.partial_fill.apply_to_trade(dict(ev))
        result["evaluations"] = evals
        return result

    @staticmethod
    def _analyze_root_cause(scenario_results: dict) -> str:
        """Identify which perturbation type causes the most instability."""
        worst_delta = -1.0
        worst_scenario = ""
        worst_layer = ""

        for name, m in scenario_results.items():
            if name == "CLEAN BASELINE":
                continue
            gd = m.get("geometry_detection_stability", {})
            gd_delta = float(gd.get("lead_time_delta", 0))
            rs = 1.0 - m.get("regime_classification_stability", 1.0)
            gs = 1.0 - m.get("governor_decision_stability", 1.0)

            if gd_delta > worst_delta:
                worst_delta = gd_delta
                worst_scenario = name
                worst_layer = "Geometry Detection"
            if rs > worst_delta:
                worst_delta = rs
                worst_scenario = name
                worst_layer = "Regime Classifier"
            if gs > worst_delta:
                worst_delta = gs
                worst_scenario = name
                worst_layer = "Governor Decisions"

        if worst_scenario:
            return (
                f"Most affected: {worst_layer} "
                f"({worst_delta:.3f} delta under {worst_scenario.lower()})"
            )
        return "No significant degradation detected"

    @staticmethod
    def _generate_recommendation(layer_stability: dict) -> str:
        """Generate deployment recommendations based on layer stability."""
        points: List[str] = []
        for key, info in layer_stability.items():
            if "DEGRADED" in info["status"] or "WARNING" in info["status"]:
                if key == "geometry_detection":
                    points.append(
                        "Geometry Forecaster needs temporal buffer for "
                        "latency tolerance"
                    )
                    points.append(
                        "Add 2-cycle minimum history requirement before "
                        "PRE_COLLAPSE alarm"
                    )
                elif key == "regime_classifier":
                    points.append(
                        "Regime Classifier may need hysteresis window expansion"
                    )
                elif key == "governor":
                    points.append(
                        "Governor parameter tuning may be needed for "
                        "execution noise tolerance"
                    )
                elif key == "exit_rate":
                    points.append(
                        "Exit rate sensitivity to perturbation within "
                        "acceptable bounds"
                    )

        if not points:
            points.append("Governor robust enough for immediate deployment")

        return "\n".join(points)

    @staticmethod
    def _generate_verdict(layer_stability: dict) -> str:
        """Generate deployment readiness verdict."""
        degraded = sum(
            1 for v in layer_stability.values() if "DEGRADED" in v["status"]
        )
        warnings = sum(
            1 for v in layer_stability.values() if "WARNING" in v["status"]
        )

        if degraded == 0 and warnings == 0:
            return "DEPLOY"
        elif degraded == 0:
            return "DEPLOY WITH LATENCY WARNING"
        elif degraded == 1 and warnings <= 1:
            return "CONDITIONAL DEPLOY"
        else:
            return "HOLD - REQUIRES HARDENING"


# ---------------------------------------------------------------------------
# 4. Test Data Generator
# ---------------------------------------------------------------------------


def generate_test_data() -> dict:
    """Generate realistic test data for shadow execution validation.

    Creates 30 cycles of data with:
    - 3 symbols (AUDUSD, EURUSD, USDJPY)
    - 15 cycles of STABLE_FLOW (high coherence, low pressure)
    - 15 cycles of SLOW_DISSOLUTION (decaying coherence, rising pressure)

    Returns
    -------
    dict with keys:
        cluster_states : list of dict
        rfe_outputs    : list of dict
        price_history   : dict of symbol -> list of prices
        labels          : dict of symbol -> list of regime labels
    """
    random.seed(42)
    symbols = ["AUDUSD", "EURUSD", "USDJPY"]
    clusters = {
        "AUDUSD": "AUD_NZD",
        "EURUSD": "EUR",
        "USDJPY": "JPY",
    }
    n_cycles = 30

    cluster_states: List[dict] = []
    rfe_outputs: List[dict] = []
    price_history: Dict[str, List[float]] = {sym: [] for sym in symbols}
    labels: Dict[str, List[str]] = {sym: [] for sym in symbols}

    for cycle in range(n_cycles):
        # Determine phase: first 15 = STABLE_FLOW, last 15 = SLOW_DISSOLUTION
        if cycle < 15:
            # STABLE_FLOW: high coherence, low divergence, moderate net direction
            coherence = 0.82 + random.uniform(-0.05, 0.05)
            divergence = 1.0 - coherence
            net_dir = random.uniform(-0.2, 0.2)
            pressure_base = 0.12 + random.uniform(-0.04, 0.04)
            regime_label = RegimeMetaType.STABLE_FLOW
            price_slope = random.uniform(-0.02, 0.02)
        else:
            # SLOW_DISSOLUTION: decaying coherence, rising divergence
            t = (cycle - 15) / 14.0  # 0.0 to 1.0
            coherence = max(0.25, 0.80 - t * 0.55)
            divergence = 1.0 - coherence
            net_dir = random.uniform(-0.5, -0.1)
            pressure_base = 0.35 + t * 0.40 + random.uniform(-0.05, 0.05)
            regime_label = RegimeMetaType.SLOW_DISSOLUTION
            price_slope = -0.005 - t * 0.015

        # Build cluster state for this cycle
        cs: dict = {}
        for sym in symbols:
            cluster = clusters[sym]
            if cluster not in cs:
                cs[cluster] = {
                    "coherence": round(coherence, 4),
                    "divergence": round(divergence, 4),
                    "net_direction": round(net_dir, 4),
                    "net_pressure": (
                        "BEARISH" if net_dir < -0.1 else "BULLISH" if net_dir > 0.1 else "NEUTRAL"
                    ),
                }

        # Build RFE output for this cycle
        evaluations: dict = {}
        for sym in symbols:
            # Add some per-symbol variation
            sym_offset = hash(sym) % 100 / 1000.0
            score = round(max(0.0, min(1.0, pressure_base + sym_offset)), 4)

            if score < 0.15:
                state = "INFO"
            elif score < 0.35:
                state = "WATCH"
            elif score < 0.60:
                state = "WARNING"
            elif score < 0.85:
                state = "EXIT_PREP"
            else:
                state = "EXIT"

            evaluations[sym] = {
                "state": state,
                "score": score,
                "components": {
                    "divergence": round(max(0.0, divergence - 0.1 + sym_offset * 0.5), 4),
                    "persistence": round(min(1.0, score * 0.6), 4),
                    "hysteresis_decay": round(max(0.0, score * 0.3 - 0.1), 4),
                    "pnl_regime": round(min(1.0, score * 0.4), 4),
                },
                "exit_allowed": score >= 0.85,
                "cycles_in_state": min(cycle + 1, 10),
                "divergence_cycles": min(cycle, 8),
                "current_price": 0.0,
            }

        rfe_output = {
            "evaluations": evaluations,
            "summary": {
                "max_pressure": max(e["score"] for e in evaluations.values()),
                "max_state": max(e["state"] for e in evaluations.values()),
                "any_exit_allowed": any(e["exit_allowed"] for e in evaluations.values()),
                "trades_at_risk": [
                    sym for sym, e in evaluations.items() if e["state"] not in ("INFO", "WATCH")
                ],
                "overall_risk": (
                    "HIGH"
                    if any(e["state"] in ("EXIT_PREP", "EXIT") for e in evaluations.values())
                    else "MEDIUM"
                    if any(e["state"] == "WARNING" for e in evaluations.values())
                    else "LOW"
                ),
            },
            "transitions": {},
            "temporal": {},
            "breaches": [],
            "timestamp": datetime.now().isoformat(),
        }

        # Generate price
        for sym in symbols:
            prev = price_history[sym][-1] if price_history[sym] else 1.0
            noise = random.gauss(0, 0.002)
            new_price = round(prev * (1.0 + price_slope + noise), 5)
            price_history[sym].append(new_price)

        cluster_states.append(cs)
        rfe_outputs.append(rfe_output)
        for sym in symbols:
            labels[sym].append(regime_label)

    return {
        "cluster_states": cluster_states,
        "rfe_outputs": rfe_outputs,
        "price_history": price_history,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# 5. Report Generator
# ---------------------------------------------------------------------------


def format_shadow_execution_report(validation_result: dict) -> str:
    """Format comprehensive shadow execution validation report.

    Parameters
    ----------
    validation_result : dict
        Output from ``ShadowExecutionLoop.validate_all()``.

    Returns
    -------
    str
        Formatted report with scenario comparison, layer stability,
        invariance analysis, and deployment readiness verdict.
    """
    lines: List[str] = []
    scenario_results = validation_result.get("scenario_results", {})
    layer_stability = validation_result.get("layer_stability", {})

    lines.append("")
    lines.append("SHADOW EXECUTION FEEDBACK LOOP \u2014 VALIDATION REPORT")
    lines.append("=" * 78)

    # -- Scenario Comparison --
    lines.append("")
    lines.append("SCENARIO COMPARISON")
    lines.append("=" * 78)
    header = (
        f"{'Scenario':<18s} {'RegimeStab':<12s} {'GovStab':<12s} "
        f"{'GeoStab':<10s} {'FalseExDel':<10s} {'MissExDel':<10s}"
    )
    lines.append(header)
    lines.append("-" * 78)

    for scenario in [
        "CLEAN BASELINE",
        "SLIPPAGE ONLY",
        "LATENCY ONLY",
        "PARTIAL FILL",
        "FULL COMBINED",
    ]:
        m = scenario_results.get(scenario, {})
        rs = f"{m.get('regime_classification_stability', 1.0):.3f}"
        gs = f"{m.get('governor_decision_stability', 1.0):.3f}"
        geo_stab = m.get("geometry_detection_stability", {})
        geo_str = f"{geo_stab.get('lead_time_delta', 0)}cyc"
        fe = f"{m.get('false_exit_rate_change', 0.0):.3f}"
        me = f"{m.get('missed_exit_rate_change', 0.0):.3f}"
        lines.append(
            f"{scenario:<18s} {rs:<12s} {gs:<12s} "
            f"{geo_str:<10s} {fe:<10s} {me:<10s}"
        )

    # -- Layer Stability --
    lines.append("")
    lines.append("LAYER STABILITY (worst-case across scenarios)")
    lines.append("=" * 78)
    lines.append(
        f"{'Layer':<22s} {'Clean':<10s} {'Perturbed':<12s} "
        f"{'Delta':<10s} {'Status':<18s}"
    )
    lines.append("-" * 78)

    for key, info in layer_stability.items():
        name = info.get("name", key)
        clean_val = info.get("clean", 1.0)
        delta = info.get("max_delta", 0.0)
        pert_val = max(0.0, clean_val - delta)
        status = info.get("status", "")
        lines.append(
            f"{name:<22s} {clean_val:<10.3f} {pert_val:<12.3f} "
            f"{delta:<10.3f} {status:<18s}"
        )

    # -- Invariance Analysis --
    lines.append("")
    lines.append("INVARIANCE ANALYSIS")
    lines.append("=" * 78)

    root_cause = validation_result.get("root_cause", "")
    lines.append(f"Most affected: {root_cause}")
    least = validation_result.get("least_affected", ("", {}))
    if least and least[1]:
        lines.append(
            f"Least affected: {least[1].get('name', least[0])} "
            f"({least[1].get('max_delta', 0):.3f} delta)"
        )

    # Provide root cause rationale
    # Find which scenario caused the most damage to geometry
    for scenario in ["LATENCY ONLY", "SLIPPAGE ONLY", "PARTIAL FILL", "FULL COMBINED"]:
        m = scenario_results.get(scenario, {})
        geo_stab = m.get("geometry_detection_stability", {})
        if geo_stab.get("lead_time_delta", 0) >= 2:
            lines.append(
                f"Root cause: {scenario} shifts coherence curvature "
                f"calculation timing"
            )
            break

    # -- Deployment Readiness --
    lines.append("")
    lines.append("DEPLOYMENT READINESS VERDICT")
    lines.append("=" * 78)
    verdict = validation_result.get("verdict", "UNKNOWN")
    surviving = validation_result.get("surviving_layers", 0)
    total = validation_result.get("total_layers", 0)
    lines.append(f"Overall: {verdict}")
    lines.append("")
    lines.append(
        f"Layers that survive feedback noise: {surviving}/{total}"
    )
    lines.append(
        f"Layers that degrade: "
        f"{validation_result.get('degraded_layers', 0)}/{total}"
    )
    lines.append("")
    lines.append("Recommendation:")
    rec = validation_result.get("recommendation", "")
    for r_line in rec.split("\n"):
        lines.append(f"  - {r_line}")
    lines.append("")
    lines.append(f"Verdict: {verdict}")
    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)
