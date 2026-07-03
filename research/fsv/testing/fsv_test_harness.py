from ..core.fsv_schema import FundamentalStateVector, NormalizedEvent, neutral_fsv, validate_fsv, validate_event
from ..core.fsv_engine import FSVEngine
from ..integration.fsv_modulator import FSVModulator
from ..simulation.synthetic_event_generator import SyntheticMacroGenerator
from ..runtime.fsv_event_loop import FSVEventLoop
import time
import math
import random
import statistics


class FSVTestHarness:

    def __init__(self) -> None:
        self.engine: FSVEngine = None
        self.modulator: FSVModulator = None
        self.generator: SyntheticMacroGenerator = None
        self.loop: FSVEventLoop = None
        self.results: dict = {}
        self.passed: int = 0
        self.failed: int = 0

    def test_decay_correctness(self) -> bool:
        fsv: FundamentalStateVector = neutral_fsv("EURUSD")
        fsv.bias_alignment = 0.8
        fsv.macro_pressure = 0.6
        fsv.sentiment_gradient = 0.4
        fsv.event_risk = 0.8
        fsv.regime_stability = 0.9
        now: float = time.time()
        decayed: FundamentalStateVector = fsv.apply_decay(now + 100)
        if not (decayed.bias_alignment < fsv.bias_alignment):
            return False
        if not (decayed.macro_pressure < fsv.macro_pressure):
            return False
        if not (decayed.sentiment_gradient < fsv.sentiment_gradient):
            return False
        if not (0.5 < decayed.event_risk < fsv.event_risk):
            return False
        if not (0.5 < decayed.regime_stability < fsv.regime_stability):
            return False
        decayed2: FundamentalStateVector = decayed.apply_decay(now + 200)
        change1: float = fsv.bias_alignment - decayed.bias_alignment
        change2: float = decayed.bias_alignment - decayed2.bias_alignment
        if not (change2 < change1):
            return False
        return True

    def test_event_accumulation_stability(self) -> bool:
        engine: FSVEngine = FSVEngine()
        generator: SyntheticMacroGenerator = SyntheticMacroGenerator()
        symbols: list[str] = ["EURUSD", "GBPUSD", "USDJPY"]
        events: list[NormalizedEvent] = generator.generate_event_stream(symbols, duration_seconds=3000, events_per_minute=2)
        for event in events:
            engine.update_with_event(event)
        states: dict[str, FundamentalStateVector] = engine.get_all_states()
        for sym in symbols:
            if sym not in states:
                return False
        for state in states.values():
            if not validate_fsv(state):
                return False
            for val in [
                state.bias_alignment,
                state.macro_pressure,
                state.sentiment_gradient,
                state.event_risk,
                state.regime_stability,
            ]:
                if math.isnan(val):
                    return False
        neutral_ref: FundamentalStateVector = neutral_fsv("EURUSD")
        all_neutral: bool = True
        for state in states.values():
            if abs(state.bias_alignment) > 0.001 or abs(state.macro_pressure) > 0.001 or abs(state.event_risk - 0.5) > 0.001:
                all_neutral = False
                break
        if all_neutral:
            return False
        return True

    def test_shock_response_behavior(self) -> bool:
        engine: FSVEngine = FSVEngine()
        neutral_state: FundamentalStateVector = engine.get_state("EURUSD")
        cpi_shock: NormalizedEvent = NormalizedEvent(
            symbol="EURUSD",
            event_type="CPI",
            surprise_score=0.8,
            direction_bias=0.7,
            impact_weight=0.8,
        )
        post_shock: FundamentalStateVector = engine.update_with_event(cpi_shock)
        if post_shock.bias_alignment <= neutral_state.bias_alignment:
            return False
        if post_shock.event_risk <= 0.5:
            return False
        reverse_shock: NormalizedEvent = NormalizedEvent(
            symbol="EURUSD",
            event_type="CPI",
            surprise_score=-0.8,
            direction_bias=-0.7,
            impact_weight=0.8,
        )
        post_reverse: FundamentalStateVector = engine.update_with_event(reverse_shock)
        if post_reverse.bias_alignment >= post_shock.bias_alignment:
            return False
        return True

    def test_multi_symbol_interference(self) -> bool:
        engine: FSVEngine = FSVEngine()
        engine.update_with_event(NormalizedEvent("EURUSD", "CPI", 0.5, 0.6, 0.7))
        engine.update_with_event(NormalizedEvent("GBPUSD", "RATE", -0.3, -0.4, 0.6))
        engine.update_with_event(NormalizedEvent("EURUSD", "GDP", 0.4, 0.5, 0.5))
        engine.update_with_event(NormalizedEvent("GBPUSD", "NEWS", 0.2, 0.3, 0.4))
        usdjpy_state: FundamentalStateVector = engine.get_state("USDJPY")
        if abs(usdjpy_state.bias_alignment) > 0.001:
            return False
        if abs(usdjpy_state.macro_pressure) > 0.001:
            return False
        eur_state: FundamentalStateVector = engine.get_state("EURUSD")
        gbp_state: FundamentalStateVector = engine.get_state("GBPUSD")
        if abs(eur_state.bias_alignment - gbp_state.bias_alignment) < 0.01:
            return False
        return True

    def test_modulation_bounds(self) -> bool:
        modulator: FSVModulator = FSVModulator()
        extreme_fsv: FundamentalStateVector = FundamentalStateVector(
            symbol="EURUSD",
            bias_alignment=1.0,
            macro_pressure=1.0,
            sentiment_gradient=1.0,
            event_risk=1.0,
            regime_stability=1.0,
        )
        result: tuple[float, dict] = modulator.modulate(0.5, extreme_fsv)
        adjusted: float = result[0]
        factor: float = result[1]["modulation_factor"]
        if not (0.0 <= adjusted <= 1.0):
            return False
        if not (-0.15 <= factor <= 0.15):
            return False
        extreme_neg: FundamentalStateVector = FundamentalStateVector(
            symbol="EURUSD",
            bias_alignment=-1.0,
            macro_pressure=-1.0,
            sentiment_gradient=-1.0,
            event_risk=1.0,
            regime_stability=0.0,
        )
        result_neg: tuple[float, dict] = modulator.modulate(0.5, extreme_neg)
        adjusted_neg: float = result_neg[0]
        factor_neg: float = result_neg[1]["modulation_factor"]
        if not (0.0 <= adjusted_neg <= 1.0):
            return False
        if not (-0.15 <= factor_neg <= 0.15):
            return False
        neutral: FundamentalStateVector = neutral_fsv("EURUSD")
        result_neutral: tuple[float, dict] = modulator.modulate(0.5, neutral)
        factor_neutral: float = result_neutral[1]["modulation_factor"]
        if abs(factor_neutral) > 0.01:
            return False
        return True

    def test_decay_during_data_gap(self) -> bool:
        engine: FSVEngine = FSVEngine()
        event: NormalizedEvent = NormalizedEvent("EURUSD", "CPI", 0.7, 0.6, 0.8)
        initial_state: FundamentalStateVector = engine.update_with_event(event)
        initial_bias: float = initial_state.bias_alignment
        t1: float = initial_state.last_update_ts + 1000.0
        engine.decay_all(t1)
        state_after_gap: FundamentalStateVector = engine.get_state("EURUSD")
        if abs(state_after_gap.bias_alignment) >= abs(initial_bias):
            return False
        t2: float = t1 + 10000.0
        engine.decay_all(t2)
        state_after_long_gap: FundamentalStateVector = engine.get_state("EURUSD")
        if abs(state_after_long_gap.bias_alignment) > 0.05:
            return False
        return True

    def test_conflicting_events_stability(self) -> bool:
        engine: FSVEngine = FSVEngine()
        conflicting_events: list[NormalizedEvent] = [
            NormalizedEvent("EURUSD", "CPI", 0.8, 0.7, 0.8),
            NormalizedEvent("EURUSD", "RATE", -0.6, -0.5, 0.7),
            NormalizedEvent("EURUSD", "GDP", 0.3, 0.4, 0.5),
            NormalizedEvent("EURUSD", "NEWS", -0.9, -0.8, 0.9),
        ]
        for event in conflicting_events:
            engine.update_with_event(event)
        state: FundamentalStateVector = engine.get_state("EURUSD")
        if not validate_fsv(state):
            return False
        for val in [
            state.bias_alignment,
            state.macro_pressure,
            state.sentiment_gradient,
            state.event_risk,
            state.regime_stability,
        ]:
            if math.isnan(val):
                return False
        return True

    def test_high_volatility_news_cluster(self) -> bool:
        engine: FSVEngine = FSVEngine()
        generator: SyntheticMacroGenerator = SyntheticMacroGenerator()
        events: list[NormalizedEvent] = generator.stress_scenario("crisis")
        for event in events:
            engine.update_with_event(event)
        primary_symbol: str = events[0].symbol if events else "EURUSD"
        state: FundamentalStateVector = engine.get_state(primary_symbol)
        if not validate_fsv(state):
            return False
        if state.event_risk <= 0.5:
            return False
        if state.regime_stability >= 0.7:
            return False
        return True

    def test_api_failure_graceful_degradation(self) -> bool:
        engine: FSVEngine = FSVEngine()
        for _ in range(5):
            engine.update_with_event(NormalizedEvent("EURUSD", "CPI", 0.3, 0.2, 0.5))
        now: float = time.time()
        engine.decay_all(now + 500.0)
        for _ in range(3):
            engine.update_with_event(NormalizedEvent("EURUSD", "RATE", -0.2, -0.3, 0.4))
        state: FundamentalStateVector = engine.get_state("EURUSD")
        if not validate_fsv(state):
            return False
        return True

    def test_fsv_to_feature_vector(self) -> bool:
        fsv: FundamentalStateVector = FundamentalStateVector(
            symbol="EURUSD",
            bias_alignment=0.5,
            macro_pressure=-0.3,
            sentiment_gradient=0.2,
            event_risk=0.7,
            regime_stability=0.4,
            decay_lambda=0.01,
        )
        vec: list[float] = fsv.to_feature_vector()
        if not isinstance(vec, list):
            return False
        if len(vec) != 5:
            return False
        if not all(isinstance(v, float) for v in vec):
            return False
        expected: list[float] = [0.5, -0.3, 0.2, 0.7, 0.4]
        for i in range(5):
            if vec[i] != expected[i]:
                return False
        return True

    def test_normalized_event_validation(self) -> bool:
        valid_event: NormalizedEvent = NormalizedEvent("EURUSD", "CPI", 0.5, 0.3, 0.7)
        if not validate_event(valid_event):
            return False
        invalid_surprise: NormalizedEvent = NormalizedEvent("EURUSD", "CPI", 5.0, 0.3, 0.7)
        if validate_event(invalid_surprise):
            return False
        invalid_type: NormalizedEvent = NormalizedEvent("EURUSD", "INVALID_TYPE", 0.5, 0.3, 0.7)
        if validate_event(invalid_type):
            return False
        invalid_bias: NormalizedEvent = NormalizedEvent("EURUSD", "CPI", 0.5, 2.0, 0.7)
        if validate_event(invalid_bias):
            return False
        invalid_weight: NormalizedEvent = NormalizedEvent("EURUSD", "CPI", 0.5, 0.3, 1.5)
        if validate_event(invalid_weight):
            return False
        return True

    def test_loop_integration(self) -> bool:
        loop: FSVEventLoop = FSVEventLoop()
        loop.start_loop()
        generator: SyntheticMacroGenerator = SyntheticMacroGenerator()
        events: list[NormalizedEvent] = generator.generate_event_stream(["EURUSD"], duration_seconds=300, events_per_minute=2)
        for event in events:
            loop.ingest_event(event)
        loop.run_cycle()
        states: dict[str, FundamentalStateVector] = loop.engine.get_all_states()
        if "EURUSD" not in states:
            return False
        if loop.get_queue_size() > 0:
            return False
        return True

    def test_zero_impact_on_execution_path(self) -> bool:
        modulator: FSVModulator = FSVModulator()
        fsves: list[FundamentalStateVector] = [
            neutral_fsv("EURUSD"),
            FundamentalStateVector("EURUSD", 1.0, 1.0, 1.0, 1.0, 1.0),
            FundamentalStateVector("EURUSD", -1.0, -1.0, -1.0, 1.0, 0.0),
            FundamentalStateVector("EURUSD", 0.5, -0.5, 0.3, 0.7, 0.5),
        ]
        base_conviction: float = 0.5
        for fsv in fsves:
            result: tuple[float, dict] = modulator.modulate(base_conviction, fsv)
            adjusted: float = result[0]
            factor: float = result[1]["modulation_factor"]
            if not isinstance(adjusted, float):
                return False
            if not isinstance(factor, float):
                return False
            if not (0.0 <= adjusted <= 1.0):
                return False
        return True

    def run_all(self) -> dict:
        self.passed = 0
        self.failed = 0
        self.results = {}
        failures: list[str] = []
        tests: list[tuple[str, callable]] = [
            ("test_decay_correctness", self.test_decay_correctness),
            ("test_event_accumulation_stability", self.test_event_accumulation_stability),
            ("test_shock_response_behavior", self.test_shock_response_behavior),
            ("test_multi_symbol_interference", self.test_multi_symbol_interference),
            ("test_modulation_bounds", self.test_modulation_bounds),
            ("test_decay_during_data_gap", self.test_decay_during_data_gap),
            ("test_conflicting_events_stability", self.test_conflicting_events_stability),
            ("test_high_volatility_news_cluster", self.test_high_volatility_news_cluster),
            ("test_api_failure_graceful_degradation", self.test_api_failure_graceful_degradation),
            ("test_fsv_to_feature_vector", self.test_fsv_to_feature_vector),
            ("test_normalized_event_validation", self.test_normalized_event_validation),
            ("test_loop_integration", self.test_loop_integration),
            ("test_zero_impact_on_execution_path", self.test_zero_impact_on_execution_path),
        ]
        for name, method in tests:
            try:
                result: bool = method()
                self.results[name] = result
                if result:
                    self.passed += 1
                else:
                    self.failed += 1
                    failures.append(name)
            except Exception:
                self.results[name] = False
                self.failed += 1
                failures.append(name)
        total: int = self.passed + self.failed
        return {
            "passed": self.passed,
            "failed": self.failed,
            "total": total,
            "results": dict(self.results),
            "failures": failures,
            "stability_score": self.passed / total if total > 0 else 0.0,
            "timestamp": time.time(),
        }

    def stress_suite(self) -> dict:
        self.passed = 0
        self.failed = 0
        self.results = {}
        failures: list[str] = []
        stress_tests: list[tuple[str, callable]] = [
            ("test_high_volatility_news_cluster", self.test_high_volatility_news_cluster),
            ("test_decay_during_data_gap", self.test_decay_during_data_gap),
            ("test_event_accumulation_stability", self.test_event_accumulation_stability),
            ("test_api_failure_graceful_degradation", self.test_api_failure_graceful_degradation),
            ("test_shock_response_behavior", self.test_shock_response_behavior),
            ("test_conflicting_events_stability", self.test_conflicting_events_stability),
        ]
        for name, method in stress_tests:
            try:
                result: bool = method()
                self.results[name] = result
                if result:
                    self.passed += 1
                else:
                    self.failed += 1
                    failures.append(name)
            except Exception:
                self.results[name] = False
                self.failed += 1
                failures.append(name)
        total: int = self.passed + self.failed
        return {
            "passed": self.passed,
            "failed": self.failed,
            "total": total,
            "results": dict(self.results),
            "failures": failures,
            "stability_score": self.passed / total if total > 0 else 0.0,
            "timestamp": time.time(),
        }

    def get_report(self) -> dict:
        start: float = time.time()
        results: dict = self.run_all()
        duration: float = time.time() - start
        recommendations: list[str] = []
        if results["stability_score"] < 1.0:
            for failure in results["failures"]:
                if "decay" in failure:
                    recommendations.append("Review decay lambda configuration and decay logic")
                if "shock" in failure:
                    recommendations.append("Inspect shock response thresholds and event impact weights")
                if "conflict" in failure:
                    recommendations.append("Investigate conflicting event resolution in merge logic")
                if "modulation" in failure:
                    recommendations.append("Check modulation bounds and conviction adjustment")
                if "validation" in failure:
                    recommendations.append("Validate event and state validation rules")
                if "loop" in failure:
                    recommendations.append("Check event loop processing and queue management")
                if "feature" in failure or "vector" in failure:
                    recommendations.append("Verify to_feature_vector output format matches schema")
                if "multi" in failure or "interference" in failure:
                    recommendations.append("Review multi-symbol state isolation in engine")
                if "gap" in failure:
                    recommendations.append("Examine decay behavior during extended data gaps")
                if "accumulation" in failure:
                    recommendations.append("Check event accumulation stability and state bounds")
                if "crisis" in failure or "volatility" in failure or "cluster" in failure:
                    recommendations.append("Review high-volatility event cluster handling")
                if "api" in failure or "graceful" in failure or "degradation" in failure:
                    recommendations.append("Inspect graceful degradation after data gaps")
                if "zero" in failure or "execution" in failure or "impact" in failure:
                    recommendations.append("Verify modulation never blocks or rejects execution")
        if results["stability_score"] >= 0.9:
            recommendations.append("System is stable. Monitor for regressions.")
        elif results["stability_score"] >= 0.7:
            recommendations.append("Moderate issues detected. Review failing tests.")
        else:
            recommendations.append("Critical issues detected. Immediate investigation required.")
        return {
            "summary": {
                "passed": results["passed"],
                "failed": results["failed"],
                "total": results["total"],
                "stability_score": results["stability_score"],
                "duration_seconds": duration,
            },
            "results": results["results"],
            "failures": results["failures"],
            "recommendations": recommendations,
            "timestamp": results["timestamp"],
        }
