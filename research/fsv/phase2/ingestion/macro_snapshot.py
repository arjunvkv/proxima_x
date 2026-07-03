from typing import Optional
from dataclasses import asdict
import time
import random
from statistics import stdev, mean
from ...core.fsv_schema import NormalizedEvent, FundamentalStateVector
from ...core.fsv_engine import FSVEngine


class MacroSnapshotEngine:
    def __init__(self, fsv_engine: FSVEngine = None) -> None:
        if fsv_engine is None:
            fsv_engine = FSVEngine()
        self.fsv_engine: FSVEngine = fsv_engine

    def take_snapshot(self, symbols: list[str] = None) -> dict:
        all_states = self.fsv_engine.get_all_states()
        if symbols is None:
            symbols = list(all_states.keys())

        fsves: dict[str, FundamentalStateVector] = {}
        for sym in symbols:
            if sym in all_states:
                fsves[sym] = all_states[sym]
            else:
                fsves[sym] = FundamentalStateVector(symbol=sym)

        now = time.time()

        if not fsves:
            return {
                "timestamp": now,
                "symbols_covered": [],
                "environment": "neutral",
                "macro_pressure_avg": 0.0,
                "sentiment_avg": 0.0,
                "regime_stability_avg": 0.5,
                "event_risk_avg": 0.5,
                "bias_dispersion": 0.0,
                "top_events": [],
                "data_freshness": 0.0,
                "coverage_score": 0.0,
                "alert_flags": {
                    "stale_data": False,
                    "high_risk_cluster": False,
                    "regime_shift_detected": False,
                },
            }

        macro_pressures = [f.macro_pressure for f in fsves.values()]
        sentiments = [f.sentiment_gradient for f in fsves.values()]
        stabilities = [f.regime_stability for f in fsves.values()]
        event_risks = [f.event_risk for f in fsves.values()]
        biases = [f.bias_alignment for f in fsves.values()]
        freshness = [now - f.last_update_ts for f in fsves.values()]

        macro_pressure_avg = mean(macro_pressures)
        sentiment_avg = mean(sentiments)
        regime_stability_avg = mean(stabilities)
        event_risk_avg = mean(event_risks)
        bias_dispersion = stdev(biases) if len(biases) > 1 else 0.0
        data_freshness = mean(freshness)
        symbols_with_fresh = sum(1 for f in freshness if f < 3600.0)
        coverage_score = symbols_with_fresh / len(fsves) * 100.0

        environment = self.compute_environment(fsves)

        recent_events = self.get_recent_events(limit=50)
        high_impact = [e for e in recent_events if e.get("impact_weight", 0) > 0.5]
        high_impact.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        top_events = high_impact[:5]

        stale_data = any(f > 3600.0 for f in freshness)
        high_risk_symbols = [sym for sym, f in fsves.items() if f.event_risk > 0.7]
        high_risk_cluster = len(high_risk_symbols) > 1
        regime_shift_detected = abs(regime_stability_avg - 0.5) > 0.25

        return {
            "timestamp": now,
            "symbols_covered": symbols,
            "environment": environment,
            "macro_pressure_avg": macro_pressure_avg,
            "sentiment_avg": sentiment_avg,
            "regime_stability_avg": regime_stability_avg,
            "event_risk_avg": event_risk_avg,
            "bias_dispersion": bias_dispersion,
            "top_events": top_events,
            "data_freshness": data_freshness,
            "coverage_score": coverage_score,
            "alert_flags": {
                "stale_data": stale_data,
                "high_risk_cluster": high_risk_cluster,
                "regime_shift_detected": regime_shift_detected,
            },
        }

    def get_recent_events(self, symbol: str = None, limit: int = 10) -> list[dict]:
        events = self.fsv_engine.get_event_history(symbol=symbol, limit=limit)
        return [asdict(e) for e in events]

    def simulate_macro_update(self) -> NormalizedEvent:
        event_type = random.choice(("CPI", "NEWS", "RATE", "GDP", "SENTIMENT", "UNKNOWN"))
        symbol = random.choice(("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"))
        return NormalizedEvent(
            symbol=symbol,
            event_type=event_type,
            surprise_score=random.uniform(-1.0, 1.0),
            direction_bias=random.uniform(-1.0, 1.0),
            impact_weight=random.uniform(0.3, 1.0),
            timestamp=time.time(),
            source="simulated_macro_snapshot",
            raw_data={"generated_by": "MacroSnapshotEngine.simulate_macro_update"},
        )

    def compute_coverage(self, fsves: dict[str, FundamentalStateVector]) -> dict:
        now = time.time()
        sorted_fsves = list(fsves.values())
        if not sorted_fsves:
            return {
                "symbols_with_data": 0,
                "avg_freshness": 0.0,
                "coverage_pct": 0.0,
                "stale_symbols": [],
            }

        symbols_with_data = len(sorted_fsves)
        freshness = [now - f.last_update_ts for f in sorted_fsves]
        avg_freshness = mean(freshness)
        fresh_count = sum(1 for f in freshness if f < 3600.0)
        coverage_pct = fresh_count / symbols_with_data * 100.0
        stale_symbols = [f.symbol for f in sorted_fsves if now - f.last_update_ts >= 3600.0]

        return {
            "symbols_with_data": symbols_with_data,
            "avg_freshness": avg_freshness,
            "coverage_pct": coverage_pct,
            "stale_symbols": stale_symbols,
        }

    def compute_environment(self, fsves: dict[str, FundamentalStateVector]) -> str:
        if not fsves:
            return "neutral"

        risk_on_count = 0
        risk_off_count = 0
        for fsv in fsves.values():
            if fsv.sentiment_gradient > 0.2 and fsv.macro_pressure > 0.1 and fsv.event_risk < 0.4 and fsv.regime_stability > 0.6:
                risk_on_count += 1
            elif fsv.sentiment_gradient < -0.2 and fsv.macro_pressure < -0.1 and fsv.event_risk > 0.6 and fsv.regime_stability < 0.4:
                risk_off_count += 1

        total = len(fsves)
        risk_on_pct = risk_on_count / total
        risk_off_pct = risk_off_count / total

        if risk_on_pct > 0.6:
            return "risk_on"
        elif risk_off_pct > 0.6:
            return "risk_off"
        elif risk_on_pct > 0.3 and risk_off_pct > 0.3:
            return "mixed"
        else:
            return "neutral"

    def get_macro_events_for_period(self, start_ts: float, end_ts: float) -> list[dict]:
        events = [e for e in self.fsv_engine.event_log if start_ts <= e.timestamp <= end_ts]
        return [asdict(e) for e in events]
