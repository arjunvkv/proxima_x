from __future__ import annotations

import json
import math
import random
import time
from typing import Any

from ..core.unified_conviction_field import UnifiedConvictionField
from ..integration.regime_adaptive_modulator import RegimeAdaptiveModulator
from ..integration.ucf_pipeline_bridge import UCFPipelineBridge
from ...fsv.core.fsv_engine import FSVEngine
from ...fsv.core.fsv_schema import neutral_fsv
from .tick_adapter import TickAdapter
from .replay_macro_adapter import ReplayMacroAdapter


class ReplayOrchestrator:
    def __init__(self) -> None:
        self.ucf: UnifiedConvictionField = UnifiedConvictionField()
        self.bridge: UCFPipelineBridge = UCFPipelineBridge()
        self.fsv_engine: FSVEngine = FSVEngine()
        self.macro_adapter: ReplayMacroAdapter = ReplayMacroAdapter(self.fsv_engine)
        self.tick_adapter: TickAdapter = TickAdapter()
        self.log: list[dict[str, Any]] = []
        self.symbols: list[str] = []
        self._ticks: list[dict[str, Any]] = []

    def load_ticks(self, ticks: list[dict[str, Any]]) -> None:
        self._ticks = ticks
        self.symbols = list({t["symbol"] for t in ticks})

    def generate_synthetic_ticks(
        self, num_ticks: int = 1000, symbols: list[str] | None = None
    ) -> list[dict[str, Any]]:
        if symbols is None:
            symbols = ["EURUSD"]

        base_prices: dict[str, float] = {}
        for sym in symbols:
            if sym == "EURUSD":
                base_prices[sym] = 1.10000
            elif sym == "GBPUSD":
                base_prices[sym] = 1.25000
            elif sym == "USDJPY":
                base_prices[sym] = 110.000
            elif sym == "USDCHF":
                base_prices[sym] = 0.92000
            elif sym == "AUDUSD":
                base_prices[sym] = 0.72000
            elif sym == "USDCAD":
                base_prices[sym] = 1.35000
            elif sym == "NZDUSD":
                base_prices[sym] = 0.68000
            else:
                base_prices[sym] = 1.10000

        ticks: list[dict[str, Any]] = []
        start_time: float = time.time()
        current_prices: dict[str, float] = dict(base_prices)
        num_syms: int = len(symbols)

        for i in range(num_ticks):
            symbol: str = symbols[i % num_syms]
            price_change: float = random.gauss(0, 0.0001)
            current_prices[symbol] += price_change
            bid: float = current_prices[symbol]
            spread: int = random.randint(1, 5)
            pip_size: float = 0.00001 if symbol != "USDJPY" else 0.001
            ask: float = bid + spread * pip_size
            volume: int = random.randint(1, 100)

            tick: dict[str, Any] = {
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "spread": spread,
                "time": start_time + i * 0.1,
                "volume": volume,
            }
            ticks.append(tick)

        return ticks

    def run_cycle(self, current_ticks: list[dict[str, Any]]) -> dict[str, Any]:
        now_ts = time.time()
        macro_ctx: dict[str, Any] = self.macro_adapter.get_regime_context(now_ts)

        adapted: list[dict[str, Any]] = self.tick_adapter.adapt_tick_batch(current_ticks)
        technical_state: dict[str, dict[str, Any]] = self.tick_adapter.ticks_to_technical_state(adapted)
        fsv_raw = {s: self.fsv_engine.get_state(s, now_ts) for s in self.symbols}
        fsv_states: dict[str, dict[str, Any]] = {}
        for sym, state in fsv_raw.items():
            fsv_states[sym] = {
                "conviction": abs(state.bias_alignment) * 0.8 + 0.2,
                "direction": 1 if state.bias_alignment > 0.1 else (-1 if state.bias_alignment < -0.1 else 0),
                "stability": state.regime_stability,
            }
        regime_context: dict[str, Any] = self._build_regime_context(technical_state, fsv_states)
        regime_context["regime"] = macro_ctx.get("regime", regime_context["regime"])

        result: dict[str, Any] = self.bridge.process(
            symbols=self.symbols,
            technical_states=technical_state,
            fsv_states=fsv_states,
            cev_state=None,
            regime_state=regime_context,
        )

        log_entry: dict[str, Any] = {
            "timestamp": time.time(),
            "batch_size": len(current_ticks),
            "technical_state": technical_state,
            "regime": result.get("regime", "neutral"),
            "selected_symbol": result.get("selected_symbol", ""),
            "ranked_symbols": result.get("ranked_symbols", []),
            "field": result.get("field", {}),
            "weights_used": result.get("weights_used", {}),
            "field_coherence": result.get("field_coherence", 0.0),
            "dominant_direction": result.get("dominant_direction", 0),
        }
        self.log.append(log_entry)

        return result

    def _build_regime_context(
        self,
        technical_state: dict[str, dict[str, Any]],
        fsv_states: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        volatilities: list[float] = [
            v.get("stability", 0.5) for v in technical_state.values()
        ]
        avg_stability: float = (
            sum(volatilities) / len(volatilities) if volatilities else 0.5
        )

        technical_volatility: float = 1.0 - avg_stability

        if technical_volatility < 0.3:
            regime: str = "stable"
        elif technical_volatility < 0.6:
            regime = "neutral"
        else:
            regime = "transition"

        fsv_entropy = 0.0
        fsv_directions: list[int] = [
            v.get("direction", 0) if isinstance(v, dict) else 0 for v in fsv_states.values()
        ]
        if fsv_directions:
            unique_dirs: set[int] = set(fsv_directions)
            counts: list[int] = [fsv_directions.count(d) for d in unique_dirs]
            total: int = len(fsv_directions)
            fsv_entropy = -sum(
                (c / total) * math.log(c / total) if c > 0 else 0
                for c in counts
            )

        return {
            "regime": regime,
            "regime_stability": avg_stability,
            "fsv_entropy": fsv_entropy,
            "technical_volatility": technical_volatility,
            "recent_prediction_error": 0.0,
            "exposure_concentration": 0.0,
        }

    def run_replay(self, batch_size: int = 100) -> dict[str, Any]:
        total_ticks: int = len(self._ticks)
        cycles: int = 0
        all_logs: list[dict[str, Any]] = []

        for i in range(0, total_ticks, batch_size):
            batch: list[dict[str, Any]] = self._ticks[i : i + batch_size]
            if not batch:
                continue
            self.run_cycle(batch)
            cycles += 1

        all_logs = self.log

        avg_confidence: float = 0.0
        direction_distribution: dict[str, int] = {"-1": 0, "0": 0, "1": 0}
        symbol_selection_frequency: dict[str, int] = {}
        regime_distribution: dict[str, int] = {}

        if all_logs:
            confidences: list[float] = []
            for entry in all_logs:
                ranked: list[dict[str, Any]] = entry.get("ranked_symbols", [])
                if ranked:
                    confidences.append(ranked[0].get("ucf_score", 0.0))

                sel: str = entry.get("selected_symbol", "")
                if sel:
                    symbol_selection_frequency[sel] = (
                        symbol_selection_frequency.get(sel, 0) + 1
                    )

                regime_name: str = entry.get("regime", "neutral")
                regime_distribution[regime_name] = (
                    regime_distribution.get(regime_name, 0) + 1
                )

                dom_dir: int = entry.get("dominant_direction", 0)
                direction_distribution[str(dom_dir)] = (
                    direction_distribution.get(str(dom_dir), 0) + 1
                )

            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

        summary: dict[str, Any] = {
            "avg_confidence": avg_confidence,
            "direction_distribution": direction_distribution,
            "symbol_selection_frequency": symbol_selection_frequency,
            "regime_distribution": regime_distribution,
        }

        return {
            "total_ticks": total_ticks,
            "symbols": self.symbols,
            "cycles": cycles,
            "logs": all_logs,
            "summary": summary,
        }

    def get_logs(self) -> list[dict[str, Any]]:
        return list(self.log)

    def export_logs(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.log, f, indent=2)
