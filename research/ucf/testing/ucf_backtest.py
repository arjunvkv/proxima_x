import random
import time
import math
import statistics
from typing import Optional

from ...phase2.integration.fundamental_selector import FundamentalSelector
from ...phase2.core.fundamental_ranker import FundamentalRanker
from ...phase2.core.macro_alignment import MacroAlignmentEngine
from ...phase2.core.symbol_comparator import FundamentalComparator
from ..core.unified_conviction_field import UnifiedConvictionField
from ..integration.regime_adaptive_modulator import RegimeAdaptiveModulator
from ...core.fsv_engine import FSVEngine
from ...simulation.synthetic_event_generator import SyntheticMacroGenerator


class UCFBacktestEngine:
    def __init__(self) -> None:
        self.fsv_engine = FSVEngine()
        self.fundamental_selector = FundamentalSelector()
        self.fundamental_ranker = FundamentalRanker()
        self.macro_alignment = MacroAlignmentEngine()
        self.symbol_comparator = FundamentalComparator()
        self.ucf = UnifiedConvictionField()
        self.regime_modulator = RegimeAdaptiveModulator()
        self.event_generator = SyntheticMacroGenerator()
        self._symbols: list[str] = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
                                     "NZDUSD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY"]

    def run_comparison(self, num_cycles: int = 100, symbols: Optional[list[str]] = None) -> dict:
        if symbols is not None:
            self._symbols = symbols
        active_symbols = list(self._symbols)

        phase2_results: list[dict] = []
        ucf_results: list[dict] = []

        for cycle_index in range(num_cycles):
            num_events = random.randint(2, 5)
            for _ in range(num_events):
                event = self.event_generator.generate_event()
                self.fsv_engine.ingest_event(event)

            fsv_states: dict[str, float] = {}
            for sym in active_symbols:
                fsv_states[sym] = self.fsv_engine.get_state(sym)

            top3_candidates = random.sample(active_symbols, min(3, len(active_symbols)))

            phase2_pick = self.fundamental_selector.select_best(top3_candidates)
            phase2_confidence = 0.0
            if isinstance(phase2_pick, dict):
                phase2_pick_symbol = phase2_pick.get("symbol", "")
                phase2_confidence = phase2_pick.get("confidence", 0.0)
            elif isinstance(phase2_pick, str):
                phase2_pick_symbol = phase2_pick
            else:
                phase2_pick_symbol = ""

            state_bundle = {
                "symbols": active_symbols,
                "fsv_states": fsv_states,
                "fsv_engine": self.fsv_engine,
                "fundamental_ranker": self.fundamental_ranker,
                "macro_alignment": self.macro_alignment,
                "symbol_comparator": self.symbol_comparator,
                "regime_modulator": self.regime_modulator,
            }

            ucf_output = self.ucf.compute(state_bundle)
            ucf_pick_symbol = ""
            ucf_confidence = 0.0
            if isinstance(ucf_output, dict):
                scores = ucf_output.get("scores", {})
                if scores:
                    ucf_pick_symbol = max(scores, key=lambda k: scores[k])
                    ucf_confidence = scores.get(ucf_pick_symbol, 0.0)
            elif isinstance(ucf_output, list):
                if ucf_output:
                    best_entry = max(ucf_output, key=lambda x: x.get("score", 0.0) if isinstance(x, dict) else 0.0)
                    if isinstance(best_entry, dict):
                        ucf_pick_symbol = best_entry.get("symbol", "")
                        ucf_confidence = best_entry.get("score", 0.0)

            phase2_entry = {
                "cycle": cycle_index,
                "pick": phase2_pick_symbol,
                "confidence": phase2_confidence,
                "candidates": top3_candidates,
            }
            phase2_results.append(phase2_entry)

            ucf_entry = {
                "cycle": cycle_index,
                "pick": ucf_pick_symbol,
                "confidence": ucf_confidence,
                "candidates": active_symbols,
            }
            ucf_results.append(ucf_entry)

        agreement_count = 0
        for p2, ucf in zip(phase2_results, ucf_results):
            if p2["pick"] == ucf["pick"] and p2["pick"] != "":
                agreement_count += 1

        total_cycles = num_cycles if num_cycles > 0 else 1
        agreement_rate = agreement_count / total_cycles
        divergence_rate = 1.0 - agreement_rate

        phase2_confidences = [r["confidence"] for r in phase2_results]
        ucf_confidences = [r["confidence"] for r in ucf_results]

        phase2_avg_confidence = statistics.mean(phase2_confidences) if phase2_confidences else 0.0
        ucf_avg_confidence = statistics.mean(ucf_confidences) if ucf_confidences else 0.0

        phase2_unique = len(set(r["pick"] for r in phase2_results if r["pick"]))
        ucf_unique = len(set(r["pick"] for r in ucf_results if r["pick"]))

        phase2_stability = self._compute_stability([r["pick"] for r in phase2_results])
        ucf_stability = self._compute_stability([r["pick"] for r in ucf_results])

        comparison = {
            "agreement_rate": round(agreement_rate, 4),
            "divergence_rate": round(divergence_rate, 4),
            "phase2_avg_confidence": round(phase2_avg_confidence, 4),
            "ucf_avg_confidence": round(ucf_avg_confidence, 4),
            "phase2_selection_diversity": float(phase2_unique),
            "ucf_selection_diversity": float(ucf_unique),
            "phase2_stability": round(phase2_stability, 4),
            "ucf_stability": round(ucf_stability, 4),
        }

        return {
            "cycles": num_cycles,
            "phase2_results": phase2_results,
            "ucf_results": ucf_results,
            "comparison": comparison,
        }

    def _compute_stability(self, picks: list[str]) -> float:
        if len(picks) < 2:
            return 0.0
        consecutive = 0
        for i in range(1, len(picks)):
            if picks[i] == picks[i - 1] and picks[i] != "":
                consecutive += 1
        return consecutive / (len(picks) - 1)

    def generate_comparison_report(self, comparison: dict) -> str:
        comp = comparison.get("comparison", {})
        lines: list[str] = []
        lines.append("=" * 64)
        lines.append("UCF BACKTEST COMPARISON REPORT")
        lines.append("=" * 64)
        lines.append(f"Total Cycles:          {comparison.get('cycles', 0)}")
        lines.append("")
        lines.append("--- Agreement Metrics ---")
        lines.append(f"  Agreement Rate:      {comp.get('agreement_rate', 0.0):.2%}")
        lines.append(f"  Divergence Rate:     {comp.get('divergence_rate', 0.0):.2%}")
        lines.append("")
        lines.append("--- Confidence Metrics ---")
        lines.append(f"  Phase 2 Avg Confidence: {comp.get('phase2_avg_confidence', 0.0):.4f}")
        lines.append(f"  UCF Avg Confidence:     {comp.get('ucf_avg_confidence', 0.0):.4f}")
        lines.append("")
        lines.append("--- Selection Diversity ---")
        lines.append(f"  Phase 2 Unique Picks:  {comp.get('phase2_selection_diversity', 0.0):.0f}")
        lines.append(f"  UCF Unique Picks:      {comp.get('ucf_selection_diversity', 0.0):.0f}")
        lines.append("")
        lines.append("--- Stability ---")
        lines.append(f"  Phase 2 Stability:     {comp.get('phase2_stability', 0.0):.4f}")
        lines.append(f"  UCF Stability:         {comp.get('ucf_stability', 0.0):.4f}")
        lines.append("=" * 64)
        return "\n".join(lines)

    def compute_divergence_heatmap(self, comparison: dict) -> list[dict]:
        phase2_results = comparison.get("phase2_results", [])
        ucf_results = comparison.get("ucf_results", [])

        all_symbols: set[str] = set()
        for r in phase2_results:
            s = r.get("pick", "")
            if s:
                all_symbols.add(s)
        for r in ucf_results:
            s = r.get("pick", "")
            if s:
                all_symbols.add(s)

        sorted_symbols = sorted(all_symbols)
        heatmap: list[dict] = []

        for symbol in sorted_symbols:
            phase2_picks = sum(1 for r in phase2_results if r.get("pick") == symbol)
            ucf_picks = sum(1 for r in ucf_results if r.get("pick") == symbol)
            overlap = 0
            for p2, u in zip(phase2_results, ucf_results):
                if p2.get("pick") == symbol and u.get("pick") == symbol:
                    overlap += 1
            total_phase2 = max(len(phase2_results), 1)
            divergence = 1.0 - (overlap / max(phase2_picks + ucf_picks - overlap, 1)) if (phase2_picks + ucf_picks - overlap) > 0 else 0.0

            heatmap.append({
                "symbol": symbol,
                "phase2_picks": phase2_picks,
                "ucf_picks": ucf_picks,
                "overlap": overlap,
                "divergence": round(divergence, 4),
            })

        return heatmap

    def evaluate_regime_sensitivity(self, comparison: dict) -> dict:
        phase2_results = comparison.get("phase2_results", [])
        ucf_results = comparison.get("ucf_results", [])

        regimes = ["bullish", "bearish", "neutral", "volatile", "calm"]
        regime_data: dict[str, dict[str, float]] = {}

        for regime in regimes:
            phase2_conf: list[float] = []
            ucf_conf: list[float] = []
            agreement_local = 0
            total_local = 0

            for p2, u in zip(phase2_results, ucf_results):
                if p2.get("regime", regime) == regime:
                    total_local += 1
                    phase2_conf.append(p2.get("confidence", 0.0))
                    ucf_conf.append(u.get("confidence", 0.0))
                    if p2.get("pick") == u.get("pick") and p2.get("pick", "") != "":
                        agreement_local += 1

            regime_data[regime] = {
                "phase2_avg_confidence": round(statistics.mean(phase2_conf), 4) if phase2_conf else 0.0,
                "ucf_avg_confidence": round(statistics.mean(ucf_conf), 4) if ucf_conf else 0.0,
                "agreement_rate": round(agreement_local / total_local, 4) if total_local > 0 else 0.0,
                "sample_count": float(total_local),
            }

        return regime_data

    def run_parameter_sweep(self, cycles_list: Optional[list[int]] = None) -> dict:
        if cycles_list is None:
            cycles_list = [10, 25, 50, 100, 200, 500]
        sweep_results: dict[int, dict] = {}

        for num_c in cycles_list:
            random.seed(42)
            result = self.run_comparison(num_cycles=num_c)
            comp = result.get("comparison", {})
            sweep_results[num_c] = {
                "agreement_rate": comp.get("agreement_rate", 0.0),
                "divergence_rate": comp.get("divergence_rate", 0.0),
                "phase2_avg_confidence": comp.get("phase2_avg_confidence", 0.0),
                "ucf_avg_confidence": comp.get("ucf_avg_confidence", 0.0),
                "phase2_selection_diversity": comp.get("phase2_selection_diversity", 0.0),
                "ucf_selection_diversity": comp.get("ucf_selection_diversity", 0.0),
                "phase2_stability": comp.get("phase2_stability", 0.0),
                "ucf_stability": comp.get("ucf_stability", 0.0),
            }

        metrics_over_cycles: dict[str, float] = {}
        metric_keys = [
            "agreement_rate", "divergence_rate",
            "phase2_avg_confidence", "ucf_avg_confidence",
            "phase2_selection_diversity", "ucf_selection_diversity",
            "phase2_stability", "ucf_stability",
        ]
        for key in metric_keys:
            values = [sweep_results[c][key] for c in sorted(sweep_results.keys())]
            if len(values) >= 2:
                metrics_over_cycles[key + "_stability"] = round(
                    1.0 - (statistics.stdev(values) / max(abs(statistics.mean(values)), 0.0001)), 4
                )
            else:
                metrics_over_cycles[key + "_stability"] = 1.0

        return {
            "sweep_results": {str(k): v for k, v in sweep_results.items()},
            "metrics_over_cycles": metrics_over_cycles,
        }
