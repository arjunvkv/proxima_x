from ...simulation.synthetic_event_generator import SyntheticMacroGenerator
from ...core.fsv_engine import FSVEngine
from ...core.fsv_schema import FundamentalStateVector, NormalizedEvent, neutral_fsv
from ...integration.fsv_modulator import FSVModulator
from ..core.fundamental_ranker import FundamentalRanker
from ..core.macro_alignment import MacroAlignmentEngine
from ..core.symbol_comparator import FundamentalComparator
from ..integration.fundamental_selector import FundamentalSelector
from ..ingestion.macro_snapshot import MacroSnapshotEngine

import time
import math
import random
import statistics


class FundamentalBacktest:

    def __init__(self) -> None:
        self.generator = SyntheticMacroGenerator()
        self.engine = FSVEngine()
        self.selector = FundamentalSelector()
        self.results: dict = {}

    def run_backtest(self, num_cycles: int = 100, symbols: list[str] = None) -> dict:
        if symbols is None:
            symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF"]

        results: list[dict] = []
        base_time: float = time.time()

        for cycle in range(num_cycles):
            num_events = random.randint(2, 5)
            events: list[NormalizedEvent] = []
            for _ in range(num_events):
                sym = random.choice(symbols)
                raw_event = self.generator.generate_event(symbol=sym, time_offset=cycle * 60)
                events.append(raw_event)

            for event in events:
                self.engine.update_with_event(event)

            random.shuffle(symbols)
            technical_top3 = symbols[:3]

            fsves: dict[str, FundamentalStateVector] = {}
            for s in technical_top3:
                fsves[s] = self.engine.get_state(s, time.time())

            directions: dict[str, int] = {s: random.choice([1, -1]) for s in technical_top3}
            convictions: dict[str, float] = {s: random.uniform(0.4, 0.8) for s in technical_top3}

            try:
                selection_result = self.selector.select_best(technical_top3, fsves, directions, convictions)
            except Exception:
                selection_result = self.selector.select_with_fallback(technical_top3, fsves, directions, convictions)

            selected_symbol = selection_result.get("selected_symbol", technical_top3[0])
            confidence = selection_result.get("recommendation", {}).get("confidence", 0.0)
            ranking_vector = selection_result.get("ranking_vector", {})
            accuracy_events = self._compute_cycle_accuracy(events, selected_symbol, technical_top3)

            snapshot = MacroSnapshotEngine(self.engine)
            environment = snapshot.compute_environment(fsves)

            cycle_result = {
                "cycle": cycle,
                "technical_top3": technical_top3,
                "selected_symbol": selected_symbol,
                "confidence": confidence,
                "ranking_vector": ranking_vector,
                "environment": environment,
                "accuracy_events": accuracy_events
            }
            results.append(cycle_result)

        output: dict = {
            "cycles": num_cycles,
            "symbols": symbols,
            "results": results
        }
        self.results = output
        return output

    def _compute_cycle_accuracy(self, events: list[NormalizedEvent], selected: str, top3: list[str]) -> dict:
        positive_counts: dict[str, int] = {}
        for event in events:
            if event.surprise_score > 0:
                positive_counts[event.symbol] = positive_counts.get(event.symbol, 0) + 1

        best_symbol = max(positive_counts, key=positive_counts.get) if positive_counts else top3[0]

        return {
            "best_by_events": best_symbol,
            "selected": selected,
            "match": selected == best_symbol,
            "positive_event_counts": positive_counts.copy()
        }

    def evaluate_accuracy(self, backtest_results: dict, ground_truth: dict[str, float] = None) -> dict:
        results = backtest_results.get("results", [])
        total_cycles = len(results)

        if total_cycles == 0:
            return {
                "selection_accuracy": 0.0,
                "random_baseline": 0.0,
                "lift_over_random": 0.0,
                "consistency_score": 0.0,
                "best_symbol_count": {},
                "regime_accuracy_breakdown": {
                    "risk_on": 0.0, "risk_off": 0.0, "neutral": 0.0, "transition": 0.0
                },
                "total_cycles": 0
            }

        correct_count = 0
        regime_correct: dict[str, list[bool]] = {
            "risk_on": [], "risk_off": [], "neutral": [], "transition": []
        }
        selection_frequency: dict[str, int] = {}

        for cycle_result in results:
            selected = cycle_result.get("selected_symbol", "")
            selection_frequency[selected] = selection_frequency.get(selected, 0) + 1

            accuracy_events = cycle_result.get("accuracy_events", {})
            best_by_events = accuracy_events.get("best_by_events", "")

            if ground_truth is not None:
                best_symbol = max(ground_truth, key=ground_truth.get)
            else:
                best_symbol = best_by_events

            is_correct = selected == best_symbol
            if is_correct:
                correct_count += 1

        selection_accuracy = correct_count / total_cycles if total_cycles > 0 else 0.0
        random_baseline = 1.0 / 3.0
        lift_over_random = selection_accuracy - random_baseline

        if len(selection_frequency) <= 1:
            consistency_score = 1.0
        else:
            frequencies = list(selection_frequency.values())
            total_selections = sum(frequencies)
            proportions = [f / total_selections for f in frequencies]
            consistency_score = 1.0 - statistics.stdev(proportions) if len(proportions) > 1 else 1.0

        regime_accuracy_breakdown: dict[str, float] = {}
        for regime_name, outcomes in regime_correct.items():
            regime_accuracy_breakdown[regime_name] = sum(outcomes) / len(outcomes) if outcomes else 0.0

        return {
            "selection_accuracy": selection_accuracy,
            "random_baseline": random_baseline,
            "lift_over_random": lift_over_random,
            "consistency_score": consistency_score,
            "best_symbol_count": selection_frequency.copy(),
            "regime_accuracy_breakdown": regime_accuracy_breakdown,
            "total_cycles": total_cycles
        }

    def compare_strategies(self, num_cycles: int = 200) -> dict:
        backtest_results = self.run_backtest(num_cycles=num_cycles)
        results = backtest_results.get("results", [])

        strategy_a_selections: list[str] = []
        strategy_b_selections: list[str] = []
        strategy_b_confidence: list[float] = []
        strategy_a_diversity: set[str] = set()
        strategy_b_diversity: set[str] = set()

        for cycle_result in results:
            top3 = cycle_result.get("technical_top3", [])
            selected_b = cycle_result.get("selected_symbol", "")
            confidence_b = cycle_result.get("confidence", 0.0)

            selected_a = top3[0] if top3 else ""
            strategy_a_selections.append(selected_a)
            strategy_b_selections.append(selected_b)
            strategy_a_diversity.add(selected_a)
            strategy_b_diversity.add(selected_b)
            strategy_b_confidence.append(confidence_b)

        selection_diversity_a = len(strategy_a_diversity)
        selection_diversity_b = len(strategy_b_diversity)

        confidence_variance = statistics.variance(strategy_b_confidence) if len(strategy_b_confidence) > 1 else 0.0

        noise_stability = self._compute_noise_stability(results)

        return {
            "strategy_a": {
                "name": "technical_only",
                "selection_diversity": selection_diversity_a,
                "selections": strategy_a_selections
            },
            "strategy_b": {
                "name": "technical_plus_fundamental",
                "selection_diversity": selection_diversity_b,
                "confidence_variance": confidence_variance,
                "selections": strategy_b_selections
            },
            "stability_under_noise": noise_stability,
            "total_cycles": num_cycles
        }

    def _compute_noise_stability(self, results: list[dict]) -> float:
        if len(results) < 2:
            return 1.0

        flips = 0
        for i in range(1, len(results)):
            prev_selected = results[i - 1].get("selected_symbol", "")
            curr_selected = results[i].get("selected_symbol", "")
            prev_conf = results[i - 1].get("confidence", 0.0)
            curr_conf = results[i].get("confidence", 0.0)

            if prev_selected != curr_selected and abs(curr_conf - prev_conf) < 0.1:
                flips += 1

        max_possible_flips = len(results) - 1
        stability = 1.0 - (flips / max_possible_flips) if max_possible_flips > 0 else 1.0
        return stability

    def compute_selection_metrics(self, backtest_results: dict) -> dict:
        results = backtest_results.get("results", [])
        total_cycles = len(results)

        if total_cycles == 0:
            return {"selection_stability": 0.0, "concentration": 0.0, "turnover_rate": 0.0}

        changes = 0
        symbol_counts: dict[str, int] = {}
        previous_symbol = ""

        for i, cycle_result in enumerate(results):
            symbol = cycle_result.get("selected_symbol", "")
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

            if i > 0 and symbol != previous_symbol:
                changes += 1
            previous_symbol = symbol

        selection_stability = 1.0 - (changes / (total_cycles - 1)) if total_cycles > 1 else 1.0

        total_selections = sum(symbol_counts.values())
        concentration = max(symbol_counts.values()) / total_selections if total_selections > 0 else 0.0
        turnover_rate = changes / total_cycles if total_cycles > 0 else 0.0

        return {
            "selection_stability": selection_stability,
            "concentration": concentration,
            "turnover_rate": turnover_rate
        }

    def generate_report(self, num_cycles: int = 100) -> str:
        backtest_results = self.run_backtest(num_cycles=num_cycles)
        accuracy = self.evaluate_accuracy(backtest_results)
        metrics = self.compute_selection_metrics(backtest_results)
        comparison = self.compare_strategies(num_cycles=num_cycles)

        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("FUNDAMENTAL BACKTEST REPORT")
        lines.append("=" * 70)
        lines.append(f"Number of Cycles:      {backtest_results.get('cycles', 0)}")
        lines.append(f"Total Symbols:         {len(backtest_results.get('symbols', []))}")
        lines.append(f"Symbols:               {', '.join(backtest_results.get('symbols', []))}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("ACCURACY ANALYSIS")
        lines.append("-" * 70)
        lines.append(f"Selection Accuracy:     {accuracy.get('selection_accuracy', 0.0):.2%}")
        lines.append(f"Random Baseline:        {accuracy.get('random_baseline', 0.0):.2%}")
        lines.append(f"Lift Over Random:       {accuracy.get('lift_over_random', 0.0):+.2%}")
        lines.append(f"Consistency Score:      {accuracy.get('consistency_score', 0.0):.4f}")
        lines.append(f"Total Cycles Evaluated: {accuracy.get('total_cycles', 0)}")
        lines.append("")
        lines.append("Regime Accuracy Breakdown:")
        regime_breakdown = accuracy.get("regime_accuracy_breakdown", {})
        for regime_name, regime_acc in regime_breakdown.items():
            lines.append(f"  {regime_name:12s}: {regime_acc:.2%}")
        lines.append("")
        lines.append("Best Symbol Selection Count:")
        best_symbol_count = accuracy.get("best_symbol_count", {})
        for symbol, count in sorted(best_symbol_count.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {symbol:6s}: {count}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("STRATEGY COMPARISON")
        lines.append("-" * 70)
        strat_a = comparison.get("strategy_a", {})
        strat_b = comparison.get("strategy_b", {})
        lines.append(f"Strategy A (Technical Only):")
        lines.append(f"  Selection Diversity:     {strat_a.get('selection_diversity', 0)}")
        lines.append(f"Strategy B (Technical + Fundamental):")
        lines.append(f"  Selection Diversity:     {strat_b.get('selection_diversity', 0)}")
        lines.append(f"  Confidence Variance:     {strat_b.get('confidence_variance', 0.0):.6f}")
        lines.append(f"Stability Under Noise:     {comparison.get('stability_under_noise', 0.0):.4f}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("SELECTION METRICS")
        lines.append("-" * 70)
        lines.append(f"Selection Stability:      {metrics.get('selection_stability', 0.0):.4f}")
        lines.append(f"Concentration:            {metrics.get('concentration', 0.0):.4f}")
        lines.append(f"Turnover Rate:            {metrics.get('turnover_rate', 0.0):.4f}")
        lines.append("")
        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)
