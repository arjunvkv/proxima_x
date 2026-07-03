from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numba
from numpy.typing import NDArray

from research.mechanism_discovery.temporal_topology import TemporalTopology
from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.memory_landscape import MemoryLandscape
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality

from research.causal_physics.generator_graph import GeneratorGraphBuilder
from research.causal_physics.memory_physics import MemoryPhysicsAnalyzer
from research.causal_physics.survival_validator import GeneratorSurvivalValidator
from research.causal_physics.latent_driver_search import LatentDriverSearch
from research.causal_physics.market_physics_model import MarketPhysicsModel

try:
    from research.causal_physics.generator_discovery import GeneratorDiscoveryEngine
except ImportError:
    GeneratorDiscoveryEngine = None

try:
    from research.causal_physics.causal_ordering import CausalOrderingEngine
except ImportError:
    CausalOrderingEngine = None

try:
    from research.causal_physics.adaptive_time_origins import AdaptiveTimeOriginsAnalyzer
except ImportError:
    AdaptiveTimeOriginsAnalyzer = None

try:
    from research.causal_physics.mutation_origins import MutationOriginsAnalyzer
except ImportError:
    MutationOriginsAnalyzer = None

try:
    from research.causal_physics.energy_physics import EnergyPhysicsAnalyzer
except ImportError:
    EnergyPhysicsAnalyzer = None


@numba.jit(nopython=True, cache=True)
def _numba_mutation_rate(
    states: NDArray[np.int64], window: int
) -> NDArray[np.float64]:
    n = len(states)
    result = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        changes = 0
        for j in range(i - window + 1, i):
            if states[j] != states[j - 1]:
                changes += 1
        result[i] = float(changes) / float(window)
    return result


@numba.jit(nopython=True, cache=True)
def _numba_regime_prob(
    states: NDArray[np.int64], window: int
) -> NDArray[np.float64]:
    n = len(states)
    result = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        current = states[i]
        diff_count = 0
        for j in range(i - window, i):
            if states[j] != current:
                diff_count += 1
        result[i] = float(diff_count) / float(window)
    return result


class CausalPipeline:
    def __init__(self, data_dir: str = "data/market"):
        self.data_dir = Path(data_dir)
        self.tt = TemporalTopology()
        self.ed = EnergyDynamics()
        self.ml = MemoryLandscape()
        self._max_lag = 50
        self._candidates: list[dict] = []

    def run(
        self, asset: str, cross_assets: Optional[list[str]] = None
    ) -> dict[str, Any]:
        wall = time.time()
        timing: dict[str, float] = {}
        output: dict[str, Any] = {"asset": asset}

        t0 = time.time()
        data = self._load_data(asset)
        timing["load_data"] = time.time() - t0
        print(f"[{asset}] Loaded {len(data['price'])} rows ({timing['load_data']:.2f}s)")

        t0 = time.time()
        analysis_data = self._compute_signals(data)
        timing["compute_signals"] = time.time() - t0
        print(f"[{asset}] Signals computed ({timing['compute_signals']:.2f}s)")

        t0 = time.time()
        analysis_data["state_mutation_rate"] = _numba_mutation_rate(
            analysis_data["states"], 20
        )
        analysis_data["regime_change_probability"] = _numba_regime_prob(
            analysis_data["states"], 20
        )
        timing["derive_mutation_regime"] = time.time() - t0
        print(f"[{asset}] Mutation/regime derived ({timing['derive_mutation_regime']:.2f}s)")

        output["signals"] = {k: v for k, v in analysis_data.items()
                             if isinstance(v, np.ndarray)}

        t0 = time.time()
        generator_result = self._run_generator_discovery(analysis_data)
        timing["generator_discovery"] = time.time() - t0
        output["generator_discovery"] = generator_result

        raw_candidates = generator_result.get("candidates", [])
        self._candidates = [{
            "source": c.get("source_variable", c.get("source", "")),
            "target": c.get("target_variable", c.get("target", "")),
            "causal_strength": c.get("causal_strength", 0.0),
            "information_flow": c.get("information_flow", c.get("transfer_entropy", 0.0)),
            "peak_lag": c.get("peak_lag", 0),
            "peak_corr": c.get("peak_corr", 0.0),
        } for c in raw_candidates if isinstance(c, dict)]
        print(f"[{asset}] Generator discovery done ({timing['generator_discovery']:.2f}s)")

        t0 = time.time()
        ordering_result = self._run_causal_ordering(self._candidates)
        timing["causal_ordering"] = time.time() - t0
        output["causal_ordering"] = ordering_result
        print(f"[{asset}] Causal ordering done ({timing['causal_ordering']:.2f}s)")

        t0 = time.time()
        at_result = self._run_adaptive_time_origins(analysis_data)
        timing["adaptive_time_origins"] = time.time() - t0
        output["adaptive_time_origins"] = at_result
        print(f"[{asset}] AT origins done ({timing['adaptive_time_origins']:.2f}s)")

        t0 = time.time()
        mutation_result = self._run_mutation_origins(analysis_data)
        timing["mutation_origins"] = time.time() - t0
        output["mutation_origins"] = mutation_result
        print(f"[{asset}] Mutation origins done ({timing['mutation_origins']:.2f}s)")

        t0 = time.time()
        energy_result = self._run_energy_physics(analysis_data)
        timing["energy_physics"] = time.time() - t0
        output["energy_physics"] = energy_result
        print(f"[{asset}] Energy physics done ({timing['energy_physics']:.2f}s)")

        t0 = time.time()
        memory_result = self._run_memory_physics(analysis_data)
        timing["memory_physics"] = time.time() - t0
        output["memory_physics"] = memory_result
        print(f"[{asset}] Memory physics done ({timing['memory_physics']:.2f}s)")

        t0 = time.time()
        graph_result = self._run_generator_graph(analysis_data, ordering_result)
        timing["generator_graph"] = time.time() - t0
        output["generator_graph"] = graph_result
        print(f"[{asset}] Generator graph built ({timing['generator_graph']:.2f}s)")

        t0 = time.time()
        latent_result = self._run_latent_driver_search(analysis_data)
        timing["latent_driver_search"] = time.time() - t0
        output["latent_driver_search"] = latent_result
        print(f"[{asset}] Latent driver search done ({timing['latent_driver_search']:.2f}s)")

        t0 = time.time()
        model_result = self._run_market_physics_model(
            analysis_data, ordering_result
        )
        timing["market_physics_model"] = time.time() - t0
        output["market_physics_model"] = model_result
        print(f"[{asset}] Market physics model built ({timing['market_physics_model']:.2f}s)")

        t0 = time.time()
        survival_result = self._run_survival_validation(
            analysis_data, ordering_result
        )
        timing["survival_validation"] = time.time() - t0
        output["survival_validation"] = survival_result
        print(f"[{asset}] Survival validation done ({timing['survival_validation']:.2f}s)")

        timing["total"] = time.time() - wall
        output["timing"] = timing
        print(f"[{asset}] CPE total: {timing['total']:.2f}s")

        return output

    def _load_data(self, asset: str) -> dict[str, NDArray]:
        path = self.data_dir / f"{asset}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Data not found: {path}")
        import polars as pl
        df = pl.read_parquet(str(path))
        price = df["close"].to_numpy().astype(np.float64)
        returns = (
            df["log_return"].to_numpy().astype(np.float64)
            if "log_return" in df.columns
            else np.diff(np.log(price), prepend=np.log(price[0]))
        )
        volume = (
            df["volume"].to_numpy().astype(np.float64)
            if "volume" in df.columns
            else np.ones(len(price), dtype=np.float64)
        )
        high = df["high"].to_numpy().astype(np.float64) if "high" in df.columns else price.copy()
        low = df["low"].to_numpy().astype(np.float64) if "low" in df.columns else price.copy()
        return {"price": price, "returns": returns, "volume": volume, "high": high, "low": low}

    def _compute_signals(self, data: dict) -> dict[str, Any]:
        price = data["price"]
        returns = data["returns"]
        n = len(price)

        result_tt = self.tt.compute(data)
        result_ed = self.ed.compute(data)
        result_ml = self.ml.compute(data)

        vol = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            vol[i] = np.std(returns[i - 20 : i])

        states = result_tt.get("time_regime", np.zeros(n, dtype=np.int64)).astype(np.int64)

        analysis: dict[str, Any] = {
            "price": price,
            "returns": returns,
            "volume": data.get("volume", np.ones(n, dtype=np.float64)),
            "adaptive_time": result_tt.get("adaptive_time_coordinate", np.zeros(n)),
            "adaptive_time_coordinate": result_tt.get("adaptive_time_coordinate", np.zeros(n)),
            "time_density": result_tt.get("time_density", np.zeros(n)),
            "event_density": result_tt.get("event_density", np.zeros(n)),
            "information_density": result_tt.get("information_density", np.zeros(n)),
            "behavior_density": result_tt.get("behavior_density", np.zeros(n)),
            "time_regime": states,
            "states": states,
            "energy_creation": result_ed.get("energy_creation", np.zeros(n)),
            "energy_storage": result_ed.get("energy_storage", np.zeros(n)),
            "energy_release": result_ed.get("energy_release", np.zeros(n)),
            "energy_dissipation": result_ed.get("energy_dissipation", np.zeros(n)),
            "energy_balance": result_ed.get("energy_balance", np.zeros(n)),
            "energy_regime": result_ed.get("energy_regime", np.ones(n, dtype=np.int64)),
            "memory_density": result_ml.get("memory_density", np.zeros(n)),
            "memory_gradient": result_ml.get("memory_gradient", np.zeros(n)),
            "memory_interference": result_ml.get("memory_interference", np.zeros(n)),
            "memory_landscape": result_ml.get("memory_landscape", np.zeros(n)),
            "memory_regime": result_ml.get("memory_regime", np.zeros(n, dtype=np.int32)),
            "volatility": vol,
        }
        return analysis

    def _run_generator_discovery(self, data: dict) -> dict[str, Any]:
        if GeneratorDiscoveryEngine is not None:
            engine = GeneratorDiscoveryEngine()
            candidates = engine.compute(data)
            return {"candidates": [{
                "source_variable": c.source_variable,
                "target_variable": c.target_variable,
                "peak_lag": c.peak_lag,
                "peak_corr": c.peak_corr,
                "transfer_entropy": c.transfer_entropy,
                "causal_strength": c.causal_strength,
            } for c in candidates]}
        return {"status": "skipped", "reason": "GeneratorDiscoveryEngine not available"}

    def _run_causal_ordering(self, candidates: list[dict]) -> dict[str, Any]:
        if CausalOrderingEngine is not None:
            engine = CausalOrderingEngine()
            from research.causal_physics.generator_discovery import GeneratorCandidate
            gc_list = [GeneratorCandidate(
                source_variable=c["source"], target_variable=c["target"],
                peak_lag=c.get("peak_lag", 0), peak_corr=c.get("peak_corr", 0.0),
                transfer_entropy=c.get("information_flow", 0.0),
                causal_strength=c.get("causal_strength", 0.0),
            ) for c in candidates if c.get("source") and c.get("target")]
            if not gc_list:
                return {"status": "skipped", "reason": "no candidates"}
            return engine.compute(gc_list)
        return {"status": "skipped", "reason": "CausalOrderingEngine not available"}

    def _run_adaptive_time_origins(self, data: dict) -> dict[str, Any]:
        if AdaptiveTimeOriginsAnalyzer is not None:
            analyzer = AdaptiveTimeOriginsAnalyzer()
            return analyzer.compute(data)
        at = np.asarray(data.get("adaptive_time", []), dtype=np.float64)
        n = len(at)
        candidates: dict[str, NDArray[np.float64]] = {
            "returns": np.asarray(data.get("returns", np.zeros(n)), dtype=np.float64),
            "volatility": np.asarray(data.get("volatility", np.zeros(n)), dtype=np.float64),
            "energy_creation": np.asarray(data.get("energy_creation", np.zeros(n)), dtype=np.float64),
            "memory_density": np.asarray(data.get("memory_density", np.zeros(n)), dtype=np.float64),
        }
        best_lag = 0
        best_corr = -1.0
        best_gen = "none"
        for name, sig in candidates.items():
            if len(sig) < self._max_lag * 2 + 1:
                continue
            corr = AdaptiveTimeCausality._cross_correlate(at, sig, self._max_lag)
            peak_idx = int(np.argmax(np.abs(corr)))
            pc = float(corr[peak_idx])
            if abs(pc) > abs(best_corr):
                best_corr = pc
                best_lag = int(np.arange(-self._max_lag, self._max_lag + 1)[peak_idx])
                best_gen = name
        return {
            "status": "inline",
            "primary_generator": best_gen,
            "peak_lag": best_lag,
            "peak_corr": best_corr,
        }

    def _run_mutation_origins(self, data: dict) -> dict[str, Any]:
        if MutationOriginsAnalyzer is not None:
            analyzer = MutationOriginsAnalyzer()
            return analyzer.compute(data)
        smr = np.asarray(data.get("state_mutation_rate", np.zeros(1)), dtype=np.float64)
        n = len(smr)
        at = np.asarray(data.get("adaptive_time", np.zeros(n)), dtype=np.float64)
        es = np.asarray(data.get("energy_storage", np.zeros(n)), dtype=np.float64)
        md = np.asarray(data.get("memory_density", np.zeros(n)), dtype=np.float64)
        candidates = {"adaptive_time": at, "energy_storage": es, "memory_density": md}
        best_lag = 0
        best_corr = -1.0
        best_gen = "none"
        for name, sig in candidates.items():
            if len(sig) < self._max_lag * 2 + 1:
                continue
            corr = AdaptiveTimeCausality._cross_correlate(smr, sig, self._max_lag)
            peak_idx = int(np.argmax(np.abs(corr)))
            pc = float(corr[peak_idx])
            if abs(pc) > abs(best_corr):
                best_corr = pc
                best_lag = int(np.arange(-self._max_lag, self._max_lag + 1)[peak_idx])
                best_gen = name
        return {
            "status": "inline",
            "primary_driver": best_gen,
            "peak_lag": best_lag,
            "peak_corr": best_corr,
        }

    def _run_energy_physics(self, data: dict) -> dict[str, Any]:
        if EnergyPhysicsAnalyzer is not None:
            analyzer = EnergyPhysicsAnalyzer()
            price = np.asarray(data.get("price", np.zeros(1)), dtype=np.float64)
            returns = np.asarray(data.get("returns", np.zeros(1)), dtype=np.float64)
            return analyzer.compute(data, price, returns)
        es = np.asarray(data.get("energy_storage", np.zeros(1)), dtype=np.float64)
        n = len(es)
        ec = np.asarray(data.get("energy_creation", np.zeros(n)), dtype=np.float64)
        eb = np.asarray(data.get("energy_balance", np.zeros(n)), dtype=np.float64)
        at = np.asarray(data.get("adaptive_time", np.zeros(n)), dtype=np.float64)
        return {
            "status": "inline",
            "energy_creation_mean": float(np.mean(ec)),
            "energy_balance_mean": float(np.mean(eb)),
            "storage_dissipation_ratio": float(np.mean(es) / max(np.mean(np.abs(eb)), 1e-12)),
            "adaptive_time_corr": float(np.corrcoef(at, es)[0, 1]) if n > 1 else 0.0,
        }

    def _run_memory_physics(self, data: dict) -> dict[str, Any]:
        analyzer = MemoryPhysicsAnalyzer(max_lag=self._max_lag)
        return analyzer.compute(data)

    def _run_generator_graph(
        self, data: dict, ordering_result: dict
    ) -> dict[str, Any]:
        ordering = ordering_result.get("ordering", {}) if isinstance(ordering_result, dict) else {}
        candidates = list(self._candidates)
        if not candidates:
            n = len(data.get("adaptive_time", np.zeros(1)))
            candidates = [
                {"source": "adaptive_time", "target": "state_mutation_rate",
                 "causal_strength": 0.3, "information_flow": 0.2},
                {"source": "state_mutation_rate", "target": "regime_change_probability",
                 "causal_strength": 0.3, "information_flow": 0.2},
                {"source": "energy_storage", "target": "adaptive_time",
                 "causal_strength": 0.2, "information_flow": 0.15},
                {"source": "memory_density", "target": "adaptive_time",
                 "causal_strength": 0.2, "information_flow": 0.15},
            ]
        builder = GeneratorGraphBuilder()
        graph = builder.build(candidates, ordering)
        chain = graph.get_market_physics_chain()
        return {"graph": graph.to_dict(), "physics_chain": chain}

    def _run_latent_driver_search(self, data: dict) -> dict[str, Any]:
        searcher = LatentDriverSearch()
        return searcher.search(data)

    def _run_market_physics_model(
        self, data: dict, ordering_result: dict
    ) -> dict[str, Any]:
        candidates = self._candidates
        model = MarketPhysicsModel()
        hierarchy = model.build_hierarchy(data, candidates)
        tested = model.test_hierarchy(data, hierarchy)
        best = model.get_best_model()
        return {
            "built_hierarchy": hierarchy,
            "tested": tested,
            "best_model": best,
        }

    def _run_survival_validation(
        self, data: dict, ordering_result: dict
    ) -> dict[str, Any]:
        candidates = list(self._candidates)
        if not candidates:
            n = len(data.get("adaptive_time", np.zeros(1)))
            at = np.asarray(data.get("adaptive_time", np.zeros(n)), dtype=np.float64)
            smr = np.asarray(data.get("state_mutation_rate", np.zeros(n)), dtype=np.float64)
            rcp = np.asarray(data.get("regime_change_probability", np.zeros(n)), dtype=np.float64)
            ret = np.asarray(data.get("returns", np.zeros(n)), dtype=np.float64)
            candidates = [
                {"source": "adaptive_time", "target": "state_mutation_rate",
                 "causal_strength": float(np.corrcoef(at[:min(n, 100)], smr[:min(n, 100)])[0, 1]) if n > 1 else 0.0},
                {"source": "state_mutation_rate", "target": "regime_change_probability",
                 "causal_strength": float(np.corrcoef(smr[:min(n, 100)], rcp[:min(n, 100)])[0, 1]) if n > 1 else 0.0},
            ]
        validator = GeneratorSurvivalValidator()
        results = validator.validate_single(candidates, data)
        return {
            "survival_results": {
                k: {
                    "generator": r.generator,
                    "survival_probability": r.survival_probability,
                    "validated": r.validated,
                    "bootstrap_stability": r.bootstrap_stability,
                    "noise_stability": r.noise_stability,
                    "regime_split_consistency": r.regime_split_consistency,
                }
                for k, r in results.items()
            },
        }
