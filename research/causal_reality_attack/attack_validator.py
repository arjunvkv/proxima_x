from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numba
from numpy.typing import NDArray

from research.mechanism_discovery.temporal_topology import TemporalTopology
from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.memory_landscape import MemoryLandscape
from research.causal_physics.generator_discovery import GeneratorDiscoveryEngine, GeneratorCandidate
from research.causal_physics.causal_ordering import CausalOrderingEngine
from research.causal_physics.generator_graph import GeneratorGraphBuilder, GeneratorGraph
from research.causal_physics.market_physics_model import MarketPhysicsModel
from research.causal_physics.survival_validator import GeneratorSurvivalValidator
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality


@numba.jit(nopython=True, cache=True)
def _numba_mutation_rate(states: NDArray[np.int64], window: int) -> NDArray[np.float64]:
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
def _numba_regime_prob(states: NDArray[np.int64], window: int) -> NDArray[np.float64]:
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


@dataclass
class AttackResult:
    attack_name: str
    status: str  # PASSED, FAILED, INCONCLUSIVE
    metrics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_name": self.attack_name,
            "status": self.status,
            "metrics": _clean_serializable(self.metrics),
            "details": _clean_serializable(self.details),
        }


def _clean_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return list(_clean_serializable(v) for v in obj)
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return obj


TARGET_ASSETS = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD"]
TARGET_VARIABLES = [
    "adaptive_time", "energy_storage", "memory_density",
    "state_mutation_rate", "regime_change_probability",
]
CANONICAL_CHAIN = ["compression", "energy_storage", "memory_density", "adaptive_time", "state_mutation_rate", "regime_change_probability"]


class AttackValidator:
    """Shared utilities for all attacks."""

    def __init__(self, data_dir: str = "data/market"):
        self.data_dir = Path(data_dir)
        self.tt = TemporalTopology()
        self.ed = EnergyDynamics()
        self.ml = MemoryLandscape()
        self._max_lag = 50

    def load_asset_data(self, asset: str) -> dict[str, NDArray]:
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

    def load_data_window(self, asset: str, start: str, end: str) -> dict[str, NDArray]:
        path = self.data_dir / f"{asset}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Data not found: {path}")
        import polars as pl
        import datetime
        df = pl.read_parquet(str(path))
        dt_start = datetime.datetime.strptime(start, "%Y-%m-%d")
        dt_end = datetime.datetime.strptime(end, "%Y-%m-%d")
        df = df.filter((pl.col("timestamp") >= dt_start) & (pl.col("timestamp") < dt_end))
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

    def compute_signals(self, data: dict) -> dict[str, Any]:
        price = data["price"]
        returns = data["returns"]
        n = len(price)

        result_tt = self.tt.compute(data)
        result_ed = self.ed.compute(data)
        result_ml = self.ml.compute(data)

        vol = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            vol[i] = np.std(returns[i - 20:i])

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

        analysis["state_mutation_rate"] = _numba_mutation_rate(states, 20)
        analysis["regime_change_probability"] = _numba_regime_prob(states, 20)

        return analysis

    def build_causal_graph(self, analysis_data: dict) -> tuple[GeneratorGraph, list[dict], dict]:
        engine = GeneratorDiscoveryEngine()
        candidates = engine.compute(analysis_data)

        candidate_dicts = [{
            "source": c.source_variable,
            "target": c.target_variable,
            "causal_strength": c.causal_strength,
            "information_flow": c.transfer_entropy,
            "peak_lag": c.peak_lag,
            "peak_corr": c.peak_corr,
        } for c in candidates]

        order_engine = CausalOrderingEngine()
        ordering_result = order_engine.compute(candidates)
        ordering = ordering_result.get("adjacency_matrix", {})

        builder = GeneratorGraphBuilder()
        graph = builder.build(candidate_dicts, ordering)
        chain = graph.get_market_physics_chain()

        return graph, candidate_dicts, ordering

    def build_graph_with_removed_vars(self, analysis_data: dict, removed_vars: set[str]) -> tuple[GeneratorGraph, list[dict]]:
        engine = GeneratorDiscoveryEngine()
        all_candidates = engine.compute(analysis_data)

        filtered = [c for c in all_candidates
                    if c.source_variable not in removed_vars
                    and c.target_variable not in removed_vars]

        candidate_dicts = [{
            "source": c.source_variable,
            "target": c.target_variable,
            "causal_strength": c.causal_strength,
            "information_flow": c.transfer_entropy,
            "peak_lag": c.peak_lag,
            "peak_corr": c.peak_corr,
        } for c in filtered]

        if not filtered:
            return GeneratorGraph(
                nodes=[n for n in TARGET_VARIABLES if n not in removed_vars],
                edges=[], topological_order=[]
            ), candidate_dicts

        order_engine = CausalOrderingEngine()
        ordering_result = order_engine.compute(filtered)
        ordering = ordering_result.get("adjacency_matrix", {})

        builder = GeneratorGraphBuilder()
        graph = builder.build(candidate_dicts, ordering)
        return graph, candidate_dicts

    def graph_similarity(self, g1: GeneratorGraph, g2: GeneratorGraph) -> dict[str, float]:
        nodes1, nodes2 = set(g1.nodes), set(g2.nodes)
        common_nodes = nodes1 & nodes2
        all_nodes = nodes1 | nodes2
        node_jaccard = len(common_nodes) / max(len(all_nodes), 1)

        edges1 = {(e.source, e.target) for e in g1.edges}
        edges2 = {(e.source, e.target) for e in g2.edges}
        common_edges = edges1 & edges2
        all_edges = edges1 | edges2
        edge_jaccard = len(common_edges) / max(len(all_edges), 1)

        order1 = {n: i for i, n in enumerate(g1.topological_order)}
        order2 = {n: i for i, n in enumerate(g2.topological_order)}
        common = [n for n in common_nodes if n in order1 and n in order2]
        if len(common) > 1:
            ranks1 = np.array([order1[n] for n in common], dtype=np.float64)
            ranks2 = np.array([order2[n] for n in common], dtype=np.float64)
            order_sim = float(np.corrcoef(ranks1, ranks2)[0, 1]) if np.std(ranks1) > 0 and np.std(ranks2) > 0 else 0.0
        else:
            order_sim = 0.0

        strength_sim = 0.0
        common_edge_list = common_edges
        if common_edge_list:
            s1_vals = []
            s2_vals = []
            e1_map = {(e.source, e.target): e.causal_strength for e in g1.edges}
            e2_map = {(e.source, e.target): e.causal_strength for e in g2.edges}
            for e in common_edge_list:
                s1_vals.append(e1_map.get(e, 0.0))
                s2_vals.append(e2_map.get(e, 0.0))
            if len(s1_vals) > 1 and np.std(s1_vals) > 0 and np.std(s2_vals) > 0:
                strength_sim = float(np.corrcoef(s1_vals, s2_vals)[0, 1])

        return {
            "node_jaccard": float(node_jaccard),
            "edge_jaccard": float(edge_jaccard),
            "order_similarity": float(order_sim),
            "strength_similarity": float(strength_sim),
        }

    def graph_information_score(self, graph: GeneratorGraph) -> float:
        if not graph.edges:
            return 0.0
        n_nodes = len(graph.nodes)

        strengths = np.array([abs(e.causal_strength) for e in graph.edges], dtype=np.float64)
        s_sum = np.sum(strengths) + 1e-12
        p = strengths / s_sum
        weight_entropy = -np.sum(p * np.log(p + 1e-12))
        weight_entropy_norm = weight_entropy / max(np.log(len(strengths) + 1e-12), 1e-12)

        node_centrality = {n: 0.0 for n in graph.nodes}
        for e in graph.edges:
            node_centrality[e.source] += abs(e.causal_strength)
            node_centrality[e.target] += abs(e.causal_strength)
        centrality = np.array(list(node_centrality.values()), dtype=np.float64)
        cent_mean = float(np.mean(centrality)) + 1e-12
        centrality_var = float(np.var(centrality)) / cent_mean

        adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        node_index = {n: i for i, n in enumerate(graph.nodes)}
        for e in graph.edges:
            adj[node_index[e.source], node_index[e.target]] = abs(e.causal_strength)
        eigenvals = np.linalg.eigvals(adj)
        spectral_spread = float(np.std(np.real(eigenvals))) / max(float(np.mean(np.abs(eigenvals)) + 1e-12), 1e-12)

        asymmetry = len(set((e.source, e.target) for e in graph.edges)) / max(len(graph.edges), 1)

        return (
            0.30 * weight_entropy_norm +
            0.30 * min(centrality_var, 1.0) +
            0.25 * min(spectral_spread, 1.0) +
            0.15 * asymmetry
        )

    def explained_variance_removed(self, analysis_data: dict,
                                   target_var: str, removed_source: str) -> float:
        target = np.asarray(analysis_data.get(target_var, np.zeros(1)), dtype=np.float64)
        source = np.asarray(analysis_data.get(removed_source, np.zeros(1)), dtype=np.float64)

        n = min(len(target), len(source))
        if n < self._max_lag * 2 + 1:
            return 1.0  # no change detectable

        corr_with = AdaptiveTimeCausality._cross_correlate(source[:n], target[:n], self._max_lag)
        peak_with = float(np.max(np.abs(corr_with)))

        return 1.0 - abs(peak_with)

    def information_flow_between(self, analysis_data: dict, source: str, target: str) -> float:
        from research.information_discovery.mi_estimator import _fast_conditional_mutual_info
        s = np.asarray(analysis_data.get(source, np.zeros(1)), dtype=np.float64)
        t = np.asarray(analysis_data.get(target, np.zeros(1)), dtype=np.float64)
        n = min(len(s), len(t))
        if n < 3:
            return 0.0
        return float(_fast_conditional_mutual_info(s[:n - 1], t[1:n], t[:n - 1], 20))


VERDICT_MAP = {
    "cross_asset": {"PASSED": "", "FAILED": "asset_specific"},
    "cross_time": {"PASSED": "", "FAILED": "regime_specific"},
    "node_removal": {"PASSED": "", "FAILED": "redundant"},
    "mediator": {"PASSED": "", "FAILED": "decorative"},
    "random_graph": {"PASSED": "", "FAILED": "not_special"},
    "bootstrap": {"PASSED": "", "FAILED": "unstable"},
    "noise": {"PASSED": "", "FAILED": "fragile"},
    "hidden_variable": {"PASSED": "", "FAILED": "replaceable"},
    "chain_collapse": {"PASSED": "", "FAILED": "not_essential"},
}
