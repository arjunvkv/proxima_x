from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import polars as pl
from numpy.typing import NDArray

from research.memory_field import MemoryFieldResearch
from research.temporal_dna import TemporalDNAResearch
from research.information_pressure import InformationPressureResearch
from research.liquidity_migration import LiquidityMigrationResearch
from research.cohort_simulation import CohortSimulationResearch
from research.state_entanglement import StateEntanglementResearch
from research.tension_tensor import TensionTensorResearch
from research.behavioral_echoes import BehavioralEchoesResearch
from research.state_discovery import StateDiscoveryEngine
from research.state_compressor import StateCompressor
from research.novel_states import NovelStateGenerator
from research.sid_calculator import SIDCalculator
from research.forward_analyzer import ForwardAnalyzer
from research.persistence_analyzer import PersistenceAnalyzer
from research.transition_analyzer_v2 import TransitionGraphAnalyzer
from research.cross_validator import CrossValidator
from ml.clustering import StateClusterer
from utils.serialization import save_json, load_json
from config.settings import settings
from research.information_discovery.discovery_pipeline import DiscoveryPipeline as InfoDiscoveryPipeline
from research.mechanism_discovery.mechanism_pipeline import MechanismPipeline as MDEPipeline


class ResearchPipeline:
    def __init__(self, output_dir: str | Path = "research/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, Any] = {}

    def _collect_features(self, module_features: dict, discovery_features: dict, novel_state_vec: NDArray | None) -> dict[str, NDArray]:
        features: dict[str, NDArray] = {}
        for module_name, result in module_features.items():
            if not isinstance(result, dict):
                continue
            for key, value in result.items():
                if isinstance(value, np.ndarray) and value.ndim == 1:
                    features[f"{module_name}_{key}"] = value
        for key, value in discovery_features.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    features[f"discovery_{key}"] = value
                else:
                    for i in range(value.shape[1]):
                        features[f"discovery_{key}_{i}"] = value[:, i]
        if novel_state_vec is not None and novel_state_vec.ndim == 2:
            for i in range(novel_state_vec.shape[1]):
                features[f"novel_state_{i}"] = novel_state_vec[:, i]
        return features

    def _build_feature_matrix(self, features: dict[str, NDArray]) -> NDArray[np.float32]:
        arrays = []
        min_len = min(arr.shape[0] for arr in features.values() if arr.ndim >= 1)
        for arr in features.values():
            if arr.ndim == 1:
                arrays.append(arr[:min_len].reshape(-1, 1))
            elif arr.ndim == 2:
                arrays.append(arr[:min_len])
        return np.hstack(arrays).astype(np.float32)

    def run_discovery_cycle(self, data: dict) -> dict:
        price = data["price"].astype(np.float64)
        high = data["high"].astype(np.float64)
        low = data["low"].astype(np.float64)
        returns = data["returns"].astype(np.float64)
        volume = data["volume"].astype(np.float64)
        n = len(price)

        modules: dict[str, dict] = {}

        mf = MemoryFieldResearch()
        modules["memory_field"] = mf.compute_all(price, returns)

        dna = TemporalDNAResearch()
        ohlc = {"high": high, "low": low, "close": price, "volume": volume, "returns": returns}
        modules["temporal_dna"] = dna.compute_all(ohlc)

        ip = InformationPressureResearch()
        modules["information_pressure"] = ip.compute_all(returns, volume)

        lm = LiquidityMigrationResearch()
        modules["liquidity_migration"] = lm.compute_all(price)

        cs = CohortSimulationResearch()
        cs_res = cs.compute_all(volume, price, returns)
        flat_cs = {}
        for k, v in cs_res.items():
            if v.ndim == 1:
                flat_cs[k] = v
            else:
                for i in range(v.shape[1]):
                    flat_cs[f"{k}_{i}"] = v[:, i]
        modules["cohort_simulation"] = flat_cs

        tf_returns = {tf: returns for tf in StateEntanglementResearch.TIMEFRAMES}
        se = StateEntanglementResearch()
        se_res = se.compute_all(tf_returns)
        flat_se = {}
        for k, v in se_res.items():
            if v.ndim == 1:
                flat_se[k] = v
        modules["state_entanglement"] = flat_se

        tension_memory = modules["memory_field"].get("memory_strength", np.zeros(n, dtype=np.float32))
        tension_pressure = modules["information_pressure"].get("pressure_build", np.zeros(n, dtype=np.float32))
        tension_liquidity = modules["liquidity_migration"].get("liquidity_mass", np.zeros(n, dtype=np.float32))
        tension_cohort = modules["cohort_simulation"].get("cohort_alignment", np.zeros(n, dtype=np.float32))
        tension_vol = modules["temporal_dna"].get("volatility", np.zeros(n, dtype=np.float32))
        tension_inputs = {
            "memory": tension_memory,
            "pressure": tension_pressure,
            "liquidity": tension_liquidity,
            "cohort": tension_cohort,
            "volatility": tension_vol,
            "state_alignment": np.ones(n, dtype=np.float32),
        }
        tt = TensionTensorResearch()
        modules["tension_tensor"] = tt.compute_all(tension_inputs)

        modules["behavioral_echoes"] = {}

        discovery = StateDiscoveryEngine()
        discovery_features = discovery.compute_all_features(price, high, low, returns, volume)

        generator = NovelStateGenerator()
        novel_state_vec = generator.compute_novel_state_vector(price, high, low, returns, volume)

        all_features = self._collect_features(modules, discovery_features, novel_state_vec)
        feature_names = list(all_features.keys())
        hybrid_vector = self._build_feature_matrix(all_features)

        trim_start = 0
        row_sums = np.sum(np.abs(hybrid_vector), axis=1)
        nonzero_idx = np.argmax(row_sums > 1e-8)
        if nonzero_idx > 0:
            trim_start = nonzero_idx
            hybrid_vector = hybrid_vector[trim_start:]

        compressor = StateCompressor(method="umap", n_components=min(20, hybrid_vector.shape[1] - 1))
        try:
            compressed = compressor.fit_transform(hybrid_vector)
        except Exception:
            compressor = StateCompressor(method="pca", n_components=min(20, hybrid_vector.shape[1] - 1))
            compressed = compressor.fit_transform(hybrid_vector)

        mcs = max(5, min(30, len(compressed) // 40))
        clusterer = StateClusterer(method="hdbscan", params={"min_cluster_size": mcs})
        labels = clusterer.fit_predict(compressed)

        noise_mask = labels != -1
        clean_labels = labels[noise_mask]
        unique_clean = np.unique(clean_labels)
        label_map = {old: new for new, old in enumerate(sorted(unique_clean))}
        remapped = np.full_like(labels, -1)
        for old, new in label_map.items():
            remapped[labels == old] = new
        labels = remapped
        n_clusters = int(np.sum(np.unique(labels) >= 0))

        self._state["compressed_states"] = compressed
        self._state["cluster_labels"] = labels
        self._state["n_clusters"] = n_clusters

        return {
            "raw_state_vector": hybrid_vector,
            "compressed_states": compressed,
            "cluster_labels": labels,
            "n_clusters": n_clusters,
            "feature_names": feature_names,
        }

    def run_validation_cycle(self, states: NDArray[np.int32], price: NDArray[np.float64], returns: NDArray[np.float64]) -> dict:
        valid_mask = states != -1
        valid_states = states[valid_mask]

        fa = ForwardAnalyzer(horizons=[1, 5, 20, 50])
        forward_metrics = fa.compute_all_forward_metrics(price, returns)

        sid_data = {}
        for h in [1, 5, 20, 50]:
            key = f"forward_return_{h}"
            if key in forward_metrics:
                fm = forward_metrics[key][valid_mask]
                valid_mask2 = ~np.isnan(fm)
                sid_data[f"forward_returns_{h}"] = fm[valid_mask2]
                vs = valid_states[valid_mask2]
                sid_data[f"forward_returns_{h}_states"] = vs

        sid_calc = SIDCalculator(n_bins=20)
        sid_scores = {}
        for h in [1, 5, 20, 50]:
            rk = f"forward_returns_{h}"
            sk = f"forward_returns_{h}_states"
            if rk in sid_data and sk in sid_data:
                try:
                    result = sid_calc.compute_all_sid(sid_data[sk], {rk: sid_data[rk]})
                    for k, v in result.items():
                        if isinstance(v, dict) and "avg_sid" in v:
                            sid_scores[f"sid_return_{h}"] = v["avg_sid"]
                except Exception:
                    pass

        pa = PersistenceAnalyzer(min_duration=2)
        persistence_all = pa.compute_all(valid_states)
        classification = pa.classify_states(valid_states)

        tga = TransitionGraphAnalyzer(max_lag=10)
        n_states = int(np.max(valid_states)) + 1 if len(valid_states) > 0 else 0
        if n_states > 0:
            transitions = tga.compute_all(valid_states, n_states)
        else:
            transitions = {}

        return {
            "sid_scores": sid_scores,
            "forward_metrics": str({k: v.shape for k, v in forward_metrics.items()}),
            "persistence": persistence_all,
            "state_classification": classification,
            "transition_analysis": transitions,
        }

    def run_cross_validation(self, state_discovery_fn: Callable, assets_data: dict, regime_data: dict) -> dict:
        cv = CrossValidator()
        try:
            return cv.validate_all(state_discovery_fn, assets_data, regime_data)
        except Exception as e:
            return {"error": str(e)}

    def run_full_research_cycle(self, data: dict, assets_data: Optional[dict] = None, regime_data: Optional[dict] = None) -> dict:
        discovery = self.run_discovery_cycle(data)
        price = data["price"]
        returns = data["returns"]
        validation = self.run_validation_cycle(discovery["cluster_labels"], price, returns)

        results: dict[str, Any] = {
            "discovery": discovery,
            "validation": validation,
        }

        if assets_data is not None and regime_data is not None:
            def discovery_wrapper(data):
                disc = self.run_discovery_cycle(data)
                return disc["cluster_labels"]
            results["cross_validation"] = self.run_cross_validation(
                discovery_wrapper,
                assets_data,
                regime_data,
            )

        return results

    def run_information_discovery_cycle(self, data: dict) -> dict:
        price = data.get("price", np.array([]))
        high = data.get("high", price)
        low = data.get("low", price)
        returns = data.get("returns", np.array([]))
        volume = data.get("volume", np.array([]))
        n = len(price)

        modules: dict[str, dict] = {}
        mf = MemoryFieldResearch()
        modules["memory_field"] = mf.compute_all(price, returns)

        dna = TemporalDNAResearch()
        dna_input = {"high": high, "low": low, "close": price, "volume": volume, "returns": returns}
        modules["temporal_dna"] = dna.compute_all(dna_input)

        ip = InformationPressureResearch()
        modules["information_pressure"] = ip.compute_all(returns, volume)

        lm = LiquidityMigrationResearch()
        modules["liquidity_migration"] = lm.compute_all(price)

        cs = CohortSimulationResearch()
        cs_res = cs.compute_all(volume, price, returns)
        for k, v in cs_res.items():
            if v.ndim == 1:
                modules[f"cohort_{k}"] = v
            else:
                for i in range(v.shape[1]):
                    modules[f"cohort_{k}_{i}"] = v[:, i]

        se = StateEntanglementResearch()
        tf_returns = {tf: returns for tf in se.TIMEFRAMES}
        se_res = se.compute_all(tf_returns)
        for k, v in se_res.items():
            if v.ndim == 1:
                modules[f"entanglement_{k}"] = v
            else:
                for i in range(v.shape[1]):
                    modules[f"entanglement_{k}_{i}"] = v[:, i]

        tt = TensionTensorResearch()
        tension_inputs = {
            "memory": modules["memory_field"]["memory_strength"],
            "pressure": modules["information_pressure"]["pressure_build"],
            "liquidity": modules["liquidity_migration"]["liquidity_mass"],
            "cohort": modules.get("cohort_cohort_alignment", np.ones(n, dtype=np.float32)),
            "volatility": modules["temporal_dna"]["volatility"],
            "state_alignment": np.ones(n, dtype=np.float32),
        }
        modules["tension_tensor"] = tt.compute_all(tension_inputs)
        modules["behavioral_echoes"] = {}

        sde = StateDiscoveryEngine()
        disc_features = sde.compute_all_features(price, high, low, returns, volume)
        nsg = NovelStateGenerator()
        novel = nsg.compute_novel_state_vector(price, high, low, returns, volume)

        all_features: dict[str, NDArray] = {}
        for mod_name, result in modules.items():
            if not isinstance(result, dict):
                continue
            for key, value in result.items():
                if isinstance(value, np.ndarray) and value.ndim == 1:
                    all_features[f"{mod_name}_{key}"] = value.astype(np.float64)

        for key, value in disc_features.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    all_features[f"disc_{key}"] = value.astype(np.float64)
                else:
                    for i in range(value.shape[1]):
                        all_features[f"disc_{key}_{i}"] = value[:, i].astype(np.float64)

        if novel is not None and novel.ndim == 2:
            for i in range(novel.shape[1]):
                all_features[f"novel_{i}"] = novel[:, i].astype(np.float64)

        id_pipe = InfoDiscoveryPipeline()
        results = id_pipe.run_full_pipeline(all_features, price, returns, volume)
        results["all_features"] = all_features
        return results

    def generate_info_discovery_report(self, results: dict) -> str:
        id_pipe = InfoDiscoveryPipeline()
        return id_pipe.generate_report(results)

    def generate_report(self, results: dict) -> str:
        discovery = results.get("discovery", {})
        validation = results.get("validation", {})
        cross_validation = results.get("cross_validation")

        lines: list[str] = []
        lines.append("# PROXIMA X Research Report")
        lines.append("")

        n_states = discovery.get("n_clusters", 0)
        lines.append("## Discovery Summary")
        lines.append(f"- **States Discovered**: {n_states}")
        feature_names = discovery.get("feature_names", [])
        if feature_names:
            lines.append(f"- **Total Features**: {len(feature_names)}")
        compressed = discovery.get("compressed_states")
        if compressed is not None and isinstance(compressed, np.ndarray):
            lines.append(f"- **Compressed Dimension**: {compressed.shape[1]}")
        lines.append("")

        sid_scores = validation.get("sid_scores", {})
        lines.append("## SID Scores")
        if sid_scores:
            for k, v in sorted(sid_scores.items()):
                lines.append(f"- **{k}**: {v:.6f}")
        else:
            lines.append("- No SID scores available")
        lines.append("")

        persistence = validation.get("persistence", {})
        lines.append("## Persistence Analysis")
        if isinstance(persistence, dict):
            for k, v in persistence.items():
                if k == "transition_prob_by_duration":
                    continue
                if isinstance(v, dict):
                    vals = [f"{sk}: {sv}" for sk, sv in v.items()]
                    lines.append(f"- **{k}**: {vals}")
                elif isinstance(v, (int, float)):
                    lines.append(f"- **{k}**: {v}")
        else:
            lines.append("- No persistence data available")
        lines.append("")

        classification = validation.get("state_classification", {})
        lines.append("## State Classification")
        if isinstance(classification, dict):
            for cls_name, state_list in classification.items():
                lines.append(f"- **{cls_name}**: {len(state_list)} states {state_list}")
        lines.append("")

        transitions = validation.get("transition_analysis", {})
        lines.append("## Transition Matrix Summary")
        if isinstance(transitions, dict):
            tm = transitions.get("transition_matrix")
            if tm is not None and isinstance(tm, np.ndarray):
                lines.append(f"- **Matrix Shape**: {tm.shape}")
                diag_mean = float(np.mean(np.diag(tm)))
                lines.append(f"- **Self-Transition Rate (mean)**: {diag_mean:.4f}")
            for key, value in transitions.items():
                if key == "transition_matrix":
                    continue
                if isinstance(value, np.ndarray):
                    lines.append(f"- **{key}**: {value}")
                elif isinstance(value, (int, float)):
                    lines.append(f"- **{key}**: {value:.4f}")
                elif isinstance(value, list):
                    lines.append(f"- **{key}**: {value}")
        lines.append("")

        lines.append("## Top States by SID")
        if sid_scores:
            sorted_sids = sorted(sid_scores.items(), key=lambda x: x[1], reverse=True)
            for sid_key, score in sorted_sids[:5]:
                lines.append(f"- **{sid_key}**: {score:.6f}")
        lines.append("")

        if cross_validation and any(k != "error" for k in cross_validation):
            lines.append("## Cross-Validation Results")
            lines.append("")
            ca = cross_validation.get("consistency", {})
            if isinstance(ca, dict) and ca:
                for key, value in ca.items():
                    lines.append(f"- **{key}**: {value}")
            lines.append("")

        lines.append("## Recommended Next Experiments")
        recommendations: list[str] = []
        if n_states < 3:
            recommendations.append("Increase min_cluster_size or reduce n_components to merge states")

        if isinstance(persistence, dict):
            avg_dur = persistence.get("average_duration", {})
            if avg_dur:
                durs = [float(v) for v in avg_dur.values()]
                if durs and float(np.mean(durs)) < 5:
                    recommendations.append("States are short-lived - increase lookback windows or use coarser state definitions")
                elif durs and float(np.mean(durs)) > 100:
                    recommendations.append("States are overly stable - consider finer-grained clustering")

        if sid_scores:
            sid_vals = list(sid_scores.values())
            if sid_vals and float(np.mean(sid_vals)) < 0.0:
                recommendations.append("SID scores are negative - states are not reducing uncertainty, try alternative feature combinations")

        if not recommendations:
            recommendations.append("Run with real market data across multiple assets")
            recommendations.append("Increase state vector dimension (50-500 target)")
            recommendations.append("Try spectral embedding as alternative compressor")
            recommendations.append("Explore HDBSCAN with different min_cluster_size values")

        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    def save_results(self, results: dict, name: str = "research_cycle") -> None:
        path = self.output_dir / f"{name}.json"
        save_json(path, results)

    def load_results(self, name: str = "research_cycle") -> dict:
        path = self.output_dir / f"{name}.json"
        return load_json(path)

    def run_mechanism_discovery_cycle(self, data: dict) -> dict:
        info_results = self.run_information_discovery_cycle(data)
        states = info_results["state_construction"]["state_result"]["labels"]
        compressed_dim = info_results["state_construction"]["state_result"].get("compressed", np.zeros((len(data.get("price", [])), 1))).shape[1]
        price = data.get("price", np.array([]))
        returns = data.get("returns", np.array([]))
        forward_analyzer = ForwardAnalyzer(horizons=[1])
        fwd = forward_analyzer.compute_forward_returns(price, 1)
        mde = MDEPipeline()
        mechanism_results = mde.run_full_pipeline(data, states, fwd, compressed_dim)
        mechanism_results["phase3_results"] = info_results
        return mechanism_results

    def generate_mechanism_report(self, results: dict) -> str:
        mde = MDEPipeline()
        return mde.generate_report(results)

    def print_summary(self, results: dict) -> None:
        report = self.generate_report(results)
        print(report)
