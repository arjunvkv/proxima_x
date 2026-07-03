from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from config.settings import settings
from ml.clustering import StateClusterer
from research.information_discovery.mi_estimator import MIEstimator
from research.state_compressor import StateCompressor


class StateConstructor:
    def __init__(self, mi_estimator: Optional[MIEstimator] = None):
        self.mi = mi_estimator or MIEstimator(n_bins=20)

    def build_state_matrix(self, features: dict[str, NDArray], feature_names: list[str]) -> NDArray:
        arrays: list[NDArray] = []
        min_len = min(arr.shape[0] for arr in features.values() if arr.ndim >= 1)
        for name in feature_names:
            arr = features[name]
            if arr.ndim == 1:
                arrays.append(arr[:min_len].reshape(-1, 1))
            elif arr.ndim == 2:
                arrays.append(arr[:min_len])
        if not arrays:
            return np.zeros((min_len, 1), dtype=np.float32)
        return np.hstack(arrays).astype(np.float32)

    def discover_states(self, feature_matrix: NDArray, compressor: Optional[StateCompressor] = None, clusterer: Optional[StateClusterer] = None) -> dict:
        if feature_matrix.shape[1] == 0:
            return {"compressed": np.zeros((len(feature_matrix), 1), dtype=np.float32), "labels": np.full(len(feature_matrix), -1, dtype=np.int32), "n_clusters": 0}
        trim_start = 0
        row_sums = np.sum(np.abs(feature_matrix), axis=1)
        nonzero_idx = np.argmax(row_sums > 1e-8)
        if nonzero_idx > 0:
            trim_start = nonzero_idx
            feature_matrix = feature_matrix[trim_start:]
        n_comp = min(20, feature_matrix.shape[1] - 1)
        if compressor is None:
            compressor = StateCompressor(method="pca", n_components=max(2, n_comp))
        try:
            compressed = compressor.fit_transform(feature_matrix)
        except Exception:
            compressor = StateCompressor(method="pca", n_components=max(2, n_comp))
            compressed = compressor.fit_transform(feature_matrix)
        mcs = max(5, min(30, len(compressed) // 40))
        if clusterer is None:
            clusterer = StateClusterer(method="hdbscan", params={"min_cluster_size": mcs})
        labels = clusterer.fit_predict(compressed)
        noise_mask = labels != -1
        clean_labels = labels[noise_mask]
        unique_clean = np.unique(clean_labels)
        label_map = {old: new for new, old in enumerate(sorted(unique_clean))}
        remapped = np.full_like(labels, -1)
        for old, new in label_map.items():
            remapped[labels == old] = new
        n_clusters = int(np.sum(np.unique(remapped) >= 0))
        return {
            "compressed": compressed,
            "labels": remapped,
            "n_clusters": n_clusters,
            "raw_matrix": feature_matrix,
        }

    def construct_from_survivors(self, features: dict[str, NDArray], survivor_names: list[str],
                                 compressor: Optional[StateCompressor] = None, clusterer: Optional[StateClusterer] = None) -> dict:
        surviving: dict[str, NDArray] = {}
        for name in survivor_names:
            if name in features:
                surviving[name] = features[name]
        matrix = self.build_state_matrix(surviving, survivor_names)
        return self.discover_states(matrix, compressor, clusterer)

    def iterative_state_construction(self, features: dict[str, NDArray], survivor_names: list[str],
                                     mi_target: NDArray, min_sid: float = 0.0) -> dict:
        from research.information_discovery.sid_sir import SIDCalculator
        sid_calc = SIDCalculator(mi_estimator=self.mi)
        best_result = None
        best_sid = -float("inf")
        for n_features in range(max(2, len(survivor_names) // 4), len(survivor_names) + 1, max(1, len(survivor_names) // 8)):
            subset = survivor_names[:n_features]
            result = self.construct_from_survivors(features, subset)
            if result["n_clusters"] < 2:
                continue
            sid = sid_calc.compute_sid(result["labels"], mi_target)
            avg_sid = float(np.mean(list(sid.get("sid_per_state", {}).values()))) if sid.get("sid_per_state") else -float("inf")
            if avg_sid > best_sid:
                best_sid = avg_sid
                best_result = result
            if avg_sid > min_sid:
                break
        if best_result is None:
            return self.construct_from_survivors(features, survivor_names[:max(5, len(survivor_names) // 4)])
        return best_result
