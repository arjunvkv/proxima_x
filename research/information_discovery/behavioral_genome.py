from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import MIEstimator


class BehavioralGenomeEngine:
    def __init__(self, mi_estimator: Optional[MIEstimator] = None, genome_length: int = 50):
        self.mi = mi_estimator or MIEstimator(n_bins=20)
        self.genome_length = genome_length
        self._genome_store: dict[str, NDArray] = {}
        self._outcome_store: dict[str, NDArray] = {}

    def build_genome(self, states: NDArray[np.int32], n_gram: int = 3) -> NDArray:
        valid = states >= 0
        if valid.sum() < self.genome_length:
            idx = np.where(valid)[0]
            padded = np.full(self.genome_length, -1, dtype=np.int32)
            padded[: len(idx)] = states[idx]
            return padded
        idx = np.where(valid)[0]
        start = np.random.randint(0, max(1, len(idx) - self.genome_length + 1))
        genome_indices = idx[start : start + self.genome_length]
        genome = states[genome_indices]
        if n_gram > 1:
            genome_str = np.array([
                int("".join(str(max(0, s)) for s in genome[i:i + n_gram]))
                for i in range(len(genome) - n_gram + 1)
            ], dtype=np.int64)
            return genome_str.astype(np.int32)
        return genome

    def encode_genome(self, genome: NDArray, max_state: int = 10) -> NDArray:
        genome_clean = np.maximum(genome, 0)
        multipliers = np.array([max_state ** (len(genome_clean) - 1 - i) for i in range(len(genome_clean))], dtype=np.int64)
        fingerprint = int(np.dot(genome_clean.astype(np.int64), multipliers))
        return np.array([fingerprint], dtype=np.int64)

    def store_genome(self, genome_id: str, genome: NDArray, future_outcome: NDArray) -> None:
        self._genome_store[genome_id] = genome
        self._outcome_store[genome_id] = future_outcome

    def genome_similarity(self, g1: NDArray, g2: NDArray) -> float:
        min_len = min(len(g1), len(g2))
        if min_len < 2:
            return 0.0
        matches = np.sum(g1[:min_len] == g2[:min_len])
        return matches / min_len

    def future_outcome_similarity(self, genome_id1: str, genome_id2: str) -> float:
        o1 = self._outcome_store.get(genome_id1)
        o2 = self._outcome_store.get(genome_id2)
        if o1 is None or o2 is None:
            return 0.0
        min_len = min(len(o1), len(o2))
        if min_len < 2:
            return 0.0
        corr = float(np.corrcoef(o1[:min_len], o2[:min_len])[0, 1])
        if np.isnan(corr):
            return 0.0
        return max(0.0, corr)

    def find_similar_genomes(self, query_genome: NDArray, top_k: int = 5) -> list[tuple[str, float]]:
        similarities: list[tuple[str, float]] = []
        for gid, stored in self._genome_store.items():
            sim = self.genome_similarity(query_genome, stored)
            similarities.append((gid, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def compute_genome_information(self, genome: NDArray, future_target: NDArray) -> float:
        min_len = min(len(genome), len(future_target))
        if min_len < 5:
            return 0.0
        fingerprint = self.encode_genome(genome).astype(np.float64)
        fingerprints = np.full(min_len, fingerprint, dtype=np.float64)
        return self.mi.mutual_info(fingerprints, future_target[:min_len])

    def batch_build_genomes(self, states_dict: dict[str, NDArray]) -> dict[str, NDArray]:
        genomes: dict[str, NDArray] = {}
        for asset_id, state_series in states_dict.items():
            genomes[asset_id] = self.build_genome(state_series)
        return genomes

    def cross_asset_genome_similarity(self, genomes: dict[str, NDArray]) -> dict[tuple[str, str], float]:
        similarities: dict[tuple[str, str], float] = {}
        assets = list(genomes.keys())
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                sim = self.genome_similarity(genomes[assets[i]], genomes[assets[j]])
                similarities[(assets[i], assets[j])] = sim
        return similarities
