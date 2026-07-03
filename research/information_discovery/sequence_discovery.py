from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import MIEstimator


class SequenceDiscovery:
    def __init__(self, mi_estimator: Optional[MIEstimator] = None, max_sequence_length: int = 5):
        self.mi = mi_estimator or MIEstimator(n_bins=20)
        self.max_sequence_length = max_sequence_length

    def extract_sequences(self, states: NDArray[np.int32], length: int) -> NDArray:
        valid = states >= 0
        if valid.sum() < length:
            return np.empty((0, length), dtype=np.int32)
        indices = np.where(valid)[0]
        n_seq = len(indices) - length + 1
        if n_seq < 1:
            return np.empty((0, length), dtype=np.int32)
        sequences = np.zeros((n_seq, length), dtype=np.int32)
        for i in range(length):
            sequences[:, i] = states[indices[i : i + n_seq]]
        return sequences

    def sequence_to_id(self, sequences: NDArray, max_states: int = 10) -> NDArray:
        multipliers = np.array([max_states ** (sequences.shape[1] - 1 - i) for i in range(sequences.shape[1])], dtype=np.int64)
        return np.dot(sequences.astype(np.int64), multipliers)

    def compute_sequence_mi(self, sequences: NDArray, target: NDArray) -> float:
        n_seq = len(sequences)
        if n_seq < 2 or n_seq > len(target):
            return 0.0
        seq_ids = self.sequence_to_id(sequences)
        target_aligned = target[:n_seq]
        return self.mi.mutual_info(seq_ids.astype(np.float64), target_aligned)

    def find_informative_sequences(self, states: NDArray[np.int32], target: NDArray) -> list[dict]:
        results: list[dict] = []
        for length in range(2, self.max_sequence_length + 1):
            sequences = self.extract_sequences(states, length)
            if len(sequences) < 2:
                continue
            seq_mi = self.compute_sequence_mi(sequences, target)
            seq_ids = self.sequence_to_id(sequences)
            unique_seq, counts = np.unique(seq_ids, return_counts=True)
            n_unique = len(unique_seq)
            freq = counts / counts.sum()
            seq_entropy = -float(np.sum(freq * np.log(freq + 1e-10)))
            results.append({
                "length": length,
                "sequence_mi": seq_mi,
                "n_sequences": len(sequences),
                "n_unique_sequences": n_unique,
                "sequence_entropy": seq_entropy,
                "information_per_step": seq_mi / length if length > 0 else 0.0,
            })
        results.sort(key=lambda x: x["sequence_mi"], reverse=True)
        return results

    def compute_sequence_sid(self, sequences: NDArray, forward_values: NDArray) -> float:
        n_seq = len(sequences)
        if n_seq < 2:
            return 0.0
        seq_ids = self.sequence_to_id(sequences)
        valid = n_seq
        fwd = forward_values[:valid]
        h_y = self.mi.entropy(fwd)
        h_y_given_seq = 0.0
        unique_ids = np.unique(seq_ids)
        for sid in unique_ids:
            mask = seq_ids == sid
            if mask.sum() < 2:
                continue
            cond_h = self.mi.entropy(fwd[mask])
            h_y_given_seq += (mask.sum() / n_seq) * cond_h
        return h_y - h_y_given_seq

    def compute_sequence_sir(self, sequences: NDArray, forward_values: NDArray, complexity_penalty: float = 1.0) -> float:
        sid = self.compute_sequence_sid(sequences, forward_values)
        if sid <= 0:
            return sid
        n_unique = len(np.unique(self.sequence_to_id(sequences)))
        complexity = n_unique * complexity_penalty
        if complexity < 1e-10:
            return 0.0
        return sid / complexity

    def find_best_sequence_length(self, states: NDArray[np.int32], target: NDArray) -> dict:
        best: dict = {"length": 0, "mi": -float("inf"), "sir": -float("inf")}
        for length in range(2, self.max_sequence_length + 1):
            sequences = self.extract_sequences(states, length)
            if len(sequences) < 2:
                continue
            mi = self.compute_sequence_mi(sequences, target)
            sir = self.compute_sequence_sir(sequences, target)
            if mi > best["mi"]:
                best = {"length": length, "mi": mi, "sir": sir}
        return best
