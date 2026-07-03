from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import numba
from numpy.typing import NDArray

from config.settings import settings
from core.event_engine import EventEngine, Event, EventType


@numba.jit(nopython=True)
def _cosine_sim(v1: NDArray[np.float32], v2: NDArray[np.float32]) -> float:
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for i in range(len(v1)):
        dot += v1[i] * v2[i]
        n1 += v1[i] * v1[i]
        n2 += v2[i] * v2[i]
    denom = math.sqrt(n1) * math.sqrt(n2)
    if denom < 1e-10:
        return 0.0
    return dot / denom


@numba.jit(nopython=True)
def _find_similar(
    query: NDArray[np.float32], storage: list[NDArray[np.float32]],
    indices: NDArray[np.int64], scores: NDArray[np.float32]
) -> None:
    n = len(storage)
    sims = np.zeros(n, dtype=np.float32)
    for i in range(n):
        sims[i] = np.float32(_cosine_sim(query, storage[i]))
    order = np.argsort(-sims)
    k = min(len(indices), n)
    for i in range(k):
        indices[i] = order[i]
        scores[i] = sims[order[i]]


@numba.jit(nopython=True)
def _echo_strength_inner(
    chain_events: list[NDArray[np.float32]],
    chain_responses: list[NDArray[np.float32]],
    decay: bool, half_life: float
) -> float:
    k = len(chain_events)
    if k < 2:
        return 0.0
    total = 0.0
    weight_sum = 0.0
    for i in range(1, k):
        sim = _cosine_sim(chain_events[i - 1], chain_events[i])
        if decay:
            w = math.exp(-float(i) * math.log(2.0) / half_life)
        else:
            w = 1.0
        total += sim * w
        weight_sum += w
    if weight_sum < 1e-10:
        return 0.0
    return total / weight_sum


class BehavioralEchoesResearch:

    def __init__(
        self, max_chain: int = settings.research.echo_max_chain,
        half_life: int = settings.research.echo_decay_half_life
    ) -> None:
        self._max_chain = max_chain
        self._half_life = half_life
        self._echo_events: list[NDArray[np.float32]] = []
        self._echo_responses: list[NDArray[np.float32]] = []

    def store_event_response(
        self, event_vector: NDArray[np.float32],
        response_vector: NDArray[np.float32]
    ) -> None:
        self._echo_events.append(event_vector)
        self._echo_responses.append(response_vector)

    def compute_echo_strength(
        self,
        echo_chain: list[tuple[NDArray[np.float32], NDArray[np.float32]]],
        decay: bool = True
    ) -> float:
        events: list[NDArray[np.float32]] = [e for e, r in echo_chain]
        responses: list[NDArray[np.float32]] = [r for e, r in echo_chain]
        return _echo_strength_inner(events, responses, decay, float(self._half_life))

    def compute_echo_decay(self, chain_length: int) -> float:
        if chain_length == 0:
            return 1.0
        return float(np.exp2(-float(chain_length) / float(self._half_life)))

    def compute_echo_similarity(
        self, v1: NDArray[np.float32], v2: NDArray[np.float32]
    ) -> float:
        return _cosine_sim(v1, v2)

    def find_similar_echoes(
        self, query: NDArray[np.float32], top_k: int = 10
    ) -> list[int]:
        if not self._echo_events:
            return []
        k = min(top_k, len(self._echo_events))
        indices = np.zeros(k, dtype=np.int64)
        scores = np.zeros(k, dtype=np.float32)
        _find_similar(query, self._echo_events, indices, scores)
        return [int(indices[i]) for i in range(k) if scores[i] > 0]

    def build_echo_pattern(
        self, event_sequences: list[list[NDArray[np.float32]]]
    ) -> dict:
        patterns: dict[str, int] = {}
        for seq in event_sequences:
            if len(seq) < 2:
                continue
            max_len = min(self._max_chain, len(seq))
            for length in range(2, max_len + 1):
                for start in range(len(seq) - length + 1):
                    subseq = seq[start : start + length]
                    key_parts: list[str] = []
                    for vec in subseq:
                        dom = float(np.argmax(np.abs(vec)))
                        key_parts.append(f"{dom:.0f}")
                    key = "_".join(key_parts)
                    patterns[key] = patterns.get(key, 0) + 1
        return {k: v for k, v in sorted(patterns.items(), key=lambda x: -x[1])}

    def emit_events(
        self, timestamps: list[int], echo_strength: NDArray[np.float32],
        event_engine: EventEngine, threshold: float = 0.8
    ) -> None:
        events: list[Event] = []
        n = min(len(timestamps), len(echo_strength))
        for i in range(n):
            if echo_strength[i] >= threshold:
                events.append(Event(
                    event_type=EventType.ECHO_TRIGGERED,
                    timestamp=timestamps[i],
                    data={"echo_strength": float(echo_strength[i])},
                    source="behavioral_echoes",
                    confidence=float(min(echo_strength[i], 1.0)),
                ))
        event_engine.emit_batch(events)
