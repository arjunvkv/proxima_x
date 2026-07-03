from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import numba
from numpy.typing import NDArray

from research.temporal_reality.causality_analysis import AdaptiveTimeCausality
from research.information_discovery.mi_estimator import (
    _fast_mutual_info,
    _fast_conditional_mutual_info,
    _fast_percentile,
    _fast_digitize,
)


@dataclass
class CausalLink:
    source: str
    target: str
    lead_lag: int
    peak_corr: float
    transfer_entropy: float
    survives_validation: bool


@numba.jit(nopython=True, cache=True)
def _numba_transfer_entropy(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    lag: int,
    n_bins: int,
) -> float:
    n = min(len(source), len(target))
    if n <= lag + 2:
        return 0.0
    s = source[: n - lag]
    t = target[lag:]
    t_past = target[: n - lag]
    valid = ~(np.isnan(s) | np.isnan(t) | np.isnan(t_past))
    s_c = s[valid]
    t_c = t[valid]
    tp_c = t_past[valid]
    if len(s_c) < 2:
        return 0.0
    q = np.linspace(0.0, 1.0, n_bins + 1)
    s_bins = _fast_percentile(s_c, q)
    t_bins = _fast_percentile(t_c, q)
    tp_bins = _fast_percentile(tp_c, q)
    u_s = np.unique(s_bins)
    u_t = np.unique(t_bins)
    u_tp = np.unique(tp_bins)
    if len(u_s) < 2 or len(u_t) < 2 or len(u_tp) < 2:
        return 0.0
    return _fast_conditional_mutual_info(t_c, s_c, tp_c, n_bins)


def _numba_regime_split_stability(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    regime_labels: NDArray[np.int32],
    n_regimes: int,
    max_lag: int,
) -> float:
    strengths = np.zeros(n_regimes, dtype=np.float64)
    counts = np.zeros(n_regimes, dtype=np.int32)
    for r in range(n_regimes):
        mask = regime_labels == r
        n_pts = int(np.sum(mask))
        if n_pts > max_lag * 2:
            s = source[mask]
            t = target[mask]
            corr = AdaptiveTimeCausality._cross_correlate(s, t, max_lag)
            peak = 0.0
            for k in range(len(corr)):
                if abs(corr[k]) > abs(peak):
                    peak = corr[k]
            strengths[r] = peak
            counts[r] = 1
    valid_count = int(np.sum(counts))
    if valid_count < 2:
        return 0.5
    valid_strengths = strengths[counts > 0]
    mean_s = np.mean(valid_strengths)
    var_s = 0.0
    for i in range(valid_count):
        var_s += (valid_strengths[i] - mean_s) ** 2
    std_s = np.sqrt(var_s / valid_count)
    if abs(mean_s) < 1e-12:
        return 0.0
    return float(1.0 - std_s / (abs(mean_s) + 1e-12))


@numba.jit(nopython=True, cache=True)
def _numba_rolling_mutation_rate(
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
def _numba_rolling_regime_prob(
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


class MarketPhysicsModel:
    CANONICAL_CHAIN = [
        "generator",
        "adaptive_time",
        "state_mutation_rate",
        "regime_change_probability",
        "outcome",
    ]

    def __init__(self, max_lag: int = 50, n_bins: int = 20):
        self.max_lag = max_lag
        self.n_bins = n_bins
        self._models: list[dict[str, Any]] = []

    def build_hierarchy(
        self, data: dict, candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        chain = list(self.CANONICAL_CHAIN)
        signals = self._extract_signals(data)
        if not signals:
            return self._empty_hierarchy()

        chain_vars = [c for c in chain if c in signals]

        link_strengths: dict[str, float] = {}
        information_flow: dict[str, float] = {}
        all_links: list[CausalLink] = []

        for i in range(len(chain_vars) - 1):
            src = chain_vars[i]
            tgt = chain_vars[i + 1]
            s_sig = signals[src]
            t_sig = signals[tgt]
            n = min(len(s_sig), len(t_sig))
            if n < self.max_lag * 2 + 1:
                continue

            corr = AdaptiveTimeCausality._cross_correlate(
                s_sig[:n], t_sig[:n], self.max_lag
            )
            lags_arr = np.arange(-self.max_lag, self.max_lag + 1, dtype=np.intp)
            peak_idx = int(np.argmax(np.abs(corr)))
            lead_lag = int(lags_arr[peak_idx])
            peak_corr = float(corr[peak_idx])

            te = _numba_transfer_entropy(s_sig[:n], t_sig[:n], 1, self.n_bins)

            regimes = np.asarray(
                data.get("time_regime", np.zeros(n, dtype=np.int64)),
                dtype=np.int32,
            )
            n_regimes = max(1, int(np.max(regimes)) + 1) if len(regimes) > 0 else 1
            stability = _numba_regime_split_stability(
                s_sig[:n], t_sig[:n], regimes[:n], n_regimes, self.max_lag
            )
            survives = stability > 0.3

            link_key = f"{src}\u2192{tgt}"
            link_strengths[link_key] = peak_corr
            information_flow[link_key] = te
            all_links.append(CausalLink(
                source=src,
                target=tgt,
                lead_lag=lead_lag,
                peak_corr=peak_corr,
                transfer_entropy=te,
                survives_validation=survives,
            ))

        chain_coherence = self._compute_chain_coherence(
            link_strengths, information_flow, len(chain_vars)
        )

        alternative_chains = self._discover_alternative_chains(
            signals, chain_vars
        )

        model = {
            "hierarchy": chain_vars,
            "link_strengths": link_strengths,
            "information_flow": information_flow,
            "chain_coherence": chain_coherence,
            "alternative_chains": alternative_chains,
            "links": all_links,
        }
        self._models.append(model)
        return model

    def test_hierarchy(
        self, data: dict, hierarchy: dict
    ) -> dict[str, Any]:
        tested: dict[str, Any] = {}
        chain = hierarchy.get("hierarchy", [])
        signals = self._extract_signals(data)
        results: list[dict[str, Any]] = []

        for i in range(len(chain) - 1):
            src = chain[i]
            tgt = chain[i + 1]
            s_sig = signals.get(src)
            t_sig = signals.get(tgt)
            if s_sig is None or t_sig is None:
                continue
            n = min(len(s_sig), len(t_sig))
            if n < self.max_lag * 2 + 1:
                continue

            corr = AdaptiveTimeCausality._cross_correlate(
                s_sig[:n], t_sig[:n], self.max_lag
            )
            lags_arr = np.arange(-self.max_lag, self.max_lag + 1, dtype=np.intp)
            peak_idx = int(np.argmax(np.abs(corr)))
            lead_lag = int(lags_arr[peak_idx])
            peak_corr = float(corr[peak_idx])

            te = _numba_transfer_entropy(s_sig[:n], t_sig[:n], 1, self.n_bins)

            cond_mi = _fast_conditional_mutual_info(
                t_sig[:n], s_sig[:n], t_sig[:n], self.n_bins
            )

            results.append({
                "source": src,
                "target": tgt,
                "lead_lag": lead_lag,
                "peak_corr": peak_corr,
                "transfer_entropy": te,
                "conditional_mutual_info": cond_mi,
                "expected_direction": lead_lag <= 0,
            })

        tested["link_tests"] = results
        tested["chain_integrity"] = all(
            r["expected_direction"] for r in results if results
        )
        return tested

    def get_best_model(self) -> dict[str, Any]:
        if not self._models:
            return self._empty_hierarchy()
        best = max(self._models, key=lambda m: m["chain_coherence"])
        return best

    def _extract_signals(self, data: dict) -> dict[str, NDArray[np.float64]]:
        signals: dict[str, NDArray[np.float64]] = {}
        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) > 0:
                signals[key] = np.asarray(val, dtype=np.float64)
        return signals

    def _compute_chain_coherence(
        self,
        link_strengths: dict[str, float],
        information_flow: dict[str, float],
        n_links: int,
    ) -> float:
        if n_links < 2:
            return 0.0
        avg_strength = float(
            np.mean([abs(v) for v in link_strengths.values()]) if link_strengths else 0.0
        )
        avg_flow = float(
            np.mean([v for v in information_flow.values()]) if information_flow else 0.0
        )
        return 0.5 * avg_strength + 0.5 * min(1.0, avg_flow * 10.0)

    def _discover_alternative_chains(
        self,
        signals: dict[str, NDArray[np.float64]],
        chain_vars: list[str],
    ) -> list[dict[str, Any]]:
        alternatives: list[dict[str, Any]] = []
        if len(chain_vars) < 3:
            return alternatives

        for i in range(1, len(chain_vars) - 1):
            swapped = list(chain_vars)
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            alt_strengths: dict[str, float] = {}
            for j in range(len(swapped) - 1):
                src = swapped[j]
                tgt = swapped[j + 1]
                s_sig = signals.get(src)
                t_sig = signals.get(tgt)
                if s_sig is None or t_sig is None:
                    continue
                n = min(len(s_sig), len(t_sig))
                if n < self.max_lag * 2 + 1:
                    continue
                corr = AdaptiveTimeCausality._cross_correlate(
                    s_sig[:n], t_sig[:n], self.max_lag
                )
                peak = float(corr[np.argmax(np.abs(corr))])
                alt_strengths[f"{src}\u2192{tgt}"] = peak
            if alt_strengths:
                avg_s = float(np.mean([abs(v) for v in alt_strengths.values()]))
                alternatives.append({
                    "chain": swapped,
                    "average_strength": avg_s,
                })

        alternatives.sort(key=lambda x: x["average_strength"], reverse=True)
        return alternatives[:3]

    def _empty_hierarchy(self) -> dict[str, Any]:
        return {
            "hierarchy": [],
            "link_strengths": {},
            "information_flow": {},
            "chain_coherence": 0.0,
            "alternative_chains": [],
            "links": [],
        }
