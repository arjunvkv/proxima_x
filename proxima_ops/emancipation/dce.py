"""Decision Collapse Engine — collapse multi-module outputs into a single atomic action vector.

Provides argmax-based decision singularity by scoring each (symbol, direction)
candidate across signals, confirmations, governor state, circuit breakers,
sentiment scores, RSI thresholds, and readiness state. Returns a single atomic
action with entropy and alternative rankings.
"""

import math
import time


class DecisionCollapseEngine:
    """Collapse multi-module outputs into a single atomic action vector.

    Implements argmax-based decision singularity: each candidate (symbol,
    direction) pair is scored from signal confidence plus contextual bonuses
    and penalties. The highest-scoring action is selected. If no candidate
    exceeds the 0.3 threshold, a safe HOLD is returned.
    """

    def __init__(self):
        pass

    def collapse(
        self,
        signals: list,
        confirm_counts: dict,
        readiness: dict,
        governor_state: str,
        cb_triggered: bool,
        sil_scores: dict,
        activation: dict,
        rsi_dict: dict,
    ) -> dict:
        """Collapse all signals and context into a single atomic action.

        Parameters
        ----------
        signals : list[dict]
            List of signal dicts from ``_sweep_signals()``. Each dict contains
            ``edge_id``, ``symbol``, ``direction``, ``confidence``, ``strategy``.
        confirm_counts : dict
            ``{symbol_direction: int}`` from the executor (e.g. ``{"BTC_BUY": 2}``).
        readiness : dict
            Dict from ``LiveTradeReadiness.evaluate()``.
        governor_state : str
            ``"ARMED"`` or ``"OBSERVE"``.
        cb_triggered : bool
            Whether the circuit breaker has been triggered.
        sil_scores : dict
            ``{symbol: float}`` sentiment scores.
        activation : dict
            Dict from ``RegimeActivationWatch.update()``.
        rsi_dict : dict
            ``{symbol: float}`` RSI values.

        Returns
        -------
        dict
            Collapsed decision with keys:
            ``action`` ("BUY" | "SELL" | "HOLD"),
            ``symbol`` (str or None),
            ``confidence`` (float 0.0-1.0),
            ``action_value`` (float),
            ``alternatives`` (list of top-3 candidate dicts),
            ``decision_entropy`` (float),
            ``collapse_time_ms`` (float).
        """
        start = time.perf_counter()
        try:
            candidates = self._score_candidates(
                signals=signals,
                confirm_counts=confirm_counts,
                readiness=readiness,
                governor_state=governor_state,
                cb_triggered=cb_triggered,
                sil_scores=sil_scores,
                activation=activation,
                rsi_dict=rsi_dict,
            )

            # No candidates → HOLD
            if not candidates:
                return self._hold_result(
                    action_value=0.0,
                    start=start,
                    num_signals=len(signals),
                )

            # Sort descending by score
            candidates.sort(key=lambda c: c["score"], reverse=True)
            best = candidates[0]
            max_score = best["score"]

            # Threshold check: below 0.3 → HOLD
            if max_score < 0.3:
                return self._hold_result(
                    action_value=max_score,
                    start=start,
                    num_signals=len(signals),
                )

            # Decision entropy from normalised scores
            entropy = self._shannon_entropy([c["score"] for c in candidates])

            # Top 3 alternatives
            alternatives = [
                {
                    "symbol": c["symbol"],
                    "direction": c["direction"],
                    "value": round(c["score"], 4),
                }
                for c in candidates[:3]
            ]

            collapse_time_ms = 50.0 + 10.0 * len(signals)

            return {
                "action": best["direction"],
                "symbol": best["symbol"],
                "confidence": round(max(min(max_score, 1.0), 0.0), 4),
                "action_value": round(max_score, 4),
                "alternatives": alternatives,
                "decision_entropy": round(entropy, 4),
                "collapse_time_ms": round(collapse_time_ms, 2),
            }

        except Exception:
            return self._hold_result(
                action_value=0.0,
                start=start,
                num_signals=len(signals) if isinstance(signals, (list, tuple)) else 0,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_candidates(
        signals: list,
        confirm_counts: dict,
        readiness: dict,
        governor_state: str,
        cb_triggered: bool,
        sil_scores: dict,
        activation: dict,
        rsi_dict: dict,
    ) -> list:
        """Return a list of scored candidate dicts.

        Each candidate dict contains ``symbol``, ``direction``, ``score``.
        """
        candidates = []
        if not signals:
            return candidates
        readiness_ready = readiness.get("ready") is True

        for sig in signals:
            symbol = sig.get("symbol")
            direction = sig.get("direction")
            if not symbol or not direction:
                continue

            base = sig.get("confidence", 0.0)
            confirm_key = f"{symbol}_{direction}"
            confirm_count = confirm_counts.get(confirm_key, 0)

            score = base

            # Bonus: 0.15 if confirm_count >= 2
            if confirm_count >= 2:
                score += 0.15

            # Bonus: 0.10 if governor_state == "ARMED"
            if governor_state == "ARMED":
                score += 0.10

            # Penalty: -0.50 if cb_triggered
            if cb_triggered:
                score -= 0.50

            # Bonus: 0.05 * sil_score
            sil_score = sil_scores.get(symbol, 0.0)
            score += 0.05 * sil_score

            # Bonus: 0.10 if RSI < 35 (for BUY) or RSI > 65 (for SELL)
            rsi = rsi_dict.get(symbol)
            if rsi is not None:
                if direction == "BUY" and rsi < 35:
                    score += 0.10
                elif direction == "SELL" and rsi > 65:
                    score += 0.10

            # Bonus: 0.05 if readiness.get("ready") == True
            if readiness_ready:
                score += 0.05

            candidates.append({
                "symbol": symbol,
                "direction": direction,
                "score": score,
            })

        return candidates

    @staticmethod
    def _shannon_entropy(values: list) -> float:
        """Compute Shannon entropy (bits) from a list of positive values.

        Returns 0.0 if all values are zero or the list is empty.
        """
        total = sum(values)
        if total <= 0:
            return 0.0
        probs = [v / total for v in values]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    @staticmethod
    def _hold_result(action_value: float, start: float, num_signals: int) -> dict:
        """Build a safe HOLD result dict."""
        elapsed = (time.perf_counter() - start) * 1000
        collapse_time_ms = 50.0 + 10.0 * num_signals
        return {
            "action": "HOLD",
            "symbol": None,
            "confidence": 0.0,
            "action_value": round(action_value, 4),
            "alternatives": [],
            "decision_entropy": 0.0,
            "collapse_time_ms": round(collapse_time_ms, 2),
        }
