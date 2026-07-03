import json
import logging
import math
from statistics import mean, variance
from typing import Any

logger = logging.getLogger("proxima_ops.diagnostics.emrs")


class ExecutionManifoldReinjection:
    """Execution Manifold Reinjection System.

    Defines an execution manifold from historical states, projects the current
    state onto the manifold, and computes re-entry convergence metrics.

    The execution manifold represents the ideal state conditions under which
    execution has historically occurred: ARMED state + no CB + signals present
    + mof_state INFORMATION_RICH.
    """

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl") -> None:
        self.log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict:
        """Run EMRS analysis over recent cycles from the cycle log.

        Parameters
        ----------
        n_recent_cycles : int
            How many of the most recent log entries to consider.

        Returns
        -------
        dict
            Schema defined in the task specification.
        """
        try:
            records = self._load_records(n_recent_cycles)
            if not records:
                logger.warning("No cycle records loaded — returning empty EMRS.")
                return self._empty_result()

            # Extract raw feature vectors
            raw_vectors: list[dict] = []
            for rec in records:
                try:
                    fv = self._extract_features(rec)
                    raw_vectors.append(fv)
                except Exception as inner:
                    logger.debug(
                        "Skipping cycle %s due to error: %s",
                        rec.get("cycle", "?"), inner,
                    )
                    continue

            if not raw_vectors:
                return self._empty_result()

            # Normalize total_signals across all cycles
            max_signal = max(fv["total_signals_raw"] for fv in raw_vectors)
            max_signal = max(max_signal, 1.0)  # avoid division by zero

            # Build final feature vectors with normalized values
            feature_vectors: list[dict] = []
            for fv in raw_vectors:
                feature_vectors.append({
                    "cycle": fv["cycle"],
                    "mof_score": fv["mof_score"],
                    "total_signals": fv["total_signals_raw"] / max_signal,
                    "cb_active": fv["cb_active"],
                    "segl_armed": fv["segl_armed"],
                    "confirm_progress": fv["confirm_progress"],
                })

            # Manifold center = ideal execution state
            # [mof=1.0, signals=1.0, cb=0, segl=1, confirm=1]
            manifold_center = [1.0, 1.0, 0.0, 1.0, 1.0]

            # Compute distances to manifold center for each cycle
            distances: list[float] = []
            for fv in feature_vectors:
                vec = [
                    fv["mof_score"],
                    fv["total_signals"],
                    fv["cb_active"],
                    fv["segl_armed"],
                    fv["confirm_progress"],
                ]
                dist = self._euclidean_distance(vec, manifold_center)
                distances.append(dist)

            # Normalize distances to [0, 1]
            if distances:
                max_dist = max(distances) if max(distances) > 0 else 1.0
                normalized_distances = [d / max_dist for d in distances]
            else:
                normalized_distances = []

            # Build manifold_distance_trajectory keyed by cycle number
            manifold_distance_trajectory: dict[int, float] = {}
            for i, fv in enumerate(feature_vectors):
                cycle_num = fv["cycle"]
                manifold_distance_trajectory[cycle_num] = round(
                    normalized_distances[i], 4
                )

            # Current manifold distance (latest cycle)
            current_manifold_distance = (
                normalized_distances[-1] if normalized_distances else 1.0
            )

            # Convergence speed = 1st derivative over last 10 cycles
            convergence_speed = self._compute_convergence_speed(
                normalized_distances
            )

            # Re-entry probability (heuristic)
            reentry_probability = round(1.0 - current_manifold_distance, 4)

            # Manifold dimensions: feature loadings from closest cycles
            manifold_dimensions = self._compute_manifold_dimensions(
                feature_vectors, normalized_distances
            )

            # Projection quality = 1.0 - variance of distances
            projection_quality = self._compute_projection_quality(
                normalized_distances
            )

            # Estimated convergence cycles until re-entry
            estimated_convergence_cycles = self._compute_estimated_convergence(
                current_manifold_distance, convergence_speed
            )

            return {
                "manifold_distance_trajectory": manifold_distance_trajectory,
                "current_manifold_distance": round(current_manifold_distance, 4),
                "convergence_speed": round(convergence_speed, 4),
                "reentry_probability": reentry_probability,
                "manifold_dimensions": manifold_dimensions,
                "projection_quality": round(projection_quality, 4),
                "estimated_convergence_cycles": estimated_convergence_cycles,
            }

        except Exception as exc:
            logger.error("EMRS analysis failed: %s", exc, exc_info=True)
            return self._empty_result()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_records(self, n_recent: int) -> list[dict[str, Any]]:
        """Load up to *n_recent* cycle log entries from the JSONL file."""
        try:
            records: list[dict[str, Any]] = []
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        records.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        logger.debug(
                            "Skipping malformed JSONL line: %.80s", stripped
                        )
                        continue
            # Return the most recent N
            return records[-n_recent:] if n_recent < len(records) else records
        except FileNotFoundError:
            logger.warning("Cycle log not found at %s", self.log_path)
            return []
        except Exception as exc:
            logger.error("Error loading cycle log: %s", exc)
            return []

    @staticmethod
    def _extract_features(rec: dict) -> dict:
        """Extract manifold feature vector from a cycle record.

        Returns raw values; total_signals is normalized later across all
        cycles.
        """
        # mof_score — already in [0,1] range typically
        mof_score = float(rec.get("mof_score", 0.0) or 0.0)
        mof_score = max(0.0, min(1.0, mof_score))

        # total_signals (raw, normalized later across all cycles)
        total_signals_raw = float(rec.get("total_signals", 0) or 0)

        # cb_active: 1 if circuit breaker decision is not "Allowed"
        cb_decision = rec.get("cb_decision", "Allowed") or "Allowed"
        cb_active = 0.0 if cb_decision == "Allowed" else 1.0

        # segl_armed: 1 if segl_state is "ARMED"
        segl_state = rec.get("segl_state", "OBSERVE") or "OBSERVE"
        segl_armed = 1.0 if segl_state.upper() == "ARMED" else 0.0

        # confirm_progress: confirm_cycles / 2 capped at 1.0
        confirm_cycles = float(rec.get("confirm_cycles", 0) or 0)
        confirm_progress = min(confirm_cycles / 2.0, 1.0)

        return {
            "cycle": int(rec.get("cycle", 0)),
            "mof_score": mof_score,
            "total_signals_raw": total_signals_raw,
            "cb_active": cb_active,
            "segl_armed": segl_armed,
            "confirm_progress": confirm_progress,
        }

    @staticmethod
    def _euclidean_distance(
        vec_a: list[float], vec_b: list[float]
    ) -> float:
        """Compute Euclidean distance between two vectors."""
        if len(vec_a) != len(vec_b):
            return 1.0
        squared_sum = sum((a - b) ** 2 for a, b in zip(vec_a, vec_b))
        return math.sqrt(squared_sum)

    @staticmethod
    def _compute_convergence_speed(
        normalized_distances: list[float],
    ) -> float:
        """Compute 1st derivative of distance over last 10 cycles.

        Positive = moving away from manifold (diverging).
        Negative = moving toward manifold (converging).
        """
        if len(normalized_distances) < 2:
            return 0.0
        window = (
            normalized_distances[-10:]
            if len(normalized_distances) >= 10
            else normalized_distances
        )
        if len(window) < 2:
            return 0.0

        # Linear regression slope as 1st derivative
        n = len(window)
        x_vals = list(range(n))
        x_mean = mean(x_vals)
        y_mean = mean(window)
        numerator = sum(
            (x - x_mean) * (y - y_mean) for x, y in zip(x_vals, window)
        )
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        slope = numerator / denominator if denominator != 0 else 0.0
        return slope

    @staticmethod
    def _compute_manifold_dimensions(
        feature_vectors: list[dict],
        normalized_distances: list[float],
    ) -> dict[str, float]:
        """Compute feature loadings from cycles closest to manifold center.

        Takes the tertile of cycles with lowest distance to the manifold
        center, then averages each feature within that group to determine
        its loading on the execution manifold.
        """
        if len(feature_vectors) < 3:
            return {
                "mof_loading": 0.0,
                "signal_loading": 0.0,
                "cb_loading": 0.0,
                "confirm_loading": 0.0,
            }

        # Pair features with distances and sort by distance (ascending)
        paired = list(zip(feature_vectors, normalized_distances))
        paired.sort(key=lambda x: x[1])

        # Take the tertile closest to manifold center
        n_close = max(1, len(paired) // 3)
        close_cycles = [p[0] for p in paired[:n_close]]

        # Compute mean feature values in the close cycles
        mof_loading = mean(fv["mof_score"] for fv in close_cycles)

        # signal_loading: mean of normalized total_signals
        signal_loading = mean(fv["total_signals"] for fv in close_cycles)

        # cb_loading: invert so that low cb_active → high loading (good)
        mean_cb_active = mean(fv["cb_active"] for fv in close_cycles)
        cb_loading = 1.0 - mean_cb_active

        # confirm_loading: mean of confirm_progress
        confirm_loading = mean(fv["confirm_progress"] for fv in close_cycles)

        return {
            "mof_loading": round(mof_loading, 4),
            "signal_loading": round(signal_loading, 4),
            "cb_loading": round(cb_loading, 4),
            "confirm_loading": round(confirm_loading, 4),
        }

    @staticmethod
    def _compute_projection_quality(
        normalized_distances: list[float],
    ) -> float:
        """Projection quality = 1.0 - variance of distances.

        Lower variance means the state consistently projects near the
        manifold, indicating higher quality.
        """
        if len(normalized_distances) < 2:
            return 0.0
        var = variance(normalized_distances)
        # Clamp to [0, 1]
        return max(0.0, min(1.0, 1.0 - var))

    @staticmethod
    def _compute_estimated_convergence(
        current_distance: float,
        convergence_speed: float,
    ) -> int:
        """Estimated cycles until re-entry.

        If the system is converging toward the manifold
        (convergence_speed < 0), compute: current_distance / |speed|.
        If diverging or stationary, return 9999 (infinity).
        """
        if convergence_speed < 0:
            speed = max(abs(convergence_speed), 0.01)
            est = current_distance / speed
            return int(round(est))
        return 9999

    @staticmethod
    def _empty_result() -> dict:
        """Return an empty result dict for error / no-data cases."""
        return {
            "manifold_distance_trajectory": {},
            "current_manifold_distance": 1.0,
            "convergence_speed": 0.0,
            "reentry_probability": 0.0,
            "manifold_dimensions": {
                "mof_loading": 0.0,
                "signal_loading": 0.0,
                "cb_loading": 0.0,
                "confirm_loading": 0.0,
            },
            "projection_quality": 0.0,
            "estimated_convergence_cycles": 9999,
        }
