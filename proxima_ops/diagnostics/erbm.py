import json
import logging
import math
from statistics import stdev
from typing import Any

logger = logging.getLogger("proxima_ops.diagnostics.erbm")

# ---------------------------------------------------------------------------
# ERBM — Execution Restart Basin Mapping
#
# Map state space to identify basins where execution becomes possible.
# Compute distance to nearest basin.
# ---------------------------------------------------------------------------


class ExecutionRestartBasin:
    """Map state space to identify basins of execution viability.

    An "execution basin" is a region in feature space where past cycles
    had a decision other than HOLD.  When no such cycles exist (e.g. all
    HOLD), a single basin is fit over all states and centered on the
    mean of ARMED + high-signal states.
    """

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl") -> None:
        self.log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict:
        """Run ERBM analysis over recent cycles from the cycle log.

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
                logger.warning("No cycle records loaded — returning empty ERBM.")
                return self._empty_result()

            # ---- 1. Build feature vectors per cycle ----
            features, cycle_map, _ = self._extract_features(records)

            if len(features) < 2:
                logger.warning("Fewer than 2 cycles with features — returning empty.")
                return self._empty_result()

            # ---- 2. Identify execution basins via clustering ----
            execute_mask = [
                rec.get("decision", "HOLD") != "HOLD" for rec in records
            ]
            num_execute = sum(execute_mask)

            if num_execute > 0:
                k = min(3, num_execute)
                basin_centers, basin_labels, basin_radii, basin_counts = (
                    self._kmeans_cluster(features, k, execute_mask)
                )
            else:
                # Fallback: k=1 over ALL states centred on ARMED+signal centroid
                k = 1
                # Weighted centroid: give more weight to ARMED + high-signal states
                weights = []
                for i, rec in enumerate(records):
                    w = 1.0
                    segl = (rec.get("segl_state") or "").upper()
                    if segl == "ARMED":
                        w += 2.0
                    sigs = rec.get("total_signals", 0) or 0
                    if sigs > 0:
                        w += float(sigs)
                    weights.append(w)
                # Weighted mean
                dim = len(features[0])
                wsum = sum(weights)
                centroid = [
                    sum(features[j][d] * weights[j] for j in range(len(features)))
                    / wsum
                    for d in range(dim)
                ]
                basin_centers = [centroid]
                # Assign every point to basin 0
                basin_labels = [0] * len(features)
                # Radius = mean distance from centroid
                dists = [
                    self._euclidean(f, centroid) for f in features
                ]
                basin_radii = [sum(dists) / len(dists) if dists else 0.5]
                basin_counts = [len(features)]

            # ---- 3. Per-cycle basin distance ----
            basin_distance_trajectory: dict[int, float] = {}
            nearest_basin_per_cycle: dict[int, int] = {}
            for i, (cycle_num, fv) in enumerate(cycle_map.items()):
                min_dist = float("inf")
                best_bid = 0
                for bid, center in enumerate(basin_centers):
                    d = self._euclidean(fv, center)
                    if d < min_dist:
                        min_dist = d
                        best_bid = bid
                # Normalise 0-1 (cap at 1.0)
                nd = min(1.0, min_dist / math.sqrt(len(fv)))
                basin_distance_trajectory[cycle_num] = round(nd, 4)
                nearest_basin_per_cycle[cycle_num] = best_bid

            # ---- 4. Escape probability ----
            escape_prob = self._compute_escape_probability(
                records, features, basin_centers, basin_labels
            )

            # ---- 5. Basin topology summary ----
            all_dists = list(basin_distance_trajectory.values())
            avg_basin_distance = (
                sum(all_dists) / len(all_dists) if all_dists else 0.0
            )
            all_dists_sorted = sorted(all_dists)
            n = len(all_dists_sorted)
            median_dist = (
                all_dists_sorted[n // 2]
                if n > 0
                else 0.0
            )
            within_median = sum(1 for d in all_dists if d <= median_dist) if n > 0 else 0
            execution_friendly_ratio = within_median / n if n > 0 else 0.0

            # Build execution_basins output
            execution_basins: list[dict] = []
            for bid in range(len(basin_centers)):
                c = basin_centers[bid]
                execution_basins.append(
                    {
                        "basin_id": bid,
                        "center": {
                            "mof_score": round(c[0], 4) if len(c) > 0 else 0.0,
                            "total_signals": round(c[1], 4) if len(c) > 1 else 0.0,
                            "spread": round(c[2], 4) if len(c) > 2 else 0.0,
                            "confirm_pct": round(c[3], 4) if len(c) > 3 else 0.0,
                        },
                        "radius": round(basin_radii[bid], 4),
                        "num_points": basin_counts[bid],
                    }
                )

            return {
                "basin_distance_trajectory": basin_distance_trajectory,
                "execution_basins": execution_basins,
                "nearest_basin_per_cycle": nearest_basin_per_cycle,
                "escape_probability": round(escape_prob, 4),
                "basin_topology": {
                    "num_basins": len(basin_centers),
                    "avg_basin_distance": round(avg_basin_distance, 4),
                    "execution_friendly_ratio": round(execution_friendly_ratio, 4),
                },
            }

        except Exception as exc:
            logger.error("ERBM analysis failed: %s", exc, exc_info=True)
            return self._empty_result()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self, records: list[dict]
    ) -> tuple[list[list[float]], dict[int, list[float]], list[str]]:
        """Build normalised feature vectors from cycle records.

        Returns
        -------
        features : list[list[float]]
            Feature vectors (one per cycle).
        cycle_map : dict[int, list[float]]
            Cycle number -> feature vector.
        feature_names : list[str]
            Names of the feature dimensions.
        """
        raw_vectors: list[dict[str, float]] = []
        feature_names = [
            "mof_score",
            "total_signals",
            "spread",
            "confirm_pct",
            "segl_armed",
        ]

        for rec in records:
            try:
                vec: dict[str, float] = {}

                # mof_score (already 0-1)
                vec["mof_score"] = float(rec.get("mof_score", 0.5) or 0.5)

                # total_signals (raw)
                vec["total_signals"] = float(rec.get("total_signals", 0) or 0)

                # spread proxy from denial_reason (embedded in pipeline_trace)
                vec["spread"] = self._compute_spread_proxy(rec)

                # confirm_pct = confirm_cycles / max_possible (cap at 1)
                confirm_cycles = float(rec.get("confirm_cycles", 0) or 0)
                # Use a denominator of 10 as a reasonable max confirm cycles
                vec["confirm_pct"] = min(1.0, confirm_cycles / 10.0)

                # segl_armed: 1 if ARMED, 0 otherwise
                segl = (rec.get("segl_state") or "OBSERVE").upper()
                vec["segl_armed"] = 1.0 if segl == "ARMED" else 0.0

                raw_vectors.append(vec)
            except Exception:
                continue

        if not raw_vectors:
            return [], {}, []

        # Min-max normalise each dimension across the dataset
        normalized: list[list[float]] = []
        feat_keys = ["mof_score", "total_signals", "spread", "confirm_pct", "segl_armed"]

        for key in feat_keys:
            vals = [v[key] for v in raw_vectors]
            mn = min(vals)
            mx = max(vals)
            rng = mx - mn if mx > mn else 1.0
            for i, v in enumerate(raw_vectors):
                v[key] = (v[key] - mn) / rng

        for v in raw_vectors:
            normalized.append([v[k] for k in feat_keys])

        # Build cycle map
        cycle_map: dict[int, list[float]] = {}
        for i, rec in enumerate(records):
            if i < len(normalized):
                cycle_map[rec.get("cycle", i)] = normalized[i]

        return normalized, cycle_map, feat_keys

    @staticmethod
    def _compute_spread_proxy(rec: dict) -> float:
        """Derive a spread proxy from denial information in pipeline_trace."""
        # Use erp_pressure inverted as the primary spread signal
        erp = float(rec.get("erp_pressure", 0.5) or 0.5)
        base_spread = 1.0 - erp  # invert: high pressure => wide spread

        # Check pipeline_trace for denial information
        trace = rec.get("pipeline_trace", {}) or {}
        execution_msg = str(trace.get("execution", ""))
        governor_gate = trace.get("governor_gate", []) or []

        # Amplify spread when execution was actively denied (not just HOLD)
        denial_keywords = [
            "BLOCKED", "blocked", "DENIED", "denied",
            "NO_SIGNAL", "no best_signal",
        ]
        boost = 0.0
        if any(kw in execution_msg for kw in denial_keywords):
            boost = 0.2
        for msg in governor_gate:
            if any(kw in msg for kw in denial_keywords):
                boost = max(boost, 0.15)

        spread = min(1.0, base_spread + boost)
        return spread

    # ------------------------------------------------------------------
    # Clustering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _euclidean(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _kmeans_cluster(
        self,
        features: list[list[float]],
        k: int,
        execute_mask: list[bool],
    ) -> tuple[list[list[float]], list[int], list[float], list[int]]:
        """Simple k-means over execute cycles only.

        Returns
        -------
        centers : list of centroids
        labels  : cluster label per feature (only execute points are labelled;
                  non-execute points get label -1)
        radii   : mean distance from centroid in each cluster
        counts  : number of execute points per cluster
        """
        # Only cluster the execute cycles
        exec_indices = [i for i, m in enumerate(execute_mask) if m]
        exec_features = [features[i] for i in exec_indices]

        if len(exec_features) < k:
            k = max(1, len(exec_features))

        if not exec_features:
            return [], [], [], []

        dim = len(exec_features[0])
        # Initialise centres: spread across the range
        centres: list[list[float]] = []
        step = max(1, len(exec_features) // k)
        for ki in range(k):
            idx = min(ki * step, len(exec_features) - 1)
            centres.append(list(exec_features[idx]))

        # Iterate (max 50, or until convergence)
        prev_labels: list[int] = [-1] * len(exec_features)
        for _iteration in range(50):
            # Assign
            labels: list[int] = []
            for fv in exec_features:
                dists = [self._euclidean(fv, c) for c in centres]
                labels.append(dists.index(min(dists)))
            if labels == prev_labels:
                break
            prev_labels = list(labels)

            # Update centres
            new_centres: list[list[float]] = []
            for ki in range(k):
                members = [
                    exec_features[j] for j, lb in enumerate(labels) if lb == ki
                ]
                if members:
                    new_c = [
                        sum(m[d] for m in members) / len(members)
                        for d in range(dim)
                    ]
                else:
                    new_c = list(centres[ki])
                new_centres.append(new_c)
            centres = new_centres

        # Build full labels (only execute points get a label)
        full_labels: list[int] = [-1] * len(features)
        for jj, idx in enumerate(exec_indices):
            full_labels[idx] = labels[jj]

        # Compute radii and counts
        radii: list[float] = []
        counts: list[int] = []
        for ki in range(k):
            members = [
                exec_features[j] for j, lb in enumerate(labels) if lb == ki
            ]
            counts.append(len(members))
            if members:
                avg_dist = sum(
                    self._euclidean(m, centres[ki]) for m in members
                ) / len(members)
                radii.append(avg_dist)
            else:
                radii.append(0.0)

        return centres, full_labels, radii, counts

    # ------------------------------------------------------------------
    # Escape probability
    # ------------------------------------------------------------------

    def _compute_escape_probability(
        self,
        records: list[dict],
        features: list[list[float]],
        basin_centers: list[list[float]],
        basin_labels: list[int],
    ) -> float:
        """Probability of reaching a basin in the next 10 cycles.

        Computed as the fraction of past recovery attempts (transitions from
        outside to inside basin) that succeeded.
        """
        try:
            if len(features) < 2 or not basin_centers:
                return 0.0

            # Determine "inside basin" for each cycle
            inside: list[bool] = []
            for i, fv in enumerate(features):
                min_d = float("inf")
                for center in basin_centers:
                    d = self._euclidean(fv, center)
                    if d < min_d:
                        min_d = d
                # Inside if distance < threshold (use median distance as threshold)
                inside.append(min_d < 0.3)  # threshold tuned to ~0.3 normalised

            # Count recovery attempts: transitions from outside to inside
            attempts = 0
            successes = 0
            for i in range(1, len(inside)):
                if not inside[i - 1] and inside[i]:
                    attempts += 1
                    # A recovery "succeeds" if the decision was EXECUTE or
                    # the state stayed inside for at least 3 consecutive cycles
                    j = i
                    consecutive_inside = 0
                    while j < len(inside) and inside[j]:
                        consecutive_inside += 1
                        j += 1
                    if consecutive_inside >= 3:
                        successes += 1
                    elif records[i].get("decision") in ("EXECUTE",):
                        successes += 1

            if attempts == 0:
                return 0.0
            return successes / attempts

        except Exception as exc:
            logger.debug("Error computing escape probability: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # I/O helpers
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
                        logger.debug("Skipping malformed JSONL line: %.80s", stripped)
                        continue
            return records[-n_recent:] if n_recent < len(records) else records
        except FileNotFoundError:
            logger.warning("Cycle log not found at %s", self.log_path)
            return []
        except Exception as exc:
            logger.error("Error loading cycle log: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Empty result
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict:
        return {
            "basin_distance_trajectory": {},
            "execution_basins": [],
            "nearest_basin_per_cycle": {},
            "escape_probability": 0.0,
            "basin_topology": {
                "num_basins": 0,
                "avg_basin_distance": 0.0,
                "execution_friendly_ratio": 0.0,
            },
        }
