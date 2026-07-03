"""HNC — Hysteresis Neutralization Controller.

Measure trigger/reset asymmetry in CircuitBreaker, compute symmetry
convergence score.
"""

from __future__ import annotations

import json
from typing import Any


class HysteresisNeutralization:
    """Analyse CircuitBreaker hysteresis neutralisation from a wave cycle log.

    Identifies CB trigger events and their subsequent reset, then compares
    the market conditions (mof_state, spread, open_positions) at trigger
    vs reset to quantify asymmetry and propose topology corrections.

    Parameters
    ----------
    log_path : str
        Path to the JSONL cycle log (default
        ``"state/wave12_cycle_log.jsonl"``).
    """

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl") -> None:
        self.log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Run the full HNC analysis.

        Parameters
        ----------
        n_recent_cycles : int
            Number of most-recent cycles to analyse (default 500).

        Returns
        -------
        dict
            See class docstring for output schema.
        """
        try:
            rows = self._load_log()
        except FileNotFoundError:
            return self._empty_result()
        except json.JSONDecodeError:
            return self._empty_result()

        if not rows:
            return self._empty_result()

        # Keep only the N most recent cycles
        rows = rows[-n_recent_cycles:]

        try:
            return self._compute(rows)
        except Exception:
            return self._empty_result()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_log(self) -> list[dict[str, Any]]:
        """Parse every JSON line from *log_path*."""
        rows: list[dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows

    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "trigger_reset_asymmetry": 0.0,
            "hysteresis_depth_trajectory": {},
            "symmetry_convergence_score": 0.0,
            "basin_overlap": 0.0,
            "trigger_conditions_profile": {"mof_state": "", "spread_range": ""},
            "reset_conditions_profile": {"mof_state": "", "spread_range": ""},
            "topology_correction_suggestion": "",
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _is_cb_triggered(record: dict[str, Any]) -> bool:
        """Return True when *record* has a CircuitBreaker denial reason."""
        denial = record.get("denial_reason", "")
        return isinstance(denial, str) and "CircuitBreaker" in denial

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_condition_profile(row: dict[str, Any]) -> dict[str, Any]:
        """Extract the condition snapshot at a given cycle row.

        Returns
        -------
        dict with keys: mof_state, mof_score, open_positions
        """
        return {
            "mof_state": row.get("mof_state", ""),
            "mof_score": row.get("mof_score", 0.0),
            "open_positions": row.get("open_positions", 0),
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _spread_range(
        profiles: list[dict[str, Any]],
    ) -> str:
        """Describe the spread range across a list of condition profiles.

        Returns a compact string like ``"0.85-0.92"`` or ``""`` when
        no score data is available.
        """
        scores = [p["mof_score"] for p in profiles if p["mof_score"] is not None]
        if not scores:
            return ""
        lo, hi = min(scores), max(scores)
        if lo == hi:
            return f"{lo:.3f}"
        return f"{lo:.3f}-{hi:.3f}"

    # ------------------------------------------------------------------

    @staticmethod
    def _dominant_mof_state(profiles: list[dict[str, Any]]) -> str:
        """Return the most frequent mof_state among *profiles*.

        Ties are broken by first occurrence.  Returns ``""`` when
        *profiles* is empty.
        """
        if not profiles:
            return ""
        counts: dict[str, int] = {}
        for p in profiles:
            state = p.get("mof_state", "")
            if state:
                counts[state] = counts.get(state, 0) + 1
        if not counts:
            return ""
        # most-frequent; tie → higher first occurrence
        best = ""
        best_count = -1
        for p in profiles:
            state = p.get("mof_state", "")
            if state and counts.get(state, 0) > best_count:
                best_count = counts[state]
                best = state
        return best

    # ------------------------------------------------------------------

    @staticmethod
    def _topology_suggestion(
        trigger_profile: dict[str, Any],
        reset_profile: dict[str, Any],
    ) -> str:
        """Generate a topology-correction suggestion string.

        Compares mof_state, mof_score, and open_positions between trigger
        and reset profiles to decide which knob would most reduce
        hysteresis.
        """
        t_mof = trigger_profile.get("mof_state", "")
        r_mof = reset_profile.get("mof_state", "")
        t_score = trigger_profile.get("mof_score", 0.0) or 0.0
        r_score = reset_profile.get("mof_score", 0.0) or 0.0
        t_pos = trigger_profile.get("open_positions", 0) or 0
        r_pos = reset_profile.get("open_positions", 0) or 0

        diffs: list[str] = []

        if t_mof != r_mof:
            diffs.append("mof_state")
        if abs(t_score - r_score) > 0.01:
            diffs.append("spread")
        if t_pos != r_pos:
            diffs.append("open_positions")

        if not diffs:
            return "No correction needed — trigger and reset conditions are symmetric."

        if "mof_state" in diffs:
            return (
                "Flatten mof_state trigger threshold — "
                "different mof_state at trigger vs reset indicates "
                "state-dependent hysteresis."
            )
        if "spread" in diffs:
            return (
                "Adjust spread sensitivity — "
                "mof_score differs between trigger and reset, "
                "suggesting spread-driven asymmetry."
            )
        if "open_positions" in diffs:
            return (
                "Add time-based decay to CB reset — "
                "open_positions differ between trigger and reset, "
                "indicating position-dependent latch behaviour."
            )

        return "No correction needed."

    # ------------------------------------------------------------------

    def _compute(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Core analysis logic on pre-loaded, trimmed *rows*."""
        # ── 1. Build CB trigger-event blocks ──────────────────────────
        # A CB trigger event starts when denial_reason first contains
        # "CircuitBreaker" and ends when denial_reason no longer mentions
        # it for at least one complete cycle.
        trigger_events: list[dict[str, Any]] = []

        i = 0
        while i < len(rows):
            if self._is_cb_triggered(rows[i]):
                event_start_cycle = rows[i].get("cycle", 0)

                # Snapshot conditions at trigger onset
                trigger_profile = self._extract_condition_profile(rows[i])

                # Collect all profiles during the CB block (for
                # hysteresis_depth_trajectory and spread_range)
                block_profiles: list[dict[str, Any]] = [trigger_profile]

                # Walk forward while CB is still triggered
                j = i
                while j < len(rows) and self._is_cb_triggered(rows[j]):
                    block_profiles.append(
                        self._extract_condition_profile(rows[j])
                    )
                    j += 1

                # j is the first index where CB is no longer triggered
                reset_profile: dict[str, Any] | None = None
                event_end_cycle: int | None = None

                if j < len(rows):
                    # The cycle immediately after the CB block is the
                    # reset cycle
                    reset_profile = self._extract_condition_profile(rows[j])
                    event_end_cycle = rows[j - 1].get("cycle", 0)
                else:
                    # CB is still active at end of data — no reset
                    event_end_cycle = rows[-1].get("cycle", 0)

                # Duration in cycles (inclusive of start and end)
                if event_end_cycle is not None:
                    duration = event_end_cycle - event_start_cycle + 1
                else:
                    duration = len(rows) - i

                trigger_events.append({
                    "start_cycle": event_start_cycle,
                    "end_cycle": event_end_cycle,
                    "duration": duration,
                    "trigger_profile": trigger_profile,
                    "reset_profile": reset_profile,
                    "block_profiles": block_profiles,
                })

                i = j  # skip past this event
            else:
                i += 1

        # ── 2. Compute asymmetry and overlap ──────────────────────────
        # Compare trigger vs reset profiles across all events that have
        # a reset.

        total_conditions = 0
        matching_conditions = 0

        # Collect mof_state and spread info for aggregate profiles
        all_trigger_mofs: list[str] = []
        all_reset_mofs: list[str] = []
        all_trigger_scores: list[float] = []
        all_reset_scores: list[float] = []

        for ev in trigger_events:
            t_profile = ev["trigger_profile"]
            r_profile = ev["reset_profile"]

            if r_profile is None:
                continue  # still active, no reset to compare

            # Collect for aggregate profiles
            t_mof = t_profile.get("mof_state", "")
            r_mof = r_profile.get("mof_state", "")
            t_score = t_profile.get("mof_score", 0.0) or 0.0
            r_score = r_profile.get("mof_score", 0.0) or 0.0
            t_pos = t_profile.get("open_positions", 0) or 0
            r_pos = r_profile.get("open_positions", 0) or 0

            if t_mof:
                all_trigger_mofs.append(t_mof)
            if r_mof:
                all_reset_mofs.append(r_mof)
            if t_score is not None:
                all_trigger_scores.append(float(t_score))
            if r_score is not None:
                all_reset_scores.append(float(r_score))

            # Compare condition 1: mof_state
            if t_mof and r_mof:
                total_conditions += 1
                if t_mof == r_mof:
                    matching_conditions += 1

            # Compare condition 2: mof_score (spread proxy)
            if t_score is not None and r_score is not None:
                total_conditions += 1
                if abs(t_score - r_score) <= 0.01:
                    matching_conditions += 1

            # Compare condition 3: open_positions
            total_conditions += 1
            if t_pos == r_pos:
                matching_conditions += 1

        # Asymmetry = 1 - (matching / total)
        if total_conditions > 0:
            asymmetry = 1.0 - (matching_conditions / total_conditions)
            basin_overlap = matching_conditions / total_conditions
        else:
            asymmetry = 0.0
            basin_overlap = 0.0

        symmetry_convergence = 1.0 - asymmetry

        # ── 3. Hysteresis depth trajectory ────────────────────────────
        # For each trigger event, record a normalised hysteresis depth
        # score: duration_as_fraction_of_row_window.  We use len(rows)
        # as the window denominator because cycle numbers in the log may
        # wrap around (e.g. 35489 → 1).
        hysteresis_trajectory: dict[str, float] = {}
        row_window = len(rows)
        if row_window > 0:
            for ev in trigger_events:
                key = f"cycle_{ev['start_cycle']}"
                hysteresis_trajectory[key] = round(
                    ev["duration"] / row_window, 4
                )

        # ── 4. Aggregate condition profiles ───────────────────────────
        trigger_mof = self._dominant_mof_state(
            [{"mof_state": m} for m in all_trigger_mofs]
        )
        reset_mof = self._dominant_mof_state(
            [{"mof_state": m} for m in all_reset_mofs]
        )

        trigger_spread = self._spread_range(
            [{"mof_score": s} for s in all_trigger_scores]
        )
        reset_spread = self._spread_range(
            [{"mof_score": s} for s in all_reset_scores]
        )

        # ── 5. Topology suggestion ────────────────────────────────────
        # Use the aggregate profiles for the suggestion (most recent
        # complete trigger/reset pair if available, else the last event
        # with a reset).
        suggestion = ""
        complete_events = [
            ev for ev in trigger_events if ev["reset_profile"] is not None
        ]
        if complete_events:
            last_complete = complete_events[-1]
            agg_trigger = {
                "mof_state": trigger_mof or last_complete["trigger_profile"].get("mof_state", ""),
                "mof_score": (
                    sum(all_trigger_scores) / len(all_trigger_scores)
                    if all_trigger_scores
                    else last_complete["trigger_profile"].get("mof_score", 0.0)
                ),
                "open_positions": last_complete["trigger_profile"].get("open_positions", 0),
            }
            agg_reset = {
                "mof_state": reset_mof or last_complete["reset_profile"].get("mof_state", ""),
                "mof_score": (
                    sum(all_reset_scores) / len(all_reset_scores)
                    if all_reset_scores
                    else last_complete["reset_profile"].get("mof_score", 0.0)
                ),
                "open_positions": last_complete["reset_profile"].get("open_positions", 0),
            }
            suggestion = self._topology_suggestion(agg_trigger, agg_reset)
        else:
            suggestion = "No complete trigger/reset events to evaluate."

        return {
            "trigger_reset_asymmetry": round(asymmetry, 4),
            "hysteresis_depth_trajectory": hysteresis_trajectory,
            "symmetry_convergence_score": round(symmetry_convergence, 4),
            "basin_overlap": round(basin_overlap, 4),
            "trigger_conditions_profile": {
                "mof_state": trigger_mof,
                "spread_range": trigger_spread,
            },
            "reset_conditions_profile": {
                "mof_state": reset_mof,
                "spread_range": reset_spread,
            },
            "topology_correction_suggestion": suggestion,
        }
