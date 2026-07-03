"""
AEEM — Attractor Escape Energy Model

Model system state as an energy landscape over decision dynamics.
Compute basin depth, escape energy, gradient trajectory, minimal
escape path conditions, and spontaneous escape frequency.

Reads from state/wave12_cycle_log.jsonl.
"""

import json
import logging
from collections import Counter
from statistics import mean, stdev
from typing import Any

logger = logging.getLogger("proxima_ops.diagnostics.aeem")

# ---------------------------------------------------------------------------
# MoF readiness lookup (mirrors ERF convention)
# ---------------------------------------------------------------------------
_MOF_READINESS: dict[str, float] = {
    "INFORMATION_RICH": 0.9,
    "STRUCTURE_LIMITED": 0.5,
    "NOISE": 0.1,
}

# ---------------------------------------------------------------------------
# SEGL state → energy contribution
# ---------------------------------------------------------------------------
_SEGL_ENERGY: dict[str, float] = {
    "OBSERVE": 0.8,  # high entropy / low energy
    "ARMED": 0.2,  # low entropy / high energy
}

# ---------------------------------------------------------------------------
# Conditions evaluated for correlation with HOLD→non-HOLD transitions
# ---------------------------------------------------------------------------
_ESCAPE_CONDITIONS = [
    "mof_score",
    "spread_proxy",
    "cb_cleared",
    "confirm_cycles",
]


class AttractorEscapeEnergy:
    """Model system state as energy landscape over decision dynamics.

    Parameters
    ----------
    log_path : str
        Path to the wave12 cycle log JSONL file.
    window : int
        Rolling window size for gradient computation (default 20).
    """

    def __init__(
        self, log_path: str = "state/wave12_cycle_log.jsonl", window: int = 20
    ) -> None:
        self.log_path = log_path
        self.window = window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Run AEEM analysis over the most recent cycles.

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
                logger.warning("No cycle records loaded — returning empty AEEM.")
                return self._empty_result()

            # ---- 1. Basin depth (fraction of cycles in HOLD) ----
            basin_depth = self._compute_basin_depth(records)

            # ---- 2. Escape energy required ----
            escape_energy = self._compute_escape_energy(records)

            # ---- 3. Escape gradient trajectory ----
            gradient = self._compute_gradient_trajectory(records)

            # ---- 4. Minimal escape path (top-3 correlated conditions) ----
            minimal_escape_path = self._minimal_escape_path(records)

            # ---- 5. Spontaneous escape frequency ----
            spontaneous_freq = self._spontaneous_escape_frequency(records)

            # ---- 6. Energy landscape summary ----
            landscape = self._energy_landscape_summary(records)

            return {
                "basin_depth": round(basin_depth, 6),
                "escape_energy_required": round(escape_energy, 6),
                "escape_gradient_trajectory": {
                    str(k): round(v, 6) for k, v in gradient.items()
                },
                "minimal_escape_path": minimal_escape_path,
                "spontaneous_escape_frequency": round(spontaneous_freq, 6),
                "energy_landscape_summary": {
                    "current_net_energy": round(landscape["current_net_energy"], 6),
                    "barrier_height": round(landscape["barrier_height"], 6),
                    "nearest_saddle": landscape["nearest_saddle"],
                },
            }

        except Exception as exc:
            logger.error(
                "AEEM analysis failed: %s", exc, exc_info=True
            )
            return self._empty_result()

    # ------------------------------------------------------------------
    # Internals — basin & escape
    # ------------------------------------------------------------------

    def _compute_basin_depth(self, records: list[dict]) -> float:
        """Fraction of cycles with decision == HOLD (0.0–1.0)."""
        if not records:
            return 0.0
        hold_count = sum(
            1 for r in records if str(r.get("decision", "")).upper() == "HOLD"
        )
        return hold_count / len(records)

    def _compute_escape_energy(self, records: list[dict]) -> float:
        """Energy required to escape the HOLD attractor.

        Uses current ERF-like readiness if available, else derives from
        mof_score and signal_quality-like proxy.

        Returns 0.0–1.0 where higher means more energy needed to escape.
        """
        if not records:
            return 1.0

        latest = records[-1]
        # Attempt to derive readiness: use mof_score as primary signal
        mof_score = float(latest.get("mof_score", 0.0) or 0.0)

        # Signal quality proxy: fraction of active_signals relative to core_signals
        active_sigs = float(latest.get("active_signals", 0) or 0)
        core_sigs = float(latest.get("core_signals", 0) or 0)
        signal_quality = (
            min(1.0, active_sigs / core_sigs) if core_sigs > 0 else 0.0
        )

        # Readiness = min(mof_score, signal_quality)
        readiness = min(mof_score, signal_quality)

        # Escape energy = 1.0 - readiness, clamped to [0, 1]
        return max(0.0, min(1.0, 1.0 - readiness))

    def _compute_gradient_trajectory(
        self, records: list[dict]
    ) -> dict[int, float]:
        """Change in basin depth over rolling windows (∇D per window).

        Each window-sized chunk of cycles gets a basin depth.  The gradient
        is the difference between consecutive windows: positive means the
        basin is deepening (worse).
        """
        if len(records) < self.window + 1:
            return {}

        gradient: dict[int, float] = {}
        depths: list[float] = []

        # Slide a window of size `self.window` across records
        for i in range(0, len(records) - self.window + 1):
            chunk = records[i : i + self.window]
            hold_c = sum(
                1
                for r in chunk
                if str(r.get("decision", "")).upper() == "HOLD"
            )
            depths.append(hold_c / self.window)

        # Gradients at each window boundary
        for i in range(1, len(depths)):
            # Use the cycle number at end of the second window as key
            end_idx = i * self.window + self.window - 1
            if end_idx < len(records):
                cycle_key = records[end_idx].get("cycle", end_idx)
                gradient[cycle_key] = depths[i] - depths[i - 1]

        return gradient

    # ------------------------------------------------------------------
    # Internals — minimal escape path
    # ------------------------------------------------------------------

    def _minimal_escape_path(self, records: list[dict]) -> list[str]:
        """Return top-3 conditions that correlate with HOLD→non-HOLD escape.

        Conditions evaluated:
          - mof_score improvement (increase)
          - spread_proxy improvement (decrease in erp_pressure)
          - cb_cleared (cb_decision == "Allowed")
          - confirm_cycles increase

        Returns human-readable strings describing the highest-correlation
        conditions, sorted by strength descending.
        """
        # We need at least a few transitions to compute correlations
        transitions = self._find_hold_exit_transitions(records)
        if len(transitions) < 3:
            return self._fallback_escape_path(records)

        # For each transition, compute delta of each condition
        deltas: dict[str, list[float]] = {cond: [] for cond in _ESCAPE_CONDITIONS}
        # We also track whether the transition is HOLD→non-HOLD (True) or non-HOLD→HOLD (False)
        # For minimal escape path we care about HOLD→non-HOLD specifically.
        escape_flags: list[bool] = []

        for i, j in transitions:
            before = records[i]
            after = records[j]
            # Is this a HOLD→non-HOLD transition?
            decision_before = str(before.get("decision", "")).upper()
            decision_after = str(after.get("decision", "")).upper()
            is_escape = decision_before == "HOLD" and decision_after != "HOLD"
            escape_flags.append(is_escape)

            # Delta: after - before for each condition
            # mof_score
            mof_b = float(before.get("mof_score", 0.0) or 0.0)
            mof_a = float(after.get("mof_score", 0.0) or 0.0)
            deltas["mof_score"].append(mof_a - mof_b)

            # spread_proxy: use 1.0 - erp_pressure as inverse spread
            ep_b = float(before.get("erp_pressure", 0.5) or 0.5)
            ep_a = float(after.get("erp_pressure", 0.5) or 0.5)
            spread_b = 1.0 - ep_b
            spread_a = 1.0 - ep_a
            deltas["spread_proxy"].append(spread_a - spread_b)

            # cb_cleared: was cb_decision "Allowed" (1.0) vs not (0.0)?
            cb_b = 1.0 if str(before.get("cb_decision", "")).lower() == "allowed" else 0.0
            cb_a = 1.0 if str(after.get("cb_decision", "")).lower() == "allowed" else 0.0
            deltas["cb_cleared"].append(cb_a - cb_b)

            # confirm_cycles
            conf_b = float(before.get("confirm_cycles", 0) or 0)
            conf_a = float(after.get("confirm_cycles", 0) or 0)
            deltas["confirm_cycles"].append(conf_a - conf_b)

        # Compute point-biserial-like correlation for each condition with escape_flags
        # For simplicity, compare mean delta when escape=True vs escape=False.
        # Higher absolute difference = stronger correlation.
        scores: list[tuple[float, str, str]] = []  # (strength, direction_label, condition_name)

        for cond in _ESCAPE_CONDITIONS:
            vals = deltas[cond]
            if not vals:
                continue
            escape_deltas = [
                vals[k] for k in range(len(escape_flags)) if escape_flags[k]
            ]
            non_escape_deltas = [
                vals[k] for k in range(len(escape_flags)) if not escape_flags[k]
            ]

            if not escape_deltas or not non_escape_deltas:
                continue

            mean_esc = mean(escape_deltas)
            mean_non = mean(non_escape_deltas)
            diff = abs(mean_esc - mean_non)

            # Determine direction that helps escape
            if mean_esc > mean_non:
                # Higher delta helps escape
                if cond == "mof_score":
                    label = "increase mof_score"
                elif cond == "spread_proxy":
                    label = "decrease erp_pressure (narrower spread)"
                elif cond == "cb_cleared":
                    label = "clear circuit breaker"
                elif cond == "confirm_cycles":
                    label = "increase confirm_cycles"
                else:
                    label = f"increase {cond}"
            else:
                # Lower delta helps escape
                if cond == "mof_score":
                    label = "decrease mof_score (less relevant)"
                elif cond == "spread_proxy":
                    label = "increase erp_pressure (wider spread)"
                elif cond == "cb_cleared":
                    label = "trigger circuit breaker"
                elif cond == "confirm_cycles":
                    label = "decrease confirm_cycles"
                else:
                    label = f"decrease {cond}"

            scores.append((diff, label, cond))

        # Sort by strength descending, take top 3
        scores.sort(key=lambda x: x[0], reverse=True)
        top3 = scores[:3]

        if not top3:
            return self._fallback_escape_path(records)

        return [t[1] for t in top3]

    def _find_hold_exit_transitions(
        self, records: list[dict]
    ) -> list[tuple[int, int]]:
        """Find indices where decision changes between consecutive records.

        Returns list of (before_idx, after_idx) pairs.
        """
        transitions: list[tuple[int, int]] = []
        for i in range(len(records) - 1):
            d1 = str(records[i].get("decision", "")).upper()
            d2 = str(records[i + 1].get("decision", "")).upper()
            if d1 != d2:
                transitions.append((i, i + 1))
        return transitions

    def _fallback_escape_path(self, records: list[dict]) -> list[str]:
        """Fallback when not enough transitions exist for correlation analysis.

        Return generic guidance based on current state.
        """
        if not records:
            return [
                "increase mof_score",
                "decrease erp_pressure (narrower spread)",
                "clear circuit breaker",
            ]

        latest = records[-1]
        suggestions: list[str] = []

        # mof_state
        mof_state = str(latest.get("mof_state", "")).upper()
        if mof_state != "INFORMATION_RICH":
            suggestions.append("improve mof_state toward INFORMATION_RICH")

        # erp_pressure
        ep = float(latest.get("erp_pressure", 0.5) or 0.5)
        if ep > 0.7:
            suggestions.append("reduce erp_pressure (narrow spreads)")

        # cb_decision
        cb = str(latest.get("cb_decision", "")).lower()
        if "triggered" in cb:
            suggestions.append("clear circuit breaker (cb_decision = Allowed)")

        # confirm_cycles
        conf = int(latest.get("confirm_cycles", 0) or 0)
        if conf < 2:
            suggestions.append("increase confirm_cycles >= 2")

        # Fill remaining slots with generic advice
        while len(suggestions) < 3:
            suggestions.append("increase signal quality via mof_score improvement")

        return suggestions[:3]

    # ------------------------------------------------------------------
    # Internals — spontaneous escape
    # ------------------------------------------------------------------

    def _spontaneous_escape_frequency(self, records: list[dict]) -> float:
        """Fraction of cycles where decision changes FROM HOLD TO non-HOLD.

        A "spontaneous escape" is a transition from HOLD in cycle N to
        something other than HOLD in cycle N+1.
        """
        if len(records) < 2:
            return 0.0

        escapes = 0
        for i in range(len(records) - 1):
            d1 = str(records[i].get("decision", "")).upper()
            d2 = str(records[i + 1].get("decision", "")).upper()
            if d1 == "HOLD" and d2 != "HOLD":
                escapes += 1

        return escapes / (len(records) - 1)

    # ------------------------------------------------------------------
    # Internals — energy landscape
    # ------------------------------------------------------------------

    def _energy_landscape_summary(
        self, records: list[dict]
    ) -> dict[str, Any]:
        """Compute net energy, barrier height, and nearest saddle point.

        *Current net energy* = SE - GE (Signal Energy - Gate Entropy),
        borrowing the DST module's concepts but computing directly:
            SE ≈ mof_score × signal_quality
            GE ≈ (1 - cb_state) where cb_state == 0 if triggered, 1 if clear

        *Barrier height* = max(SE - GE) in ARMED states - min(SE - GE) in OBSERVE states

        *Nearest saddle* = state condition closest to the transition boundary
        (where the decision flips).
        """
        if not records:
            return {
                "current_net_energy": 0.0,
                "barrier_height": 0.0,
                "nearest_saddle": "unknown",
            }

        # --- Per-cycle net energy ---
        net_energies: list[float] = []
        segl_states: list[str] = []

        for rec in records:
            se = self._signal_energy(rec)
            ge = self._gate_entropy(rec)
            net_energies.append(se - ge)
            segl_states.append(
                str(rec.get("segl_state", "OBSERVE")).upper()
            )

        current_net = net_energies[-1] if net_energies else 0.0

        # --- Barrier height ---
        armed_net = [
            net_energies[i]
            for i, s in enumerate(segl_states)
            if s == "ARMED"
        ]
        observe_net = [
            net_energies[i]
            for i, s in enumerate(segl_states)
            if s == "OBSERVE"
        ]

        max_armed = max(armed_net) if armed_net else 0.0
        min_observe = min(observe_net) if observe_net else 0.0
        barrier_height = abs(max_armed - min_observe)

        # --- Nearest saddle ---
        # Find the condition that is closest to a decision transition.
        # Evaluate conditions at the mid-point between transition boundaries.
        nearest_saddle = self._find_nearest_saddle(records, net_energies)

        return {
            "current_net_energy": current_net,
            "barrier_height": barrier_height,
            "nearest_saddle": nearest_saddle,
        }

    def _signal_energy(self, rec: dict) -> float:
        """Compute Signal Energy for one cycle.

        SE = mof_score × signal_quality
        signal_quality = active_signals / max(1, core_signals)
        """
        mof_score = float(rec.get("mof_score", 0.0) or 0.0)
        active_sigs = float(rec.get("active_signals", 0) or 0)
        core_sigs = float(rec.get("core_signals", 0) or 0)
        signal_quality = (
            min(1.0, active_sigs / core_sigs) if core_sigs > 0 else 0.0
        )
        return mof_score * signal_quality

    def _gate_entropy(self, rec: dict) -> float:
        """Compute Gate Entropy for one cycle.

        GE = 1.0 - cb_state
        where cb_state = 0 if cb_decision contains "triggered", else 1.0.
        """
        cb = str(rec.get("cb_decision", "")).lower()
        cb_state = 0.0 if ("triggered" in cb) else 1.0
        return 1.0 - cb_state

    def _find_nearest_saddle(
        self, records: list[dict], net_energies: list[float]
    ) -> str:
        """Find the state condition closest to a decision transition boundary.

        A saddle point occurs where net_energy is near zero (transition zone).
        Examine the state variables at cycles where net_energy ≈ 0.
        """
        # Find transition cycles (where decision changes)
        saddle_conditions: list[str] = []

        for i in range(1, len(records)):
            d1 = str(records[i - 1].get("decision", "")).upper()
            d2 = str(records[i].get("decision", "")).upper()
            if d1 == d2:
                continue

            # This is a transition — examine the "before" state
            rec = records[i - 1]
            net_i = net_energies[i - 1] if i - 1 < len(net_energies) else 0.0

            conditions: list[str] = []
            mof_state = str(rec.get("mof_state", "")).upper()
            segl_state = str(rec.get("segl_state", "")).upper()
            ep = float(rec.get("erp_pressure", 0.5) or 0.5)
            cb = str(rec.get("cb_decision", "")).lower()

            conditions.append(f"mof_state={mof_state}")
            conditions.append(f"segl_state={segl_state}")
            conditions.append(f"erp_pressure={ep:.2f}")
            conditions.append(
                "cb_triggered=True" if "triggered" in cb else "cb_cleared=True"
            )

            # Saddle is near net_energy ≈ 0
            if abs(net_i) < 0.3:
                saddle_conditions.append(", ".join(conditions))

        if saddle_conditions:
            # Return the most frequently appearing saddle condition
            counter = Counter(saddle_conditions)
            return counter.most_common(1)[0][0]

        # Fallback: current state
        if records:
            latest = records[-1]
            mof_state = str(latest.get("mof_state", "")).upper()
            segl_state = str(latest.get("segl_state", "")).upper()
            ep = float(latest.get("erp_pressure", 0.5) or 0.5)
            cb = str(latest.get("cb_decision", "")).lower()
            cb_str = "cb_triggered=True" if "triggered" in cb else "cb_cleared=True"
            return f"mof_state={mof_state}, segl_state={segl_state}, erp_pressure={ep:.2f}, {cb_str}"

        return "unknown"

    # ------------------------------------------------------------------
    # Internals — I/O
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
            return records[-n_recent:] if n_recent < len(records) else records
        except FileNotFoundError:
            logger.warning("Cycle log not found at %s", self.log_path)
            return []
        except Exception as exc:
            logger.error("Error loading cycle log: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internals — empty result
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return a zero-filled result dict for error / no-data cases."""
        return {
            "basin_depth": 0.0,
            "escape_energy_required": 1.0,
            "escape_gradient_trajectory": {},
            "minimal_escape_path": [
                "increase mof_score",
                "decrease erp_pressure (narrower spread)",
                "clear circuit breaker",
            ],
            "spontaneous_escape_frequency": 0.0,
            "energy_landscape_summary": {
                "current_net_energy": 0.0,
                "barrier_height": 0.0,
                "nearest_saddle": "unknown",
            },
        }


# ---------------------------------------------------------------------------
# Quick CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    aeem = AttractorEscapeEnergy()
    result = aeem.analyze(n_recent_cycles=500)
    print(json.dumps(result, indent=2, default=str))
