"""GovernorSimplificationSim: Simulate simplified governor variants.

Simulates what WOULD happen if governor rules were removed or relaxed —
WITHOUT affecting the live system. Reads historical cycle logs and
re-plays each blocked cycle through hypothetical variant rule sets.

Variants
--------
- no_frequency_budget : remove VEL frequency / burst prevention rules
- no_spread_filter    : remove spread-based filtering (if present)
- no_confirm_gate     : remove cross-projection confirm gate dependency
- minimal_governor    : keep only circuit_breaker and VEL rules

Pipeline
--------
Cycle log contains one JSON object per line.  Each log line stores the
execution outcome in ``pipeline_trace.execution``.  Seven distinct outcomes
are observed in the 27 548‑line data set:

    NO_SIGNAL no best_signal passed all gates           (11 494)
    DENIED CB: Circuit breaker already triggered        (10 264)
    DENIED segl_state=OBSERVE intent=True                (5 755)
    DENIED cross_confirm=1/2                                (12)
    DENIED VEL: exposure_smoothing: ...                     (10)
    DENIED VEL: burst_prevention: ...                       (10)
    FAILED MT5 place_order returned None                    (3)

Simulation logic
----------------
A "blocked" cycle becomes a simulated "additional trade" for a variant
*iff* the ONLY blockers in that cycle belong to the rule set the variant
removes.  Since every blocked cycle in the log has exactly one denial
reason, this reduces to: count blocked cycles whose denial reason matches
the removed rules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("proxima_ops.simulation.governor_simplification_sim")


class GovernorSimplificationSim:
    """Simulate simplified governor variants on historical cycle data.

    Parameters
    ----------
    log_path : str
        Path to the JSONL cycle log file (default
        ``"state/wave12_cycle_log.jsonl"``).
    """

    # ------------------------------------------------------------------
    # Denial-reason classifiers — maps execution string → category
    # ------------------------------------------------------------------
    _CATEGORY_VEL = "vel"
    _CATEGORY_SPREAD = "spread"
    _CATEGORY_CONFIRM = "confirm"
    _CATEGORY_SEGL = "segl"
    _CATEGORY_CB = "circuit_breaker"
    _CATEGORY_OTHER = "other"

    @staticmethod
    def _classify_execution(execution: str) -> str:
        """Return the rule category for an execution outcome string.

        Classification
        -------------
        * ``vel``           — any ``DENIED VEL: …``  (frequency / burst)
        * ``spread``        — any ``DENIED spread…`` (if present)
        * ``confirm``       — ``DENIED cross_confirm=…``
        * ``segl``          — ``DENIED segl_state=…``
        * ``circuit_breaker`` — ``DENIED CB: …``
        * ``other``         — ``FAILED …``, ``NO_SIGNAL …``, empty, etc.
        """
        if not isinstance(execution, str):
            return GovernorSimplificationSim._CATEGORY_OTHER

        if execution.startswith("DENIED VEL:"):
            return GovernorSimplificationSim._CATEGORY_VEL
        if execution.startswith("DENIED spread"):
            return GovernorSimplificationSim._CATEGORY_SPREAD
        if execution.startswith("DENIED cross_confirm="):
            return GovernorSimplificationSim._CATEGORY_CONFIRM
        if execution.startswith("DENIED segl_state="):
            return GovernorSimplificationSim._CATEGORY_SEGL
        if execution.startswith("DENIED CB:"):
            return GovernorSimplificationSim._CATEGORY_CB
        return GovernorSimplificationSim._CATEGORY_OTHER

    # ------------------------------------------------------------------
    # Variant definitions: which categories are *removed* in each variant
    # ------------------------------------------------------------------
    _VARIANTS: dict[str, dict[str, Any]] = {
        "no_frequency_budget": {
            "removed_categories": {_CATEGORY_VEL},
            "description": "Remove VEL frequency budget and burst prevention rules",
        },
        "no_spread_filter": {
            "removed_categories": {_CATEGORY_SPREAD},
            "description": "Remove spread-based filter rule (no spread denials exist in current log)",
        },
        "no_confirm_gate": {
            "removed_categories": {_CATEGORY_CONFIRM},
            "description": "Remove cross-projection confirm gate dependency",
        },
        "minimal_governor": {
            "removed_categories": {_CATEGORY_SEGL, _CATEGORY_CONFIRM},
            "description": "Keep only circuit breaker and VEL rules; drop SEGL state gating and confirm gate",
        },
    }

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl") -> None:
        self._log_path = Path(log_path)
        self._log_lines: list[dict[str, Any]] = []
        logger.debug("GovernorSimplificationSim initialised with log_path=%s", log_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Run all variant simulations and return results.

        Parameters
        ----------
        n_recent_cycles : int
            Number of most-recent cycles to analyse (default 500).
            Pass a large number (e.g. 100_000) to process all available data.

        Returns
        -------
        dict
            ``baseline_blocks``, ``variant_results`` (per-variant dicts),
            ``best_variant`` (name of variant with most additional trades),
            ``warning`` (simulation notice).
        """
        try:
            self._load_logs(n_recent_cycles)
        except Exception:
            logger.exception("Failed to load or parse cycle log")
            return self._empty_result("Failed to load cycle log — see logs")

        try:
            baseline = self._compute_baseline()
        except Exception:
            logger.exception("Failed to compute baseline")
            return self._empty_result("Failed to compute baseline — see logs")

        variant_results: dict[str, dict[str, Any]] = {}
        for name, vdef in self._VARIANTS.items():
            try:
                variant_results[name] = self._simulate_variant(
                    removed_categories=vdef["removed_categories"],
                    description=vdef["description"],
                )
            except Exception:
                logger.exception("Variant '%s' simulation failed", name)
                variant_results[name] = {
                    "blocks": baseline["blocks"],
                    "additional_trades": 0,
                    "description": vdef["description"] + " [SIMULATION ERROR]",
                }

        # Determine best variant (most additional trades)
        best_variant: str = "none"
        best_extra: int = -1
        for name, vr in variant_results.items():
            extra = vr.get("additional_trades", 0)
            if extra > best_extra:
                best_extra = extra
                best_variant = name

        return {
            "baseline_blocks": baseline["blocks"],
            "variant_results": variant_results,
            "best_variant": best_variant,
            "warning": "SIMULATION ONLY — governor unchanged",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_logs(self, n_recent_cycles: int) -> None:
        """Load up to *n_recent_cycles* log entries from the JSONL file."""
        self._log_lines = []
        path = self._log_path

        if not path.is_file():
            raise FileNotFoundError(f"Cycle log not found: {path.resolve()}")

        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning("Skipping unparseable log line: %s", stripped[:80])
                    continue
                self._log_lines.append(record)

        if not self._log_lines:
            raise ValueError(f"No valid log entries loaded from {path}")

        logger.debug(
            "Loaded %d total log entries; will analyse most recent %d",
            len(self._log_lines),
            n_recent_cycles,
        )

        # Keep only the *n_recent_cycles* most recent entries
        if len(self._log_lines) > n_recent_cycles:
            self._log_lines = self._log_lines[-n_recent_cycles:]

    def _compute_baseline(self) -> dict[str, int]:
        """Count blocked cycles under the current (real) governor."""
        blocked = 0
        for record in self._log_lines:
            execution = self._get_execution(record)
            if self._is_blocked(execution):
                blocked += 1
        return {"blocks": blocked}

    def _simulate_variant(
        self,
        removed_categories: set[str],
        description: str,
    ) -> dict[str, Any]:
        """Simulate what happens if certain rule categories are removed.

        A previously-blocked cycle becomes an *additional trade* if
        its blocker belongs to one of the *removed_categories*.
        Cycles blocked by categories NOT in *removed_categories* remain
        blocked, and cycles that were never blocked remain unchanged.
        """
        baseline_blocked = 0
        still_blocked = 0

        for record in self._log_lines:
            execution = self._get_execution(record)
            if not self._is_blocked(execution):
                continue  # never blocked → no change

            baseline_blocked += 1
            category = self._classify_execution(execution)

            if category in removed_categories:
                # This denial would be removed → trade becomes allowed
                continue
            else:
                still_blocked += 1

        additional_trades = baseline_blocked - still_blocked

        return {
            "blocks": still_blocked,
            "additional_trades": additional_trades,
            "description": description,
        }

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _get_execution(record: dict[str, Any]) -> str:
        """Extract the execution outcome string from a log record."""
        pt = record.get("pipeline_trace")
        if isinstance(pt, dict):
            return str(pt.get("execution", ""))
        return ""

    @staticmethod
    def _is_blocked(execution: str) -> bool:
        """Return True if the execution outcome indicates a blocked trade."""
        if not isinstance(execution, str):
            return False
        return execution.startswith("DENIED") or execution.startswith("FAILED")

    @staticmethod
    def _empty_result(error_hint: str = "") -> dict[str, Any]:
        """Return a safe fallback result when simulation cannot run."""
        return {
            "baseline_blocks": 0,
            "variant_results": {
                name: {
                    "blocks": 0,
                    "additional_trades": 0,
                    "description": vdef["description"] + " [SKIPPED]",
                }
                for name, vdef in GovernorSimplificationSim._VARIANTS.items()
            },
            "best_variant": "none",
            "warning": f"SIMULATION ONLY — governor unchanged | {error_hint}",
        }
