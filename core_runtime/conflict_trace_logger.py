"""
Conflict Trace Logger — Record every layer disagreement event so the system
can learn from past conflicts.

Logs when SDIL disagreed with CSRF, SAAL overrode ALT, MRSRL changed
resolution mid-decision, etc.  This becomes the system's "decision memory."

Usage
-----
    from core_runtime.conflict_trace_logger import ConflictTraceLogger

    logger = ConflictTraceLogger()
    entry = logger.log_conflict(
        cycle_id=42,
        symbol="EURUSD",
        layer_a="sdil",
        layer_b="csfr",
        conflict_type="SIGNAL_DIVERGENCE",
        detail={"sdil_signal": 0, "csfr_signal": 1},
    )
    logger.log_resolution(42, entry["entry_id"], "SAAL_OVERRIDE", "saal")
    summary = logger.get_conflict_summary()
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid conflict types
# ---------------------------------------------------------------------------

VALID_CONFLICT_TYPES = frozenset({
    "AUTHORITY_MISMATCH",
    "SIGNAL_DIVERGENCE",
    "RESOLUTION_CONFLICT",
    "VETO_TRIGGERED",
    "WEIGHT_SHIFT",
    "STABILITY_BREACH",
})

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances: Dict[str, "_ConflictTraceLogger"] = {}


def ConflictTraceLogger(instance_id="default"):
    """Singleton accessor for ``_ConflictTraceLogger``.

    Parameters
    ----------
    instance_id : str
        Unique identifier.  Callers sharing the same *instance_id* share the
        same underlying logger.

    Returns
    -------
    _ConflictTraceLogger
    """
    if instance_id not in _instances:
        _instances[instance_id] = _ConflictTraceLogger(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _ConflictTraceLogger:
    """Records layer disagreement events and their resolutions.

    Maintains an ordered log of conflicts (newest last) and a separate
    resolution index.  Automatically trims to *max_entries* when the log
    exceeds capacity.

    Parameters
    ----------
    instance_id : str
        Instance identifier forwarded from the singleton accessor.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        self._max_entries: int = 5000

        # Ordered list of conflict entries (newest last)
        self._conflicts: List[Dict[str, Any]] = []

        # conflict_entry_id -> resolution dict
        self._resolutions: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "ConflictTraceLogger(%r) initialised (max_entries=%d)",
            instance_id,
            self._max_entries,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_conflict(
        self,
        cycle_id: int,
        symbol: str,
        layer_a: str,
        layer_b: str,
        conflict_type: str,
        detail: dict,
    ) -> dict:
        """Record a conflict event.

        Parameters
        ----------
        cycle_id : int
            Tick cycle when the conflict occurred.
        symbol : str
            Instrument identifier.
        layer_a, layer_b : str
            Which layers conflicted (e.g. ``"saal"``, ``"csfr"``,
            ``"sdil"``, ``"mrsrl"``).
        conflict_type : str
            One of ``"AUTHORITY_MISMATCH"``, ``"SIGNAL_DIVERGENCE"``,
            ``"RESOLUTION_CONFLICT"``, ``"VETO_TRIGGERED"``,
            ``"WEIGHT_SHIFT"``, ``"STABILITY_BREACH"``.
        detail : dict
            Specific information about the conflict (what each layer said).

        Returns
        -------
        dict
            The logged entry with added ``entry_id`` and ``timestamp``.
        """
        conflict_type = conflict_type.upper()
        if conflict_type not in VALID_CONFLICT_TYPES:
            logger.warning(
                "Unknown conflict_type %r — recording anyway",
                conflict_type,
            )

        entry: Dict[str, Any] = {
            "entry_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "cycle_id": cycle_id,
            "symbol": symbol,
            "layer_a": str(layer_a),
            "layer_b": str(layer_b),
            "conflict_type": conflict_type,
            "detail": dict(detail) if detail else {},
            "resolved": False,
        }
        self._conflicts.append(entry)

        # Trim to max_entries
        if len(self._conflicts) > self._max_entries:
            self._conflicts = self._conflicts[-self._max_entries:]

        logger.debug(
            "log_conflict type=%s layers=%s/%s cycle=%d symbol=%s "
            "entry_id=%s",
            conflict_type, layer_a, layer_b, cycle_id, symbol,
            entry["entry_id"],
        )
        return dict(entry)

    def log_resolution(
        self,
        cycle_id: int,
        conflict_entry_id: str,
        resolution: str,
        resolved_by: str,
    ) -> dict:
        """Record how a conflict was resolved.

        Parameters
        ----------
        cycle_id : int
            Tick cycle when the resolution occurred.
        conflict_entry_id : str
            The ``entry_id`` returned by :meth:`log_conflict`.
        resolution : str
            Description of the resolution (e.g. ``"SAAL_OVERRIDE"``).
        resolved_by : str
            Which layer or component resolved it (e.g. ``"saal"``,
            ``"priority_arbiter"``).

        Returns
        -------
        dict
            The resolution record.
        """
        resolution_entry: Dict[str, Any] = {
            "cycle_id": cycle_id,
            "conflict_entry_id": conflict_entry_id,
            "resolution": str(resolution),
            "resolved_by": str(resolved_by),
            "timestamp": time.time(),
        }
        self._resolutions[conflict_entry_id] = resolution_entry

        # Mark the original conflict as resolved
        for conflict in self._conflicts:
            if conflict["entry_id"] == conflict_entry_id:
                conflict["resolved"] = True
                break

        logger.debug(
            "log_resolution entry=%s resolution=%s by=%s",
            conflict_entry_id, resolution, resolved_by,
        )
        return dict(resolution_entry)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_conflicts(
        self,
        symbol: Optional[str] = None,
        conflict_type: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        """Query conflicts with optional filters.

        Parameters
        ----------
        symbol : str or None
            If provided, only conflicts for this instrument are returned.
        conflict_type : str or None
            If provided, only conflicts of this type are returned.
        limit : int
            Maximum number of entries to return (default 100, newest first).

        Returns
        -------
        list of dict
        """
        result: list = self._conflicts
        if symbol is not None:
            result = [c for c in result if c["symbol"] == symbol]
        if conflict_type is not None:
            ct = conflict_type.upper()
            result = [c for c in result if c["conflict_type"] == ct]
        return result[-limit:]

    def get_conflict_summary(
        self,
        symbol: Optional[str] = None,
    ) -> dict:
        """Return summary statistics.

        Parameters
        ----------
        symbol : str or None
            If provided, statistics are scoped to this instrument.

        Returns
        -------
        dict
            ``total_conflicts``, ``by_type``, ``by_layer_pair``,
            ``most_common_resolution``, ``unresolved_count``,
            ``conflict_rate`` (conflicts per 100 ticks).
        """
        conflicts = self._conflicts
        if symbol is not None:
            conflicts = [c for c in conflicts if c["symbol"] == symbol]

        total = len(conflicts)
        by_type: Dict[str, int] = {}
        by_layer_pair: Dict[str, int] = {}
        unresolved = 0

        for c in conflicts:
            ct = c["conflict_type"]
            by_type[ct] = by_type.get(ct, 0) + 1
            pair = f"{c['layer_a']}_vs_{c['layer_b']}"
            by_layer_pair[pair] = by_layer_pair.get(pair, 0) + 1
            if not c.get("resolved"):
                unresolved += 1

        # Most common resolution (scoped to filtered conflicts)
        resolution_counts: Dict[str, int] = {}
        for c in conflicts:
            rid = c["entry_id"]
            res = self._resolutions.get(rid)
            if res is not None:
                resolution_counts[res["resolution"]] = (
                    resolution_counts.get(res["resolution"], 0) + 1
                )

        most_common_resolution = (
            max(resolution_counts, key=resolution_counts.get)
            if resolution_counts else "N/A"
        )

        # Conflict rate: conflicts per 100 ticks
        max_cycle = max(
            (c["cycle_id"] for c in conflicts), default=0,
        )
        if max_cycle > 0 and total > 0:
            conflict_rate = (total / max_cycle) * 100.0
        else:
            conflict_rate = 0.0

        return {
            "total_conflicts": total,
            "by_type": dict(by_type),
            "by_layer_pair": dict(by_layer_pair),
            "most_common_resolution": most_common_resolution,
            "unresolved_count": unresolved,
            "conflict_rate": round(conflict_rate, 4),
        }

    def get_layer_disagreement_rate(self, layer_name: str) -> float:
        """How often a specific layer disagrees with the final decision.

        Returns a value between 0 and 1 representing the fraction of all
        conflicts that involve *layer_name*.

        Parameters
        ----------
        layer_name : str
            Layer identifier (e.g. ``"saal"``, ``"csfr"``).

        Returns
        -------
        float
        """
        if not self._conflicts:
            return 0.0
        ln = str(layer_name).lower()
        layer_conflicts = [
            c for c in self._conflicts
            if c["layer_a"].lower() == ln or c["layer_b"].lower() == ln
        ]
        return round(len(layer_conflicts) / len(self._conflicts), 4)

    def get_recent_conflicts(self, n: int = 10) -> list:
        """Return the last *n* conflicts (newest first).

        Parameters
        ----------
        n : int
            Number of entries to return (default 10).

        Returns
        -------
        list of dict
        """
        return self._conflicts[-n:]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all logged conflicts and resolutions.

        The instance and its *max_entries* setting are preserved.
        """
        self._conflicts.clear()
        self._resolutions.clear()
        logger.info(
            "ConflictTraceLogger(%r) reset", self._instance_id,
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    """Exercise the ConflictTraceLogger with various conflict types and
    verify query, filtering, summary statistics, and singleton behaviour.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("ConflictTraceLogger — Self-Test")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            logger.info("  [PASS] %s", msg)
        else:
            failed += 1
            logger.error("  [FAIL] %s", msg)

    # ==================================================================
    # Scenario 1 — Log conflicts of all types
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 1: Log 5 conflicts of different types ---")

    ctl = ConflictTraceLogger("selftest")

    conflicts = [
        (1, "EURUSD", "sdil", "csfr", "AUTHORITY_MISMATCH",
         {"sdil": "OSS", "csfr": "ALT"}),
        (2, "EURUSD", "csfr", "saal", "SIGNAL_DIVERGENCE",
         {"csfr_signal": 0, "saal_signal": 1}),
        (3, "GBPUSD", "saal", "mrsrl", "RESOLUTION_CONFLICT",
         {"saal_resolution": "TICK", "mrsrl_resolution": "1M"}),
        (4, "EURUSD", "sdil", "mrsrl", "VETO_TRIGGERED",
         {"sdil": "HOLD", "mrsrl": "BUY"}),
        (5, "USDJPY", "csfr", "saal", "WEIGHT_SHIFT",
         {"csfr_weight": 0.3, "saal_weight": 0.35}),
    ]

    entries = []
    for (cycle_id, symbol, la, lb, ctype, detail) in conflicts:
        entry = ctl.log_conflict(cycle_id, symbol, la, lb, ctype, detail)
        entries.append(entry)
        _check(
            entry["entry_id"] is not None and len(entry["entry_id"]) > 0,
            f"Scenario 1 entry_id present for {ctype}",
        )
        _check(entry["conflict_type"] == ctype, f"Scenario 1 type={ctype}")
        _check(entry["cycle_id"] == cycle_id, f"Scenario 1 cycle_id={cycle_id}")
        _check(entry["symbol"] == symbol, f"Scenario 1 symbol={symbol}")
        _check(entry["resolved"] is False, f"Scenario 1 unresolved initially")

    _check(len(ctl._conflicts) == 5, "Scenario 1 5 conflicts stored")

    # ==================================================================
    # Scenario 2 — Log resolutions
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 2: Log resolutions ---")

    resolutions = [
        (1, entries[0]["entry_id"], "SAAL_OVERRIDE", "saal"),
        (2, entries[1]["entry_id"], "WEIGHT_REBALANCE", "weight_controller"),
        (3, entries[2]["entry_id"], "MRSRL_PREEMPT", "mrsrl"),
        (4, entries[3]["entry_id"], "SDIL_VETO_SUSTAINED", "sdil"),
    ]

    for (cycle_id, eid, resolution, resolved_by) in resolutions:
        res_entry = ctl.log_resolution(cycle_id, eid, resolution, resolved_by)
        _check(
            res_entry["conflict_entry_id"] == eid,
            f"Scenario 2 resolution linked to {eid[:8]}...",
        )
        _check(
            res_entry["resolution"] == resolution,
            f"Scenario 2 resolution={resolution}",
        )
        _check(
            res_entry["resolved_by"] == resolved_by,
            f"Scenario 2 resolved_by={resolved_by}",
        )

    # Verify conflicts are marked resolved
    for i in range(4):
        c = ctl._conflicts[i]
        _check(
            c["resolved"] is True,
            f"Scenario 2 conflict {i} resolved=True",
        )

    # The 5th conflict was never resolved
    _check(
        ctl._conflicts[4]["resolved"] is False,
        "Scenario 2 last conflict remains unresolved",
    )

    # ==================================================================
    # Scenario 3 — Query / filter
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 3: Query and filter ---")

    all_conflicts = ctl.get_conflicts()
    _check(len(all_conflicts) == 5, "Scenario 3 get_conflicts returns 5")

    eurusd = ctl.get_conflicts(symbol="EURUSD")
    _check(len(eurusd) == 3, "Scenario 3 3 conflicts for EURUSD")

    gbpusd = ctl.get_conflicts(symbol="GBPUSD")
    _check(len(gbpusd) == 1, "Scenario 3 1 conflict for GBPUSD")

    sig_div = ctl.get_conflicts(conflict_type="SIGNAL_DIVERGENCE")
    _check(len(sig_div) == 1, "Scenario 3 1 SIGNAL_DIVERGENCE")

    filtered = ctl.get_conflicts(symbol="EURUSD", conflict_type="AUTHORITY_MISMATCH")
    _check(len(filtered) == 1, "Scenario 3 1 EURUSD AUTHORITY_MISMATCH")

    recent = ctl.get_recent_conflicts(n=3)
    _check(len(recent) == 3, "Scenario 3 get_recent_conflicts returns 3")
    _check(recent[-1]["cycle_id"] == 5, "Scenario 3 newest conflict is cycle 5")

    # ==================================================================
    # Scenario 4 — get_conflict_summary
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 4: Conflict summary ---")

    summary = ctl.get_conflict_summary()
    _check(summary["total_conflicts"] == 5, "Scenario 4 total=5")
    _check(
        summary["by_type"]["AUTHORITY_MISMATCH"] == 1,
        "Scenario 4 by_type AUTHORITY_MISMATCH=1",
    )
    _check(
        summary["by_type"]["SIGNAL_DIVERGENCE"] == 1,
        "Scenario 4 by_type SIGNAL_DIVERGENCE=1",
    )
    _check(
        summary["by_type"]["RESOLUTION_CONFLICT"] == 1,
        "Scenario 4 by_type RESOLUTION_CONFLICT=1",
    )
    _check(
        summary["by_type"]["VETO_TRIGGERED"] == 1,
        "Scenario 4 by_type VETO_TRIGGERED=1",
    )
    _check(
        summary["by_type"]["WEIGHT_SHIFT"] == 1,
        "Scenario 4 by_type WEIGHT_SHIFT=1",
    )
    _check(
        "sdil_vs_csfr" in summary["by_layer_pair"],
        "Scenario 4 by_layer_pair sdil_vs_csfr present",
    )
    _check(
        summary["most_common_resolution"] in ("SAAL_OVERRIDE", "WEIGHT_REBALANCE",
                                              "MRSRL_PREEMPT", "SDIL_VETO_SUSTAINED"),
        "Scenario 4 most_common_resolution is one of the logged ones",
    )
    _check(
        summary["unresolved_count"] == 1,
        "Scenario 4 1 unresolved conflict",
    )
    _check(
        summary["conflict_rate"] > 0,
        "Scenario 4 conflict_rate > 0",
    )

    # Symbol-scoped summary
    eurusd_summary = ctl.get_conflict_summary(symbol="EURUSD")
    _check(
        eurusd_summary["total_conflicts"] == 3,
        "Scenario 4 EURUSD summary total=3",
    )
    _check(
        eurusd_summary["unresolved_count"] == 0,
        "Scenario 4 EURUSD all resolved",
    )

    # ==================================================================
    # Scenario 5 — get_layer_disagreement_rate
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 5: Layer disagreement rate ---")

    saal_rate = ctl.get_layer_disagreement_rate("saal")
    _check(
        saal_rate > 0,
        f"Scenario 5 saal disagreement rate > 0 (got {saal_rate})",
    )

    unknown_rate = ctl.get_layer_disagreement_rate("nonexistent")
    _check(
        unknown_rate == 0.0,
        "Scenario 5 unknown layer rate = 0.0",
    )

    # Empty logger
    ctl_empty = ConflictTraceLogger("selftest_empty")
    _check(
        ctl_empty.get_layer_disagreement_rate("saal") == 0.0,
        "Scenario 5 empty logger rate = 0.0",
    )

    # ==================================================================
    # Scenario 6 — Reset
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 6: Reset ---")

    ctl_reset = ConflictTraceLogger("selftest_reset")
    ctl_reset.log_conflict(10, "AUDUSD", "saal", "csfr",
                           "STABILITY_BREACH", {"msg": "test"})
    _check(len(ctl_reset._conflicts) == 1, "Scenario 6 1 conflict before reset")
    ctl_reset.reset()
    _check(len(ctl_reset._conflicts) == 0, "Scenario 6 0 conflicts after reset")
    _check(len(ctl_reset._resolutions) == 0, "Scenario 6 0 resolutions after reset")

    # ==================================================================
    # Scenario 7 — Singleton identity
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 7: Singleton identity ---")

    a = ConflictTraceLogger("selftest_singleton")
    b = ConflictTraceLogger("selftest_singleton")
    c = ConflictTraceLogger("selftest_singleton_other")
    _check(a is b, "Scenario 7 same instance_id -> same object")
    _check(a is not c, "Scenario 7 different instance_id -> different object")

    # ==================================================================
    # Scenario 8 — Edge: limit parameter
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 8: Limit parameter ---")

    ctl_limit = ConflictTraceLogger("selftest_limit")
    for i in range(20):
        ctl_limit.log_conflict(
            cycle_id=i,
            symbol="EURUSD",
            layer_a="sdil",
            layer_b="csfr",
            conflict_type="SIGNAL_DIVERGENCE",
            detail={"i": i},
        )

    all_20 = ctl_limit.get_conflicts()
    _check(len(all_20) == 20, "Scenario 8 20 conflicts stored")

    limited = ctl_limit.get_conflicts(limit=5)
    _check(len(limited) == 5, "Scenario 8 limit=5 works")

    # ==================================================================
    # Scenario 9 — Edge: max_entries trimming
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 9: max_entries trimming ---")

    ctl_trim = ConflictTraceLogger("selftest_trim")
    ctl_trim._max_entries = 10

    for i in range(15):
        ctl_trim.log_conflict(
            cycle_id=i,
            symbol="EURUSD",
            layer_a="sdil",
            layer_b="csfr",
            conflict_type="SIGNAL_DIVERGENCE",
            detail={"i": i},
        )

    _check(
        len(ctl_trim._conflicts) == 10,
        f"Scenario 9 trimmed to 10 (got {len(ctl_trim._conflicts)})",
    )
    _check(
        ctl_trim._conflicts[0]["cycle_id"] == 5,
        "Scenario 9 oldest entry is cycle 5",
    )
    _check(
        ctl_trim._conflicts[-1]["cycle_id"] == 14,
        "Scenario 9 newest entry is cycle 14",
    )

    # ==================================================================
    # Summary
    # ==================================================================
    logger.info("")
    logger.info("-" * 60)
    total = passed + failed
    logger.info(
        "Results:  %d / %d passed  (%s)",
        passed,
        total,
        "ALL PASSED" if failed == 0 else f"{failed} FAILED",
    )

    if failed > 0:
        logger.error(">>> SELF-TEST FAILED <<<")
    else:
        logger.info(">>> SELF-TEST PASSED <<<")

    return failed == 0


if __name__ == "__main__":
    _selftest()
