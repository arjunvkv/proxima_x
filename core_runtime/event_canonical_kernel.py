"""
Event Canonical Kernel — SINGLE canonical event format that ALL subsystems
(SDIL, CSRF, SAAL, execution) must write to.

One format, one source of truth per tick cycle.

Every event is a ``CanonicalEvent`` dataclass stored in an in-memory dict
keyed by ``cycle_id``.  Events are immutable after creation — updates are
applied via explicit methods that also recompute the integrity hash.
"""

import hashlib
import logging
from dataclasses import dataclass, fields, astuple
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances: Dict[str, "_EventCanonicalKernel"] = {}


def EventCanonicalKernel(instance_id: str = "default", **kwargs: object) -> "_EventCanonicalKernel":
    """Singleton accessor — returns the same ``_EventCanonicalKernel`` for a
    given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Logical identifier for the kernel instance (default ``"default"``).
    **kwargs
        Additional keyword arguments forwarded to ``_EventCanonicalKernel``
        on first creation (e.g. ``max_size=5000``).  Subsequent calls with
        the same *instance_id* ignore these.

    Returns
    -------
    _EventCanonicalKernel
    """
    if instance_id not in _instances:
        _instances[instance_id] = _EventCanonicalKernel(instance_id, **kwargs)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Canonical Event data model
# ---------------------------------------------------------------------------

@dataclass
class CanonicalEvent:
    """Single canonical event representing one tick cycle across all layers.

    All fields except ``event_hash`` are set at creation or via update
    methods.  The ``event_hash`` is a SHA-256 digest of every other field
    for tamper detection.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    cycle_id: int               # unique cycle counter
    timestamp: float            # wall clock time
    symbol: str                 # instrument

    # ── Raw data ──────────────────────────────────────────────────────────
    bid: float
    ask: float
    spread: float
    volume: Optional[float] = None

    # ── Signal layer (OSS + ALT) ──────────────────────────────────────────
    oss_p_cont: float = 0.0      # OSS continuation probability
    oss_ev: float = 0.0          # OSS expected value
    oss_signal: int = 0          # -1, 0, +1
    oss_confidence: float = 0.0  # 0.0-1.0
    alt_signal: int = 0          # -1, 0, +1
    alt_confidence: float = 0.0  # 0.0-1.0

    # ── CSRF layer ────────────────────────────────────────────────────────
    duality_verdict: Optional[str] = None      # SignalDualityEngine verdict
    collapse_verdict: Optional[str] = None     # OSSSurfaceDiagnostic verdict
    entropy_assessment: Optional[str] = None   # SignalSpaceEntropy verdict
    truth_label: Optional[str] = None          # SignalTruthLabeler verdict

    # ── SAAL layer ────────────────────────────────────────────────────────
    authority: Optional[str] = None            # SignalAuthorityArbiter verdict
    active_policy: Optional[str] = None        # ExecutionPolicySwitcher policy
    consensus_signal: Optional[int] = None     # SignalConsensusModel output
    economic_value: Optional[float] = None     # SignalEconomicValueRanker output
    stability_verdict: Optional[str] = None    # AuthorityStabilityTracker verdict

    # ── Execution layer ───────────────────────────────────────────────────
    execution_decision: Optional[str] = None   # "EXECUTE"|"SKIP"|None
    execution_signal: Optional[int] = None     # final signal sent to MT5
    execution_reason: Optional[str] = None     # skip reason or confirmation

    # ── Integrity ─────────────────────────────────────────────────────────
    event_hash: str = ""  # SHA-256 of all other fields; computed automatically

    # ── Field ordering helper for hashing ─────────────────────────────────
    _HASH_EXCLUDE = {"event_hash"}

    def compute_hash(self) -> str:
        """Compute SHA-256 digest of all fields except ``event_hash``.

        Fields are joined as their string representations in declaration
        order (the order of the dataclass fields).
        """
        raw = "|".join(
            str(getattr(self, f.name)) if getattr(self, f.name) is not None else ""
            for f in fields(self)
            if f.name not in self._HASH_EXCLUDE
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        """Check whether the stored ``event_hash`` matches a fresh
        recomputation.  Returns ``True`` if the event is untampered.
        """
        return self.event_hash == self.compute_hash()


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _EventCanonicalKernel:
    """In-memory canonical event store.

    Parameters
    ----------
    instance_id : str
        Logical identifier for this kernel instance.
    max_size : int
        Maximum number of events to retain.  When exceeded, the oldest
        events are pruned automatically (default 10 000).
    """

    def __init__(self, instance_id: str = "default", max_size: int = 10000) -> None:
        self._instance_id = instance_id
        self._max_size = max_size
        self._events: Dict[int, CanonicalEvent] = {}
        self._sorted_ids: List[int] = []  # chronological order of cycle_ids
        logger.info(
            "EventCanonicalKernel('%s') initialised (max_size=%d)",
            instance_id,
            max_size,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def create_event(
        self,
        cycle_id: int,
        timestamp: float,
        symbol: str,
        bid: float,
        ask: float,
        spread: float,
        volume: Optional[float] = None,
        oss_p_cont: float = 0.0,
        oss_ev: float = 0.0,
        oss_signal: int = 0,
        oss_confidence: float = 0.0,
        alt_signal: int = 0,
        alt_confidence: float = 0.0,
    ) -> CanonicalEvent:
        """Create a new canonical event with raw + signal data and an
        auto-generated integrity hash.

        Parameters
        ----------
        cycle_id : int
            Unique cycle counter.
        timestamp : float
            Wall clock time.
        symbol : str
            Instrument symbol.
        bid : float
            Bid price.
        ask : float
            Ask price.
        spread : float
            Bid-ask spread.
        volume : Optional[float]
            Trading volume (optional).
        oss_p_cont : float
            OSS continuation probability.
        oss_ev : float
            OSS expected value.
        oss_signal : int
            OSS signal (-1, 0, +1).
        oss_confidence : float
            OSS confidence (0.0–1.0).
        alt_signal : int
            ALT signal (-1, 0, +1).
        alt_confidence : float
            ALT confidence (0.0–1.0).

        Returns
        -------
        CanonicalEvent
            The newly created event.
        """
        if cycle_id in self._events:
            logger.warning("Overwriting existing event for cycle_id=%d", cycle_id)

        event = CanonicalEvent(
            cycle_id=cycle_id,
            timestamp=timestamp,
            symbol=symbol,
            bid=bid,
            ask=ask,
            spread=spread,
            volume=volume,
            oss_p_cont=oss_p_cont,
            oss_ev=oss_ev,
            oss_signal=oss_signal,
            oss_confidence=oss_confidence,
            alt_signal=alt_signal,
            alt_confidence=alt_confidence,
        )
        # Compute hash after all fields are set
        event.event_hash = event.compute_hash()

        self._events[cycle_id] = event
        self._sorted_ids.append(cycle_id)
        self._prune()

        logger.debug("Created event cycle_id=%d symbol=%s", cycle_id, symbol)
        return event

    def get_event(self, cycle_id: int) -> Optional[CanonicalEvent]:
        """Retrieve an event by *cycle_id*.

        Returns ``None`` if not found.
        """
        return self._events.get(cycle_id)

    def get_latest_event(self) -> Optional[CanonicalEvent]:
        """Return the most recently created event, or ``None`` if no events
        exist.
        """
        if not self._sorted_ids:
            return None
        return self._events[self._sorted_ids[-1]]

    def update_event(self, cycle_id: int, **updates: object) -> Optional[CanonicalEvent]:
        """Update fields on an existing event and recompute its integrity
        hash.

        Parameters
        ----------
        cycle_id : int
            The cycle to update.
        **updates
            Field names and their new values.  Only dataclass fields are
            accepted; ``event_hash`` cannot be set directly.

        Returns
        -------
        CanonicalEvent or None
            The updated event, or ``None`` if *cycle_id* was not found.
        """
        event = self._events.get(cycle_id)
        if event is None:
            logger.warning("Cannot update — cycle_id=%d not found", cycle_id)
            return None

        for key, value in updates.items():
            if key == "event_hash":
                logger.warning("Ignoring attempt to set event_hash directly")
                continue
            if not hasattr(event, key):
                logger.warning("Ignoring unknown field '%s'", key)
                continue
            setattr(event, key, value)

        # Recompute integrity hash
        event.event_hash = event.compute_hash()
        return event

    def append_sdil(
        self,
        cycle_id: int,
        duality_verdict: Optional[str] = None,
        collapse_verdict: Optional[str] = None,
        entropy_assessment: Optional[str] = None,
        truth_label: Optional[str] = None,
    ) -> Optional[CanonicalEvent]:
        """Convenience method to update CSRF-layer fields on an event.

        Returns the updated event, or ``None`` if *cycle_id* was not found.
        """
        return self.update_event(
            cycle_id,
            duality_verdict=duality_verdict,
            collapse_verdict=collapse_verdict,
            entropy_assessment=entropy_assessment,
            truth_label=truth_label,
        )

    def append_saal(
        self,
        cycle_id: int,
        authority: Optional[str] = None,
        policy: Optional[str] = None,
        consensus: Optional[int] = None,
        economic_value: Optional[float] = None,
        stability: Optional[str] = None,
    ) -> Optional[CanonicalEvent]:
        """Convenience method to update SAAL-layer fields on an event.

        Returns the updated event, or ``None`` if *cycle_id* was not found.
        """
        return self.update_event(
            cycle_id,
            authority=authority,
            active_policy=policy,
            consensus_signal=consensus,
            economic_value=economic_value,
            stability_verdict=stability,
        )

    def append_execution(
        self,
        cycle_id: int,
        decision: Optional[str] = None,
        signal: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> Optional[CanonicalEvent]:
        """Convenience method to update execution-layer fields on an event.

        Returns the updated event, or ``None`` if *cycle_id* was not found.
        """
        return self.update_event(
            cycle_id,
            execution_decision=decision,
            execution_signal=signal,
            execution_reason=reason,
        )

    def get_cycle_count(self) -> int:
        """Return the total number of events stored."""
        return len(self._events)

    def get_all_events(self, start: int = 0, end: Optional[int] = None) -> List[CanonicalEvent]:
        """Return a contiguous range of events by creation order.

        Parameters
        ----------
        start : int
            Starting index (0-based, default 0).
        end : int or None
            Ending index (exclusive).  ``None`` means all events from *start*
            to the end.

        Returns
        -------
        list[CanonicalEvent]
        """
        ids = self._sorted_ids[start:end]
        return [self._events[cid] for cid in ids if cid in self._events]

    def reset(self) -> None:
        """Clear all events from this kernel instance."""
        self._events.clear()
        self._sorted_ids.clear()
        logger.info("EventCanonicalKernel('%s') reset", self._instance_id)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _prune(self) -> None:
        """Remove the oldest events when the store exceeds ``_max_size``."""
        while len(self._events) > self._max_size:
            oldest_id = self._sorted_ids.pop(0)
            removed = self._events.pop(oldest_id, None)
            if removed is not None:
                logger.debug("Pruned event cycle_id=%d", oldest_id)


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("EventCanonicalKernel self-test")
    logger.info("=" * 60)

    # 1. Obtain singleton instance
    kernel = EventCanonicalKernel("test")
    assert kernel.get_cycle_count() == 0
    logger.info("✓ Singleton obtained, cycle_count=%d", kernel.get_cycle_count())

    # 2. Create events
    ev1 = kernel.create_event(
        cycle_id=1,
        timestamp=time.time(),
        symbol="EURUSD",
        bid=1.1050,
        ask=1.1052,
        spread=0.0002,
        volume=100.0,
        oss_p_cont=0.75,
        oss_ev=0.0025,
        oss_signal=1,
        oss_confidence=0.80,
        alt_signal=0,
        alt_confidence=0.50,
    )
    assert kernel.get_cycle_count() == 1
    assert ev1.cycle_id == 1
    assert ev1.symbol == "EURUSD"
    assert ev1.event_hash != ""  # hash was auto-computed
    logger.info("✓ Created event 1 (hash=%s…)", ev1.event_hash[:16])

    ev2 = kernel.create_event(
        cycle_id=2,
        timestamp=time.time(),
        symbol="GBPUSD",
        bid=1.2500,
        ask=1.2503,
        spread=0.0003,
        volume=200.0,
        oss_p_cont=0.60,
        oss_ev=0.0010,
        oss_signal=-1,
        oss_confidence=0.65,
        alt_signal=1,
        alt_confidence=0.55,
    )
    assert kernel.get_cycle_count() == 2
    logger.info("✓ Created event 2 (hash=%s…)", ev2.event_hash[:16])

    # 3. Verify hash integrity
    assert ev1.verify_integrity()
    assert ev2.verify_integrity()
    logger.info("✓ Hash integrity verified for both events")

    # 4. Tamper detection
    original_hash = ev1.event_hash
    ev1.bid = 9.9999  # simulate tamper
    assert not ev1.verify_integrity()
    ev1.bid = 1.1050  # restore
    ev1.event_hash = original_hash
    logger.info("✓ Tamper detection works (hash mismatch detected)")

    # 5. Retrieval
    fetched = kernel.get_event(1)
    assert fetched is not None
    assert fetched.cycle_id == 1
    assert fetched.symbol == "EURUSD"
    logger.info("✓ get_event(1) returned correct event")

    latest = kernel.get_latest_event()
    assert latest is not None
    assert latest.cycle_id == 2
    logger.info("✓ get_latest_event() returned cycle_id=2")

    # 6. Update layers
    # SDIL
    updated = kernel.append_sdil(
        1,
        duality_verdict="CONFLICT",
        collapse_verdict="ACTIVE",
        entropy_assessment="HIGH",
        truth_label="FLAT",
    )
    assert updated is not None
    assert updated.duality_verdict == "CONFLICT"
    assert updated.collapse_verdict == "ACTIVE"
    assert updated.entropy_assessment == "HIGH"
    assert updated.truth_label == "FLAT"
    assert updated.verify_integrity()
    logger.info("✓ append_sdil updated event 1 and hash is valid")

    # SAAL
    updated = kernel.append_saal(
        1,
        authority="OSS",
        policy="HYBRID",
        consensus=1,
        economic_value=0.0032,
        stability="STABLE",
    )
    assert updated is not None
    assert updated.authority == "OSS"
    assert updated.active_policy == "HYBRID"
    assert updated.consensus_signal == 1
    assert updated.economic_value == 0.0032
    assert updated.stability_verdict == "STABLE"
    assert updated.verify_integrity()
    logger.info("✓ append_saal updated event 1 and hash is valid")

    # Execution
    updated = kernel.append_execution(
        1,
        decision="EXECUTE",
        signal=1,
        reason="OSS authority confirmed",
    )
    assert updated is not None
    assert updated.execution_decision == "EXECUTE"
    assert updated.execution_signal == 1
    assert updated.execution_reason == "OSS authority confirmed"
    assert updated.verify_integrity()
    logger.info("✓ append_execution updated event 1 and hash is valid")

    # 7. Update non-existent cycle
    result = kernel.update_event(999, bid=1.0)
    assert result is None
    logger.info("✓ update_event on missing cycle returns None")

    # 8. All events range
    all_events = kernel.get_all_events()
    assert len(all_events) == 2
    assert all_events[0].cycle_id == 1
    assert all_events[1].cycle_id == 2
    logger.info("✓ get_all_events() returned both events in order")

    partial = kernel.get_all_events(start=1)
    assert len(partial) == 1
    assert partial[0].cycle_id == 2
    logger.info("✓ get_all_events(start=1) returned only event 2")

    # 9. Cycle count
    assert kernel.get_cycle_count() == 2
    logger.info("✓ get_cycle_count() == 2")

    # 10. Reset
    kernel.reset()
    assert kernel.get_cycle_count() == 0
    assert kernel.get_latest_event() is None
    logger.info("✓ reset() cleared all events")

    # 11. Pruning behaviour
    small_kernel = EventCanonicalKernel("prune_test", max_size=3)
    for i in range(1, 6):
        small_kernel.create_event(
            cycle_id=i,
            timestamp=time.time(),
            symbol="EURUSD",
            bid=1.10 + i * 0.0001,
            ask=1.1002 + i * 0.0001,
            spread=0.0002,
        )
    assert small_kernel.get_cycle_count() == 3
    # Oldest cycle_ids (1, 2) should have been pruned
    assert small_kernel.get_event(1) is None
    assert small_kernel.get_event(2) is None
    assert small_kernel.get_event(3) is not None
    assert small_kernel.get_event(4) is not None
    assert small_kernel.get_event(5) is not None
    logger.info("✓ Pruning works (max_size=3, only last 3 remain)")

    # 12. Singleton isolation
    default_kernel = EventCanonicalKernel()
    assert default_kernel.get_cycle_count() == 0  # separate from test kernel
    assert kernel is not default_kernel
    logger.info("✓ Singleton instances are isolated per instance_id")

    logger.info("=" * 60)
    logger.info("ALL SELF-TESTS PASSED")
    logger.info("=" * 60)
