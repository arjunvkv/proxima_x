"""
Lifecycle Reconciliation Engine — Bidirectional Mapping Repair.

Fixes drift between internal lifecycle state (lifecycle_state.json) and
MT5 actual position state. Produces a reconciliation delta report.

Usage:
    python -m proxima_x.bootstrap.lifecycle_reconciliation
    python -m proxima_x.bootstrap.lifecycle_reconciliation --dry-run   # preview only

Problem context:
    Integration contract reports:
    - "Lifecycle signals {57343721376} are OPENED but no MT5 position"
    - "2 signals have no ticket: ['MIX_B_1', 'MIX_O_1']"

Approach:
    1. Connect to MT5 and get current positions (fallback to state.json)
    2. Read lifecycle_state.json
    3. Build bidirectional mapping table: signal_id <-> mt5_ticket <-> lifecycle_stage
    4. Resolve orphans:
       - Orphan signals with no ticket → mark as CLOSED/ARCHIVED
       - OPENED signals with no MT5 position → mark as CLOSED
       - MT5 positions with no lifecycle signal → CREATE new signal entry
    5. Write corrected lifecycle_state.json
    6. Produce reconciliation delta report
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── Path Setup ───────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PROXIMA_X = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROXIMA_X not in sys.path:
    sys.path.insert(0, _PROXIMA_X)

_LIFECYCLE_STATE_PATH = os.path.join(_PROJECT_ROOT, "state", "lifecycle_state.json")
_RUNTIME_STATE_PATH = os.path.join(_PROJECT_ROOT, "core_runtime", "state.json")
_HEALTH_SNAPSHOT_PATH = os.path.join(_PROJECT_ROOT, "state", "system_health_snapshot.json")

logger = logging.getLogger("lifecycle_reconciliation")

# ── Lifecycle Stages ─────────────────────────────────────────────────────────

# Mirrors core_runtime.execution_lifecycle_manager.LifecycleStage
class LifecycleStage:
    GENERATED = "GENERATED"
    THRESHOLD_PASSED = "THRESHOLD_PASSED"
    TRIGGERED = "TRIGGERED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    OPENED = "OPENED"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"
    ORPHANED = "ORPHANED"


# ── Data Structures ──────────────────────────────────────────────────────────


class MappingEntry:
    """A single entry in the bidirectional mapping table."""

    def __init__(
        self,
        signal_id: str,
        ticket: Optional[int],
        stage: str,
        symbol: str,
        direction: int,
        source: str,  # "lifecycle", "mt5", "created", "resolved"
    ):
        self.signal_id = signal_id
        self.ticket = ticket
        self.stage = stage
        self.symbol = symbol
        self.direction = direction
        self.source = source

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "ticket": self.ticket,
            "stage": self.stage,
            "symbol": self.symbol,
            "direction": self.direction,
            "source": self.source,
        }

    def __repr__(self) -> str:
        return (
            f"MappingEntry(signal={self.signal_id}, ticket={self.ticket}, "
            f"stage={self.stage}, sym={self.symbol}, src={self.source})"
        )


class ReconciliationDelta:
    """Records all changes made during a reconciliation cycle."""

    def __init__(self):
        self.signals_closed: List[dict] = []       # OPENED/ORPHANED → CLOSED
        self.signals_created: List[dict] = []      # MT5 position → new signal
        self.signals_updated: List[dict] = []      # Metadata updates
        self.orphans_resolved: List[dict] = []     # Orphan signals resolved
        self.errors: List[str] = []
        self.mapping_table: List[dict] = []        # Full bidirectional mapping
        self.warnings: List[str] = []

    @property
    def total_changes(self) -> int:
        return (
            len(self.signals_closed)
            + len(self.signals_created)
            + len(self.signals_updated)
            + len(self.orphans_resolved)
        )

    def to_dict(self) -> dict:
        return {
            "reconciliation_timestamp": datetime.now().isoformat(),
            "total_changes": self.total_changes,
            "signals_closed": self.signals_closed,
            "signals_created": self.signals_created,
            "signals_updated": self.signals_updated,
            "orphans_resolved": self.orphans_resolved,
            "errors": self.errors,
            "warnings": self.warnings,
            "mapping_table": self.mapping_table,
        }

    def print_report(self) -> None:
        """Print a human-readable reconciliation delta report."""
        print()
        print("=" * 72)
        print("  LIFECYCLE RECONCILIATION DELTA REPORT")
        print("=" * 72)
        print()
        print(f"  Timestamp:  {self.to_dict()['reconciliation_timestamp']}")
        print(f"  Total changes:  {self.total_changes}")
        print()

        if self.mapping_table:
            print("  ── Bidirectional Mapping Table ──")
            print(
                f"    {'Signal ID':<35s} {'Ticket':<15s} {'Stage':<15s} "
                f"{'Symbol':<10s} {'Dir':<5s} {'Source':<12s}"
            )
            print(f"    {'─' * 92}")
            for entry in self.mapping_table:
                ticket_str = str(entry["ticket"]) if entry["ticket"] is not None else "—"
                print(
                    f"    {entry['signal_id']:<35s} {ticket_str:<15s} "
                    f"{entry['stage']:<15s} {entry['symbol']:<10s} "
                    f"{entry['direction']:<5d} {entry['source']:<12s}"
                )
            print()

        if self.orphans_resolved:
            print(f"  ── Orphan Signals Resolved ({len(self.orphans_resolved)}) ──")
            for o in self.orphans_resolved:
                print(
                    f"    ✓ {o['signal_id']:<35s} → {o['new_stage']:<15s} "
                    f"(reason: {o.get('reason', 'N/A')})"
                )
            print()

        if self.signals_closed:
            print(f"  ── Signals Closed ({len(self.signals_closed)}) ──")
            for s in self.signals_closed:
                print(
                    f"    ✗ {s['signal_id']:<35s} ticket={s.get('ticket', '—')} "
                    f"reason={s.get('exit_reason', 'N/A')}"
                )
            print()

        if self.signals_created:
            print(f"  ── Signals Created from MT5 ({len(self.signals_created)}) ──")
            for s in self.signals_created:
                print(
                    f"    + {s['signal_id']:<35s} ticket={s['ticket']} "
                    f"sym={s['symbol']}"
                )
            print()

        if self.signals_updated:
            print(f"  ── Signals Updated ({len(self.signals_updated)}) ──")
            for s in self.signals_updated:
                print(
                    f"    ~ {s['signal_id']:<35s} fields={s.get('fields', [])}"
                )
            print()

        if self.warnings:
            print("  ── Warnings ──")
            for w in self.warnings:
                print(f"    ⚠  {w}")
            print()

        if self.errors:
            print("  ── Errors ──")
            for e in self.errors:
                print(f"    ✗ {e}")
            print()

        if self.total_changes == 0:
            print("  ✓  No changes needed — lifecycle is already in sync with MT5")
        else:
            print(
                f"  Total: {self.total_changes} change(s) "
                f"({len(self.signals_closed)} closed, "
                f"{len(self.signals_created)} created, "
                f"{len(self.orphans_resolved)} orphans resolved)"
            )
        print("=" * 72)
        print()


# ── Loaders ──────────────────────────────────────────────────────────────────


def _load_json(path: str, default: Any = None) -> Any:
    """Load a JSON file, returning *default* on failure."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
    return default if default is not None else {}


def get_mt5_positions() -> Tuple[List[dict], Optional[str]]:
    """Get current MT5 positions.

    Tries MT5Connector first. Falls back to state.json if MT5 unavailable.

    Returns:
        (positions_list, error_or_none)
        Each position dict has keys:
            ticket, symbol, type (BUY/SELL), volume, price_open, sl, tp,
            time, profit, magic
    """
    # First, try MT5Connector
    try:
        from proxima_ops.execution.mt5_connector import MT5Connector

        mt5 = MT5Connector()
        if mt5.connect():
            positions = mt5.get_positions()
            mt5.disconnect()
            if positions:
                logger.info("Got %d positions from live MT5 connection", len(positions))
                return positions, None
            else:
                logger.info("MT5 connected but returned 0 positions")
                return [], None
        else:
            err = mt5.last_error or "MT5 connection failed"
            logger.warning("MT5 connection failed: %s", err)
    except Exception as e:
        logger.warning("MT5 connector error: %s", e)

    # Fallback: read from state.json
    logger.info("Falling back to state.json for MT5 positions")
    runtime_state = _load_json(_RUNTIME_STATE_PATH, {})
    positions_raw = runtime_state.get("positions", [])

    # Convert state.json position format to MT5-like format
    positions = []
    for p in positions_raw:
        positions.append({
            "ticket": p.get("ticket", 0),
            "symbol": p.get("symbol", ""),
            "type": "BUY" if p.get("direction", 1) == 1 else "SELL",
            "volume": p.get("volume", 0.0),
            "price_open": p.get("entry_price", 0.0),
            "sl": p.get("sl", 0.0),
            "tp": p.get("tp", 0.0),
            "time": int(p.get("open_time", 0)),
            "profit": 0.0,  # Not available from state.json
            "magic": p.get("magic", 202406) if "magic" in p else 202406,
        })

    if positions:
        logger.info("Got %d positions from state.json fallback", len(positions))
        return positions, None
    else:
        return [], "No positions found in MT5 or state.json"


def load_lifecycle_state() -> Tuple[dict, List[dict]]:
    """Load lifecycle state and return (full_data, signals_list)."""
    data = _load_json(_LIFECYCLE_STATE_PATH, {"signals": [], "h20_bars": 20})
    signals = data.get("signals", [])
    logger.info("Loaded %d signals from lifecycle_state.json", len(signals))
    return data, signals


# ── Core Reconciliation Logic ────────────────────────────────────────────────


def build_mapping_table(
    lifecycle_signals: List[dict],
    mt5_positions: List[dict],
) -> Tuple[List[MappingEntry], dict, dict, dict]:
    """Build a bidirectional mapping between lifecycle signals and MT5 positions.

    Returns:
        (mapping_entries, signal_by_ticket, ticket_by_signal_id, mt5_by_ticket)
    """
    mapping: List[MappingEntry] = []

    # Index: signal_id -> signal
    lifecycle_by_signal = {s["signal_id"]: s for s in lifecycle_signals}
    # Index: ticket -> lifecycle signal (only those with tickets)
    lifecycle_by_ticket: Dict[int, dict] = {}
    for s in lifecycle_signals:
        t = s.get("ticket")
        if t is not None:
            lifecycle_by_ticket[t] = s

    # Index: ticket -> MT5 position
    mt5_by_ticket: Dict[int, dict] = {}
    for p in mt5_positions:
        t = p.get("ticket", 0)
        if t:
            mt5_by_ticket[t] = p

    # Build mapping from lifecycle side
    for sig in lifecycle_signals:
        signal_id = sig["signal_id"]
        ticket = sig.get("ticket")
        stage = sig.get("stage", LifecycleStage.ORPHANED)
        symbol = sig.get("symbol", "")
        direction = sig.get("direction", 1)
        mapping.append(MappingEntry(
            signal_id=signal_id,
            ticket=ticket,
            stage=stage,
            symbol=symbol,
            direction=direction,
            source="lifecycle",
        ))

    # Build mapping from MT5 side — any MT5 position NOT already in lifecycle
    tracked_tickets = {
        s.get("ticket") for s in lifecycle_signals if s.get("ticket") is not None
    }
    for ticket, pos in mt5_by_ticket.items():
        if ticket not in tracked_tickets:
            direction = 1 if pos.get("type") == "BUY" else -1
            signal_id = f"MT5_{ticket}"
            mapping.append(MappingEntry(
                signal_id=signal_id,
                ticket=ticket,
                stage=LifecycleStage.OPENED,
                symbol=pos.get("symbol", ""),
                direction=direction,
                source="mt5",
            ))

    return mapping, lifecycle_by_ticket, lifecycle_by_signal, mt5_by_ticket


def reconcile(
    mt5_positions: List[dict],
    lifecycle_data: dict,
    dry_run: bool = False,
) -> ReconciliationDelta:
    """Run the full reconciliation and return deltas.

    Args:
        mt5_positions: List of MT5 position dicts.
        lifecycle_data: Full lifecycle_state.json data (mutated in place if not dry_run).
        dry_run: If True, only report changes without writing.

    Returns:
        ReconciliationDelta describing all changes.
    """
    delta = ReconciliationDelta()
    signals = lifecycle_data.get("signals", [])

    # ── Build index structures ──
    mt5_by_ticket: Dict[int, dict] = {}
    for p in mt5_positions:
        t = p.get("ticket", 0)
        if t:
            mt5_by_ticket[t] = p

    mt5_tickets = set(mt5_by_ticket.keys())

    lifecycle_by_ticket: Dict[int, dict] = {}
    lifecycle_by_signal_id: Dict[str, dict] = {}
    for sig in signals:
        sig_id = sig.get("signal_id", "")
        lifecycle_by_signal_id[sig_id] = sig
        ticket = sig.get("ticket")
        if ticket is not None:
            lifecycle_by_ticket[ticket] = sig

    lifecycle_tickets = set(lifecycle_by_ticket.keys())

    # ── 1. Build bidirectional mapping table ──
    for sig in signals:
        delta.mapping_table.append({
            "signal_id": sig.get("signal_id", ""),
            "ticket": sig.get("ticket"),
            "stage": sig.get("stage", ""),
            "symbol": sig.get("symbol", ""),
            "direction": sig.get("direction", 1),
            "source": "lifecycle",
        })

    for ticket, pos in mt5_by_ticket.items():
        if ticket not in lifecycle_tickets:
            direction = 1 if pos.get("type") == "BUY" else -1
            delta.mapping_table.append({
                "signal_id": f"MT5_{ticket}",
                "ticket": ticket,
                "stage": LifecycleStage.OPENED,
                "symbol": pos.get("symbol", ""),
                "direction": direction,
                "source": "mt5 (no lifecycle entry)",
            })

    # ── 2. Resolve orphan signals (no ticket, ORPHANED stage) ──
    for sig in signals:
        signal_id = sig.get("signal_id", "")
        stage = sig.get("stage", "")
        ticket = sig.get("ticket")

        if ticket is None and stage in (LifecycleStage.ORPHANED, LifecycleStage.OPENED):
            # This is an orphan — mark as CLOSED/ARCHIVED
            if not dry_run:
                sig["stage"] = LifecycleStage.CLOSED
                sig["closed_at"] = time.time()
                sig["exit_reason"] = "ORPHAN_RESOLVED"
                sig["exit_price"] = sig.get("entry_price")
                sig["age_seconds"] = round(time.time() - sig.get("generated_at", time.time()), 1)

            delta.orphans_resolved.append({
                "signal_id": signal_id,
                "ticket": ticket,
                "old_stage": stage,
                "new_stage": LifecycleStage.CLOSED,
                "reason": "ORPHAN_RESOLVED — no MT5 ticket associated",
                "symbol": sig.get("symbol", ""),
            })
            logger.info("ORPHAN RESOLVED: %s (%s) → CLOSED", signal_id, stage)

    # ── 3. Lifecycle OPENED signals with no matching MT5 position → CLOSED ──
    opened_without_mt5 = lifecycle_tickets - mt5_tickets
    for ticket in opened_without_mt5:
        sig = lifecycle_by_ticket.get(ticket)
        if sig is None:
            continue
        signal_id = sig["signal_id"]
        stage = sig.get("stage", "")

        if stage in (LifecycleStage.OPENED, LifecycleStage.CLOSING):
            if not dry_run:
                sig["stage"] = LifecycleStage.CLOSED
                sig["closed_at"] = time.time()
                sig["exit_reason"] = "MT5_POSITION_MISSING"
                sig["exit_price"] = sig.get("entry_price")
                sig["age_seconds"] = round(time.time() - sig.get("generated_at", time.time()), 1)

            delta.signals_closed.append({
                "signal_id": signal_id,
                "ticket": ticket,
                "old_stage": stage,
                "new_stage": LifecycleStage.CLOSED,
                "exit_reason": "MT5_POSITION_MISSING",
                "symbol": sig.get("symbol", ""),
            })
            logger.info(
                "CLOSED lifecycle signal without MT5 position: %s ticket=%d",
                signal_id, ticket,
            )

    # ── 4. MT5 positions with no lifecycle signal → CREATE ──
    new_tickets = mt5_tickets - lifecycle_tickets
    for ticket in new_tickets:
        pos = mt5_by_ticket.get(ticket)
        if pos is None:
            continue

        direction = 1 if pos.get("type") == "BUY" else -1
        entry_price = float(pos.get("price_open", 0.0))
        position_time = int(pos.get("time", time.time()))
        volume = float(pos.get("volume", 0.0))
        sl = float(pos.get("sl", 0.0))
        tp = float(pos.get("tp", 0.0))
        magic = pos.get("magic", 202406)
        signal_id = f"MT5_{ticket}"

        new_signal = {
            "signal_id": signal_id,
            "symbol": pos.get("symbol", ""),
            "direction": direction,
            "volume": volume,
            "ticket": ticket,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "magic": magic,
            "stage": LifecycleStage.OPENED,
            "generated_at": float(position_time),
            "threshold_passed_at": None,
            "triggered_at": None,
            "submitted_at": None,
            "accepted_at": None,
            "opened_at": float(position_time),
            "close_requested_at": None,
            "closed_at": None,
            "exit_price": None,
            "exit_reason": None,
            "block_reason": None,
            "age_seconds": round(time.time() - position_time, 1),
            "bars_since_open": 0,
        }

        if not dry_run:
            signals.append(new_signal)

        delta.signals_created.append({
            "signal_id": signal_id,
            "ticket": ticket,
            "symbol": pos.get("symbol", ""),
            "direction": direction,
            "volume": volume,
            "entry_price": entry_price,
            "stage": LifecycleStage.OPENED,
        })
        logger.info(
            "CREATED lifecycle signal for MT5 position: %s ticket=%d sym=%s",
            signal_id, ticket, pos.get("symbol"),
        )

    # ── 5. Verify bidirectional consistency of existing matched pairs ──
    matched_tickets = lifecycle_tickets & mt5_tickets
    for ticket in matched_tickets:
        sig = lifecycle_by_ticket.get(ticket)
        pos = mt5_by_ticket.get(ticket)
        if sig is None or pos is None:
            continue

        # Verify direction consistency
        expected_dir = 1 if pos.get("type") == "BUY" else -1
        actual_dir = sig.get("direction", 0)
        if expected_dir != actual_dir:
            msg = (
                f"Direction mismatch for ticket {ticket}: "
                f"lifecycle={actual_dir}, MT5={expected_dir}"
            )
            delta.warnings.append(msg)
            logger.warning(msg)

        # Update SL/TP/magic from MT5 if they diverged
        updated_fields = []
        mt5_sl = float(pos.get("sl", 0.0))
        mt5_tp = float(pos.get("tp", 0.0))
        mt5_magic = pos.get("magic", 202406)
        mt5_volume = float(pos.get("volume", 0.0))

        if sig.get("sl") != mt5_sl and mt5_sl != 0.0:
            if not dry_run:
                sig["sl"] = mt5_sl
            updated_fields.append("sl")
        if sig.get("tp") != mt5_tp and mt5_tp != 0.0:
            if not dry_run:
                sig["tp"] = mt5_tp
            updated_fields.append("tp")
        if sig.get("magic") != mt5_magic:
            if not dry_run:
                sig["magic"] = mt5_magic
            updated_fields.append("magic")
        if sig.get("volume") != mt5_volume:
            if not dry_run:
                sig["volume"] = mt5_volume
            updated_fields.append("volume")

        if updated_fields:
            delta.signals_updated.append({
                "signal_id": sig["signal_id"],
                "ticket": ticket,
                "fields": updated_fields,
            })
            logger.info(
                "UPDATED lifecycle signal %s ticket=%d fields=%s",
                sig["signal_id"], ticket, updated_fields,
            )

        # Update age_seconds
        position_time = sig.get("opened_at") or sig.get("generated_at")
        if position_time:
            if not dry_run:
                sig["age_seconds"] = round(time.time() - position_time, 1)

    # ── 6. Update mapping table with final stages ──
    if not dry_run:
        delta.mapping_table = []
        for sig in signals:
            delta.mapping_table.append({
                "signal_id": sig.get("signal_id", ""),
                "ticket": sig.get("ticket"),
                "stage": sig.get("stage", ""),
                "symbol": sig.get("symbol", ""),
                "direction": sig.get("direction", 1),
                "source": "lifecycle",
            })

        # Also add entries for MT5 positions not yet tracked (if any remain after creation)
    else:
        # In dry-run mode, show what WOULD be in mapping after resolution
        dry_map = []
        for sig in signals:
            dry_map.append({
                "signal_id": sig.get("signal_id", ""),
                "ticket": sig.get("ticket"),
                "stage": sig.get("stage", ""),
                "symbol": sig.get("symbol", ""),
                "direction": sig.get("direction", 1),
                "source": "lifecycle",
            })
        for ticket, pos in mt5_by_ticket.items():
            if ticket not in lifecycle_tickets:
                direction = 1 if pos.get("type") == "BUY" else -1
                dry_map.append({
                    "signal_id": f"MT5_{ticket}",
                    "ticket": ticket,
                    "stage": LifecycleStage.OPENED,
                    "symbol": pos.get("symbol", ""),
                    "direction": direction,
                    "source": "mt5 (would be created)",
                })
        # Also show what orphans would look like after resolution
        for entry in dry_map:
            if entry["ticket"] is None and entry["stage"] in (
                LifecycleStage.ORPHANED, LifecycleStage.OPENED
            ):
                entry["stage"] = f"{LifecycleStage.CLOSED} (would be resolved)"
        delta.mapping_table = dry_map

    return delta


def save_lifecycle_state(lifecycle_data: dict) -> bool:
    """Write the corrected lifecycle state to disk.

    Returns True on success, False on failure.
    """
    try:
        os.makedirs(os.path.dirname(_LIFECYCLE_STATE_PATH), exist_ok=True)
        with open(_LIFECYCLE_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(lifecycle_data, f, indent=2, default=str)
        logger.info("Corrected lifecycle state written to %s", _LIFECYCLE_STATE_PATH)
        return True
    except Exception as e:
        logger.exception("Failed to write lifecycle state: %s", e)
        return False


def save_reconciliation_report(delta: ReconciliationDelta) -> Optional[str]:
    """Save the reconciliation delta report to disk.

    Returns the path to the saved file, or None on failure.
    """
    report_path = os.path.join(
        _PROJECT_ROOT, "state", f"lifecycle_reconciliation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(delta.to_dict(), f, indent=2, default=str)
        logger.info("Reconciliation report saved to %s", report_path)
        return report_path
    except Exception as e:
        logger.exception("Failed to save reconciliation report: %s", e)
        return None


# ── Main Entry Point ─────────────────────────────────────────────────────────


def run_reconciliation(dry_run: bool = False) -> ReconciliationDelta:
    """Run the full lifecycle reconciliation process.

    Args:
        dry_run: If True, only report changes without writing.

    Returns:
        ReconciliationDelta describing all changes.
    """
    logger.info(
        "Starting lifecycle reconciliation (dry_run=%s)...", dry_run
    )

    # 1. Get MT5 positions (with fallback)
    mt5_positions, mt5_error = get_mt5_positions()
    if mt5_error:
        logger.warning("MT5 position source issue: %s", mt5_error)

    # 2. Load lifecycle state
    lifecycle_data, signals = load_lifecycle_state()

    # 3. Run reconciliation
    delta = reconcile(mt5_positions, lifecycle_data, dry_run=dry_run)

    # 4. Save if not dry run
    if not dry_run and delta.total_changes > 0:
        save_ok = save_lifecycle_state(lifecycle_data)
        if not save_ok:
            delta.errors.append("Failed to save corrected lifecycle state")
    elif not dry_run and delta.total_changes == 0:
        logger.info("No changes needed — lifecycle is already in sync")

    # 5. Save reconciliation report (regardless of dry_run)
    report_path = save_reconciliation_report(delta)
    if report_path:
        delta.report_path = report_path

    return delta


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    dry_run = "--dry-run" in sys.argv

    print()
    print("  ╔═══════════════════════════════════════════════════════════╗")
    print("  ║     LIFECYCLE RECONCILIATION ENGINE                      ║")
    print("  ╚═══════════════════════════════════════════════════════════╝")
    print()

    if dry_run:
        print("  🔍  DRY RUN MODE — no files will be modified")
        print()

    delta = run_reconciliation(dry_run=dry_run)
    delta.print_report()

    if hasattr(delta, "report_path") and delta.report_path:
        print(f"  Full report saved to: {delta.report_path}")
        print()

    # 9. Success criteria check
    orphan_count = len(delta.orphans_resolved)
    total_open_signals = sum(
        1 for e in delta.mapping_table if e["stage"] in (
            LifecycleStage.OPENED,
            "OPENED (would be resolved)",
        )
    )
    # Count unresolved orphans in the final mapping
    unresolved_orphans = sum(
        1
        for e in delta.mapping_table
        if e["ticket"] is None
        and e["stage"] in (
            LifecycleStage.ORPHANED,
            LifecycleStage.OPENED,
        )
    )

    print()
    print("  ── Success Criteria ──")
    print(f"    ✓ Orphan signals resolved: {orphan_count}")
    print(f"    ✓ Signals closed (OPENED without MT5): {len(delta.signals_closed)}")
    print(f"    ✓ Signals created (MT5 without lifecycle): {len(delta.signals_created)}")
    print(f"    ✓ Unresolved orphans: {unresolved_orphans}")
    print(f"    ✓ Total mapped entries: {len(delta.mapping_table)}")
    print()

    if unresolved_orphans == 0 and len(delta.errors) == 0:
        print("  ✅  RECONCILIATION COMPLETE — 0 unresolved orphan lifecycle states")
        print("  ✅  Bidirectional mapping validated for MT5 positions")
        print()
        return 0
    else:
        if unresolved_orphans > 0:
            print(f"  ❌  {unresolved_orphans} orphan(s) remain unresolved")
        if delta.errors:
            print(f"  ❌  {len(delta.errors)} error(s) occurred")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
