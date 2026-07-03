"""
Gate Audit Logger — Instrument ALL rejection reasons with full context.

Tracks every signal's journey through the funnel gates and records
the exact gate, value, threshold, and context for every rejection.

Output: JSONL file with per-rejection detail + aggregate stats.
"""
import os
import json
import time
import logging
from collections import defaultdict, Counter
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger("gate_audit")

# Known gate types in the Proxima funnel
GATE_TYPES = [
    "THRESHOLD",       # ECDF percentile threshold not met
    "SPREAD",          # Legacy elastic spread check
    "SPREAD_NORM",     # Adaptive spread normalization
    "INVALID_SPREAD",  # Spread invalid from MT5
    "NO_TICK",         # No tick data available
    "POSITION_EXISTS", # Position already open on symbol
    "MAX_POSITIONS",   # Max portfolio positions reached
    "POSITION_LOCK",   # 3-bar position lock active
    "FLIP_COOLDOWN",   # Direction flip cooldown
    "RISK_LIMIT",      # Risk governor limit hit
    "RHL_BLOCKED",     # RHL pre-order check rejected
    "NOT_IN_TOP3",     # Not in top-3 rotation
    "TPI_GATE",        # TPI confidence gate
    "PERSISTENCE_GATE", # Persistence gate
    "CURVATURE_GATE",  # Curvature gate
    "TPI_HARD_GATE",   # TPI hard gate
    "BROKER_REJECT",   # MT5 order rejection
    "FREQUENCY_FILTER", # Market closed / frequency filter
    "UNKNOWN",
]


class GateAuditLogger:
    """
    Instrument every rejection reason in the Proxima funnel.

    Usage:
        auditor = GateAuditLogger("state/gate_audit.jsonl")
        auditor.log_rejection(symbol="EURUSD", signal_id="SIG_001",
                              gate="SPREAD_NORM", value=45.0, threshold=30.0,
                              context={"es_rank": 0.85, "session": "LONDON"})
        auditor.log_acceptance(symbol="EURUSD", signal_id="SIG_001", ...)
        summary = auditor.summary()
    """

    def __init__(self, output_path: str = None):
        self._output_path = output_path or os.path.join(
            os.getcwd(), "state", "gate_audit.jsonl"
        )
        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)

        # Per-gate counters
        self._gate_counts: Dict[str, int] = defaultdict(int)
        self._gate_symbol_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._gate_value_buckets: Dict[str, List[float]] = defaultdict(list)
        self._total_evaluated = 0
        self._total_accepted = 0
        self._total_rejected = 0

        # Session tracking
        self._session_start = datetime.utcnow()
        self._session_gate_counts: Dict[str, int] = defaultdict(int)

        # Load existing state if available
        self._load_existing()

    def _load_existing(self):
        """Replay existing JSONL to populate counters on restart."""
        if not os.path.exists(self._output_path):
            return
        try:
            with open(self._output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        self._replay_entry(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"[GATE_AUDIT] Could not load existing log: {e}")

    def _replay_entry(self, entry: dict):
        """Replay a single log entry to rebuild counters."""
        event = entry.get("event")
        if event == "rejection":
            gate = entry.get("gate", "UNKNOWN")
            self._gate_counts[gate] += 1
            symbol = entry.get("symbol", "UNKNOWN")
            self._gate_symbol_counts[gate][symbol] += 1
            self._total_rejected += 1
            self._session_gate_counts[gate] += 1
        elif event == "acceptance":
            self._total_accepted += 1
        self._total_evaluated += 1

    def log_rejection(self, symbol: str, signal_id: str, gate: str,
                      value: Optional[float] = None,
                      threshold: Optional[float] = None,
                      context: Optional[Dict[str, Any]] = None,
                      entry_price: Optional[float] = None,
                      bar_time: Optional[int] = None):
        """
        Log a signal rejection at a specific gate.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            signal_id: Unique signal identifier
            gate: Gate type from GATE_TYPES
            value: The actual value that was tested
            threshold: The threshold it was tested against
            context: Additional context dict
            entry_price: Price at rejection time
            bar_time: Bar timestamp
        """
        gate = gate if gate in GATE_TYPES else "UNKNOWN"

        entry = {
            "ts": datetime.utcnow().isoformat(),
            "event": "rejection",
            "symbol": symbol,
            "signal_id": signal_id,
            "gate": gate,
            "value": value,
            "threshold": threshold,
            "entry_price": entry_price,
            "bar_time": bar_time,
            "context": context or {},
        }

        # Update counters
        self._gate_counts[gate] += 1
        self._gate_symbol_counts[gate][symbol] += 1
        self._total_rejected += 1
        self._total_evaluated += 1
        self._session_gate_counts[gate] += 1

        if value is not None and isinstance(value, (int, float)):
            self._gate_value_buckets[gate].append(float(value))

        self._append_entry(entry)

    def log_acceptance(self, symbol: str, signal_id: str,
                       value: Optional[float] = None,
                       context: Optional[Dict[str, Any]] = None,
                       entry_price: Optional[float] = None):
        """Log a signal that passed all gates."""
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "event": "acceptance",
            "symbol": symbol,
            "signal_id": signal_id,
            "value": value,
            "entry_price": entry_price,
            "context": context or {},
        }
        self._total_accepted += 1
        self._total_evaluated += 1
        self._append_entry(entry)

    def log_broker_rejection(self, symbol: str, signal_id: str,
                             ticket: Optional[int] = None,
                             error_code: Optional[int] = None,
                             error_msg: Optional[str] = None):
        """Log an MT5 broker rejection separately."""
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "event": "broker_rejection",
            "symbol": symbol,
            "signal_id": signal_id,
            "ticket": ticket,
            "error_code": error_code,
            "error_msg": error_msg,
        }
        self._gate_counts["BROKER_REJECT"] += 1
        self._gate_symbol_counts["BROKER_REJECT"][symbol] += 1
        self._total_rejected += 1
        self._total_evaluated += 1
        self._session_gate_counts["BROKER_REJECT"] += 1
        self._append_entry(entry)

    def _append_entry(self, entry: dict):
        """Write a single JSON line to the audit log."""
        try:
            with open(self._output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"[GATE_AUDIT] Write error: {e}")

    def summary(self) -> dict:
        """Return aggregate rejection summary."""
        total = self._total_evaluated
        accepted = self._total_accepted
        rejected = self._total_rejected

        gate_pct = {}
        for gate, count in sorted(self._gate_counts.items(), key=lambda x: -x[1]):
            if total > 0:
                gate_pct[gate] = {
                    "count": count,
                    "pct_of_total": round(count / total * 100, 2),
                    "pct_of_rejected": round(count / rejected * 100, 2) if rejected > 0 else 0,
                    "symbols": dict(sorted(self._gate_symbol_counts[gate].items(),
                                           key=lambda x: -x[1])),
                }

        return {
            "total_evaluated": total,
            "total_accepted": accepted,
            "total_rejected": rejected,
            "acceptance_rate": round(accepted / total * 100, 2) if total > 0 else 0,
            "rejection_rate": round(rejected / total * 100, 2) if total > 0 else 0,
            "gates": gate_pct,
            "session": {
                "start": self._session_start.isoformat(),
                "gate_counts": dict(self._session_gate_counts),
            },
        }

    def rejection_time_series(self, window_minutes: int = 60) -> List[dict]:
        """Return rejection rate time series, bucketed by window."""
        if not os.path.exists(self._output_path):
            return []
        rejections_per_bucket = defaultdict(int)
        total_per_bucket = defaultdict(int)
        try:
            with open(self._output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        ts_str = entry.get("ts", "")
                        if not ts_str:
                            continue
                        dt = datetime.fromisoformat(ts_str)
                        # Bucket by window
                        bucket_key = dt.strftime("%Y-%m-%d %H:") + str(dt.minute // window_minutes * window_minutes).zfill(2)
                        total_per_bucket[bucket_key] += 1
                        if entry.get("event") == "rejection":
                            rejections_per_bucket[bucket_key] += 1
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception:
            return []

        result = []
        for bucket in sorted(total_per_bucket.keys()):
            t = total_per_bucket[bucket]
            r = rejections_per_bucket.get(bucket, 0)
            result.append({
                "bucket": bucket,
                "total": t,
                "rejected": r,
                "rejection_rate": round(r / t * 100, 2) if t > 0 else 0,
            })
        return result

    def gate_ranking(self) -> List[dict]:
        """Return gates ranked by rejection count."""
        return [
            {"gate": gate, "count": count, "symbol_breakdown": dict(syms)}
            for gate, count, syms in sorted(
                [(g, c, self._gate_symbol_counts[g]) for g, c in self._gate_counts.items()],
                key=lambda x: -x[1]
            )
        ]

    def signal_preservation_rate(self) -> float:
        """SPR = accepted / total_evaluated."""
        if self._total_evaluated == 0:
            return 0.0
        return self._total_accepted / self._total_evaluated

    def false_rejection_rate(self, pnl_relevant_gates: Optional[List[str]] = None) -> float:
        """
        GFR = rejections from non-PnL gates / total evaluated.
        Non-PnL gates are those that reject for structural/logistical reasons,
        not alpha quality reasons.
        """
        if pnl_relevant_gates is None:
            pnl_relevant_gates = ["THRESHOLD", "TPI_GATE", "PERSISTENCE_GATE",
                                   "CURVATURE_GATE", "TPI_HARD_GATE"]

        non_pnl_rejections = sum(
            count for gate, count in self._gate_counts.items()
            if gate not in pnl_relevant_gates
        )
        if self._total_evaluated == 0:
            return 0.0
        return non_pnl_rejections / self._total_evaluated


# Singleton for global use
_INSTANCE: Optional[GateAuditLogger] = None


def get_gate_audit(output_path: str = None) -> GateAuditLogger:
    """Get or create the global GateAuditLogger instance."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = GateAuditLogger(output_path)
    return _INSTANCE
