"""
governor_causal_trace.py — Per-cycle governor rule-level causality tracking.

Replaces coarse "governor=block" with exact rule-level causality for every
blocked execution attempt.  Parses the pipeline trace log
(state/wave12_cycle_log.jsonl) and evaluates each governor rule in order.

Each cycle log dict contains fields such as:
    cycle, decision, denial_reason, governor_status, segl_state,
    total_signals, active_symbol, cb_decision, vel_decision,
    pipeline_trace.governor_gate, etc.

Rules evaluated (in order)
--------------------------
1. segl_state       — segl_state == "OBSERVE" or State=OBSERVE in denial_reason
2. circuit_breaker  — "CircuitBreaker" in denial_reason or cb_decision
3. confirm_gate     — "Insufficient cross-projection confirm" in denial_reason
4. vel              — denial_reason starts with "VEL blocked"
5. mt5_tick         — "No tick data" in denial_reason

Usage
-----
    from proxima_ops.analytics.governor_causal_trace import GovernorCausalTrace

    analyzer = GovernorCausalTrace()
    report = analyzer.analyze(n_recent_cycles=500)
    print(report)
"""

import json
import logging
import os
from collections import defaultdict
from typing import Any

logger = logging.getLogger("proxima_ops.analytics.governor_causal_trace")

DEFAULT_LOG_PATH = "state/wave12_cycle_log.jsonl"

# Evaluation order — the first rule that triggers is the primary_blocker.
GOVERNOR_RULES_EVALUATION_ORDER = [
    "segl_state",
    "circuit_breaker",
    "confirm_gate",
    "vel",
    "mt5_tick",
]


class GovernorCausalTrace:
    """Per-cycle governor rule-level causality tracking.

    Parameters
    ----------
    log_path : str
        Path to the JSON-lines cycle log (default
        ``state/wave12_cycle_log.jsonl``).
    """

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self._log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Analyse the trace log and return a governor-causal-trace report.

        Parameters
        ----------
        n_recent_cycles : int
            Only consider this many of the most recent cycles (default 500).
            Pass a large number (e.g. 1_000_000) to analyse all available
            cycles.

        Returns
        -------
        dict with keys:
            total_cycles_analyzed — number of cycles examined
            blocked_cycles       — count of BLOCKED decisions
            executed_cycles      — count of EXECUTED decisions
            no_signal_cycles     — count of cycles with total_signals == 0
            rule_hit_counts      — dict of rule_name -> count of blocks
            dominant_blocker     — most frequently triggered rule
            recent_traces        — last 20 per-cycle traces for inspection
        """
        try:
            return self._build_report(n_recent_cycles)
        except Exception:
            logger.exception("GovernorCausalTrace.analyze failed")
            return {
                "total_cycles_analyzed": 0,
                "blocked_cycles": 0,
                "executed_cycles": 0,
                "no_signal_cycles": 0,
                "rule_hit_counts": {},
                "dominant_blocker": "unknown",
                "recent_traces": [],
                "error": "See logs for details",
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_report(self, n_recent_cycles: int) -> dict[str, Any]:
        records = self._load_records()

        if not records:
            return {
                "total_cycles_analyzed": 0,
                "blocked_cycles": 0,
                "executed_cycles": 0,
                "no_signal_cycles": 0,
                "rule_hit_counts": {},
                "dominant_blocker": "unknown",
                "recent_traces": [],
                "warning": "No data found in log",
            }

        # Slice to the N most recent cycles.
        if n_recent_cycles < len(records):
            records = records[-n_recent_cycles:]

        traces: list[dict[str, Any]] = []
        rule_hit_counts: dict[str, int] = defaultdict(int)
        blocked = 0
        executed = 0
        no_signal = 0

        for entry in records:
            cycle_trace = self._trace_one_cycle(entry)
            traces.append(cycle_trace)

            if cycle_trace["final_decision"] == "BLOCKED":
                blocked += 1
            elif cycle_trace["final_decision"] == "EXECUTED":
                executed += 1
            else:
                no_signal += 1

            # Count rule hits (only rules that evaluated to blocking=True).
            for rule_eval in cycle_trace.get("governor_rules_evaluated", []):
                if rule_eval["result"] is True:
                    rule_hit_counts[rule_eval["rule"]] += 1

        # Dominant blocker — the rule that blocked most frequently.
        dominant_blocker: str = "unknown"
        if rule_hit_counts:
            dominant_blocker = max(
                rule_hit_counts, key=rule_hit_counts.get  # type: ignore[arg-type]
            )

        return {
            "total_cycles_analyzed": len(traces),
            "blocked_cycles": blocked,
            "executed_cycles": executed,
            "no_signal_cycles": no_signal,
            "rule_hit_counts": dict(rule_hit_counts),
            "dominant_blocker": dominant_blocker,
            "recent_traces": traces[-20:],
        }

    # ------------------------------------------------------------------
    # Per-cycle trace logic
    # ------------------------------------------------------------------

    def _trace_one_cycle(self, entry: dict) -> dict[str, Any]:
        """Evaluate every governor rule for a single cycle entry and produce
        a trace dict."""
        cycle = entry.get("cycle", 0)
        symbol = entry.get("active_symbol")
        signal = None

        # Build a minimal signal representation if one was active.
        if entry.get("total_signals", 0) > 0:
            signal = {
                "edge": entry.get("active_edge"),
                "symbol": entry.get("active_symbol"),
                "direction": entry.get("active_direction"),
                "confidence": entry.get("active_confidence"),
            }

        denial_reason = entry.get("denial_reason") or ""
        segl_state = entry.get("segl_state", "")
        decision = entry.get("decision", "")

        # Evaluate each governor rule in order.
        rules_evaluated: list[dict[str, Any]] = []
        for rule in GOVERNOR_RULES_EVALUATION_ORDER:
            result, detail = self._evaluate_rule(
                rule, entry, denial_reason, segl_state
            )
            rules_evaluated.append({
                "rule": rule,
                "result": result,
                "detail": detail,
            })

        # Determine final decision.
        final_decision: str
        if entry.get("total_signals", 0) == 0:
            final_decision = "NO_SIGNAL"
        elif self._is_executed(entry, decision):
            final_decision = "EXECUTED"
        else:
            final_decision = "BLOCKED"

        # Determine primary blocker — first rule that evaluated to True.
        primary_blocker: str | None = None
        for re in rules_evaluated:
            if re["result"] is True:
                primary_blocker = re["rule"]
                break

        return {
            "cycle_id": cycle,
            "symbol": symbol,
            "signal": signal,
            "governor_rules_evaluated": rules_evaluated,
            "final_decision": final_decision,
            "primary_blocker": primary_blocker,
        }

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_rule(
        rule: str,
        entry: dict,
        denial_reason: str,
        segl_state: str,
    ) -> tuple[bool, str]:
        """Evaluate a single governor rule against the cycle entry.

        Returns
        -------
        (blocked, detail)
            blocked  — True if this rule prevented execution
            detail   — human-readable string explaining the evaluation
        """
        # --- segl_state ---
        if rule == "segl_state":
            # Direct field check.
            if segl_state == "OBSERVE":
                return True, "segl_state=OBSERVE ready_to_exec=NO"
            # Denial-reason check.
            if "State=OBSERVE" in denial_reason:
                return True, "State=OBSERVE (from denial_reason)"
            # Pipeline-trace fallback.
            pipeline_trace = entry.get("pipeline_trace") or {}
            governor_gate = pipeline_trace.get("governor_gate") or []
            if governor_gate and any(
                "segl_state=OBSERVE" in str(g) for g in governor_gate
            ):
                return True, "segl_state=OBSERVE (from pipeline_trace)"
            return False, "segl_state=ARMED or OTHER"

        # --- circuit_breaker ---
        if rule == "circuit_breaker":
            if "CircuitBreaker" in denial_reason:
                return True, denial_reason
            cb_decision = entry.get("cb_decision", "")
            if "Circuit breaker" in cb_decision or "circuit breaker" in cb_decision.lower():
                return True, "cb_decision={}".format(cb_decision)
            return False, "cb_decision=Allowed or absent"

        # --- confirm_gate ---
        if rule == "confirm_gate":
            if "Insufficient cross-projection confirm" in denial_reason:
                return True, denial_reason
            # If signals exist but confirm_cycles < 2, the confirm gate is
            # still waiting — treat this as a block from the confirm gate.
            confirm_cycles = entry.get("confirm_cycles", 0)
            if confirm_cycles < 2 and entry.get("total_signals", 0) > 0:
                return True, "cross_confirm={}/2 (still waiting)".format(confirm_cycles)
            return False, "cross_confirm>=2/2 or no signal"

        # --- vel ---
        if rule == "vel":
            if denial_reason.startswith("VEL blocked"):
                return True, denial_reason
            vel_decision = entry.get("vel_decision", "")
            if vel_decision and vel_decision != "allowed":
                return True, "vel_decision={}".format(vel_decision)
            return False, "vel not triggered"

        # --- mt5_tick ---
        if rule == "mt5_tick":
            if "No tick data" in denial_reason:
                return True, denial_reason
            return False, "tick data available"

        return False, "unknown rule"

    # ------------------------------------------------------------------
    # Execution detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_executed(entry: dict, decision: str) -> bool:
        """Determine if a cycle resulted in an actual execution."""
        if decision == "EXECUTE":
            return True
        # Check pipeline_trace execution field.
        pipeline_trace = entry.get("pipeline_trace") or {}
        execution = pipeline_trace.get("execution", "")
        if execution and "EXECUTED" in execution:
            return True
        return False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict[str, Any]]:
        """Load and return all JSON-line records from the trace log."""
        if not os.path.exists(self._log_path):
            logger.warning("Trace log not found: %s", self._log_path)
            return []

        records: list[dict[str, Any]] = []
        try:
            with open(self._log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping unparseable line in %s", self._log_path
                        )
        except Exception:
            logger.exception("Failed to read trace log: %s", self._log_path)
            return []

        return records


# ------------------------------------------------------------------
# CLI convenience
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import sys

    n = 500
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print("Usage: python governor_causal_trace.py [n_recent_cycles]")
            sys.exit(1)

    analyzer = GovernorCausalTrace()
    report = analyzer.analyze(n_recent_cycles=n)
    print(json.dumps(report, indent=2))
