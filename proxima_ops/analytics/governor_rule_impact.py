"""
governor_rule_impact.py — Quantify which governor rules are actually
meaningful and which are redundant noise.

Analyses the pipeline trace log (state/wave12_cycle_log.jsonl) and reports
how often each governor rule triggered, which rules are effective/redundant,
a Shannon-entropy-based diversity score, and block-rate trends over time.

Rule categories (parsed from denial_reason / pipeline_trace):
    circuit_breaker   — "CircuitBreaker" in denial_reason
    vel               — "VEL blocked" in denial_reason
    mt5_tick          — "No tick data" in denial_reason
    confirm_gate      — "Insufficient cross-projection confirm" in denial_reason
    segl_state        — segl_state == "OBSERVE" / governor_gate contains "segl_state=OBSERVE"
    unknown_governor_rule — any other governor-gate denial

Usage
-----
    from proxima_ops.analytics.governor_rule_impact import GovernorRuleImpact

    analyzer = GovernorRuleImpact()
    report = analyzer.analyze(n_recent_cycles=500)
    print(report)
"""

import json
import logging
import math
import os
from collections import defaultdict
from typing import Any

logger = logging.getLogger("proxima_ops.analytics.governor_rule_impact")

DEFAULT_LOG_PATH = "state/wave12_cycle_log.jsonl"

# All known governor-rule categories in the expected output order.
RULE_LABELS = [
    "circuit_breaker",
    "vel",
    "mt5_tick",
    "confirm_gate",
    "segl_state",
    "unknown_governor_rule",
]


class GovernorRuleImpact:
    """Quantify which governor rules are actually meaningful and which are redundant noise.

    Parameters
    ----------
    log_path : str
        Path to the JSON-lines cycle log (default ``state/wave12_cycle_log.jsonl``).
    """

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self._log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Analyse the trace log and return a governor-rule-impact report.

        Parameters
        ----------
        n_recent_cycles : int
            Only consider this many of the most recent cycles (default 500).
            Pass a large number (e.g. 1_000_000) to analyse all available
            cycles.

        Returns
        -------
        dict with keys:
            rule_block_counts        — how many times each rule blocked
            effective_rules          — rules that blocked > 5% of total blocks
            redundant_rules          — rules that blocked < 1% of total blocks
            governor_entropy_score   — 0.0 (single rule dominates) to 1.0 (evenly distributed)
            block_rate_over_time     — block rate per 100-cycle bucket
        """
        try:
            return self._build_report(n_recent_cycles)
        except Exception:
            logger.exception("GovernorRuleImpact.analyze failed")
            return {
                "rule_block_counts": {label: 0 for label in RULE_LABELS},
                "effective_rules": [],
                "redundant_rules": [],
                "governor_entropy_score": 0.0,
                "block_rate_over_time": {},
                "error": "See logs for details",
            }

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(self, n_recent_cycles: int) -> dict[str, Any]:
        records = self._load_records()

        if not records:
            return {
                "rule_block_counts": {label: 0 for label in RULE_LABELS},
                "effective_rules": [],
                "redundant_rules": [],
                "governor_entropy_score": 0.0,
                "block_rate_over_time": {},
                "warning": "No data found in log",
            }

        # Slice to the N most recent cycles.
        if n_recent_cycles < len(records):
            records = records[-n_recent_cycles:]

        # --- Per-cycle classification ------------------------------------
        rule_block_counts: dict[str, int] = defaultdict(int)
        blocked_by_cycle: dict[int, bool] = {}  # cycle_number -> was blocked?

        for entry in records:
            cycle = entry.get("cycle", 0)
            rule = self._classify_governor_rule(entry)
            if rule is not None:
                rule_block_counts[rule] += 1
                blocked_by_cycle[cycle] = True
            else:
                blocked_by_cycle[cycle] = False

        # --- Effective vs redundant --------------------------------------
        total_blocks = sum(rule_block_counts.values())
        effective_rules: list[str] = []
        redundant_rules: list[str] = []

        if total_blocks > 0:
            for label in RULE_LABELS:
                count = rule_block_counts.get(label, 0)
                pct = (count / total_blocks) * 100.0
                if pct > 5.0:
                    effective_rules.append(label)
                elif pct < 1.0:
                    redundant_rules.append(label)

        # --- Governor entropy score (normalised Shannon) -----------------
        entropy = self._compute_entropy(rule_block_counts, total_blocks)

        # --- Block rate over time (buckets of 100 cycles) ---------------
        block_rate_over_time = self._compute_block_rate_over_time(
            records, blocked_by_cycle
        )

        return {
            "rule_block_counts": dict(rule_block_counts),
            "effective_rules": effective_rules,
            "redundant_rules": redundant_rules,
            "governor_entropy_score": entropy,
            "block_rate_over_time": block_rate_over_time,
        }

    # ------------------------------------------------------------------
    # Rule classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_governor_rule(entry: dict) -> str | None:
        """Return which governor rule blocked this cycle, or *None* if the
        cycle was not blocked by any governor rule.

        Priority order (first match wins):
          1. CircuitBreaker in denial_reason
          2. VEL blocked in denial_reason
          3. No tick data in denial_reason
          4. Insufficient cross-projection confirm in denial_reason
          5. segl_state=OBSERVE in pipeline_trace.governor_gate
          6. segl_state top-level field == OBSERVE
          7. ready_to_exec=NO in governor_gate  →  unknown_governor_rule
          8. FAILED MT5 / place_order None      →  unknown_governor_rule
        """
        denial_reason = entry.get("denial_reason") or ""
        pipeline_trace = entry.get("pipeline_trace") or {}
        governor_gate = pipeline_trace.get("governor_gate") or []

        # 1. CircuitBreaker
        if "CircuitBreaker" in denial_reason:
            return "circuit_breaker"

        # 2. VEL blocked
        if "VEL blocked" in denial_reason:
            return "vel"

        # 3. No tick data
        if "No tick data" in denial_reason:
            return "mt5_tick"

        # 4. Insufficient cross-projection confirm
        if "Insufficient cross-projection confirm" in denial_reason:
            return "confirm_gate"

        # 5. segl_state=OBSERVE in governor_gate
        total_signals = entry.get("total_signals", 0)
        if governor_gate and any("segl_state=OBSERVE" in str(g) for g in governor_gate):
            # Only count as a block if there were signals to trade.
            if total_signals > 0:
                return "segl_state"

        # 6. segl_state top-level field
        segl_state = entry.get("segl_state", "")
        if segl_state == "OBSERVE":
            if total_signals > 0:
                return "segl_state"

        # 7. Governor gate with ready_to_exec=NO but not OBSERVE
        if governor_gate and any("ready_to_exec=NO" in str(g) for g in governor_gate):
            if total_signals > 0:
                return "unknown_governor_rule"

        # 8. Execution-level failure
        execution = pipeline_trace.get("execution") or ""
        if "FAILED MT5" in execution or "place_order returned None" in execution:
            return "unknown_governor_rule"

        return None

    # ------------------------------------------------------------------
    # Entropy computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entropy(
        rule_block_counts: dict[str, int], total_blocks: int
    ) -> float:
        """Compute normalised Shannon entropy for governor rule distribution.

        Formula
        -------
        If ``total_blocks == 0`` or only 1 active rule → 0.0.

        Otherwise let ``active_rules = number of rules with count > 0``.

        .. math::

            H = -\\sum_{i} p_i \\log_2(p_i) \\quad
            H_{\\text{norm}} = \\frac{H}{\\log_2(\\text{active\\_rules})}

        where :math:`p_i = \\text{count}_i / \\text{total\\_blocks}`.

        Returns a float in ``[0.0, 1.0]``.
        """
        if total_blocks == 0:
            return 0.0

        # How many distinct rules actually fired?
        active_rules = sum(1 for v in rule_block_counts.values() if v > 0)
        if active_rules <= 1:
            return 0.0

        h = 0.0
        for label in RULE_LABELS:
            count = rule_block_counts.get(label, 0)
            if count == 0:
                continue
            p = count / total_blocks
            h -= p * math.log2(p)

        # Normalise by log2(number of active rules).
        h_normalised = h / math.log2(active_rules)
        return max(0.0, min(1.0, h_normalised))

    # ------------------------------------------------------------------
    # Block rate over time
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_block_rate_over_time(
        records: list[dict[str, Any]],
        blocked_by_cycle: dict[int, bool],
    ) -> dict[str, float]:
        """Group cycles into buckets of 100 and compute the fraction that
        were blocked by any governor rule.

        Returns a dict mapping ``"cycle_{start}"`` (e.g. ``"cycle_0"``,
        ``"cycle_100"``) to the block fraction (0.0 — 1.0).
        """
        if not records:
            return {}

        # Collect all unique cycle numbers in sorted order.
        cycles: set[int] = set()
        for entry in records:
            c = entry.get("cycle")
            if c is not None:
                cycles.add(c)

        if not cycles:
            return {}

        buckets: dict[int, list[bool]] = defaultdict(list)
        for c in sorted(cycles):
            bucket_start = (c // 100) * 100
            buckets[bucket_start].append(blocked_by_cycle.get(c, False))

        block_rate: dict[str, float] = {}
        for bucket_start in sorted(buckets):
            bucket_cycles = buckets[bucket_start]
            fraction = sum(bucket_cycles) / len(bucket_cycles)
            block_rate[f"cycle_{bucket_start}"] = round(fraction, 4)

        return block_rate

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict[str, Any]]:
        """Load and return all JSON-line records from the trace log.

        Returns an empty list if the file does not exist or is unreadable.
        """
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
            print("Usage: python governor_rule_impact.py [n_recent_cycles]")
            sys.exit(1)

    analyzer = GovernorRuleImpact()
    report = analyzer.analyze(n_recent_cycles=n)
    print(json.dumps(report, indent=2))
