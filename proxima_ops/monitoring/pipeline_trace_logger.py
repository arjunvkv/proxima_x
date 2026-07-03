"""
Per-cycle pipeline trace logger for live MT5 trading system.
Records structured trace entries to JSONL for pipeline bottleneck analysis.
"""

import json
import logging
import os
import re
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("proxima_ops.monitoring.pipeline_trace_logger")

TRACE_PATH = "state/live_pipeline_trace.jsonl"
MAX_RECENT_ENTRIES = 1000
RSI_PERIOD = 14
ATR_PERIOD = 14
TREND_PERIOD = 14

_GENERATED_RE = re.compile(
    r"(\S+)\s+(\S+)\s+(\S+)\s+dir=(-?\d+)\s+conf=([\d.]+)\s+->\s+(PASS|FAIL)"
)
_THRESHOLD_RE = re.compile(r"(\S+):\s+(.+)")
_CONFIRM_RE = re.compile(r"(\S+):\s+(.+)")
_CONFIRM_CYCLES_RE = re.compile(r"(?:cycles|cross_cyc)=(\d+)/2")


def _compute_rsi(closes: np.ndarray, period: int = RSI_PERIOD) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def _compute_atr(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = ATR_PERIOD
) -> float:
    if len(closes) < period + 1:
        return 0.0
    high_low = highs[1:] - lows[1:]
    high_close = np.abs(highs[1:] - closes[:-1])
    low_close = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return round(float(np.mean(tr[-period:])), 6)


def _compute_trend(closes: np.ndarray, period: int = TREND_PERIOD) -> float:
    if len(closes) < period:
        return 0.0
    x = np.arange(period, dtype=np.float64)
    y = closes[-period:].astype(np.float64)
    slope = np.polyfit(x, y, 1)[0]
    norm = float(closes[-1])
    if norm == 0.0:
        return 0.0
    return round(slope / norm, 6)


def _parse_generated(entry: str) -> Optional[dict]:
    m = _GENERATED_RE.match(entry)
    if not m:
        return None
    return {
        "edge_id": m.group(1),
        "symbol": m.group(2),
        "strategy": m.group(3),
        "direction": int(m.group(4)),
        "confidence": float(m.group(5)),
        "threshold_pass": m.group(6) == "PASS",
    }


def _parse_threshold(entry: str) -> dict:
    m = _THRESHOLD_RE.match(entry)
    if not m:
        return {"edge_id": "?", "detail": entry, "pass": False}
    eid = m.group(1)
    detail = m.group(2)
    return {"edge_id": eid, "detail": detail, "pass": detail.startswith("PASS")}


def _parse_confirm(entry: str) -> dict:
    m = _CONFIRM_RE.match(entry)
    if not m:
        return {"edge_id": "?", "detail": entry, "pass": False, "count": 0}
    eid = m.group(1)
    detail = m.group(2)
    is_pass = "CROSS_PASS" in detail
    count = 0
    cm = _CONFIRM_CYCLES_RE.search(detail)
    if cm:
        count = int(cm.group(1))
    return {"edge_id": eid, "detail": detail, "pass": is_pass, "count": count}


class PipelineTraceLogger:
    """Per-cycle pipeline trace logger.

    Records structured trace entries to a JSONL file and keeps a rolling
    in-memory window for quick diagnostic access.
    """

    def __init__(
        self, trace_path: str = TRACE_PATH, max_entries: int = MAX_RECENT_ENTRIES
    ):
        self._trace_path = trace_path
        self._recent: deque[dict] = deque(maxlen=max_entries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_trace_path(self) -> str:
        """Return the path of the current trace file."""
        return self._trace_path

    def record_cycle(
        self,
        cycle_data: dict,
        pipeline_trace: dict,
        md: dict,
        all_signals: list,
    ) -> None:
        """Build a trace entry for one cycle and persist it only on non-idle cycles."""
        try:
            entry = self._build_entry(cycle_data, pipeline_trace, md, all_signals)
            self._recent.append(entry)
            if self._is_idle_cycle(entry):
                return
            self._write_entry(entry)
        except Exception:
            logger.exception(
                "Failed to record pipeline trace for cycle %s",
                cycle_data.get("cycle", "?"),
            )

    @staticmethod
    def _is_idle_cycle(entry: dict) -> bool:
        funnel = entry.get("pipeline_funnel", {})
        execution = entry.get("execution", {})
        return (
            funnel.get("signals", 0) == 0
            and funnel.get("threshold", 0) == 0
            and funnel.get("confirm", 0) == 0
            and funnel.get("vel_blocked", 0) == 0
            and execution.get("decision") != "EXECUTE"
        )

    def recent_entries(self) -> list[dict]:
        """Return the rolling window of recent trace entries."""
        return list(self._recent)

    def funnel_summary(self, last_n: int = 100) -> dict:
        """Aggregate funnel kill distribution over the last N cycles."""
        window = list(self._recent)[-last_n:]
        if not window:
            return {}
        total = len(window)
        funnel_keys = [
            "signals", "threshold", "confirm",
            "governor_ready", "vel_blocked", "executed",
        ]
        result: dict[str, Any] = {"cycles": total}
        for key in funnel_keys:
            values = [e["pipeline_funnel"].get(key, 0) for e in window]
            result[f"avg_{key}"] = round(sum(values) / total, 2) if total else 0
            result[f"sum_{key}"] = sum(values)
        executions = [
            e for e in window if e["pipeline_funnel"].get("executed", 0) > 0
        ]
        holds = [
            e for e in window
            if e.get("execution", {}).get("decision") == "HOLD"
        ]
        result["execution_count"] = len(executions)
        result["hold_count"] = len(holds)
        kill_reasons: dict[str, int] = {}
        for e in window:
            denial = e.get("execution", {}).get("denial_reason")
            if denial:
                kill_reasons[denial] = kill_reasons.get(denial, 0) + 1
        result["kill_distribution"] = dict(
            sorted(kill_reasons.items(), key=lambda x: -x[1])
        )
        return result

    # ------------------------------------------------------------------
    # Entry construction
    # ------------------------------------------------------------------

    def _build_entry(
        self,
        cycle_data: dict,
        pipeline_trace: dict,
        md: dict,
        all_signals: list,
    ) -> dict:
        cycle = cycle_data.get("cycle", 0)
        timestamp = datetime.now(timezone.utc).isoformat()
        md_summary = self._build_md_summary(md)
        signals_generated = len(all_signals)
        signals_detail = self._build_signals_detail(
            pipeline_trace.get("generated", []), all_signals
        )
        threshold_pass_count = self._count_threshold_passes(
            pipeline_trace.get("threshold_gate", [])
        )
        confirm_gate = self._build_confirm_gate(
            pipeline_trace.get("confirm_gate", []), all_signals
        )
        governor = self._build_governor(
            cycle_data, pipeline_trace.get("governor_gate", [])
        )
        vel = self._build_vel(cycle_data)
        circuit_breaker = self._build_circuit_breaker(cycle_data)
        execution = self._build_execution(
            cycle_data, pipeline_trace.get("execution")
        )
        open_positions = cycle_data.get("open_positions", 0)
        pipeline_funnel = self._build_funnel(
            signals_generated, threshold_pass_count, cycle_data,
            governor, vel, execution,
        )
        return {
            "cycle": cycle,
            "timestamp": timestamp,
            "md_summary": md_summary,
            "signals_generated": signals_generated,
            "signals_detail": signals_detail,
            "threshold_pass_count": threshold_pass_count,
            "confirm_gate": confirm_gate,
            "governor": governor,
            "vel": vel,
            "circuit_breaker": circuit_breaker,
            "execution": execution,
            "open_positions": open_positions,
            "pipeline_funnel": pipeline_funnel,
        }

    # ------------------------------------------------------------------
    # Sub-builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_md_summary(md: dict) -> dict[str, dict]:
        summary: dict[str, dict] = {}
        closes = md.get("closes", {})
        highs = md.get("highs", {})
        lows = md.get("lows", {})
        for sym in closes:
            c = closes[sym]
            if c is None or (hasattr(c, "__len__") and len(c) == 0):
                continue
            rsi = _compute_rsi(c)
            h = highs.get(sym)
            l = lows.get(sym)
            atr = _compute_atr(h, l, c) if h is not None and l is not None else 0.0
            trend = _compute_trend(c)
            summary[sym] = {"rsi": rsi, "atr": atr, "trend": trend}
        return summary

    @staticmethod
    def _build_signal_lookup(all_signals: list) -> dict[str, dict]:
        lookup: dict[str, dict] = {}
        for s in all_signals:
            eid = s.get("edge_id")
            if eid:
                lookup[eid] = s
        return lookup

    def _build_signals_detail(
        self, generated_entries: list, all_signals: list
    ) -> list[dict]:
        lookup = self._build_signal_lookup(all_signals)
        details: list[dict] = []
        for entry in generated_entries:
            parsed = _parse_generated(entry)
            if parsed:
                eid = parsed["edge_id"]
                sig = lookup.get(eid, {})
                details.append({
                    "edge_id": eid,
                    "symbol": parsed.get("symbol", sig.get("symbol", "?")),
                    "direction": parsed.get("direction", sig.get("direction", 0)),
                    "confidence": parsed.get("confidence", sig.get("confidence", 0.0)),
                    "strategy": parsed.get("strategy", sig.get("strategy", "?")),
                    "threshold_pass": parsed.get("threshold_pass", False),
                })
            else:
                eid = entry.split()[0] if entry else "?"
                sig = lookup.get(eid, {})
                if sig:
                    details.append({
                        "edge_id": eid,
                        "symbol": sig.get("symbol", "?"),
                        "direction": sig.get("direction", 0),
                        "confidence": sig.get("confidence", 0.0),
                        "strategy": sig.get("strategy", "?"),
                        "threshold_pass": (
                            sig.get("direction", 0) != 0
                            and sig.get("confidence", 0) >= 0.40
                        ),
                    })
        return details

    @staticmethod
    def _count_threshold_passes(threshold_entries: list) -> int:
        return sum(1 for e in threshold_entries if "PASS" in e)

    def _build_confirm_gate(
        self, confirm_entries: list, all_signals: list
    ) -> dict:
        lookup = self._build_signal_lookup(all_signals)
        edges = list(confirm_entries)
        confirm_map: dict[str, int] = {}
        any_pass = False
        for entry in confirm_entries:
            parsed = _parse_confirm(entry)
            if not parsed:
                continue
            eid = parsed["edge_id"]
            sig = lookup.get(eid, {})
            sym = sig.get("symbol", "?")
            s_dir = "BUY" if sig.get("direction", 0) > 0 else "SELL"
            pair = f"{sym}_{s_dir}"
            confirm_map[pair] = max(confirm_map.get(pair, 0), parsed["count"])
            if parsed["pass"]:
                any_pass = True
        confirm_pass = any_pass or bool(
            confirm_entries and any("CROSS_PASS" in e for e in confirm_entries)
        )
        return {
            "edges": edges,
            "confirm_map": confirm_map,
            "confirm_pass": confirm_pass,
        }

    @staticmethod
    def _build_governor(cycle_data: dict, governor_entries: list) -> dict:
        segl_state = cycle_data.get("segl_state", "?")
        intent_compliant = cycle_data.get("intent_compliant", False)
        reason = " | ".join(governor_entries) if governor_entries else ""
        authorized = False
        if segl_state == "ARMED" and intent_compliant:
            denial = cycle_data.get("denial_reason") or ""
            authorized = (
                "governor" not in denial.lower()
                and "state_machine" not in denial.lower()
            )
        return {
            "segl_state": segl_state,
            "intent_compliant": intent_compliant,
            "authorized": authorized,
            "reason": reason,
        }

    @staticmethod
    def _build_vel(cycle_data: dict) -> dict:
        vel_reason = cycle_data.get("vel_decision")
        if vel_reason is None:
            return {"checked": False, "allowed": True, "reason": "not_checked"}
        denial = cycle_data.get("denial_reason") or ""
        allowed = "VEL blocked" not in denial
        return {"checked": True, "allowed": allowed, "reason": vel_reason}

    @staticmethod
    def _build_circuit_breaker(cycle_data: dict) -> dict:
        cb_reason = cycle_data.get("cb_decision")
        if cb_reason is None:
            return {"checked": False, "allowed": True, "reason": "not_checked"}
        denial = cycle_data.get("denial_reason") or ""
        allowed = "CircuitBreaker" not in denial
        return {"checked": True, "allowed": allowed, "reason": cb_reason}

    @staticmethod
    def _build_execution(
        cycle_data: dict, exec_entry: Optional[str]
    ) -> dict:
        decision = cycle_data.get("decision", "?")
        denial_reason = cycle_data.get("denial_reason")
        mt5_result = None
        if cycle_data.get("execution_result"):
            mt5_result = cycle_data["execution_result"]
        elif cycle_data.get("execution_error"):
            mt5_result = {"error": cycle_data["execution_error"]}
        return {
            "decision": decision,
            "denial_reason": denial_reason,
            "mt5_result": mt5_result,
        }

    @staticmethod
    def _build_funnel(
        signals_generated: int,
        threshold_pass_count: int,
        cycle_data: dict,
        governor: dict,
        vel: dict,
        execution: dict,
    ) -> dict:
        return {
            "signals": signals_generated,
            "threshold": threshold_pass_count,
            "confirm": cycle_data.get("active_signals", 0),
            "governor_ready": 1 if governor.get("authorized") else 0,
            "vel_blocked": 1 if not vel.get("allowed", True) else 0,
            "executed": 1 if execution.get("decision") == "EXECUTE" else 0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _write_entry(self, entry: dict) -> None:
        os.makedirs(os.path.dirname(self._trace_path) or ".", exist_ok=True)
        with open(self._trace_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
