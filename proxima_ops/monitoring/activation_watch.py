import json
import logging
import os
import time
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger("proxima_ops.monitoring.activation_watch")

RSI_PERIOD = 14
ATR_PERIOD = 14
EVENT_LOG_MAXLEN = 1000
RECENT_EVENTS_KEEP = 3

EVENT_TYPES = [
    "rsi_extreme",
    "atr_spike",
    "edge_confidence_hit",
    "threshold_pass",
    "confirm_progress",
    "governor_arm",
    "vel_block",
    "execution_attempt",
]

STATE_PATH = "state/activation_watch_state.json"


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


def _compute_rolling_atr_series(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = ATR_PERIOD
) -> np.ndarray:
    if len(closes) < period + 1:
        return np.array([])
    high_low = highs[1:] - lows[1:]
    high_close = np.abs(highs[1:] - closes[:-1])
    low_close = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_values = np.array([
        float(np.mean(tr[i - period + 1:i + 1]))
        for i in range(period - 1, len(tr))
    ])
    return atr_values


def _atr_60th_percentile(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    atr_series = _compute_rolling_atr_series(highs, lows, closes)
    if len(atr_series) < 2:
        return 0.0
    return round(float(np.percentile(atr_series, 60)), 6)


def _detect_rsi_extreme(md: dict) -> list[dict]:
    events = []
    closes = md.get("closes", {})
    for sym, c in closes.items():
        if c is None or (hasattr(c, "__len__") and len(c) == 0):
            continue
        rsi = _compute_rsi(c)
        if rsi < 30.0:
            events.append({"symbol": sym, "rsi": rsi, "extreme": "oversold"})
        elif rsi > 70.0:
            events.append({"symbol": sym, "rsi": rsi, "extreme": "overbought"})
    return events


def _detect_atr_spike(md: dict) -> list[dict]:
    events = []
    closes = md.get("closes", {})
    highs = md.get("highs", {})
    lows = md.get("lows", {})
    for sym in closes:
        c = closes[sym]
        h = highs.get(sym)
        l = lows.get(sym)
        if c is None or h is None or l is None:
            continue
        if hasattr(c, "__len__") and len(c) == 0:
            continue
        current_atr = _compute_atr(h, l, c)
        threshold = _atr_60th_percentile(h, l, c)
        if threshold > 0 and current_atr > threshold:
            events.append({
                "symbol": sym,
                "current_atr": current_atr,
                "threshold_60p": threshold,
                "excess_pct": round((current_atr / threshold - 1.0) * 100, 2),
            })
    return events


def _detect_edge_confidence_hit(all_signals: list) -> list[dict]:
    events = []
    for s in all_signals:
        conf = s.get("confidence", 0)
        if conf >= 0.40:
            events.append({
                "edge_id": s.get("edge_id", "?"),
                "symbol": s.get("symbol", "?"),
                "confidence": conf,
                "direction": s.get("direction", 0),
                "strategy": s.get("strategy", "?"),
            })
    return events


def _detect_threshold_pass(pipeline_trace: dict) -> list[dict]:
    events = []
    for entry in pipeline_trace.get("threshold_gate", []):
        if "PASS" in entry:
            parts = entry.split(":", 1)
            eid = parts[0].strip() if parts else "?"
            events.append({"edge_id": eid, "detail": entry})
    return events


def _detect_confirm_progress(cycle_data: dict) -> list[dict]:
    events = []
    confirm_cycles = cycle_data.get("confirm_cycles", 0)
    if confirm_cycles >= 1:
        events.append({
            "confirm_count": confirm_cycles,
            "symbol": cycle_data.get("active_symbol", "?"),
            "direction": cycle_data.get("active_direction", "?"),
        })
    return events


def _detect_governor_arm(cycle_data: dict) -> list[dict]:
    segl = cycle_data.get("segl_state", "")
    if segl == "ARMED":
        return [{"segl_state": segl}]
    return []


def _detect_vel_block(cycle_data: dict, pipeline_trace: dict) -> list[dict]:
    events = []
    exec_str = pipeline_trace.get("execution") or ""
    denial_reason = cycle_data.get("denial_reason") or ""
    if "VEL blocked" in denial_reason or "DENIED VEL:" in exec_str:
        events.append({
            "reason": denial_reason or exec_str,
            "vel_decision": cycle_data.get("vel_decision", ""),
        })
    return events


def _detect_execution_attempt(pipeline_trace: dict) -> list[dict]:
    events = []
    exec_str = pipeline_trace.get("execution")
    if exec_str is None:
        return events
    if exec_str.startswith("EXECUTED") or exec_str.startswith("FAILED"):
        events.append({"execution_detail": exec_str})
    return events


class ActivationWatch:
    def __init__(self, state_path: str = STATE_PATH):
        self._state_path = state_path
        self.event_log: deque[tuple] = deque(maxlen=EVENT_LOG_MAXLEN)
        self.event_counters: dict[str, int] = {}
        self.last_event_cycle: dict[str, int] = {}
        self.recent_events: dict[str, list] = {}

        for et in EVENT_TYPES:
            self.event_counters[et] = 0
            self.last_event_cycle[et] = 0
            self.recent_events[et] = []

        self._load_state()

    def _load_state(self):
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as f:
                data = json.load(f)
            counters = data.get("event_counters", {})
            last_cycle = data.get("last_event_cycle", {})
            recents = data.get("recent_events", {})
            for et in EVENT_TYPES:
                self.event_counters[et] = counters.get(et, 0)
                self.last_event_cycle[et] = last_cycle.get(et, 0)
                self.recent_events[et] = recents.get(et, [])[-RECENT_EVENTS_KEEP:]
            logger.info("ActivationWatch state loaded from %s", self._state_path)
        except Exception as e:
            logger.warning("Failed to load activation watch state: %s", e)

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
            data = {
                "event_counters": self.event_counters,
                "last_event_cycle": self.last_event_cycle,
                "recent_events": self.recent_events,
            }
            with open(self._state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save activation watch state: %s", e)

    def _log_event(self, event_type: str, details: dict, cycle: int):
        ts = time.time()
        self.event_log.append((ts, event_type, details))
        self.event_counters[event_type] += 1
        self.last_event_cycle[event_type] = cycle
        recent = self.recent_events[event_type]
        recent.append(ts)
        self.recent_events[event_type] = recent[-RECENT_EVENTS_KEEP:]
        logger.debug("ActivationWatch [%s] cycle=%s %s", event_type, cycle, details)
        self._save_state()

    def check_cycle(self, cycle_data: dict, pipeline_trace: dict, md: dict, all_signals: list):
        cycle = cycle_data.get("cycle", 0)

        rsi_events = _detect_rsi_extreme(md)
        for ev in rsi_events:
            self._log_event("rsi_extreme", ev, cycle)

        atr_events = _detect_atr_spike(md)
        for ev in atr_events:
            self._log_event("atr_spike", ev, cycle)

        conf_events = _detect_edge_confidence_hit(all_signals)
        for ev in conf_events:
            self._log_event("edge_confidence_hit", ev, cycle)

        thresh_events = _detect_threshold_pass(pipeline_trace)
        for ev in thresh_events:
            self._log_event("threshold_pass", ev, cycle)

        confirm_events = _detect_confirm_progress(cycle_data)
        for ev in confirm_events:
            self._log_event("confirm_progress", ev, cycle)

        arm_events = _detect_governor_arm(cycle_data)
        for ev in arm_events:
            self._log_event("governor_arm", ev, cycle)

        vel_events = _detect_vel_block(cycle_data, pipeline_trace)
        for ev in vel_events:
            self._log_event("vel_block", ev, cycle)

        exec_events = _detect_execution_attempt(pipeline_trace)
        for ev in exec_events:
            self._log_event("execution_attempt", ev, cycle)

    def event_summary(self) -> dict:
        total = sum(self.event_counters.values())
        return {
            "total_events": total,
            "event_counts": dict(self.event_counters),
            "last_seen": dict(self.last_event_cycle),
            "recent": {k: list(v) for k, v in self.recent_events.items()},
        }

    def has_seen_event(self, event_type: str) -> bool:
        return self.event_counters.get(event_type, 0) > 0

    def last_events(self, n: int = 10) -> list[dict]:
        result = []
        for ts, etype, details in list(self.event_log)[-n:]:
            result.append({
                "timestamp": ts,
                "event_type": etype,
                "details": details,
            })
        return result

    def clear(self):
        self.event_log.clear()
        for et in EVENT_TYPES:
            self.event_counters[et] = 0
            self.last_event_cycle[et] = 0
            self.recent_events[et] = []
        self._save_state()
        logger.info("ActivationWatch state cleared")
