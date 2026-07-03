"""Live tick and signal capture for parity validation."""
import json
import os
import time
from datetime import datetime
from typing import Dict, Optional


class LiveCapturer:
    def __init__(self, output_dir: str = None):
        self._dir = output_dir or os.path.join(os.path.dirname(__file__), "captures")
        os.makedirs(self._dir, exist_ok=True)
        self._events: list[dict] = []

    def record_tick(self, tick: dict):
        self._events.append({
            "type": "tick",
            "ts": time.time(),
            "data": tick,
        })

    def record_signal(self, event_id: str, data: dict):
        self._events.append({
            "type": "signal",
            "event_id": event_id,
            "ts": time.time(),
            "data": data,
        })

    def record_trade(self, event_id: str, trade_data: dict):
        self._events.append({
            "type": "trade",
            "event_id": event_id,
            "ts": time.time(),
            "data": trade_data,
        })

    def save(self, label: str = None):
        ts = label or datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._dir, f"capture_{ts}.json")
        with open(path, "w") as f:
            json.dump(self._events, f, indent=1)
        return path

    def load(self, path: str):
        with open(path) as f:
            self._events = json.load(f)
