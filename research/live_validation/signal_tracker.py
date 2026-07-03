import numpy as np
from dataclasses import dataclass, field
from typing import Any

@dataclass
class SignalRecord:
    timestamp: int
    asset: str
    es_percentile: float
    residual_value: float
    at_percentile: float
    threshold_used: float
    regime: str
    persistence_forecast: str
    signal_state: str
    signal_score: float

class SignalTracker:
    def __init__(self):
        self._signals: list[SignalRecord] = []
        self._daily_counts: dict[int, int] = {}
        self._weekly_counts: dict[int, int] = {}
        self._monthly_counts: dict[int, int] = {}

    def record(self, ts: int, asset: str, es_pct: float, res: float,
               at_pct: float, threshold: float, regime: str,
               persistence: str, state: str, score: float):
        rec = SignalRecord(ts, asset, es_pct, res, at_pct, threshold,
                           regime, persistence, state, score)
        self._signals.append(rec)
        day = ts // 24
        week = ts // (24 * 7)
        month = ts // (24 * 30)
        self._daily_counts[day] = self._daily_counts.get(day, 0) + 1
        self._weekly_counts[week] = self._weekly_counts.get(week, 0) + 1
        self._monthly_counts[month] = self._monthly_counts.get(month, 0) + 1
        return rec

    def signals_per_day(self) -> list[tuple[int, int]]:
        return sorted(self._daily_counts.items())

    def signals_per_week(self) -> list[tuple[int, int]]:
        return sorted(self._weekly_counts.items())

    def signals_per_month(self) -> list[tuple[int, int]]:
        return sorted(self._monthly_counts.items())

    def recent_signals(self, n: int = 100) -> list[SignalRecord]:
        return self._signals[-n:]

    def get_all(self) -> list[SignalRecord]:
        return self._signals

    def summary(self) -> dict:
        total = len(self._signals)
        days = len(self._daily_counts)
        weeks = len(self._weekly_counts)
        months = len(self._monthly_counts)
        avg_day = total / max(days, 1)
        avg_week = total / max(weeks, 1)
        avg_month = total / max(months, 1)
        states = {}
        for s in self._signals:
            states[s.signal_state] = states.get(s.signal_state, 0) + 1
        return {
            "total_signals": total,
            "avg_per_day": round(avg_day, 1),
            "avg_per_week": round(avg_week, 1),
            "avg_per_month": round(avg_month, 1),
            "days_active": days,
            "state_distribution": states}
