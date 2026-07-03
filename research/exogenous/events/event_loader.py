import sys; sys.path.insert(0, "."); sys.path.insert(0, "proxima_x")

import csv
from datetime import datetime, timezone

from research.exogenous.events.schemas import MacroEvent


class EventLoader:
    """Loads macro-economic event timestamps from CSV or built-in calendar."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # CSV loader
    # ------------------------------------------------------------------
    def load_csv(self, path: str) -> list[MacroEvent]:
        """Load events from a CSV file.

        Expected CSV columns:
            ts,currency,impact,name,actual,forecast,previous
        """
        events: list[MacroEvent] = []
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = float(row["ts"])
                currency = row["currency"]
                impact = row["impact"]
                name = row["name"]
                actual = float(row["actual"]) if row["actual"] else None
                forecast = float(row["forecast"]) if row["forecast"] else None
                previous = float(row["previous"]) if row["previous"] else None
                events.append(
                    MacroEvent(
                        ts=ts,
                        currency=currency,
                        impact=impact,
                        name=name,
                        actual=actual,
                        forecast=forecast,
                        previous=previous,
                    )
                )
        return events

    # ------------------------------------------------------------------
    # Built-in calendar  –  April 2026 high-impact events (USD & JPY)
    # ------------------------------------------------------------------
    def load_builtin_events(self) -> list[MacroEvent]:
        """Return hardcoded April 2026 high-impact macro events.

        Dates are approximate, based on recurring monthly / quarterly
        release schedules.  All timestamps are UTC.
        """
        def _dt(month: int, day: int, hour: int, minute: int) -> float:
            return datetime(2026, month, day, hour, minute, tzinfo=timezone.utc).timestamp()

        events: list[MacroEvent] = [
            # ---- USD events ----
            MacroEvent(
                ts=_dt(4, 1, 14, 0), currency="USD", impact="HIGH",
                name="ISM Manufacturing PMI", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 3, 12, 30), currency="USD", impact="HIGH",
                name="NFP", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 7, 14, 0), currency="USD", impact="HIGH",
                name="JOLTS", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 9, 18, 0), currency="USD", impact="HIGH",
                name="FOMC Minutes", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 10, 12, 30), currency="USD", impact="HIGH",
                name="CPI", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 14, 12, 30), currency="USD", impact="HIGH",
                name="PPI", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 15, 12, 30), currency="USD", impact="HIGH",
                name="Retail Sales", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 16, 12, 30), currency="USD", impact="HIGH",
                name="Housing Starts", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 17, 12, 30), currency="USD", impact="HIGH",
                name="Building Permits", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 22, 14, 0), currency="USD", impact="HIGH",
                name="Existing Home Sales", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 23, 14, 0), currency="USD", impact="HIGH",
                name="New Home Sales", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 24, 12, 30), currency="USD", impact="HIGH",
                name="Durable Goods", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 28, 14, 0), currency="USD", impact="HIGH",
                name="CB Consumer Confidence", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 29, 12, 30), currency="USD", impact="HIGH",
                name="GDP", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 30, 18, 0), currency="USD", impact="HIGH",
                name="FOMC Rate Decision", actual=None, forecast=None, previous=None,
            ),
            # ---- JPY events ----
            MacroEvent(
                ts=_dt(4, 27, 0, 0), currency="JPY", impact="HIGH",
                name="BoJ Rate Decision", actual=None, forecast=None, previous=None,
            ),
            MacroEvent(
                ts=_dt(4, 24, 0, 0), currency="JPY", impact="HIGH",
                name="Tokyo CPI", actual=None, forecast=None, previous=None,
            ),
        ]
        return events

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    def filter_by_currency(
        self, events: list[MacroEvent], currencies: list[str]
    ) -> list[MacroEvent]:
        """Return only events whose currency is in *currencies*."""
        return [e for e in events if e.currency in currencies]

    def filter_by_impact(
        self, events: list[MacroEvent], min_impact: str = "HIGH"
    ) -> list[MacroEvent]:
        """Return only events with impact >= *min_impact*.

        Impact ranking (highest → lowest): HIGH > MEDIUM > LOW.
        """
        order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        threshold = order.get(min_impact, 0)
        return [e for e in events if order.get(e.impact, 0) >= threshold]
