import random
import time
import math
from ..core.fsv_schema import NormalizedEvent, neutral_fsv


class SyntheticMacroGenerator:

    SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF"]
    EVENT_TYPES = ["CPI", "NEWS", "RATE", "GDP", "SENTIMENT"]
    SOURCES = ["tradingeconomics", "fxstreet", "investing", "fred"]

    def generate_event(self, symbol: str, event_type: str = None, time_offset: float = 0) -> NormalizedEvent:
        if event_type is None:
            event_type = random.choice(self.EVENT_TYPES)
        surprise_score = max(-1.0, min(1.0, random.gauss(0, 0.3)))
        if event_type == "CPI":
            if surprise_score > 0:
                direction_bias = -abs(random.gauss(0.4, 0.2))
            else:
                direction_bias = random.choice([-1, 1]) * abs(random.gauss(0.3, 0.15))
            direction_bias = max(-1.0, min(1.0, direction_bias))
            impact_weight = 0.8
        elif event_type == "RATE":
            base = 0.5 if surprise_score > 0 else -0.5
            direction_bias = max(-1.0, min(1.0, random.gauss(base, 0.2)))
            impact_weight = 0.9
        elif event_type == "NEWS":
            direction_bias = max(-1.0, min(1.0, random.gauss(0, 0.4)))
            impact_weight = 0.4
        elif event_type == "GDP":
            base = 0.5 if surprise_score > 0 else -0.5
            direction_bias = max(-1.0, min(1.0, random.gauss(base, 0.2)))
            impact_weight = 0.7
        elif event_type == "SENTIMENT":
            base = 0.4 if surprise_score > 0 else -0.4
            direction_bias = max(-1.0, min(1.0, random.gauss(base, 0.2)))
            impact_weight = 0.5
        else:
            direction_bias = 0.0
            impact_weight = 0.5
        timestamp = time.time() + time_offset
        source = random.choice(self.SOURCES)
        raw_data = {
            "event_type": event_type,
            "source": source,
            "value": round(random.gauss(100, 10), 2),
            "previous": round(random.gauss(100, 10), 2),
            "forecast": round(random.gauss(100, 10), 2),
        }
        return NormalizedEvent(
            symbol=symbol,
            event_type=event_type,
            surprise_score=surprise_score,
            direction_bias=direction_bias,
            impact_weight=impact_weight,
            timestamp=timestamp,
            source=source,
            raw_data=raw_data,
        )

    def generate_event_stream(
        self,
        symbols: list[str] = None,
        duration_seconds: float = 3600,
        events_per_minute: float = 2.0,
    ) -> list[NormalizedEvent]:
        if symbols is None:
            symbols = self.SYMBOLS
        total_events = int(duration_seconds / 60.0 * events_per_minute)
        events: list[NormalizedEvent] = []
        for _ in range(total_events):
            symbol = random.choice(symbols)
            event = self.generate_event(
                symbol=symbol,
                time_offset=random.uniform(0, duration_seconds),
            )
            events.append(event)
        events.sort(key=lambda e: e.timestamp)
        return events

    def _clamp_bias(self, value: float) -> float:
        return max(-1.0, min(1.0, value))

    def stress_scenario(self, mode: str = "crisis") -> list[NormalizedEvent]:
        events: list[NormalizedEvent] = []
        if mode == "crisis":
            num_events = 20
            duration = 300.0
            base_time = time.time()
            cluster_starts = [random.uniform(0, duration - 10) for _ in range(4)]
            for cluster_start in cluster_starts:
                cluster_count = random.randint(3, 5)
                for _ in range(cluster_count):
                    symbol = random.choice(self.SYMBOLS)
                    event_type = random.choice(["RATE", "NEWS"])
                    surprise_score = -abs(random.gauss(0.6, 0.15))
                    surprise_score = max(-1.0, min(1.0, surprise_score))
                    direction_bias = self._clamp_bias(-abs(random.gauss(0.5, 0.15)))
                    impact_weight = random.uniform(0.7, 1.0)
                    timestamp = base_time + cluster_start + random.uniform(0, 10)
                    source = random.choice(self.SOURCES)
                    raw_data = {
                        "event_type": event_type,
                        "source": source,
                        "value": round(random.gauss(100, 15), 2),
                        "previous": round(random.gauss(105, 10), 2),
                        "forecast": round(random.gauss(103, 10), 2),
                    }
                    events.append(NormalizedEvent(
                        symbol=symbol,
                        event_type=event_type,
                        surprise_score=surprise_score,
                        direction_bias=direction_bias,
                        impact_weight=impact_weight,
                        timestamp=timestamp,
                        source=source,
                        raw_data=raw_data,
                    ))
            remaining = num_events - len(events)
            for _ in range(remaining):
                symbol = random.choice(self.SYMBOLS)
                event_type = random.choice(["CPI", "GDP", "SENTIMENT"])
                surprise_score = -abs(random.gauss(0.5, 0.2))
                surprise_score = max(-1.0, min(1.0, surprise_score))
                direction_bias = self._clamp_bias(-abs(random.gauss(0.4, 0.15)))
                impact_weight = random.uniform(0.7, 1.0)
                timestamp = base_time + random.uniform(0, duration)
                source = random.choice(self.SOURCES)
                raw_data = {
                    "event_type": event_type,
                    "source": source,
                    "value": round(random.gauss(95, 12), 2),
                    "previous": round(random.gauss(100, 10), 2),
                    "forecast": round(random.gauss(100, 10), 2),
                }
                events.append(NormalizedEvent(
                    symbol=symbol,
                    event_type=event_type,
                    surprise_score=surprise_score,
                    direction_bias=direction_bias,
                    impact_weight=impact_weight,
                    timestamp=timestamp,
                    source=source,
                    raw_data=raw_data,
                ))
        elif mode == "trend":
            num_events = 30
            duration = 600.0
            base_time = time.time()
            direction_sign = random.choice([1, -1])
            for i in range(num_events):
                symbol = random.choice(self.SYMBOLS)
                event_type = random.choice(self.EVENT_TYPES)
                base_time_val = base_time + (i / num_events) * duration
                cluster_bump = 0.0
                if random.random() < 0.2:
                    cluster_bump = random.uniform(-5, 5)
                timestamp = base_time_val + random.uniform(0, duration / num_events) + cluster_bump
                surprise_score = direction_sign * abs(random.gauss(0.4, 0.15))
                surprise_score = max(-1.0, min(1.0, surprise_score))
                if event_type == "CPI":
                    direction_bias = self._clamp_bias(direction_sign * -abs(random.gauss(0.4, 0.15)))
                else:
                    direction_bias = self._clamp_bias(direction_sign * abs(random.gauss(0.4, 0.15)))
                impact_map = {"CPI": 0.8, "RATE": 0.9, "GDP": 0.7, "NEWS": 0.4, "SENTIMENT": 0.5}
                impact_weight = min(1.0, impact_map[event_type] * random.uniform(0.7, 1.0))
                source = random.choice(self.SOURCES)
                raw_data = {
                    "event_type": event_type,
                    "source": source,
                    "value": round(random.gauss(100, 10), 2),
                    "previous": round(random.gauss(100, 10), 2),
                    "forecast": round(random.gauss(100, 10), 2),
                }
                events.append(NormalizedEvent(
                    symbol=symbol,
                    event_type=event_type,
                    surprise_score=surprise_score,
                    direction_bias=direction_bias,
                    impact_weight=impact_weight,
                    timestamp=timestamp,
                    source=source,
                    raw_data=raw_data,
                ))
        elif mode == "conflict":
            num_events = 25
            duration = 400.0
            base_time = time.time()
            half = num_events // 2
            for _ in range(half):
                symbol = random.choice(self.SYMBOLS)
                event_type = random.choice(self.EVENT_TYPES)
                timestamp = base_time + random.uniform(0, duration)
                surprise_score = abs(random.gauss(0.5, 0.15))
                surprise_score = min(1.0, surprise_score)
                if event_type == "CPI":
                    direction_bias = self._clamp_bias(-abs(random.gauss(0.4, 0.15)))
                else:
                    direction_bias = self._clamp_bias(abs(random.gauss(0.4, 0.15)))
                impact_map = {"CPI": 0.8, "RATE": 0.9, "GDP": 0.7, "NEWS": 0.4, "SENTIMENT": 0.5}
                impact_weight = impact_map[event_type]
                source = random.choice(self.SOURCES)
                raw_data = {
                    "event_type": event_type,
                    "source": source,
                    "value": round(random.gauss(100, 10), 2),
                    "previous": round(random.gauss(100, 10), 2),
                    "forecast": round(random.gauss(100, 10), 2),
                }
                events.append(NormalizedEvent(
                    symbol=symbol,
                    event_type=event_type,
                    surprise_score=surprise_score,
                    direction_bias=direction_bias,
                    impact_weight=impact_weight,
                    timestamp=timestamp,
                    source=source,
                    raw_data=raw_data,
                ))
            for _ in range(half):
                symbol = random.choice(self.SYMBOLS)
                event_type = random.choice(self.EVENT_TYPES)
                timestamp = base_time + random.uniform(0, duration)
                surprise_score = -abs(random.gauss(0.5, 0.15))
                surprise_score = max(-1.0, surprise_score)
                if event_type == "CPI":
                    direction_bias = self._clamp_bias(abs(random.gauss(0.4, 0.15)))
                else:
                    direction_bias = self._clamp_bias(-abs(random.gauss(0.4, 0.15)))
                impact_map = {"CPI": 0.8, "RATE": 0.9, "GDP": 0.7, "NEWS": 0.4, "SENTIMENT": 0.5}
                impact_weight = impact_map[event_type]
                source = random.choice(self.SOURCES)
                raw_data = {
                    "event_type": event_type,
                    "source": source,
                    "value": round(random.gauss(100, 10), 2),
                    "previous": round(random.gauss(100, 10), 2),
                    "forecast": round(random.gauss(100, 10), 2),
                }
                events.append(NormalizedEvent(
                    symbol=symbol,
                    event_type=event_type,
                    surprise_score=surprise_score,
                    direction_bias=direction_bias,
                    impact_weight=impact_weight,
                    timestamp=timestamp,
                    source=source,
                    raw_data=raw_data,
                ))
            for _ in range(3):
                symbol = random.choice(self.SYMBOLS)
                event_type = random.choice(["CPI", "RATE"])
                timestamp = base_time + random.uniform(0, duration)
                source1 = random.choice(self.SOURCES)
                source2 = random.choice([s for s in self.SOURCES if s != source1])
                surprise_pos = abs(random.gauss(0.5, 0.15))
                surprise_pos = min(1.0, surprise_pos)
                raw_data1 = {
                    "event_type": event_type,
                    "source": source1,
                    "value": round(random.gauss(100, 10), 2),
                    "previous": round(random.gauss(100, 10), 2),
                    "forecast": round(random.gauss(100, 10), 2),
                }
                events.append(NormalizedEvent(
                    symbol=symbol,
                    event_type=event_type,
                    surprise_score=surprise_pos,
                    direction_bias=self._clamp_bias(abs(random.gauss(0.5, 0.15))),
                    impact_weight=0.8,
                    timestamp=timestamp,
                    source=source1,
                    raw_data=raw_data1,
                ))
                raw_data2 = {
                    "event_type": event_type,
                    "source": source2,
                    "value": round(random.gauss(100, 10), 2),
                    "previous": round(random.gauss(100, 10), 2),
                    "forecast": round(random.gauss(100, 10), 2),
                }
                events.append(NormalizedEvent(
                    symbol=symbol,
                    event_type=event_type,
                    surprise_score=-surprise_pos,
                    direction_bias=self._clamp_bias(-abs(random.gauss(0.5, 0.15))),
                    impact_weight=0.8,
                    timestamp=timestamp + random.uniform(-0.5, 0.5),
                    source=source2,
                    raw_data=raw_data2,
                ))
        elif mode == "api_failure":
            base_time = time.time()
            for _ in range(5):
                symbol = random.choice(self.SYMBOLS)
                event_type = random.choice(self.EVENT_TYPES)
                timestamp = base_time + random.uniform(0, 10)
                event = self.generate_event(symbol=symbol, event_type=event_type)
                event.timestamp = timestamp
                events.append(event)
            for _ in range(3):
                symbol = random.choice(self.SYMBOLS)
                event_type = random.choice(self.EVENT_TYPES)
                timestamp = base_time + 500 + random.uniform(0, 30)
                event = self.generate_event(symbol=symbol, event_type=event_type)
                event.timestamp = timestamp
                events.append(event)
        events.sort(key=lambda e: e.timestamp)
        return events

    def cpi_shock(self, symbol: str, magnitude: float = 0.8) -> NormalizedEvent:
        surprise_score = random.choice([-1, 1]) * abs(random.gauss(magnitude * 0.8, 0.1))
        surprise_score = max(-1.0, min(1.0, surprise_score))
        if surprise_score > 0:
            direction_bias = self._clamp_bias(-abs(random.gauss(0.6, 0.1)))
        else:
            direction_bias = self._clamp_bias(abs(random.gauss(0.6, 0.1)))
        source = random.choice(self.SOURCES)
        raw_data = {
            "event_type": "CPI",
            "source": source,
            "value": round(random.gauss(100, 10), 2),
            "previous": round(random.gauss(100, 10), 2),
            "forecast": round(random.gauss(100, 10), 2),
        }
        return NormalizedEvent(
            symbol=symbol,
            event_type="CPI",
            surprise_score=surprise_score,
            direction_bias=direction_bias,
            impact_weight=0.9,
            timestamp=time.time(),
            source=source,
            raw_data=raw_data,
        )

    def rate_shock(self, symbol: str, direction: str = "hawkish") -> NormalizedEvent:
        if direction == "hawkish":
            surprise_score = abs(random.gauss(0.6, 0.1))
            surprise_score = min(1.0, surprise_score)
            direction_bias = self._clamp_bias(abs(random.gauss(0.6, 0.1)))
        else:
            surprise_score = -abs(random.gauss(0.6, 0.1))
            surprise_score = max(-1.0, surprise_score)
            direction_bias = self._clamp_bias(-abs(random.gauss(0.6, 0.1)))
        source = random.choice(self.SOURCES)
        raw_data = {
            "event_type": "RATE",
            "source": source,
            "value": round(random.gauss(5, 1), 2),
            "previous": round(random.gauss(4, 1), 2),
            "forecast": round(random.gauss(4.5, 1), 2),
        }
        return NormalizedEvent(
            symbol=symbol,
            event_type="RATE",
            surprise_score=surprise_score,
            direction_bias=direction_bias,
            impact_weight=0.9,
            timestamp=time.time(),
            source=source,
            raw_data=raw_data,
        )

    def generate_multi_source_conflict(
        self,
        symbol: str,
        base_event_type: str = "CPI",
    ) -> list[NormalizedEvent]:
        events: list[NormalizedEvent] = []
        base_time = time.time()
        num_sources = random.randint(3, 4)
        selected_sources = random.sample(self.SOURCES, min(num_sources, len(self.SOURCES)))
        impact_map = {"CPI": 0.8, "RATE": 0.9, "GDP": 0.7, "NEWS": 0.4, "SENTIMENT": 0.5}
        impact_weight = impact_map.get(base_event_type, 0.5)
        for source in selected_sources:
            surprise_score = max(-1.0, min(1.0, random.gauss(0, 0.3)))
            if base_event_type in ("CPI", "RATE", "GDP"):
                direction_bias = self._clamp_bias(
                    random.choice([-1, 1]) * abs(random.gauss(0.4, 0.15))
                )
            else:
                direction_bias = self._clamp_bias(random.gauss(0, 0.4))
            raw_data = {
                "event_type": base_event_type,
                "source": source,
                "value": round(random.gauss(100, 10), 2),
                "previous": round(random.gauss(100, 10), 2),
                "forecast": round(random.gauss(100, 10), 2),
            }
            events.append(NormalizedEvent(
                symbol=symbol,
                event_type=base_event_type,
                surprise_score=surprise_score,
                direction_bias=direction_bias,
                impact_weight=impact_weight,
                timestamp=base_time + random.uniform(-1, 1),
                source=source,
                raw_data=raw_data,
            ))
        events.sort(key=lambda e: e.timestamp)
        return events
