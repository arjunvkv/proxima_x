from datetime import datetime
from ..core.fsv_schema import NormalizedEvent, validate_event


class MacroEventNormalizer:
    EVENT_TYPE_MAP = {
        "cpi": "CPI", "inflation": "CPI", "consumer price": "CPI",
        "gdp": "GDP", "gross domestic": "GDP",
        "rate decision": "RATE", "interest rate": "RATE", "central bank": "RATE",
        "sentiment": "SENTIMENT", "confidence": "SENTIMENT", "zew": "SENTIMENT",
        "news": "NEWS", "headline": "NEWS", "report": "NEWS",
    }

    def _infer_event_type(self, text: str) -> str:
        text_lower = text.lower()
        for pattern, event_type in self.EVENT_TYPE_MAP.items():
            if pattern in text_lower:
                return event_type
        return "UNKNOWN"

    @staticmethod
    def _compute_surprise(actual, forecast) -> float:
        if actual is None or forecast is None:
            return 0.0
        try:
            actual_f = float(actual)
            forecast_f = float(forecast)
            if forecast_f == 0.0:
                surprise = actual_f
            else:
                surprise = (actual_f - forecast_f) / abs(forecast_f)
            return max(-1.0, min(1.0, surprise))
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _parse_timestamp(date_val) -> float:
        if date_val is None:
            return datetime.now().timestamp()
        if isinstance(date_val, (int, float)):
            return float(date_val)
        if isinstance(date_val, str):
            formats = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%d/%m/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M:%S",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(date_val, fmt).timestamp()
                except ValueError:
                    continue
            try:
                return datetime.fromisoformat(date_val).timestamp()
            except (ValueError, AttributeError):
                pass
        return datetime.now().timestamp()

    @staticmethod
    def _infer_direction_bias(text: str) -> float:
        text_lower = text.lower()
        positive_words = [
            "bullish", "up", "gains", "positive", "rise", "rising",
            "higher", "strong", "growth", "beat", "exceeds",
        ]
        negative_words = [
            "bearish", "down", "losses", "negative", "fall", "falling",
            "lower", "weak", "decline", "miss", "misses",
        ]
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        if pos_count > neg_count:
            return min(1.0, pos_count * 0.3)
        if neg_count > pos_count:
            return max(-1.0, -neg_count * 0.3)
        return 0.0

    @staticmethod
    def infer_symbol_from_currency(currency: str) -> str:
        mapping = {
            "USD": "USD",
            "EUR": "EURUSD",
            "GBP": "GBPUSD",
            "AUD": "AUDUSD",
            "NZD": "NZDUSD",
            "JPY": "USDJPY",
            "CHF": "USDCHF",
            "CAD": "USDCAD",
        }
        return mapping.get(currency.upper(), currency.upper())

    def normalize_tradingeconomics(self, raw: dict) -> NormalizedEvent:
        symbol = raw.get("symbol", "")
        if not symbol:
            country = raw.get("country", "US")
            symbol = self.infer_symbol_from_currency(country)
        event = raw.get("event", "")
        event_type = self._infer_event_type(event)
        actual = raw.get("actual")
        forecast = raw.get("forecast")
        surprise = self._compute_surprise(actual, forecast)
        timestamp = self._parse_timestamp(raw.get("date"))
        direction_bias = 1.0 if surprise > 0 else (-1.0 if surprise < 0 else 0.0)
        return NormalizedEvent(
            symbol=symbol,
            event_type=event_type,
            surprise_score=surprise,
            direction_bias=direction_bias,
            impact_weight=0.5,
            timestamp=timestamp,
            source="tradingeconomics",
            raw_data=raw,
        )

    def normalize_fxstreet(self, raw: dict) -> NormalizedEvent:
        title = raw.get("title", "")
        content = raw.get("content", "")
        impact_str = raw.get("impact", "low")
        currency = raw.get("currency", "USD")
        timestamp = self._parse_timestamp(raw.get("timestamp"))
        impact_map = {"high": 0.9, "medium": 0.5, "low": 0.2}
        impact_weight = impact_map.get(impact_str.lower(), 0.2)
        event_type = self._infer_event_type(title)
        direction_bias = self._infer_direction_bias(title)
        symbol = self.infer_symbol_from_currency(currency)
        return NormalizedEvent(
            symbol=symbol,
            event_type=event_type,
            surprise_score=0.0,
            direction_bias=direction_bias,
            impact_weight=impact_weight,
            timestamp=timestamp,
            source="fxstreet",
            raw_data=raw,
        )

    def normalize_investing(self, raw: dict) -> NormalizedEvent:
        currency = raw.get("currency", "USD")
        event_name = raw.get("event_name", "")
        importance = raw.get("importance", 1)
        actual = raw.get("actual")
        forecast = raw.get("forecast")
        timestamp = self._parse_timestamp(raw.get("timestamp"))
        importance_map = {3: 0.9, 2: 0.5, 1: 0.2}
        impact_weight = importance_map.get(int(importance), 0.2)
        event_type = self._infer_event_type(event_name)
        surprise = self._compute_surprise(actual, forecast)
        symbol = self.infer_symbol_from_currency(currency)
        direction_bias = 1.0 if surprise > 0 else (-1.0 if surprise < 0 else 0.0)
        return NormalizedEvent(
            symbol=symbol,
            event_type=event_type,
            surprise_score=surprise,
            direction_bias=direction_bias,
            impact_weight=impact_weight,
            timestamp=timestamp,
            source="investing",
            raw_data=raw,
        )

    def normalize_fred(self, raw: dict) -> NormalizedEvent:
        series_id = raw.get("series_id", "")
        value = raw.get("value")
        name = raw.get("name", "")
        date_val = raw.get("date")
        timestamp = self._parse_timestamp(date_val)
        fred_event_map = {
            "GDP": "GDP", "GDPC1": "GDP",
            "CPIAUCSL": "CPI", "CPILFESL": "CPI",
            "FEDFUNDS": "RATE", "DFF": "RATE",
            "UNRATE": "NEWS", "PAYEMS": "NEWS",
            "UMCSENT": "SENTIMENT",
        }
        event_type = fred_event_map.get(series_id, self._infer_event_type(name))
        surprise = 0.0
        direction_bias = 0.0
        return NormalizedEvent(
            symbol="USD",
            event_type=event_type,
            surprise_score=surprise,
            direction_bias=direction_bias,
            impact_weight=0.4,
            timestamp=timestamp,
            source="fred",
            raw_data=raw,
        )

    def normalize(self, raw_event: dict, source: str) -> NormalizedEvent:
        normalizers = {
            "tradingeconomics": self.normalize_tradingeconomics,
            "fxstreet": self.normalize_fxstreet,
            "investing": self.normalize_investing,
            "fred": self.normalize_fred,
        }
        normalizer = normalizers.get(source)
        if normalizer is None:
            raise ValueError(f"Unknown source: {source}")
        return normalizer(raw_event)

    def batch_normalize(self, events: list[dict], source: str) -> list[NormalizedEvent]:
        return [self.normalize(event, source) for event in events]
