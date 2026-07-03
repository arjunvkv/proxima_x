"""Partition replay ticks by market session."""
SESSION_RULES = {
    "ASIA": (0, 7),
    "LONDON": (7, 12),
    "OVERLAP": (12, 16),
    "NY": (16, 21),
    "DEAD": (21, 24),
}

SESSION_ORDER = ["ASIA", "LONDON", "OVERLAP", "NY", "DEAD"]


class SessionPartitioner:
    def classify(self, ts: int) -> str:
        hour = (ts % 86400) // 3600
        for session, (start, end) in SESSION_RULES.items():
            if start <= hour < end:
                return session
        return "DEAD"

    def session_of(self, ts: int) -> str:
        return self.classify(ts)
