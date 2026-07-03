class ExpectationEngine:
    def __init__(self):
        self._expectations = {
            "expected_pp": 0.59,
            "expected_sharpe": 1.38,
            "expected_frequency": 30,
            "expected_mean_return": 0.0032,
            "expected_win_rate": 0.59,
            "expected_es_rank_mean": 0.75,
            "expected_at_rank_mean": 0.65,
            "expected_persistence_da": 0.83,
            "expected_regime_accuracy": 0.72}

    def get(self, key: str, default=None):
        return self._expectations.get(key, default)

    def all(self) -> dict:
        return dict(self._expectations)
