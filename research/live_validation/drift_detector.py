import numpy as np
from scipy.stats import ks_2samp

class DriftDetector:
    def __init__(self, window_30: int = 30, window_90: int = 90):
        self._window_30 = window_30
        self._window_90 = window_90
        self._feature_history: dict[str, list[float]] = {}

    def record(self, feature_name: str, value: float):
        if feature_name not in self._feature_history:
            self._feature_history[feature_name] = []
        self._feature_history[feature_name].append(value)

    def record_batch(self, features: dict[str, float]):
        for k, v in features.items():
            self.record(k, v)

    def _get_windows(self, feature: str):
        hist = self._feature_history.get(feature, [])
        if len(hist) < self._window_30 + 1:
            return None, None, None
        today = [hist[-1]]
        window_30 = hist[-self._window_30:]
        window_90 = hist[-self._window_90:] if len(hist) >= self._window_90 else hist
        return today, window_30, window_90

    def check_drift(self, feature: str) -> dict:
        today, w30, w90 = self._get_windows(feature)
        if today is None:
            return {"feature": feature, "classification": "INSUFFICIENT_DATA"}

        ks_vs_30 = ks_2samp(today, w30) if len(today) > 0 and len(w30) > 0 else (1.0, 1.0)
        ks_vs_90 = ks_2samp(today, w90) if len(today) > 0 and len(w90) > 0 else (1.0, 1.0)

        mean_today = float(np.mean(today))
        mean_30 = float(np.mean(w30))
        mean_90 = float(np.mean(w90))

        p30_change = abs(mean_today - mean_30) / max(abs(mean_30), 1e-10) if mean_30 != 0 else 0.0
        p90_change = abs(mean_today - mean_90) / max(abs(mean_90), 1e-10) if mean_90 != 0 else 0.0

        ks_dist = max(ks_vs_30[0], ks_vs_90[0])
        psi = abs(p30_change + p90_change) / 2
        overlap = 1.0 - min(ks_dist, 1.0)

        if ks_dist < 0.15 and psi < 0.10:
            classification = "STABLE"
        elif ks_dist < 0.40 and psi < 0.30:
            classification = "DRIFTING"
        else:
            classification = "BROKEN"

        return {
            "feature": feature,
            "classification": classification,
            "ks_distance": float(ks_dist),
            "psi": float(psi),
            "overlap": float(overlap),
            "mean_today": mean_today,
            "mean_30": mean_30,
            "mean_90": mean_90}

    def check_all(self) -> dict[str, dict]:
        return {f: self.check_drift(f) for f in self._feature_history}

    def summary(self) -> dict:
        results = self.check_all()
        stable = sum(1 for v in results.values() if v["classification"] == "STABLE")
        drifting = sum(1 for v in results.values() if v["classification"] == "DRIFTING")
        broken = sum(1 for v in results.values() if v["classification"] == "BROKEN")
        return {
            "features_tracked": len(results),
            "stable": stable,
            "drifting": drifting,
            "broken": broken,
            "details": results}
