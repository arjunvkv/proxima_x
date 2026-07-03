import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from research.persistence.persistence_utils import PersistenceDataLoader


class RQ9RegimeClassifier:
    """RQ9: Can persistence variables classify regimes better than existing detector?"""

    REGIME_LABELS = {"2020-2022": 0, "2022-2024": 1, "2024-2026": 2}
    REGIME_BOUNDARIES = {"2020-2022": (0, 520), "2022-2024": (520, 1040), "2024-2026": (1040, 1560)}

    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset

    def run(self) -> dict:
        loader = PersistenceDataLoader(self.asset)
        n = loader.n

        # Build persistence features at each bar
        features = []
        labels = []
        w = 252
        for rname, (r0, r1) in self.REGIME_BOUNDARIES.items():
            r0, r1 = max(w, r0), min(n, r1)
            if r1 <= r0:
                continue

            # Compute windowed persistence metrics
            for i in range(r0, r1):
                window_start = max(0, i - w)
                events_in_window = [e for e in loader.events
                                    if e["start_idx"] >= window_start and e["end_idx"] <= i]
                durations = [e["duration"] for e in events_in_window]
                if len(durations) < 2:
                    continue

                feat = [
                    float(np.mean(durations)),
                    float(np.std(durations)),
                    float(entropy(np.bincount(np.array(durations, dtype=int))) if len(durations) > 1 else 0.0),
                    float(np.median(durations)),
                    float(np.max(durations)),
                    float(np.min(durations)),
                ]
                features.append(feat)
                labels.append(self.REGIME_LABELS[rname])

        if len(features) < 50:
            return {"error": "Not enough samples", "n_samples": len(features)}

        X = np.array(features)
        y = np.array(labels)

        # Train classifier
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1_macro = float(f1_score(y_test, y_pred, average="macro"))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
        precision = float(precision_score(y_test, y_pred, average="macro"))
        recall = float(recall_score(y_test, y_pred, average="macro"))
        accuracy = float(accuracy_score(y_test, y_pred))

        # Feature importance
        importance = {
            "mean_duration": float(clf.feature_importances_[0]),
            "std_duration": float(clf.feature_importances_[1]),
            "duration_entropy": float(clf.feature_importances_[2]),
            "median_duration": float(clf.feature_importances_[3]),
            "max_duration": float(clf.feature_importances_[4]),
            "min_duration": float(clf.feature_importances_[5]),
        }

        # Compare with existing regime detector (benchmark)
        # Existing detector uses signal_frequency, signal_strength, residual_energy, etc.
        # We build a comparable benchmark using those features
        benchmark_f1 = self._build_benchmark_classifier(loader, n)

        return {
            "asset": self.asset,
            "n_samples": len(features),
            "persistence_classifier": {
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
            },
            "feature_importance": importance,
            "top_feature": max(importance, key=importance.get),
            "benchmark_f1": benchmark_f1,
            "persistence_beats_benchmark": f1_macro > benchmark_f1 if benchmark_f1 is not None else None,
            "interpretation": (
                "F1 > 0.7 means persistence alone classifies regimes well. "
                "If persistence F1 > benchmark F1, persistence is better regime classifier than existing detector."
            ),
        }

    def _build_benchmark_classifier(self, loader: PersistenceDataLoader, n: int) -> float | None:
        """Build classifier using existing detector features for comparison."""
        features = []
        labels = []
        w = 252
        for rname, (r0, r1) in self.REGIME_BOUNDARIES.items():
            r0, r1 = max(w, r0), min(n, r1)
            if r1 <= r0:
                continue
            for i in range(r0, r1):
                ws = max(0, i - w)
                feat = [
                    float(np.mean(loader.layers["residual_energy"][ws:i + 1])),
                    float(np.mean(loader.layers["energy_storage"][ws:i + 1])),
                    float(np.mean(loader.layers["adaptive_time"][ws:i + 1])),
                    float(np.mean(loader.composite[ws:i + 1])),
                    float(np.std(loader.layers["residual_energy"][ws:i + 1])),
                ]
                features.append(feat)
                labels.append(self.REGIME_LABELS[rname])

        if len(features) < 50:
            return None

        X = np.array(features)
        y = np.array(labels)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return float(f1_score(y_test, y_pred, average="macro"))
        except Exception:
            return None


def entropy(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))
