import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from research.persistence.persistence_utils import PersistenceDataLoader


class RQ7WalkForward:
    """RQ7: Can persistence be forecasted via walk-forward?"""

    REGIMES = {"2020-2022": (0, 520), "2022-2024": (520, 1040),
               "2024-2026": (1040, 1560)}

    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset

    def run(self) -> dict:
        loader = PersistenceDataLoader(self.asset)
        n = loader.n
        rolling_dur = loader.get_rolling_duration(window=252)

        window_results = []
        regime_names = list(self.REGIMES.keys())

        for i in range(len(regime_names) - 1):
            train_name = regime_names[i]
            test_name = regime_names[i + 1]
            t0, t1 = self.REGIMES[train_name]
            v0, v1 = self.REGIMES[test_name]

            t0, t1 = max(0, t0), min(n, t1)
            v0, v1 = max(0, v0), min(n, v1)

            # Build features: rolling duration + layer lags
            train_idx = np.arange(t0, t1)
            test_idx = np.arange(v0, v1)

            if len(train_idx) < 100 or len(test_idx) < 50:
                continue

            X_train, y_train = self._build_features(loader, train_idx, rolling_dur)
            X_test, y_test = self._build_features(loader, test_idx, rolling_dur)

            if X_train.shape[0] < 20 or X_test.shape[0] < 10:
                continue

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = float(r2_score(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))

            # Directional accuracy: sign of diff
            y_test_diff = np.diff(y_test)
            y_pred_diff = np.diff(y_pred)
            if len(y_test_diff) > 0 and len(y_pred_diff) > 0:
                dir_acc = float(np.mean((y_test_diff > 0) == (y_pred_diff > 0)))
            else:
                dir_acc = 0.5

            window_results.append({
                "train": train_name,
                "test": test_name,
                "n_train": X_train.shape[0],
                "n_test": X_test.shape[0],
                "r2": r2,
                "mae": mae,
                "directional_accuracy": dir_acc,
                "coef": model.coef_.tolist(),
                "intercept": float(model.intercept_),
            })

        # Overall metrics
        r2s = [w["r2"] for w in window_results]
        das = [w["directional_accuracy"] for w in window_results]

        return {
            "asset": self.asset,
            "window_results": window_results,
            "mean_r2": float(np.mean(r2s)) if r2s else 0.0,
            "mean_directional_accuracy": float(np.mean(das)) if das else 0.0,
            "forecastable": float(np.mean(das)) > 0.55 if das else False,
            "interpretation": (
                "R2 > 0 means persistence has predictable structure. "
                "Directional accuracy > 0.55 means trend forecastable."
            ),
        }

    def _build_features(self, loader: PersistenceDataLoader, idx: np.ndarray,
                        rolling_dur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lag = 20
        valid = idx[idx > lag]
        if len(valid) < 10:
            return np.zeros((0, 6)), np.zeros(0)

        X = []
        y = []
        for i in valid:
            feat = [
                rolling_dur[i - 1],
                float(rolling_dur[i - 1] - rolling_dur[max(0, i - lag)]),
                float(np.mean(rolling_dur[max(0, i - lag):i])),
                float(np.std(rolling_dur[max(0, i - lag):i])),
                loader.layers["residual_energy"][i] if i < len(loader.layers["residual_energy"]) else 0.0,
                loader.layers["energy_storage"][i] if i < len(loader.layers["energy_storage"]) else 0.0,
            ]
            X.append(feat)
            y.append(rolling_dur[i])

        return np.array(X), np.array(y)
