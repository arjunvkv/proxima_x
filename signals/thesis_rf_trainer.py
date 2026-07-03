import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional
from signals.thesis_buffer import ThesisRecord

logger = logging.getLogger("proxima_demo")

class ThesisRfTrainer:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=42):
        self._model = None
        self._trained = False
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._random_state = random_state
        self._last_train_count = 0

    def retrain(self, resolved: List[ThesisRecord]):
        if len(resolved) < 10:
            return
        X = np.array([r.features() for r in resolved], dtype=np.float64)
        y = np.array([r.label for r in resolved], dtype=np.int32)
        if len(np.unique(y)) < 2:
            return
        model = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            random_state=self._random_state,
            class_weight="balanced",
            n_jobs=1,
        )
        model.fit(X, y)
        self._model = model
        self._trained = True
        self._last_train_count = len(resolved)
        prob_pos = float(np.mean(y))
        logger.info(f"[THESIS_RF] retrained on {len(resolved)} samples "
                    f"pos_rate={prob_pos:.3f} n_features={X.shape[1]}")

    def predict(self, features: list) -> float:
        if not self._trained or self._model is None:
            return 0.5
        expected = self._model.n_features_in_
        current = len(features)
        if current != expected:
            self._model = None
            self._trained = False
            logger.warning(f"[THESIS_RF_RESET] feature_dim_mismatch "
                           f"expected={expected} current={current}")
            return 0.5
        try:
            arr = np.array(features, dtype=np.float64).reshape(1, -1)
            return float(self._model.predict_proba(arr)[0, 1])
        except Exception as e:
            logger.warning(f"[THESIS_RF] predict failed: {e}")
            return 0.5

    def ready(self) -> bool:
        return self._trained

    def feature_importance(self) -> Optional[dict]:
        if not self._trained or self._model is None:
            return None
        nf = self._model.n_features_in_
        names = (["oss_sig", "shadow_sig", "exhaustion", "ecdf", "entropy",
                  "p_cont", "drift", "rf_prob", "confidence", "horizon"]
                 + (["micro_label", "meso_label", "macro_label", "fracture"] if nf == 14 else
                    [f"feat_{i}" for i in range(nf - 10)]))
        return dict(zip(names, self._model.feature_importances_.tolist()))
