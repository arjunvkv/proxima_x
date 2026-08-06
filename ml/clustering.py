"""StateClusterer — HDBSCAN wrapper for research state discovery.

Consumer surface (restored contract, verified against both call sites):

    clusterer = StateClusterer(method="hdbscan",
                               params={"min_cluster_size": mcs})
    labels = clusterer.fit_predict(compressed)   # -1 == noise

Delegates to sklearn's bundled HDBSCAN; ``min_cluster_size`` is the only
param honored. Backward compatible with the original never-committed module:
``fit_predict`` returns an int array with -1 for noise and dense non-negative
cluster ids, matching downstream remapping in research/pipeline.py.
"""
from typing import Optional

import numpy as np

try:
    from sklearn.cluster import HDBSCAN
    _HDBSCAN_OK = True
except ImportError:  # pragma: no cover — sklearn is a core project dep
    HDBSCAN = None
    _HDBSCAN_OK = False


class StateClusterer:
    def __init__(self, method: str = "hdbscan", params: Optional[dict] = None):
        self.method = method
        self.params = params or {}
        self._model = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        if not _HDBSCAN_OK:
            raise RuntimeError("sklearn.cluster.HDBSCAN unavailable")
        mcs = int(self.params.get("min_cluster_size", 5))
        # HDBSCAN.fit_predict returns -1 for noise — exactly what the
        # consumers' remapping expects.
        self._model = HDBSCAN(min_cluster_size=mcs, metric="euclidean")
        labels = self._model.fit_predict(np.asarray(X, dtype=np.float64))
        return np.asarray(labels, dtype=np.int64)
