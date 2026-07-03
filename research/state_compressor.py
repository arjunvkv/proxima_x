from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numba
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness


class StateCompressor:
    def __init__(self, method: str = "umap", n_components: int = 20, random_state: int = 42):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model: Any = None
        self._fitted = False

    def _build_model(self) -> Any:
        if self.method == "umap":
            import umap
            return umap.UMAP(n_components=self.n_components, random_state=self.random_state)
        elif self.method == "pca":
            return PCA(n_components=self.n_components, random_state=self.random_state)
        elif self.method == "spectral":
            return SpectralEmbedding(n_components=self.n_components, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown compression method: {self.method}")

    def fit_transform(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        scaled = self._scaler.fit_transform(data).astype(np.float32)
        self._model = self._build_model()
        compressed = self._model.fit_transform(scaled).astype(np.float32)
        self._fitted = True
        return compressed

    def fit(self, data: NDArray[np.float32]) -> None:
        scaled = self._scaler.fit_transform(data).astype(np.float32)
        self._model = self._build_model()
        self._model.fit(scaled)
        self._fitted = True

    def transform(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self._fitted or self._model is None:
            raise RuntimeError("Compressor not fitted")
        scaled = self._scaler.transform(data).astype(np.float32)
        return self._model.transform(scaled).astype(np.float32)

    def compression_quality(self, original: NDArray[np.float32], compressed: NDArray[np.float32]) -> dict:
        result: dict[str, Any] = {
            "method": self.method,
            "n_components": self.n_components,
            "explained_variance": "NA",
            "reconstruction_error": "NA",
            "trustworthiness": "NA",
        }
        if self.method == "pca" and hasattr(self._model, "explained_variance_ratio_"):
            evr = self._model.explained_variance_ratio_
            result["explained_variance"] = float(np.sum(evr))
            if hasattr(self._model, "components_"):
                reconstructed = (compressed @ self._model.components_)
                reconstructed = self._scaler.inverse_transform(reconstructed).astype(np.float32)
                result["reconstruction_error"] = float(np.sqrt(np.mean((original - reconstructed) ** 2)))
        n_samples = min(len(original), 5000)
        idx = np.random.RandomState(self.random_state).choice(len(original), n_samples, replace=False)
        trust = float(trustworthiness(original[idx], compressed[idx], n_neighbors=min(20, n_samples // 2)))
        result["trustworthiness"] = round(trust, 6)
        return result

    def find_optimal_dimensions(self, data: NDArray[np.float32], dim_range: list[int] | None = None) -> dict:
        if dim_range is None:
            dim_range = [5, 10, 20, 30, 50, 100]
        results: dict[int, dict] = {}
        best_dim = dim_range[0]
        best_trust = -1.0
        for dim in dim_range:
            if dim >= data.shape[1]:
                continue
            saved = self.n_components
            self.n_components = dim
            compressed = self.fit_transform(data)
            qual = self.compression_quality(data, compressed)
            results[dim] = qual
            trust_val = qual.get("trustworthiness", -1)
            if isinstance(trust_val, (int, float)) and trust_val > best_trust:
                best_trust = trust_val
                best_dim = dim
            self.n_components = saved
        return {
            "best_dim": best_dim,
            "best_trustworthiness": best_trust,
            "results": results,
        }
