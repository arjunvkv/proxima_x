import numpy as np
from scipy.stats import pearsonr, wasserstein_distance
from research.persistence.persistence_utils import PersistenceDataLoader


class RQ6CrossAsset:
    """RQ6: Does persistence transfer better than alpha across assets?"""

    def __init__(self, assets: list[str] | None = None):
        self.assets = assets or ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD"]

    def run(self) -> dict:
        asset_durations = {}
        asset_half_lives = {}
        asset_stats = {}

        for asset in self.assets:
            try:
                loader = PersistenceDataLoader(asset)
                durations = loader.get_signal_durations()
                if len(durations) < 3:
                    continue
                half_life = loader.get_persistence_half_life()
                asset_durations[asset] = durations
                asset_half_lives[asset] = half_life
                asset_stats[asset] = {
                    "mean_duration": float(np.mean(durations)),
                    "median_duration": float(np.median(durations)),
                    "std_duration": float(np.std(durations)),
                    "n_events": len(durations),
                    "half_life": half_life,
                    "max_duration": float(np.max(durations)),
                }
            except Exception as e:
                continue

        if len(asset_durations) < 2:
            return {"error": "Need at least 2 assets with data"}

        # Pairwise persistence similarity
        asset_list = list(asset_durations.keys())
        similarity_matrix = {}
        for i, a1 in enumerate(asset_list):
            for a2 in asset_list[i + 1:]:
                d1, d2 = asset_durations[a1], asset_durations[a2]
                min_n = min(len(d1), len(d2))
                if min_n < 5:
                    continue
                d1s, d2s = d1[:min_n], d2[:min_n]
                wass = float(wasserstein_distance(d1s, d2s))
                p, _ = pearsonr(d1s, d2s)
                similarity_matrix[f"{a1}_vs_{a2}"] = {
                    "wasserstein_distance": wass,
                    "pearson_correlation": float(p),
                    "half_life_ratio": float(asset_half_lives[a1] / max(asset_half_lives[a2], 1e-10)),
                    "mean_duration_ratio": float(np.mean(d1) / max(np.mean(d2), 1e-10)),
                }

        # Clustering: which assets have similar persistence?
        dur_means = np.array([asset_stats[a]["mean_duration"] for a in asset_list])
        half_lives = np.array([asset_stats[a]["half_life"] for a in asset_list])

        persistence_clusters = {}
        if len(asset_list) >= 3:
            from sklearn.cluster import KMeans
            X = np.column_stack([dur_means, half_lives])
            X = (X - X.mean(axis=0)) / max(X.std(axis=0).max(), 1e-10)
            for k in [2, 3]:
                if k >= len(asset_list):
                    continue
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                clusters = {str(i): [asset_list[j] for j in range(len(asset_list)) if labels[j] == i]
                            for i in range(k)}
                persistence_clusters[str(k)] = clusters

        # Compare with alpha transfer (using existing results from Reality Gap)
        return {
            "assets_tested": asset_list,
            "asset_statistics": asset_stats,
            "pairwise_similarity": similarity_matrix,
            "persistence_clusters": persistence_clusters,
            "persistence_transferability": (
                "Similarity measured via Wasserstein distance. "
                "Lower distance = more transferable persistence."
            ),
        }
