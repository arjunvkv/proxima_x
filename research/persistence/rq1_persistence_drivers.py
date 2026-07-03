import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from research.persistence.persistence_utils import PersistenceDataLoader, PersistenceMeasure, align_by_min
from proxima_v1.core.signal_engine import SignalEngine

class RQ1PersistenceDrivers:
    """RQ1: Which Proxima layer best predicts persistence duration?"""
    
    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        
    def run(self) -> dict:
        loader = PersistenceDataLoader(self.asset)
        df = loader.get_events_df()
        if len(df) < 5:
            return {"error": "Not enough events", "n_events": len(df)}
        
        durations = df["duration"].values
        
        # For each layer, compute correlation between layer entry value and duration
        entry_corrs = {}
        mean_corrs = {}
        delta_corrs = {}
        
        for lk in loader.LAYER_KEYS:
            entry_col = f"{lk}_entry"
            mean_col = f"{lk}_mean"
            delta_col = f"{lk}_delta"
            
            if entry_col not in df.columns:
                continue
            
            # Mask inf/nan
            valid = ~(np.isnan(df[entry_col].values) | np.isnan(durations))
            if valid.sum() < 5:
                continue
            
            pm = PersistenceMeasure()
            
            entry_corrs[lk] = pm.pearson_spearman_mi(
                df[entry_col].values[valid], durations[valid]
            )
            
            if mean_col in df.columns:
                mean_corrs[lk] = pm.pearson_spearman_mi(
                    df[mean_col].values[valid], durations[valid]
                )
            
            if delta_col in df.columns:
                delta_corrs[lk] = pm.pearson_spearman_mi(
                    df[delta_col].values[valid], durations[valid]
                )
        
        # Build a combined score: rank by avg(|pearson| + |spearman| + mi_norm) across entry/mean/delta
        def combined_score(corr_dict: dict) -> float:
            if not corr_dict:
                return 0.0
            scores = []
            for k, v in corr_dict.items():
                s = abs(v.get("pearson", 0)) + abs(v.get("spearman", 0)) + v.get("mutual_info", 0)
                scores.append(s)
            return float(np.mean(scores)) if scores else 0.0
        
        # Rank layers by entry correlation
        ranked_by_entry = sorted(entry_corrs.keys(), key=lambda k: combined_score({k: entry_corrs[k]}), reverse=True)
        
        # Full results
        full = {}
        for lk in loader.LAYER_KEYS:
            full[lk] = {
                "entry": entry_corrs.get(lk, {}),
                "mean": mean_corrs.get(lk, {}),
                "delta": delta_corrs.get(lk, {}),
            }
        
        # Cross-layer correlation at signal start (to check collinearity)
        entry_values = {}
        for lk in loader.LAYER_KEYS:
            col = f"{lk}_entry"
            if col in df.columns:
                entry_values[lk] = df[col].values
        cross_corr = {}
        lk_list = list(entry_values.keys())
        for i, l1 in enumerate(lk_list):
            for l2 in lk_list[i+1:]:
                v1, v2 = entry_values[l1], entry_values[l2]
                valid = ~(np.isnan(v1) | np.isnan(v2))
                if valid.sum() < 5:
                    continue
                p, _ = pearsonr(v1[valid], v2[valid])
                cross_corr[f"{l1}_vs_{l2}"] = float(p)
        
        return {
            "asset": self.asset,
            "n_events": len(df),
            "duration_stats": {
                "mean": float(np.mean(durations)),
                "std": float(np.std(durations)),
                "min": float(np.min(durations)),
                "max": float(np.max(durations)),
                "median": float(np.median(durations)),
            },
            "ranked_by_entry_correlation": ranked_by_entry[:10],
            "top_entry_layer": ranked_by_entry[0] if ranked_by_entry else None,
            "top_entry_score": combined_score({ranked_by_entry[0]: entry_corrs[ranked_by_entry[0]]}) if ranked_by_entry else 0.0,
            "layer_details": full,
            "cross_layer_correlation": cross_corr,
        }
