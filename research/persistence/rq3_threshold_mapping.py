import numpy as np
from proxima_v1.core.signal_engine import SignalEngine
from research.persistence.persistence_utils import extract_persistence_events, PersistenceDataLoader


class RQ3ThresholdMapping:
    """RQ3: Map persistence duration vs percentile threshold."""

    THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]

    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset

    def run(self) -> dict:
        loader = PersistenceDataLoader(self.asset)
        composite = loader.composite

        # Compute rolling forward PP at H20 for each threshold
        n = len(composite)
        future_returns = np.diff(np.log(loader.engine._data["price"][:n] + 1e-10))
        future_returns = np.pad(future_returns, (0, 1), constant_values=0.0)
        h20_returns = np.array([np.sum(future_returns[i:i + 20]) if i + 20 < n else 0.0 for i in range(n)])

        results = {}
        for thresh in self.THRESHOLDS:
            events = extract_persistence_events(composite, threshold=thresh)
            durations = np.array([e["duration"] for e in events], dtype=float)
            frequency = len(events) / (n / 504) if n > 0 else 0  # approx annual freq

            # PP of H20 forward returns at signal start
            signal_starts = [e["start_idx"] for e in events if e["start_idx"] < len(h20_returns)]
            if signal_starts:
                start_returns = h20_returns[signal_starts]
                all_returns = h20_returns[504:]
                pp_signal = float(np.mean(start_returns > 0))
                pp_all = float(np.mean(all_returns > 0)) if len(all_returns) > 0 else 0.5
            else:
                pp_signal = 0.5
                pp_all = 0.5

            # Sharpe estimate
            if signal_starts and len(start_returns) > 2:
                sharpe = float(np.mean(start_returns) / np.std(start_returns)) * np.sqrt(252) if np.std(start_returns) > 0 else 0.0
            else:
                sharpe = 0.0

            results[str(thresh)] = {
                "n_events": len(events),
                "mean_duration": float(np.mean(durations)) if len(durations) > 0 else 0.0,
                "median_duration": float(np.median(durations)) if len(durations) > 0 else 0.0,
                "max_duration": float(np.max(durations)) if len(durations) > 0 else 0.0,
                "frequency_annual": float(frequency),
                "pp_h20": float(pp_signal),
                "pp_baseline": float(pp_all),
                "sharpe_h20": float(sharpe),
            }

        # Elasticity: % change in duration per % change in threshold
        thresholds_arr = np.array(self.THRESHOLDS)
        durations_arr = np.array([results.get(str(t), {}).get("mean_duration", 0) for t in self.THRESHOLDS])
        elasticities = []
        for i in range(len(thresholds_arr) - 1):
            if durations_arr[i] > 0 and thresholds_arr[i] > 0:
                pct_dur = (durations_arr[i+1] - durations_arr[i]) / durations_arr[i]
                pct_th = (thresholds_arr[i+1] - thresholds_arr[i]) / thresholds_arr[i]
                if pct_th != 0:
                    elasticities.append(float(pct_dur / pct_th))
                else:
                    elasticities.append(0.0)
            else:
                elasticities.append(0.0)

        return {
            "asset": self.asset,
            "thresholds": self.THRESHOLDS,
            "results": results,
            "elasticity_by_threshold_pair": elasticities,
            "mean_elasticity": float(np.mean(elasticities)) if elasticities else 0.0,
            "interpretation": (
                "Elasticity > 1 means duration is highly sensitive to threshold changes. "
                "Elasticity < 1 means duration is inelastic to threshold."
            ),
        }
