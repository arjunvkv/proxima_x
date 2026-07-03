import numpy as np
from research.persistence.persistence_utils import PersistenceDataLoader


class RQ8ThresholdDriftOrder:
    """RQ8: Does persistence collapse cause threshold drift, or vice versa?"""

    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset

    def run(self) -> dict:
        loader = PersistenceDataLoader(self.asset)
        n = loader.n
        composite = loader.composite

        # Compute rolling threshold (90th percentile of composite over 504 window)
        rolling_threshold = np.full(n, 0.5)
        w = 504
        for i in range(w, n):
            rolling_threshold[i] = float(np.percentile(composite[max(0, i - w):i + 1], 90))

        # Compute rolling persistence duration (average duration over 252 window)
        rolling_duration = np.full(n, 0.0)
        if loader.events:
            dur_series = np.zeros(n)
            for ev in loader.events:
                dur_series[ev["start_idx"]:ev["end_idx"] + 1] = ev["duration"]
            for i in range(w, n):
                rolling_duration[i] = float(np.mean(dur_series[max(0, i - 252):i + 1]))

        # Cross-correlation analysis
        valid = np.arange(w, n)
        dur = rolling_duration[valid]
        th = rolling_threshold[valid]

        dur_norm = (dur - np.mean(dur)) / max(np.std(dur), 1e-10)
        th_norm = (th - np.mean(th)) / max(np.std(th), 1e-10)

        max_lag = 100
        cross_corr = np.correlate(dur_norm, th_norm, mode="same")
        lags = np.arange(-max_lag, max_lag + 1)
        center = len(cross_corr) // 2
        lag_corrs = {}
        for lag in lags:
            idx = center + lag
            if 0 <= idx < len(cross_corr):
                lag_corrs[int(lag)] = float(cross_corr[idx] / len(dur_norm))

        # Find peak correlation lag
        peak_lag = max(lag_corrs, key=lambda k: abs(lag_corrs[k]))
        peak_corr = lag_corrs[peak_lag]

        # Lag analysis: if persistence -> threshold (positive lag = duration leads)
        # if threshold -> persistence (negative lag = threshold leads)
        pos_lags = {k: v for k, v in lag_corrs.items() if k > 0}
        neg_lags = {k: v for k, v in lag_corrs.items() if k < 0}
        mean_pos = float(np.mean(list(pos_lags.values()))) if pos_lags else 0.0
        mean_neg = float(np.mean(list(neg_lags.values()))) if neg_lags else 0.0

        if abs(mean_pos) > abs(mean_neg) * 1.2:
            causal_direction = "persistence_leads_threshold"
        elif abs(mean_neg) > abs(mean_pos) * 1.2:
            causal_direction = "threshold_leads_persistence"
        else:
            causal_direction = "mutual_or_unknown"

        # Change-point analysis: which changes first?
        # Detect major change points in both series
        dur_change = np.diff(rolling_duration[w:])
        th_change = np.diff(rolling_threshold[w:])

        # Find first large change (exceeding 2 std)
        dur_std = np.std(dur_change) if len(dur_change) > 0 else 1.0
        th_std = np.std(th_change) if len(th_change) > 0 else 1.0

        dur_break_idx = np.argmax(np.abs(dur_change) > 2 * dur_std) if np.any(np.abs(dur_change) > 2 * dur_std) else -1
        th_break_idx = np.argmax(np.abs(th_change) > 2 * th_std) if np.any(np.abs(th_change) > 2 * th_std) else -1

        if dur_break_idx >= 0 and th_break_idx >= 0:
            if dur_break_idx < th_break_idx:
                break_ordering = "persistence_breaks_first"
            elif th_break_idx < dur_break_idx:
                break_ordering = "threshold_breaks_first"
            else:
                break_ordering = "simultaneous_break"
        elif dur_break_idx >= 0:
            break_ordering = "only_persistence_breaks"
        elif th_break_idx >= 0:
            break_ordering = "only_threshold_breaks"
        else:
            break_ordering = "no_major_break_detected"

        return {
            "asset": self.asset,
            "peak_lag": peak_lag,
            "peak_correlation": peak_corr,
            "mean_correlation_positive_lags": mean_pos,
            "mean_correlation_negative_lags": mean_neg,
            "causal_direction": causal_direction,
            "break_ordering": break_ordering,
            "lag_correlations": lag_corrs,
            "interpretation": {
                "persistence_leads_threshold": "Persistence collapse causes threshold drift",
                "threshold_leads_persistence": "Threshold drift causes persistence collapse",
                "mutual_or_unknown": "No clear causal direction",
            },
        }
