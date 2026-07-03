import numpy as np
from research.persistence.persistence_utils import PersistenceDataLoader


class RQ2SurvivalCurves:
    """RQ2: Persistence survival curves via Kaplan-Meier by regime."""

    REGIMES = {"2018-2020": (0, 520), "2020-2022": (520, 1040),
               "2022-2024": (1040, 1560), "2024-2026": (1560, 2080)}

    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset

    def run(self) -> dict:
        loader = PersistenceDataLoader(self.asset)
        n = loader.n

        regime_results = {}
        for rname, (r0, r1) in self.REGIMES.items():
            r0 = max(0, r0)
            r1 = min(n, r1)
            events_in_regime = [e for e in loader.events
                                if e["start_idx"] >= r0 and e["end_idx"] <= r1]
            durations = np.array([e["duration"] for e in events_in_regime], dtype=float)
            if len(durations) < 3:
                regime_results[rname] = {
                    "n_events": len(events_in_regime),
                    "error": "insufficient events",
                    "mean_duration": 0.0,
                    "median_duration": 0.0,
                    "half_life": 0.0,
                    "kaplan_meier_bins": [],
                    "hazard_bins": [],
                }
                continue

            max_d = int(np.max(durations)) + 1
            time_points = np.arange(1, max_d + 1)

            # Kaplan-Meier: fraction of signals that survive to each time point
            km = np.array([float(np.sum(durations >= t)) / len(durations) for t in time_points])

            # Hazard: fraction of signals-at-risk that fail at each time point
            hazard = np.zeros(len(time_points))
            for i, t in enumerate(time_points):
                at_risk = int(np.sum(durations >= t))
                failed_at_t = int(np.sum(durations == t))
                hazard[i] = failed_at_t / at_risk if at_risk > 0 else 0.0

            # Half-life: point where KM survival <= 0.5
            half_life = float(time_points[np.argmax(km <= 0.5)] if np.any(km <= 0.5) else max_d)

            # Bin for reporting
            n_bins = min(20, max_d)
            bin_edges = np.linspace(1, max_d, n_bins + 1).astype(int)
            km_binned = [float(np.mean(km[(time_points >= bin_edges[i]) & (time_points < bin_edges[i+1])]))
                         if np.any((time_points >= bin_edges[i]) & (time_points < bin_edges[i+1])) else 0.0
                         for i in range(n_bins)]
            hazard_binned = [float(np.mean(hazard[(time_points >= bin_edges[i]) & (time_points < bin_edges[i+1])]))
                             if np.any((time_points >= bin_edges[i]) & (time_points < bin_edges[i+1])) else 0.0
                             for i in range(n_bins)]

            regime_results[rname] = {
                "n_events": len(events_in_regime),
                "mean_duration": float(np.mean(durations)),
                "median_duration": float(np.median(durations)),
                "std_duration": float(np.std(durations)),
                "min_duration": float(np.min(durations)),
                "max_duration": float(np.max(durations)),
                "half_life": half_life,
                "kaplan_meier_binned": km_binned,
                "hazard_binned": hazard_binned,
            }

        # Structural break detection: compare adjacent regime half-lives
        rnames = list(self.REGIMES.keys())
        half_lives = [regime_results.get(r, {}).get("half_life", 0) for r in rnames]
        breakpoints = []
        for i in range(len(half_lives) - 1):
            if half_lives[i] > 0 and half_lives[i+1] > 0:
                ratio = half_lives[i+1] / half_lives[i] if half_lives[i] > 0 else 0
                if ratio < 0.5 or ratio > 2.0:
                    breakpoints.append({
                        "from_regime": rnames[i],
                        "to_regime": rnames[i+1],
                        "half_life_before": half_lives[i],
                        "half_life_after": half_lives[i+1],
                        "ratio": float(ratio),
                    })

        return {
            "asset": self.asset,
            "regime_results": regime_results,
            "half_life_trajectory": {r: half_lives[i] for i, r in enumerate(rnames)},
            "structural_breakpoints": breakpoints,
            "persistence_decay_type": "structural_break" if breakpoints else "smooth_decay",
        }
