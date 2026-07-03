import numpy as np
from research.persistence.persistence_utils import PersistenceDataLoader


class RQ5DelayedAlpha:
    """RQ5: Does persistence length determine optimal signal-entry delay?"""

    DELAYS = [0, 1, 2, 5, 10, 15, 20]

    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset

    def run(self) -> dict:
        loader = PersistenceDataLoader(self.asset)
        n = loader.n
        price = loader.engine._data["price"][:n]
        future_returns = np.diff(np.log(price + 1e-10))
        future_returns = np.pad(future_returns, (0, 1), constant_values=0.0)

        events = loader.events
        durations = np.array([e["duration"] for e in events], dtype=float)
        if len(events) < 5:
            return {"error": "Not enough events"}

        # Stratify events by duration quartile
        if len(durations) >= 4:
            qs = [np.percentile(durations, p) for p in [25, 50, 75]]
        else:
            qs = [np.median(durations)]

        duration_groups = {
            "short": durations[durations <= qs[0]] if len(qs) >= 1 else durations,
            "medium": durations[(durations > qs[0]) & (durations <= qs[1])] if len(qs) >= 2 else np.array([]),
            "long": durations[durations > qs[1]] if len(qs) >= 2 else durations,
        }

        # For each delay, compute PP and Sharpe for each duration group
        delay_results = {}
        for delay in self.DELAYS:
            delay_key = str(delay)
            delay_results[delay_key] = {}
            for gname, gdurs in duration_groups.items():
                if len(gdurs) < 3:
                    delay_results[delay_key][gname] = {"pp": 0.5, "sharpe": 0.0, "mean_ret": 0.0, "n": 0}
                    continue
                # Find events with this duration range
                gmin, gmax = float(gdurs.min()), float(gdurs.max())
                matching = [e for e in events if gmin <= e["duration"] <= gmax]
                starts = [e["start_idx"] for e in matching]
                # Get forward returns at (start + delay) horizon H=20
                h = 20
                rets = []
                for s in starts:
                    idx = min(s + delay, n - 1)
                    if idx + h < n:
                        r = np.sum(future_returns[idx:idx + h])
                        rets.append(r)
                rets = np.array(rets)
                if len(rets) < 3:
                    delay_results[delay_key][gname] = {"pp": 0.5, "sharpe": 0.0, "mean_ret": 0.0, "n": len(rets)}
                    continue
                pp = float(np.mean(rets > 0))
                sharpe = float(np.mean(rets) / np.std(rets)) * np.sqrt(252 / h) if np.std(rets) > 0 else 0.0
                delay_results[delay_key][gname] = {
                    "pp": pp,
                    "sharpe": float(sharpe),
                    "mean_ret": float(np.mean(rets)),
                    "n": len(rets),
                }

        # Optimal delay per duration group
        optimal = {}
        for gname in duration_groups:
            if len(duration_groups[gname]) < 3:
                optimal[gname] = {"best_delay": 0, "best_sharpe": 0.0}
                continue
            best_d = 0
            best_s = -99.0
            for delay in self.DELAYS:
                v = delay_results[str(delay)].get(gname, {})
                s = v.get("sharpe", 0.0)
                if s > best_s:
                    best_s = s
                    best_d = delay
            optimal[gname] = {"best_delay": best_d, "best_sharpe": float(best_s)}

        # Correlation: does longer persistence -> longer optimal delay?
        dur_means = []
        best_delays = []
        for gname, gdurs in duration_groups.items():
            if len(gdurs) >= 3:
                dur_means.append(float(gdurs.mean()))
                best_delays.append(float(optimal[gname]["best_delay"]))

        persistence_delay_corr = {}
        if len(dur_means) >= 2:
            from scipy.stats import pearsonr
            p, _ = pearsonr(dur_means, best_delays)
            persistence_delay_corr = {"pearson": float(p)}

        return {
            "asset": self.asset,
            "n_events": len(events),
            "duration_quartiles": {"q25": float(qs[0]), "q50": float(qs[1]), "q75": float(qs[2])} if len(qs) >= 3 else {},
            "delay_results_by_duration_group": delay_results,
            "optimal_delay_by_group": optimal,
            "persistence_to_delay_correlation": persistence_delay_corr,
            "interpretation": (
                "If longer persistence -> longer optimal delay, then persistence determines optimal entry timing. "
                "If optimal delay is uniform across groups, duration does not matter for entry timing."
            ),
        }
