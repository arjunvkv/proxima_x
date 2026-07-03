"""
Collapse Causality Tracker

Tracks WHY OSS collapses to p_cont=0.5 by monitoring:
- Feature variance loss
- Normalization saturation (features hitting [0,1] boundaries)
- ECDF flattening (surface losing discriminative power)
- Entropy compression (p_cont becoming deterministic)
- p_cont self-correlation (model getting stuck)

Produces a causal chain to identify the primary collapse driver.
"""

import logging
import math
import statistics
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

_instances: dict[str, "_CollapseCausalityTracker"] = {}


def CollapseCausalityTracker(instance_id: str = "default") -> "_CollapseCausalityTracker":
    """Singleton accessor for CollapseCausalityTracker.

    Parameters
    ----------
    instance_id : str
        Unique identifier for the tracker instance (default 'default').

    Returns
    -------
    _CollapseCausalityTracker
        The shared tracker instance.
    """
    if instance_id not in _instances:
        _instances[instance_id] = _CollapseCausalityTracker(instance_id)
    return _instances[instance_id]


class _CollapseCausalityTracker:
    """Internal implementation of CollapseCausalityTracker.

    Tracks feature, OSS output, and ECDF data over a sliding window and
    computes collapse-relevant indicators on demand.
    """

    def __init__(self, instance_id: str = "default") -> None:
        self.instance_id = instance_id
        self.window_size = 100
        # Per-symbol data stores
        self._data: dict[str, dict[str, Any]] = {}
        logger.info("CollapseCausalityTracker(%s) initialized", instance_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_symbol(self, symbol: str) -> dict[str, Any]:
        """Ensure storage exists for *symbol* and return it."""
        if symbol not in self._data:
            self._data[symbol] = {
                "feature_history": deque(maxlen=self.window_size),
                "p_cont_history": deque(maxlen=self.window_size),
                "ev_history": deque(maxlen=self.window_size),
                "signal_history": deque(maxlen=self.window_size),
                "ecdf_history": deque(maxlen=self.window_size),
                "tick": 0,
            }
        return self._data[symbol]

    # ------------------------------------------------------------------
    # Public feed methods
    # ------------------------------------------------------------------

    def feed_features(self, symbol: str, feature_vector: dict[str, float]) -> None:
        """Record a feature vector for *symbol*.

        Parameters
        ----------
        symbol : str
            Trading symbol identifier.
        feature_vector : dict[str, float]
            Map of feature name -> value (e.g. {'rsi': 42.3, 'macd': 0.12}).
        """
        store = self._ensure_symbol(symbol)
        store["feature_history"].append(dict(feature_vector))
        store["tick"] += 1

    def feed_oss_output(
        self,
        symbol: str,
        p_cont: float,
        ev: float,
        signal: float,
        ecdf: list[float] | tuple[float, ...] | None,
    ) -> None:
        """Record OSS output for *symbol* at the current tick.

        Parameters
        ----------
        symbol : str
            Trading symbol identifier.
        p_cont : float
            Continuation probability (0-1).
        ev : float
            Expected value.
        signal : float
            Trading signal (-1, 0, 1).
        ecdf : list[float] | tuple[float, ...] | None
            Empirical CDF values (array) or None if unavailable.
        """
        store = self._ensure_symbol(symbol)
        store["p_cont_history"].append(p_cont)
        store["ev_history"].append(ev)
        store["signal_history"].append(signal)
        if ecdf is not None:
            store["ecdf_history"].append(list(ecdf))

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, symbol: str) -> dict[str, Any]:
        """Produce a full causal analysis for *symbol*.

        Parameters
        ----------
        symbol : str
            Trading symbol to analyse.

        Returns
        -------
        dict
            Collapse analysis report (see module docstring for schema).
        """
        store = self._ensure_symbol(symbol)
        ticks = store["tick"]

        result: dict[str, Any] = {
            "symbol": symbol,
            "ticks_analyzed": ticks,
        }

        # Bail early if too few data points
        min_required = 10
        if ticks < min_required:
            result["feature_variance"] = {"overall": 0.0, "per_feature": {}}
            result["normalization_saturation"] = 0.0
            result["ecdf_flattening"] = 0.0
            result["entropy_compression"] = {
                "current_entropy": 0.0,
                "entropy_rate_of_change": 0.0,
                "entropy_slope": "stable",
            }
            result["p_cont_autocorrelation"] = 0.0
            result["primary_cause"] = "UNKNOWN"
            result["causal_chain"] = [
                f"insufficient data ({ticks} ticks); need >= {min_required}"
            ]
            result["collapse_progression"] = "EARLY"
            return result

        # ---- compute indicators ----
        feat_var = self._compute_feature_variance(store)
        norm_sat = self._compute_normalization_saturation(store)
        ecdf_flat = self._compute_ecdf_flattening(store)
        entropy_comp = self._compute_entropy_compression(store)
        p_cont_ac = self._compute_p_cont_autocorrelation(store)

        result["feature_variance"] = feat_var
        result["normalization_saturation"] = norm_sat
        result["ecdf_flattening"] = ecdf_flat
        result["entropy_compression"] = entropy_comp
        result["p_cont_autocorrelation"] = p_cont_ac

        # ---- causal chain ----
        result["causal_chain"] = self._build_causal_chain(
            store, feat_var, norm_sat, ecdf_flat, entropy_comp
        )

        # ---- primary cause ----
        result["primary_cause"] = self._determine_primary_cause(
            feat_var, norm_sat, ecdf_flat, entropy_comp
        )

        # ---- collapse progression ----
        result["collapse_progression"] = self._determine_collapse_progression(
            feat_var, norm_sat, ecdf_flat, entropy_comp
        )

        return result

    # ------------------------------------------------------------------
    # Indicator computations
    # ------------------------------------------------------------------

    def _compute_feature_variance(self, store: dict[str, Any]) -> dict[str, Any]:
        """Compute variance of each feature over the sliding window.

        Returns
        -------
        dict
            ``{"overall": float, "per_feature": {name: {"min":, "max":, "mean":,
            "std":, "variance":}}}``
        """
        fh = store["feature_history"]
        if not fh:
            return {"overall": 0.0, "per_feature": {}}

        # Collect per-feature time series
        series: dict[str, list[float]] = {}
        for vec in fh:
            for k, v in vec.items():
                series.setdefault(k, []).append(v)

        per_feature: dict[str, dict[str, float]] = {}
        overall_variances: list[float] = []

        for fname, vals in series.items():
            if len(vals) < 2:
                continue
            mean = statistics.mean(vals)
            variance = statistics.variance(vals)
            std = math.sqrt(variance)
            per_feature[fname] = {
                "min": min(vals),
                "max": max(vals),
                "mean": mean,
                "std": std,
                "variance": variance,
            }
            overall_variances.append(variance)

        overall = statistics.mean(overall_variances) if overall_variances else 0.0
        return {"overall": overall, "per_feature": per_feature}

    def _compute_normalization_saturation(self, store: dict[str, Any]) -> float:
        """Fraction of feature values at extreme boundaries (near 0 or near 1)
        after min-max normalisation over the window.

        Returns
        -------
        float
            Saturation ratio in [0.0, 1.0].
        """
        fh = store["feature_history"]
        if not fh:
            return 0.0

        # Collect per-feature time series
        series: dict[str, list[float]] = {}
        for vec in fh:
            for k, v in vec.items():
                series.setdefault(k, []).append(v)

        epsilon = 0.05  # within 5% of a boundary counts as saturated
        total_extreme = 0
        total_values = 0

        for vals in series.values():
            if len(vals) < 2:
                continue
            vmin = min(vals)
            vmax = max(vals)
            if vmax == vmin:
                # All values identical -> all fully saturated
                total_extreme += len(vals)
                total_values += len(vals)
                continue

            for v in vals:
                normalized = (v - vmin) / (vmax - vmin)
                if normalized <= epsilon or normalized >= (1.0 - epsilon):
                    total_extreme += 1
                total_values += 1

        return total_extreme / total_values if total_values > 0 else 0.0

    def _compute_ecdf_flattening(self, store: dict[str, Any]) -> float:
        """Rolling standard deviation of ECDF arrays.

        A low value means the ECDF is nearly constant — the surface has lost
        discriminative power.

        Returns
        -------
        float
            Mean per-tick ECDF standard deviation.
        """
        ecdf_hist = store["ecdf_history"]
        if len(ecdf_hist) < 2:
            return 0.0

        stds: list[float] = []
        for ecdf_arr in ecdf_hist:
            if len(ecdf_arr) > 1:
                stds.append(statistics.stdev(ecdf_arr))

        return statistics.mean(stds) if stds else 0.0

    def _compute_entropy_compression(self, store: dict[str, Any]) -> dict[str, Any]:
        """Rolling Shannon entropy of p_cont values over a window.

        Returns
        -------
        dict
            ``{"current_entropy": float, "entropy_rate_of_change": float
            (per 100 ticks), "entropy_slope": str}``
        """
        p_hist = list(store["p_cont_history"])
        n = len(p_hist)
        if n < 5:
            return {
                "current_entropy": 0.0,
                "entropy_rate_of_change": 0.0,
                "entropy_slope": "stable",
            }

        # Sub-window for rolling entropy (half the main window, at least 10)
        sub_window = max(10, self.window_size // 2)
        recent = p_hist[-sub_window:]
        current_entropy = self._shannon_entropy(recent)

        # Compute entropy trajectory by sliding the sub-window
        step = max(1, sub_window // 5)
        entropies: list[float] = []
        for i in range(sub_window, n + 1, step):
            window = p_hist[i - sub_window : i]
            entropies.append(self._shannon_entropy(window))

        if len(entropies) < 2:
            rate = 0.0
        else:
            # Linear regression slope scaled to per-100-ticks
            xs = list(range(len(entropies)))
            n_pts = len(entropies)
            mean_x = statistics.mean(xs)
            mean_y = statistics.mean(entropies)
            num = sum((xs[i] - mean_x) * (entropies[i] - mean_y) for i in range(n_pts))
            den = sum((xs[i] - mean_x) ** 2 for i in range(n_pts))
            slope = num / den if den != 0 else 0.0
            rate = slope * (100.0 / step)

        if rate < -0.05:
            slope_label = "declining"
        elif rate > 0.05:
            slope_label = "increasing"
        else:
            slope_label = "stable"

        return {
            "current_entropy": current_entropy,
            "entropy_rate_of_change": rate,
            "entropy_slope": slope_label,
        }

    def _compute_p_cont_autocorrelation(self, store: dict[str, Any]) -> float:
        """Pearson correlation between p_cont[t] and p_cont[t-1] (lag-1).

        Returns
        -------
        float
            Autocorrelation in [-1, 1].  Values near 1.0 indicate the model
            output is stuck / not changing between ticks.
        """
        p_hist = list(store["p_cont_history"])
        if len(p_hist) < 3:
            return 0.0

        x = p_hist[:-1]  # t-1
        y = p_hist[1:]   # t

        n = len(x)
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        den_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        den_y = sum((y[i] - mean_y) ** 2 for i in range(n))

        if den_x == 0 or den_y == 0:
            return 0.0

        r = num / (math.sqrt(den_x) * math.sqrt(den_y))
        return max(-1.0, min(1.0, r))

    # ------------------------------------------------------------------
    # Entropy helper
    # ------------------------------------------------------------------

    @staticmethod
    def _shannon_entropy(values: list[float], bins: int = 20) -> float:
        """Compute normalised Shannon entropy of *values*.

        Values are clipped to [0, 1] and binned into *bins* equal-width bins.
        Entropy is normalised by log2(bins) so the result lies in [0, 1].

        Parameters
        ----------
        values : list[float]
            Input values (typically p_cont in [0, 1]).
        bins : int
            Number of histogram bins (default 20).

        Returns
        -------
        float
            Normalised entropy in [0, 1].
        """
        if not values:
            return 0.0

        clipped = [max(0.0, min(1.0, v)) for v in values]
        counts = [0] * bins
        for v in clipped:
            idx = min(int(v * bins), bins - 1)
            counts[idx] += 1

        total = len(clipped)
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)

        max_entropy = math.log2(bins)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    # ------------------------------------------------------------------
    # Causal chain & progression
    # ------------------------------------------------------------------

    @staticmethod
    def _build_causal_chain(
        store: dict[str, Any],
        feat_var: dict[str, Any],
        norm_sat: float,
        ecdf_flat: float,
        entropy_comp: dict[str, Any],
    ) -> list[str]:
        """Build a human-readable causal chain describing collapse dynamics."""
        chain: list[str] = []
        tick = store["tick"]

        fv_ov = feat_var.get("overall", 0.0)
        if fv_ov < 0.01:
            chain.append(
                f"feature variance dropped to {fv_ov:.6f} at tick {tick} "
                "(threshold < 0.01)"
            )

        if norm_sat > 0.8:
            chain.append(
                f"normalization saturation reached {norm_sat:.3f} at tick {tick} "
                "(threshold > 0.8)"
            )

        if ecdf_flat < 0.01:
            chain.append(
                f"ECDF flattening reached {ecdf_flat:.6f} at tick {tick} "
                "(threshold < 0.01)"
            )

        ent_rate = entropy_comp.get("entropy_rate_of_change", 0.0)
        if ent_rate < -0.05:
            chain.append(
                f"entropy compression rate {ent_rate:.4f} per 100 ticks at tick {tick} "
                "(threshold declining < -0.05)"
            )

        if not chain:
            chain.append(f"no collapse indicators triggered at tick {tick}")

        return chain

    @staticmethod
    def _determine_primary_cause(
        feat_var: dict[str, Any],
        norm_sat: float,
        ecdf_flat: float,
        entropy_comp: dict[str, Any],
    ) -> str:
        """Primary cause = the FIRST indicator to cross its collapse threshold.

        Order of precedence:
        1. Feature variance loss (< 0.01)
        2. Normalisation saturation (> 0.8)
        3. ECDF flattening (< 0.01)
        4. Entropy compression (rate declining < -0.05)
        """
        fv_ov = feat_var.get("overall", 1.0)
        if fv_ov < 0.01:
            return "FEATURE_VARIANCE_LOSS"

        if norm_sat > 0.8:
            return "NORMALIZATION_SATURATION"

        if ecdf_flat < 0.01:
            return "ECDF_FLATTENING"

        ent_rate = entropy_comp.get("entropy_rate_of_change", 0.0)
        if ent_rate < -0.05:
            return "ENTROPY_COMPRESSION"

        return "UNKNOWN"

    @staticmethod
    def _determine_collapse_progression(
        feat_var: dict[str, Any],
        norm_sat: float,
        ecdf_flat: float,
        entropy_comp: dict[str, Any],
    ) -> str:
        """Classify how many of the four collapse indicators are triggered."""
        triggers = 0

        if feat_var.get("overall", 1.0) < 0.01:
            triggers += 1
        if norm_sat > 0.8:
            triggers += 1
        if ecdf_flat < 0.01:
            triggers += 1
        ent_rate = entropy_comp.get("entropy_rate_of_change", 0.0)
        if ent_rate < -0.05:
            triggers += 1

        if triggers >= 4:
            return "FULL"
        if triggers >= 3:
            return "LATE"
        if triggers >= 2:
            return "MID"
        if triggers >= 1:
            return "EARLY"
        return "NONE"

    # ------------------------------------------------------------------
    # Self-test
    # ------------------------------------------------------------------

    @staticmethod
    def _self_test() -> None:
        """Run a self-test with synthetic data simulating gradual collapse.

        Three phases:
        1. Healthy dynamics  (random variance)
        2. Variance decay    (gradual loss of signal)
        3. Full collapse     (near-zero variance, flat ECDF, stuck p_cont)
        """
        import random

        print("=" * 60)
        print("CollapseCausalityTracker — Self-Test")
        print("=" * 60)

        tracker = CollapseCausalityTracker("self_test")
        symbol = "SYNTH"

        # ---- Phase 1: healthy dynamics ----
        print("\n--- Phase 1: Healthy dynamics ---")
        for _ in range(150):
            feat = {
                "rsi": 50.0 + random.gauss(0, 15),
                "macd": random.gauss(0, 0.5),
                "volume_ratio": 1.0 + random.gauss(0, 0.3),
                "volatility": random.uniform(0.1, 0.5),
            }
            tracker.feed_features(symbol, feat)
            p_cont = 0.5 + random.gauss(0, 0.15)
            p_cont = max(0.01, min(0.99, p_cont))
            ev = random.uniform(0.3, 0.7)
            signal = random.choice([-1, 0, 1])
            ecdf_vals = sorted(random.uniform(0, 1) for _ in range(20))
            tracker.feed_oss_output(symbol, p_cont, ev, signal, ecdf_vals)

        report = tracker.analyze(symbol)
        print(f"  Overall feature variance:  {report['feature_variance']['overall']:.4f}")
        print(f"  Normalization saturation:  {report['normalization_saturation']:.4f}")
        print(f"  ECDF flattening:           {report['ecdf_flattening']:.4f}")
        print(f"  Entropy:                   {report['entropy_compression']['current_entropy']:.4f}")
        print(f"  p_cont autocorrelation:    {report['p_cont_autocorrelation']:.4f}")
        print(f"  Primary cause:             {report['primary_cause']}")
        print(f"  Progression:               {report['collapse_progression']}")

        # ---- Phase 2: variance decay ----
        print("\n--- Phase 2: Variance decay ---")
        for t in range(100):
            decay = max(0.001, 1.0 - (t / 100))
            feat = {
                "rsi": 50.0 + random.gauss(0, 15 * decay),
                "macd": random.gauss(0, 0.5 * decay),
                "volume_ratio": 1.0 + random.gauss(0, 0.3 * decay),
                "volatility": 0.3 + random.gauss(0, 0.1 * decay),
            }
            tracker.feed_features(symbol, feat)
            p_cont = 0.5 + random.gauss(0, 0.15 * decay)
            p_cont = max(0.01, min(0.99, p_cont))
            ev = 0.5 + random.gauss(0, 0.1 * decay)
            signal = 0
            ecdf_vals = sorted(
                random.uniform(0.3 * decay, 1.0 - 0.3 * decay) for _ in range(20)
            )
            tracker.feed_oss_output(symbol, p_cont, ev, signal, ecdf_vals)

        report = tracker.analyze(symbol)
        print(f"  Overall feature variance:  {report['feature_variance']['overall']:.4f}")
        print(f"  Normalization saturation:  {report['normalization_saturation']:.4f}")
        print(f"  ECDF flattening:           {report['ecdf_flattening']:.4f}")
        print(f"  Entropy:                   {report['entropy_compression']['current_entropy']:.4f}")
        print(f"  p_cont autocorrelation:    {report['p_cont_autocorrelation']:.4f}")
        print(f"  Primary cause:             {report['primary_cause']}")
        print(f"  Progression:               {report['collapse_progression']}")
        print(f"  Causal chain:")
        for step in report["causal_chain"]:
            print(f"    - {step}")

        # ---- Phase 3: full collapse ----
        print("\n--- Phase 3: Full collapse ---")
        for _ in range(150):
            feat = {
                "rsi": 50.0 + random.gauss(0, 0.005),
                "macd": random.gauss(0, 0.005),
                "volume_ratio": 1.0 + random.gauss(0, 0.005),
                "volatility": 0.3 + random.gauss(0, 0.005),
            }
            tracker.feed_features(symbol, feat)
            p_cont = 0.5 + random.gauss(0, 0.01)
            p_cont = max(0.01, min(0.99, p_cont))
            ev = 0.5
            signal = 0
            # Nearly flat ECDF
            base = 0.5 + random.gauss(0, 0.005)
            ecdf_vals = sorted(base + random.gauss(0, 0.002) for _ in range(20))
            tracker.feed_oss_output(symbol, p_cont, ev, signal, ecdf_vals)

        report = tracker.analyze(symbol)
        print(f"  Overall feature variance:  {report['feature_variance']['overall']:.4f}")
        print(f"  Normalization saturation:  {report['normalization_saturation']:.4f}")
        print(f"  ECDF flattening:           {report['ecdf_flattening']:.4f}")
        print(f"  Entropy:                   {report['entropy_compression']['current_entropy']:.4f}")
        print(f"  p_cont autocorrelation:    {report['p_cont_autocorrelation']:.4f}")
        print(f"  Primary cause:             {report['primary_cause']}")
        print(f"  Progression:               {report['collapse_progression']}")
        print(f"  Causal chain:")
        for step in report["causal_chain"]:
            print(f"    - {step}")

        print("\n" + "=" * 60)
        print("Self-test complete.")
        print("=" * 60)


# ------------------------------------------------------------------
# Entry point for self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _CollapseCausalityTracker._self_test()
