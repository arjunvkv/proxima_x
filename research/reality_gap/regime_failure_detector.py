import json
import numpy as np
from proxima_v1.core.signal_engine import SignalEngine
from research.adaptive_alpha_engine.aae_validator import AAEValidator
from research.reality_gap.signal_quality_monitor import SignalQualityMonitor as SQM


class RegimeFailureDetector:
    PERIODS = ["2020-2022", "2022-2024", "2024-2026"]

    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset

    def run(self) -> dict:
        engine = SignalEngine(self.asset)
        engine.precompute_full()
        full_res = np.nan_to_num(engine._full_residual, nan=0.0)
        full_es = np.nan_to_num(engine._full_es, nan=0.0)
        full_at = np.nan_to_num(engine._full_at, nan=0.0)
        n = len(full_res)
        price = np.array(engine._data.get("price", np.zeros(n)))
        rolling_res = np.full(n, 0.5)
        rolling_es = np.full(n, 0.5)
        rolling_at = np.full(n, 0.5)
        for i in range(503, n):
            r_slice = full_res[max(0, i - 503):i + 1]
            e_slice = full_es[max(0, i - 503):i + 1]
            a_slice = full_at[max(0, i - 503):i + 1]
            rolling_res[i] = float(np.sum(r_slice <= full_res[i])) / len(r_slice)
            rolling_es[i] = float(np.sum(e_slice <= full_es[i])) / len(e_slice)
            rolling_at[i] = float(np.sum(a_slice <= full_at[i])) / len(a_slice)
        composite = np.clip(0.60 * rolling_res + 0.30 * rolling_es + 0.10 * rolling_at, 0.0, 1.0)
        mapping = SQM.date_to_index_mapping(engine._data, [("2020-01-01", "2022-01-01", "2020-2022"), ("2022-01-01", "2024-01-01", "2022-2024"), ("2024-01-01", "2026-01-01", "2024-2026")])
        period_metrics = {}
        for label in self.PERIODS:
            s, e = mapping.get(label, (0, 0))
            if e - s < 100:
                period_metrics[label] = {"signal_frequency": 0.0, "signal_strength": 0.0, "residual_energy_distribution": {"mean": 0.0, "std": 0.0, "skew": 0.0}, "energy_storage_distribution": {"mean": 0.0, "std": 0.0, "skew": 0.0}, "adaptive_time_distribution": {"mean": 0.0, "std": 0.0}, "threshold_90th": 0.0, "pp_h20": 0.5}
            else:
                period_metrics[label] = SQM._period_metrics(composite, rolling_res, rolling_es, rolling_at, price, s, e)

        pf_2022 = period_metrics.get("2022-2024", {}).get("signal_frequency", 0.0)
        pf_2020 = period_metrics.get("2020-2022", {}).get("signal_frequency", 0.0)
        ps_2022 = period_metrics.get("2022-2024", {}).get("signal_strength", 0.0)
        ps_2020 = period_metrics.get("2020-2022", {}).get("signal_strength", 0.0)
        pp_2022 = period_metrics.get("2022-2024", {}).get("pp_h20", 0.5)
        pp_2020 = period_metrics.get("2020-2022", {}).get("pp_h20", 0.5)
        t90_2022 = period_metrics.get("2022-2024", {}).get("threshold_90th", 0.0)
        t90_2020 = period_metrics.get("2020-2022", {}).get("threshold_90th", 0.0)
        failures = []
        if pf_2022 < pf_2020 * 0.7:
            failures.append("signal_frequency_drop")
        if ps_2022 < ps_2020 * 0.9:
            failures.append("signal_strength_decline")
        if pp_2022 < 0.55:
            failures.append("low_profit_probability")
        if t90_2022 < t90_2020 * 0.75:
            failures.append("threshold_collapse")
        res_skew_2022 = period_metrics.get("2022-2024", {}).get("residual_energy_distribution", {}).get("skew", 0.0)
        es_skew_2022 = period_metrics.get("2022-2024", {}).get("energy_storage_distribution", {}).get("skew", 0.0)
        if abs(res_skew_2022) > 2.0 or abs(es_skew_2022) > 2.0:
            failures.append("distribution_skew_extremes")
        if len(failures) == 0:
            failures.append("mild_degradation")

        d100_res = np.full(n, 0.5)
        d100_es = np.full(n, 0.5)
        d100_at = np.full(n, 0.5)
        for i in range(99, n):
            r_slice = full_res[max(0, i - 99):i + 1]
            e_slice = full_es[max(0, i - 99):i + 1]
            a_slice = full_at[max(0, i - 99):i + 1]
            d100_res[i] = float(np.sum(r_slice <= full_res[i])) / len(r_slice)
            d100_es[i] = float(np.sum(e_slice <= full_es[i])) / len(e_slice)
            d100_at[i] = float(np.sum(a_slice <= full_at[i])) / len(a_slice)
        composite_100 = np.clip(0.60 * d100_res + 0.30 * d100_es + 0.10 * d100_at, 0.0, 1.0)

        signal_freq_100 = SQM.compute_signal_frequency(composite_100, 100, 0.7)
        signal_strength_100 = SQM.compute_signal_strength(composite_100, 100, 0.7)
        rolling_corr = np.full(n, np.nan)
        for i in range(100, n):
            r_seg = d100_res[i - 100:i]
            e_seg = d100_es[i - 100:i]
            if np.std(r_seg) > 1e-12 and np.std(e_seg) > 1e-12:
                c = float(np.corrcoef(r_seg, e_seg)[0, 1])
                rolling_corr[i] = float(np.clip(c, -1.0, 1.0))
            else:
                rolling_corr[i] = 0.0
        rolling_corr = np.nan_to_num(rolling_corr, nan=0.0)
        baseline_90th = period_metrics.get("2020-2022", {}).get("threshold_90th", 0.7)
        threshold_90th_drift = SQM.compute_threshold_drift(composite_100, 100, 90.0)
        threshold_collapse = np.zeros(n)
        for i in range(n):
            if baseline_90th > 0.0 and threshold_90th_drift[i] < baseline_90th * 0.75:
                threshold_collapse[i] = 1.0

        freq_trend = np.zeros(n)
        for i in range(200, n):
            if np.std(signal_freq_100[i - 100:i]) > 1e-12:
                slope = float(np.polyfit(np.arange(100), signal_freq_100[i - 100:i], 1)[0])
                freq_trend[i] = float(np.clip(-slope * 100, 0.0, 1.0))

        strength_trend = np.zeros(n)
        for i in range(200, n):
            if np.std(signal_strength_100[i - 100:i]) > 1e-12:
                slope = float(np.polyfit(np.arange(100), signal_strength_100[i - 100:i], 1)[0])
                strength_trend[i] = float(np.clip(-slope * 100, 0.0, 1.0))

        corr_factor = np.clip((rolling_corr - 0.3) / 0.7, 0.0, 1.0)

        failure_probability = np.clip(
            0.30 * freq_trend +
            0.25 * strength_trend +
            0.25 * corr_factor +
            0.20 * threshold_collapse,
            0.0, 1.0
        )

        future_returns = np.full(n, np.nan)
        for i in range(n - 100):
            future_returns[i] = float(np.log(price[i + 100] / price[i]))

        fp_high = failure_probability > 0.7
        pp_next_100 = SQM.compute_rolling_pp(composite_100, future_returns, 100, 0.7)
        low_pp = pp_next_100 < 0.55

        valid = ~np.isnan(pp_next_100)
        fp_high_valid = fp_high & valid
        low_pp_valid = low_pp & valid

        tp = int(np.sum(fp_high_valid & low_pp_valid))
        fp_count = int(np.sum(fp_high_valid & ~low_pp_valid))
        fn_count = int(np.sum(~fp_high_valid & low_pp_valid & valid))

        precision = float(tp / max(tp + fp_count, 1))
        recall = float(tp / max(tp + fn_count, 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-12))

        if "threshold_collapse" in failures:
            failure_mechanism = "2022-2024 failure driven by threshold collapse: rolling 90th percentile of composite signal dropped >25% from 2020-2022 baseline, reducing signal frequency and causing profit probability to fall below 0.55."
        elif "distribution_skew_extremes" in failures:
            failure_mechanism = "2022-2024 failure driven by distribution skew extremes in residual_energy and energy_storage signals, causing unstable ranking and unreliable composite signals."
        elif "signal_frequency_drop" in failures:
            failure_mechanism = "2022-2024 failure driven by signal frequency drop: fewer bars exceeded 0.7 composite threshold compared to 2020-2022, reducing trade opportunities."
        elif "signal_strength_decline" in failures:
            failure_mechanism = "2022-2024 failure driven by signal strength decline: mean composite value above 0.7 decreased, indicating weaker conviction on generated signals."
        elif "low_profit_probability" in failures:
            failure_mechanism = "2022-2024 failure driven by low profit probability: signals above threshold had PP below 0.55, indicating poor forward predictive power."
        else:
            failure_mechanism = "Mild degradation across multiple metrics in 2022-2024 relative to 2020-2022 baseline."

        return {
            "period_metrics": period_metrics,
            "failure_mechanism": failure_mechanism,
            "failure_probability_series": failure_probability.tolist(),
            "predictive_power": {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        }

    def save(self, path: str):
        results = self.run()
        results["failure_probability_series"] = results["failure_probability_series"]
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
