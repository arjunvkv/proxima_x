"""
Regime Activation Function Mapper + Regime Scarcity Diagnosis.
Reads pipeline trace data and computes conditional probability of trade execution
given market regime state (RSI, ATR percentile, regime label).
"""

import json
import os
import logging
from collections import defaultdict

logger = logging.getLogger("proxima_ops.monitoring.regime_activation_mapper")

DEFAULT_TRACE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "state", "live_pipeline_trace.jsonl"
)

RSI_BINS = ["<20", "20-30", "30-40", "40-60", "60-70", "70-80", ">80"]
ATR_PERCENTILE_BINS = ["0-20", "20-40", "40-60", "60-80", "80-100"]
REGIME_LABELS = ["compression", "expansion", "transition"]


class RegimeActivationMapper:
    def __init__(self, trace_path=None):
        self._trace_path = trace_path or DEFAULT_TRACE_PATH
        logger.info("RegimeActivationMapper initialized with trace_path=%s", self._trace_path)

    def load_trace(self, path=None):
        resolved = path or self._trace_path
        if not os.path.exists(resolved):
            logger.warning("Trace file not found: %s", resolved)
            return []
        entries = []
        with open(resolved, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    logger.error("JSON decode error at line %d: %s", line_no, e)
        logger.info("Loaded %d trace entries from %s", len(entries), resolved)
        return entries

    @staticmethod
    def _rsi_bin(rsi):
        if rsi is None:
            return "40-60"
        if rsi < 20:
            return "<20"
        if rsi < 30:
            return "20-30"
        if rsi < 40:
            return "30-40"
        if rsi < 60:
            return "40-60"
        if rsi < 70:
            return "60-70"
        if rsi < 80:
            return "70-80"
        return ">80"

    @staticmethod
    def _regime_label(md_entry):
        rsi = md_entry.get("rsi", 50)
        if rsi is None:
            return "transition"
        if rsi < 30:
            return "compression"
        if rsi > 70:
            return "expansion"
        return "transition"

    @staticmethod
    def _compute_atr_percentiles(entries):
        atr_values = []
        for entry in entries:
            md = entry.get("md_summary", {})
            for sym_data in md.values():
                atr = sym_data.get("atr", 0)
                if atr is None:
                    atr = 0
                atr_values.append(atr)
        if not atr_values:
            return {}
        atr_values.sort()
        n = len(atr_values)
        thresholds = {}
        for p in [20, 40, 60, 80]:
            idx = max(0, int(n * p / 100) - 1)
            thresholds[p] = atr_values[idx]
        return thresholds

    @staticmethod
    def _atr_percentile_bin(atr, percentiles):
        if atr is None:
            return "0-20"
        if not percentiles:
            return "0-20"
        if atr <= percentiles.get(20, 0):
            return "0-20"
        if atr <= percentiles.get(40, 0):
            return "20-40"
        if atr <= percentiles.get(60, 0):
            return "40-60"
        if atr <= percentiles.get(80, 0):
            return "60-80"
        return "80-100"

    @staticmethod
    def _bin_template():
        return {
            "cycles": 0,
            "any_signal_rate": 0.0,
            "threshold_pass_rate": 0.0,
            "confirm_pass_rate": 0.0,
            "execution_rate": 0.0,
            "denial_reasons": {},
        }

    def _init_bin_dict(self, bin_labels):
        return {label: self._bin_template() for label in bin_labels}

    def _get_confirm_pass(self, entry):
        cg = entry.get("confirm_gate", {})
        if isinstance(cg, dict):
            return cg.get("confirm_pass", False)
        return False

    def _get_executed(self, entry):
        pf = entry.get("pipeline_funnel", {})
        if isinstance(pf, dict):
            return pf.get("executed", 0) > 0
        return False

    def _get_denial_reason(self, entry):
        ex = entry.get("execution", {})
        if isinstance(ex, dict):
            return ex.get("denial_reason")
        return None

    def _get_any_signal(self, entry):
        return entry.get("signals_generated", 0) > 0

    def _update_bin(self, bin_data, any_signal, any_threshold, confirm_pass, executed, denial_reason):
        bin_data["cycles"] += 1
        if any_signal:
            bin_data["any_signal_rate"] += 1
        if any_threshold:
            bin_data["threshold_pass_rate"] += 1
        if confirm_pass:
            bin_data["confirm_pass_rate"] += 1
        if executed:
            bin_data["execution_rate"] += 1
        if denial_reason:
            reason_str = str(denial_reason)
            bin_data["denial_reasons"][reason_str] = bin_data["denial_reasons"].get(reason_str, 0) + 1

    def _finalize_bins(self, bins):
        for bin_data in bins.values():
            c = bin_data["cycles"]
            if c > 0:
                bin_data["any_signal_rate"] = round(bin_data["any_signal_rate"] / c, 4)
                bin_data["threshold_pass_rate"] = round(bin_data["threshold_pass_rate"] / c, 4)
                bin_data["confirm_pass_rate"] = round(bin_data["confirm_pass_rate"] / c, 4)
                bin_data["execution_rate"] = round(bin_data["execution_rate"] / c, 4)

    def activation_map(self, entries):
        if not entries:
            return {"total_cycles": 0}

        total_cycles = len(entries)
        percentiles = self._compute_atr_percentiles(entries)

        rsi_bins = self._init_bin_dict(RSI_BINS)
        atr_bins = self._init_bin_dict(ATR_PERCENTILE_BINS)
        regime_bins = self._init_bin_dict(REGIME_LABELS)
        per_symbol = {}

        for entry in entries:
            md = entry.get("md_summary", {})
            signals_generated = entry.get("signals_generated", 0)
            threshold_pass_count = entry.get("threshold_pass_count", 0)
            confirm_pass = self._get_confirm_pass(entry)
            executed = self._get_executed(entry)
            denial_reason = self._get_denial_reason(entry)

            any_signal = signals_generated > 0
            any_threshold = threshold_pass_count > 0

            for sym, sym_data in md.items():
                rsi = sym_data.get("rsi", 50)
                atr = sym_data.get("atr", 0)

                r_bin = self._rsi_bin(rsi)
                a_bin = self._atr_percentile_bin(atr, percentiles)
                rl_bin = self._regime_label(sym_data)

                self._update_bin(rsi_bins[r_bin], any_signal, any_threshold, confirm_pass, executed, denial_reason)
                self._update_bin(atr_bins[a_bin], any_signal, any_threshold, confirm_pass, executed, denial_reason)
                self._update_bin(regime_bins[rl_bin], any_signal, any_threshold, confirm_pass, executed, denial_reason)

                if sym not in per_symbol:
                    per_symbol[sym] = self._init_bin_dict(RSI_BINS)
                self._update_bin(per_symbol[sym][r_bin], any_signal, any_threshold, confirm_pass, executed, denial_reason)

        self._finalize_bins(rsi_bins)
        self._finalize_bins(atr_bins)
        self._finalize_bins(regime_bins)
        for sym_bins in per_symbol.values():
            self._finalize_bins(sym_bins)

        return {
            "total_cycles": total_cycles,
            "rsi_bins": rsi_bins,
            "atr_bins": atr_bins,
            "regime_labels": regime_bins,
            "per_symbol": per_symbol,
        }

    def rsi_activation_curve(self, entries):
        if not entries:
            return []
        rsi_points = []
        for entry in entries:
            md = entry.get("md_summary", {})
            executed = self._get_executed(entry)
            for sym_data in md.values():
                rsi = sym_data.get("rsi", 50)
                if rsi is None:
                    continue
                rsi_points.append({"rsi": rsi, "executed": 1 if executed else 0})
        if not rsi_points:
            return []
        rsi_points.sort(key=lambda x: x["rsi"])
        bucket_size = max(1, len(rsi_points) // 20)
        series = []
        for i in range(0, len(rsi_points), bucket_size):
            bucket = rsi_points[i:i + bucket_size]
            avg_rsi = sum(p["rsi"] for p in bucket) / len(bucket)
            exec_rate = sum(p["executed"] for p in bucket) / len(bucket)
            series.append({
                "rsi": round(avg_rsi, 2),
                "execution_rate": round(exec_rate, 4),
                "count": len(bucket),
            })
        return series

    def activation_heatmap(self, entries):
        if not entries:
            return {}
        percentiles = self._compute_atr_percentiles(entries)
        grid = {}
        for rb in RSI_BINS:
            grid[rb] = {}
            for ab in ATR_PERCENTILE_BINS:
                grid[rb][ab] = {"cycles": 0, "executions": 0}
        for entry in entries:
            md = entry.get("md_summary", {})
            executed = self._get_executed(entry)
            for sym_data in md.values():
                rsi = sym_data.get("rsi", 50)
                atr = sym_data.get("atr", 0)
                rb = self._rsi_bin(rsi)
                ab = self._atr_percentile_bin(atr, percentiles)
                grid[rb][ab]["cycles"] += 1
                if executed:
                    grid[rb][ab]["executions"] += 1
        result = {}
        for rb in RSI_BINS:
            result[rb] = {}
            for ab in ATR_PERCENTILE_BINS:
                c = grid[rb][ab]["cycles"]
                e = grid[rb][ab]["executions"]
                result[rb][ab] = {
                    "cycles": c,
                    "execution_rate": round(e / c, 4) if c > 0 else 0.0,
                }
        return result

    def scarcity_diagnosis(self, entries):
        if not entries:
            return {
                "primary_scarcity_mode": "unknown",
                "confidence": 0.0,
                "breakdown": {},
                "diagnosis": "No data",
            }
        total = len(entries)
        cycles_with_signal = sum(1 for e in entries if self._get_any_signal(e))
        cycles_with_threshold = sum(1 for e in entries if e.get("threshold_pass_count", 0) > 0)
        cycles_confirm_pass = sum(1 for e in entries if self._get_confirm_pass(e))
        cycles_executed = sum(1 for e in entries if self._get_executed(e))

        cycles_extreme = 0
        for entry in entries:
            md = entry.get("md_summary", {})
            extreme = any(
                (sym_data.get("rsi") is not None and (sym_data["rsi"] < 30 or sym_data["rsi"] > 70))
                for sym_data in md.values()
            )
            if extreme:
                cycles_extreme += 1

        pct_signal = (cycles_with_signal / total * 100) if total else 0
        pct_confirm_of_signal = (cycles_confirm_pass / cycles_with_signal * 100) if cycles_with_signal else 0
        pct_exec_of_confirm = (cycles_executed / cycles_confirm_pass * 100) if cycles_confirm_pass else 0
        pct_extreme = (cycles_extreme / total * 100) if total else 0

        breakdown = {
            "signal_limited": {
                "label": "CORE produces too few signals for market conditions",
                "metric": "% cycles with any signal",
                "value": f"{pct_signal:.1f}%",
                "threshold": ">20% for healthy flow",
            },
            "confirmation_limited": {
                "label": "Signals exist but fail temporal persistence (confirm 2/2)",
                "metric": "% signal cycles reaching confirm pass",
                "value": f"{pct_confirm_of_signal:.1f}%",
                "threshold": ">50% expected",
            },
            "execution_limited": {
                "label": "Confirmed signals blocked by governor/VEL/CB",
                "metric": "% confirm passes reaching execution",
                "value": f"{pct_exec_of_confirm:.1f}%",
                "threshold": ">80% expected",
            },
            "market_limited": {
                "label": "Market not producing regime extreme conditions",
                "metric": "% cycles with RSI<30 or >70",
                "value": f"{pct_extreme:.1f}%",
                "threshold": ">15% for active system",
            },
        }

        modes = []
        if pct_signal < 10:
            modes.append("signal-limited")
        if pct_confirm_of_signal < 50:
            modes.append("confirmation-limited")
        if pct_exec_of_confirm < 80:
            modes.append("execution-limited")
        if pct_extreme < 15:
            modes.append("market-limited")

        priority = ["signal-limited", "confirmation-limited", "execution-limited", "market-limited"]
        primary_scarcity_mode = "none"
        for pm in priority:
            if pm in modes:
                primary_scarcity_mode = pm
                break
        if primary_scarcity_mode == "none":
            primary_scarcity_mode = "unlimited"

        n_modes = len(modes)
        if n_modes >= 3:
            confidence = 0.95
        elif n_modes == 2:
            confidence = 0.85
        elif n_modes == 1:
            confidence = 0.75
        else:
            confidence = 1.0

        diag_parts = []
        if primary_scarcity_mode == "signal-limited":
            diag_parts.append(
                f"System is signal-limited: only {pct_signal:.1f}% of cycles "
                "produce signals, below 10% threshold."
            )
        elif primary_scarcity_mode == "confirmation-limited":
            diag_parts.append(
                f"System is confirmation-limited: signals reach threshold but fail "
                f"temporal persistence requirement (2/2 confirm). "
                f"Only {pct_confirm_of_signal:.1f}% of signal cycles reach confirm pass."
            )
        elif primary_scarcity_mode == "execution-limited":
            diag_parts.append(
                f"System is execution-limited: {pct_exec_of_confirm:.1f}% of confirmed "
                "signals reach execution, below 80% threshold."
            )
        elif primary_scarcity_mode == "market-limited":
            diag_parts.append(
                f"System is market-limited: only {pct_extreme:.1f}% of cycles "
                "exhibit extreme RSI conditions."
            )
        else:
            diag_parts.append("System is not scarcity-limited.")

        if primary_scarcity_mode != "market-limited" and "market-limited" in modes:
            diag_parts.append(
                f"Market regime exposure is limited with {pct_extreme:.1f}% extreme cycles."
            )

        diagnosis = " ".join(diag_parts)

        return {
            "primary_scarcity_mode": primary_scarcity_mode,
            "confidence": confidence,
            "breakdown": breakdown,
            "diagnosis": diagnosis,
        }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    mapper = RegimeActivationMapper()
    entries = mapper.load_trace()
    if not entries:
        logger.warning("No trace entries found.")
        print(json.dumps({"error": "no trace data"}, indent=2))
        return

    amap = mapper.activation_map(entries)
    print("=== Regime Activation Map ===")
    print(json.dumps(amap, indent=2, default=str))

    diagnosis = mapper.scarcity_diagnosis(entries)
    print("\n=== Regime Scarcity Diagnosis ===")
    print(json.dumps(diagnosis, indent=2, default=str))


if __name__ == "__main__":
    main()
