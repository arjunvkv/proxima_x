import json
import logging
import math
import os

logger = logging.getLogger("proxima_ops.monitoring.layer_drift_monitor")


def _compute_staircase_weighted(pnl: float, base_volume: float, final_volume: float) -> float:
    if final_volume == 0.0:
        return 0.0
    return pnl * (base_volume / final_volume)


def _compute_amplifier_contribution(pnl: float, staircase_weighted: float) -> float:
    return pnl - staircase_weighted


def _segment_single_trade(trade: dict) -> dict:
    pnl = trade.get("pnl", 0.0)
    vol_comp = trade.get("volume_composition", {})
    base_volume = vol_comp.get("base_volume", 0.0)
    final_volume = vol_comp.get("final_volume", 0.0)

    staircase_weighted = _compute_staircase_weighted(pnl, base_volume, final_volume)
    amplifier_adjusted = _compute_amplifier_contribution(pnl, staircase_weighted)

    return {
        "pnl": pnl,
        "staircase_weighted_pnl": staircase_weighted,
        "amplifier_adjusted_pnl": amplifier_adjusted,
        "base_volume": base_volume,
        "final_volume": final_volume,
    }


def _mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stdev(values: list) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _cv(values: list) -> float:
    m = _mean(values)
    if m == 0.0:
        return 0.0
    return _stdev(values) / abs(m)


def _classify_trend(cv: float) -> str:
    if cv < 0.05:
        return "very_stable"
    if cv < 0.2:
        return "stable"
    if cv <= 0.5:
        return "variable"
    return "unstable"


class LayerDriftMonitor:

    def load_trades(self, source: str = "state/trade_lifecycle_state.json") -> list:
        if not os.path.exists(source):
            logger.warning("Trade data file not found: %s", source)
            return []
        try:
            with open(source) as f:
                state = json.load(f)
            closed_trades = state.get("closed_trades", [])
            trades = []
            for t in closed_trades:
                volume = t.get("volume", 0.0)
                trades.append({
                    "symbol": t.get("symbol", ""),
                    "direction": t.get("direction", ""),
                    "pnl": t.get("pnl", 0.0),
                    "volume_composition": {
                        "base_volume": volume,
                        "amplifier_multiplier": 1.0,
                        "final_volume": volume,
                    },
                    "entry_price": t.get("price", 0.0),
                    "exit_price": t.get("exit_price", 0.0),
                    "reason": t.get("exit_reason", ""),
                })
            logger.info("Loaded %d trades from %s", len(trades), source)
            return trades
        except Exception as e:
            logger.error("Failed to load trades from %s: %s", source, e)
            return []

    def drift_report(self, trades: list[dict], window: int = 10) -> dict:
        if not trades or window <= 0:
            return {
                "total_trades": len(trades),
                "window_size": window,
                "num_windows": 0,
                "core_stability": {
                    "expectancy_mean": 0.0,
                    "expectancy_std": 0.0,
                    "expectancy_cv": 0.0,
                    "trend": "stable",
                    "windows": [],
                },
                "staircase_stability": {
                    "expectancy_mean": 0.0,
                    "expectancy_std": 0.0,
                    "expectancy_cv": 0.0,
                    "trend": "stable",
                    "windows": [],
                },
                "amplifier_stability": {
                    "avg_contribution": 0.0,
                    "contribution_std": 0.0,
                    "contribution_cv": 0.0,
                    "trend": "stable",
                    "positive_windows": 0,
                    "negative_windows": 0,
                    "windows": [],
                },
                "overall_drift_assessment": "Insufficient data",
            }

        segmented = [_segment_single_trade(t) for t in trades]

        core_windows = []
        stair_windows = []
        amp_windows = []

        num_windows = max(1, len(segmented) - window + 1)

        for i in range(num_windows):
            window_seg = segmented[i:i + window]
            if len(window_seg) < window and len(segmented) >= window:
                break

            core_pnls = [s["pnl"] for s in window_seg]
            stair_pnls = [s["staircase_weighted_pnl"] for s in window_seg]
            amp_pnls = [s["amplifier_adjusted_pnl"] for s in window_seg]

            core_windows.append({
                "window": i + 1,
                "expectancy": round(_mean(core_pnls), 4),
                "trades": len(window_seg),
            })
            stair_windows.append({
                "window": i + 1,
                "expectancy": round(_mean(stair_pnls), 4),
                "trades": len(window_seg),
            })
            amp_windows.append({
                "window": i + 1,
                "avg_contribution": round(_mean(amp_pnls), 4),
                "trades": len(window_seg),
            })

        core_exps = [w["expectancy"] for w in core_windows]
        stair_exps = [w["expectancy"] for w in stair_windows]
        amp_avgs = [w["avg_contribution"] for w in amp_windows]

        core_mean = _mean(core_exps)
        core_std = _stdev(core_exps)
        core_cv = _cv(core_exps)

        stair_mean = _mean(stair_exps)
        stair_std = _stdev(stair_exps)
        stair_cv = _cv(stair_exps)

        amp_mean = _mean(amp_avgs)
        amp_std = _stdev(amp_avgs)
        amp_cv = _cv(amp_avgs)

        positive_windows = sum(1 for a in amp_avgs if a > 0)
        negative_windows = sum(1 for a in amp_avgs if a < 0)

        core_trend = _classify_trend(core_cv)
        stair_trend = _classify_trend(stair_cv)
        amp_trend = _classify_trend(amp_cv)

        assessment_parts = []
        assessment_parts.append("CORE " + core_trend)
        assessment_parts.append("Staircase " + stair_trend)
        assessment_parts.append("Amplifier " + amp_trend)

        return {
            "total_trades": len(trades),
            "window_size": window,
            "num_windows": len(core_windows),
            "core_stability": {
                "expectancy_mean": round(core_mean, 4),
                "expectancy_std": round(core_std, 4),
                "expectancy_cv": round(core_cv, 4),
                "trend": core_trend,
                "windows": core_windows,
            },
            "staircase_stability": {
                "expectancy_mean": round(stair_mean, 4),
                "expectancy_std": round(stair_std, 4),
                "expectancy_cv": round(stair_cv, 4),
                "trend": stair_trend,
                "windows": stair_windows,
            },
            "amplifier_stability": {
                "avg_contribution": round(amp_mean, 4),
                "contribution_std": round(amp_std, 4),
                "contribution_cv": round(amp_cv, 4),
                "trend": amp_trend,
                "positive_windows": positive_windows,
                "negative_windows": negative_windows,
                "windows": amp_windows,
            },
            "overall_drift_assessment": ", ".join(assessment_parts),
        }

    def cumulative_drift(self, trades: list[dict]) -> dict:
        if not trades:
            return {
                "total_trades": 0,
                "core": {
                    "windows": [],
                    "final_expectancy": 0.0,
                    "convergence_std": 0.0,
                },
                "staircase": {
                    "windows": [],
                    "final_expectancy": 0.0,
                    "convergence_std": 0.0,
                },
                "amplifier": {
                    "windows": [],
                    "final_avg_contribution": 0.0,
                    "convergence_std": 0.0,
                },
            }

        segmented = [_segment_single_trade(t) for t in trades]

        core_windows = []
        stair_windows = []
        amp_windows = []

        for i in range(len(segmented)):
            window_seg = segmented[:i + 1]

            core_pnls = [s["pnl"] for s in window_seg]
            stair_pnls = [s["staircase_weighted_pnl"] for s in window_seg]
            amp_pnls = [s["amplifier_adjusted_pnl"] for s in window_seg]

            core_windows.append({
                "end_index": i,
                "trades": len(window_seg),
                "expectancy": round(_mean(core_pnls), 4),
            })
            stair_windows.append({
                "end_index": i,
                "trades": len(window_seg),
                "expectancy": round(_mean(stair_pnls), 4),
            })
            amp_windows.append({
                "end_index": i,
                "trades": len(window_seg),
                "avg_contribution": round(_mean(amp_pnls), 4),
            })

        core_exps = [w["expectancy"] for w in core_windows]
        stair_exps = [w["expectancy"] for w in stair_windows]
        amp_avgs = [w["avg_contribution"] for w in amp_windows]

        return {
            "total_trades": len(trades),
            "core": {
                "windows": core_windows,
                "final_expectancy": round(core_exps[-1], 4) if core_exps else 0.0,
                "convergence_std": round(_stdev(core_exps), 4),
            },
            "staircase": {
                "windows": stair_windows,
                "final_expectancy": round(stair_exps[-1], 4) if stair_exps else 0.0,
                "convergence_std": round(_stdev(stair_exps), 4),
            },
            "amplifier": {
                "windows": amp_windows,
                "final_avg_contribution": round(amp_avgs[-1], 4) if amp_avgs else 0.0,
                "convergence_std": round(_stdev(amp_avgs), 4),
            },
        }

    def detect_regime_shift(self, trades: list[dict]) -> dict:
        if not trades:
            return {
                "shift_detected": False,
                "shift_cycle": None,
                "shift_magnitude": None,
                "current_running_expectancy": 0.0,
                "historical_mean": 0.0,
                "historical_std": 0.0,
            }

        segmented = [_segment_single_trade(t) for t in trades]
        core_pnls = [s["pnl"] for s in segmented]

        running_exps = []
        for i in range(len(core_pnls)):
            running_exps.append(_mean(core_pnls[:i + 1]))

        current_running = running_exps[-1] if running_exps else 0.0

        if len(core_pnls) < 3:
            return {
                "shift_detected": False,
                "shift_cycle": None,
                "shift_magnitude": None,
                "current_running_expectancy": round(current_running, 4),
                "historical_mean": round(current_running, 4),
                "historical_std": 0.0,
            }

        halfway = len(core_pnls) // 2
        first_half = core_pnls[:halfway]
        second_half = core_pnls[halfway:]

        hist_mean = _mean(first_half)
        hist_std = _stdev(first_half)

        second_mean = _mean(second_half)
        shift_mag = second_mean - hist_mean

        if hist_std == 0.0:
            shift_detected = second_mean != hist_mean
            return {
                "shift_detected": shift_detected,
                "shift_cycle": halfway if shift_detected else None,
                "shift_magnitude": round(shift_mag, 4) if shift_detected else None,
                "current_running_expectancy": round(current_running, 4),
                "historical_mean": round(hist_mean, 4),
                "historical_std": 0.0,
            }

        shift_detected = abs(shift_mag) > 2 * hist_std

        return {
            "shift_detected": shift_detected,
            "shift_cycle": halfway if shift_detected else None,
            "shift_magnitude": round(shift_mag, 4) if shift_detected else None,
            "current_running_expectancy": round(current_running, 4),
            "historical_mean": round(hist_mean, 4),
            "historical_std": round(hist_std, 4),
        }

    def report_all(self, trades: list[dict]) -> dict:
        return {
            "drift_report": self.drift_report(trades),
            "cumulative_drift": self.cumulative_drift(trades),
            "regime_shift": self.detect_regime_shift(trades),
        }


def _print_drift_report(drift: dict) -> None:
    print("\n  Drift Report:")
    print(f"    Total Trades: {drift['total_trades']}")
    print(f"    Window Size:  {drift['window_size']}")
    print(f"    Num Windows:  {drift['num_windows']}")

    c = drift["core_stability"]
    print("\n    CORE Stability:")
    print(f"      Expectancy Mean: {c['expectancy_mean']:.4f}")
    print(f"      Expectancy Std:  {c['expectancy_std']:.4f}")
    print(f"      Expectancy CV:   {c['expectancy_cv']:.4f}")
    print(f"      Trend:           {c['trend']}")

    s = drift["staircase_stability"]
    print("\n    Staircase Stability:")
    print(f"      Expectancy Mean: {s['expectancy_mean']:.4f}")
    print(f"      Expectancy Std:  {s['expectancy_std']:.4f}")
    print(f"      Expectancy CV:   {s['expectancy_cv']:.4f}")
    print(f"      Trend:           {s['trend']}")

    a = drift["amplifier_stability"]
    print("\n    Amplifier Stability:")
    print(f"      Avg Contribution: {a['avg_contribution']:.4f}")
    print(f"      Contribution Std: {a['contribution_std']:.4f}")
    print(f"      Contribution CV:  {a['contribution_cv']:.4f}")
    print(f"      Trend:            {a['trend']}")
    print(f"      Positive Windows: {a['positive_windows']}")
    print(f"      Negative Windows: {a['negative_windows']}")

    print(f"\n    Overall: {drift['overall_drift_assessment']}")


def _print_cumulative_drift(cumul: dict) -> None:
    print("\n  Cumulative Drift:")
    print(f"    Total Trades: {cumul['total_trades']}")
    print(f"    Core Final Expectancy:      {cumul['core']['final_expectancy']:.4f}")
    print(f"    Core Convergence Std:       {cumul['core']['convergence_std']:.4f}")
    print(f"    Staircase Final Expectancy: {cumul['staircase']['final_expectancy']:.4f}")
    print(f"    Staircase Convergence Std:  {cumul['staircase']['convergence_std']:.4f}")
    print(f"    Amplifier Final Avg:        {cumul['amplifier']['final_avg_contribution']:.4f}")
    print(f"    Amplifier Convergence Std:  {cumul['amplifier']['convergence_std']:.4f}")


def _print_regime_shift(regime: dict) -> None:
    print("\n  Regime Shift Detection:")
    print(f"    Shift Detected:           {regime['shift_detected']}")
    if regime["shift_detected"]:
        print(f"    Shift Cycle:              {regime['shift_cycle']}")
        print(f"    Shift Magnitude:          {regime['shift_magnitude']:.4f}")
    print(f"    Current Running Expectancy: {regime['current_running_expectancy']:.4f}")
    print(f"    Historical Mean:           {regime['historical_mean']:.4f}")
    print(f"    Historical Std:            {regime['historical_std']:.4f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    monitor = LayerDriftMonitor()

    trades = monitor.load_trades()
    if not trades:
        logger.warning("No trades loaded, using synthetic data for demonstration")
        trades = [
            {"symbol": "USDJPY", "direction": "SELL", "pnl": 1.41, "volume_composition": {"base_volume": 0.01, "amplifier_multiplier": 1.0, "final_volume": 0.01}, "entry_price": 162.435, "exit_price": 162.294, "reason": "TP_HIT"},
            {"symbol": "EURUSD", "direction": "BUY", "pnl": -0.85, "volume_composition": {"base_volume": 0.01, "amplifier_multiplier": 1.5, "final_volume": 0.015}, "entry_price": 1.1050, "exit_price": 1.1040, "reason": "SL_HIT"},
            {"symbol": "GBPUSD", "direction": "SELL", "pnl": 2.30, "volume_composition": {"base_volume": 0.02, "amplifier_multiplier": 2.0, "final_volume": 0.04}, "entry_price": 1.2500, "exit_price": 1.2480, "reason": "TP_HIT"},
        ]

    report = monitor.report_all(trades)

    print("\n" + "=" * 60)
    print("  Layer Contribution Drift Monitor Report")
    print("=" * 60)

    _print_drift_report(report["drift_report"])
    _print_cumulative_drift(report["cumulative_drift"])
    _print_regime_shift(report["regime_shift"])

    print("\n" + "=" * 60)
    print("  End of Report")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
