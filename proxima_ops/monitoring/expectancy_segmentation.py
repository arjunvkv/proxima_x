import json
import logging
import os
from collections import defaultdict

logger = logging.getLogger("proxima_ops.monitoring.expectancy_segmentation")


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
    amplifier_multiplier = vol_comp.get("amplifier_multiplier", 1.0)

    staircase_weighted = _compute_staircase_weighted(pnl, base_volume, final_volume)
    amplifier_adjusted = _compute_amplifier_contribution(pnl, staircase_weighted)

    return {
        "pnl": pnl,
        "staircase_weighted_pnl": staircase_weighted,
        "amplifier_adjusted_pnl": amplifier_adjusted,
        "base_volume": base_volume,
        "amplifier_multiplier": amplifier_multiplier,
        "final_volume": final_volume,
    }


def _win_rate(pnls: list) -> float:
    if not pnls:
        return 0.0
    return sum(1 for p in pnls if p > 0) / len(pnls)


def _avg(pnls: list) -> float:
    if not pnls:
        return 0.0
    return sum(pnls) / len(pnls)


def _expectancy(pnls: list) -> float:
    return _avg(pnls)


class ExpectancySegmentation:

    def load_trades_from_lifecycle(self, lifecycle_path: str = "state/trade_lifecycle_state.json") -> list:
        if not os.path.exists(lifecycle_path):
            logger.warning("Lifecycle state file not found: %s", lifecycle_path)
            return []
        try:
            with open(lifecycle_path) as f:
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
            logger.info("Loaded %d trades from lifecycle state", len(trades))
            return trades
        except Exception as e:
            logger.error("Failed to load lifecycle state: %s", e)
            return []

    def load_trades_from_trace(self, trace_path: str = "state/live_pipeline_trace.jsonl") -> list:
        if not os.path.exists(trace_path):
            logger.warning("Trace file not found: %s", trace_path)
            return []
        trades = []
        try:
            with open(trace_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    execution = entry.get("execution", {})
                    if execution.get("decision") != "EXECUTE":
                        continue
                    mt5_result = execution.get("mt5_result")
                    if not mt5_result:
                        continue
                    vol_comp = entry.get("volume_composition")
                    if vol_comp is None:
                        base_volume = mt5_result.get("volume", 0.0)
                        vol_comp = {
                            "base_volume": base_volume,
                            "amplifier_multiplier": 1.0,
                            "final_volume": base_volume,
                        }
                    trades.append({
                        "symbol": mt5_result.get("symbol", ""),
                        "direction": mt5_result.get("direction", ""),
                        "pnl": mt5_result.get("pnl", 0.0),
                        "volume_composition": {
                            "base_volume": vol_comp.get("base_volume", 0.0),
                            "amplifier_multiplier": vol_comp.get("amplifier_multiplier", 1.0),
                            "final_volume": vol_comp.get("final_volume", 0.0),
                        },
                        "entry_price": mt5_result.get("entry_price", 0.0),
                        "exit_price": mt5_result.get("exit_price", 0.0),
                        "reason": mt5_result.get("reason", ""),
                    })
            logger.info("Loaded %d trades from pipeline trace", len(trades))
            return trades
        except Exception as e:
            logger.error("Failed to load pipeline trace: %s", e)
            return []

    def segment_report(self, trades: list[dict]) -> dict:
        if not trades:
            return {
                "total_trades": 0,
                "core_only": {"total_pnl": 0.0, "win_rate": 0.0, "avg_trade": 0.0, "expectancy": 0.0},
                "staircase_weighted": {"total_pnl": 0.0, "win_rate": 0.0, "avg_trade": 0.0, "expectancy": 0.0},
                "amplifier_contribution": {"total_pnl_delta": 0.0, "avg_delta_per_trade": 0.0, "positive_contribution_count": 0, "negative_contribution_count": 0},
                "amplifier_net_effect": 0.0,
                "amplifier_efficiency": 0.0,
            }

        segmented = [_segment_single_trade(t) for t in trades]
        total_trades = len(segmented)

        core_pnls = [s["pnl"] for s in segmented]
        stair_pnls = [s["staircase_weighted_pnl"] for s in segmented]
        amp_pnls = [s["amplifier_adjusted_pnl"] for s in segmented]

        core_total = sum(core_pnls)
        stair_total = sum(stair_pnls)
        amp_net = sum(amp_pnls)

        positive_amp = [p for p in amp_pnls if p > 0]
        negative_amp = [p for p in amp_pnls if p < 0]
        pos_count = len(positive_amp)
        neg_count = len(negative_amp)
        pos_total = sum(positive_amp) if positive_amp else 0.0
        neg_total = sum(negative_amp) if negative_amp else 0.0

        total_amp_abs = pos_total + abs(neg_total)
        if total_amp_abs == 0.0:
            amp_efficiency = 0.0
        else:
            amp_efficiency = (pos_total - abs(neg_total)) / total_amp_abs

        return {
            "total_trades": total_trades,
            "core_only": {
                "total_pnl": round(core_total, 4),
                "win_rate": round(_win_rate(core_pnls), 4),
                "avg_trade": round(_avg(core_pnls), 4),
                "expectancy": round(_expectancy(core_pnls), 4),
            },
            "staircase_weighted": {
                "total_pnl": round(stair_total, 4),
                "win_rate": round(_win_rate(stair_pnls), 4),
                "avg_trade": round(_avg(stair_pnls), 4),
                "expectancy": round(_expectancy(stair_pnls), 4),
            },
            "amplifier_contribution": {
                "total_pnl_delta": round(amp_net, 4),
                "avg_delta_per_trade": round(_avg(amp_pnls), 4),
                "positive_contribution_count": pos_count,
                "negative_contribution_count": neg_count,
            },
            "amplifier_net_effect": round(amp_net, 4),
            "amplifier_efficiency": round(amp_efficiency, 4),
        }

    def per_symbol_segmentation(self, trades: list[dict]) -> dict:
        by_symbol = defaultdict(list)
        for t in trades:
            sym = t.get("symbol", "UNKNOWN")
            by_symbol[sym].append(t)

        result = {}
        for sym, sym_trades in sorted(by_symbol.items()):
            result[sym] = self.segment_report(sym_trades)
        return result

    def rolling_segmentation(self, trades: list[dict], window: int = 10) -> list[dict]:
        if not trades or window <= 0:
            return []
        result = []
        for i in range(len(trades)):
            start = max(0, i - window + 1)
            window_trades = trades[start:i + 1]
            report = self.segment_report(window_trades)
            report["index"] = i
            report["window_start"] = start
            report["window_end"] = i
            result.append(report)
        return result


def _print_report(report: dict, label: str = "") -> None:
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  Total Trades: {report['total_trades']}")
    print(f"\n  CORE-only:")
    print(f"    Total PnL:      {report['core_only']['total_pnl']:.4f}")
    print(f"    Win Rate:       {report['core_only']['win_rate']:.4f}")
    print(f"    Avg Trade:      {report['core_only']['avg_trade']:.4f}")
    print(f"    Expectancy:     {report['core_only']['expectancy']:.4f}")
    print(f"\n  Staircase-weighted:")
    print(f"    Total PnL:      {report['staircase_weighted']['total_pnl']:.4f}")
    print(f"    Win Rate:       {report['staircase_weighted']['win_rate']:.4f}")
    print(f"    Avg Trade:      {report['staircase_weighted']['avg_trade']:.4f}")
    print(f"    Expectancy:     {report['staircase_weighted']['expectancy']:.4f}")
    print(f"\n  Amplifier Contribution:")
    print(f"    Total PnL Delta: {report['amplifier_contribution']['total_pnl_delta']:.4f}")
    print(f"    Avg Delta/Trade: {report['amplifier_contribution']['avg_delta_per_trade']:.4f}")
    print(f"    Positive Count:  {report['amplifier_contribution']['positive_contribution_count']}")
    print(f"    Negative Count:  {report['amplifier_contribution']['negative_contribution_count']}")
    print(f"\n  Amplifier Net Effect:  {report['amplifier_net_effect']:.4f}")
    print(f"  Amplifier Efficiency:  {report['amplifier_efficiency']:.4f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    seg = ExpectancySegmentation()

    trades = seg.load_trades_from_lifecycle()
    if not trades:
        trades = seg.load_trades_from_trace()
    if not trades:
        trades = [
            {"symbol": "USDJPY", "direction": "SELL", "pnl": 1.41, "volume_composition": {"base_volume": 0.01, "amplifier_multiplier": 1.0, "final_volume": 0.01}, "entry_price": 162.435, "exit_price": 162.294, "reason": "TP_HIT"},
            {"symbol": "EURUSD", "direction": "BUY", "pnl": -0.85, "volume_composition": {"base_volume": 0.01, "amplifier_multiplier": 1.5, "final_volume": 0.015}, "entry_price": 1.1050, "exit_price": 1.1040, "reason": "SL_HIT"},
        ]

    report = seg.segment_report(trades)
    _print_report(report, "Expectancy Segmentation Report")

    print(f"\n{'='*60}")
    print("  Per-Symbol Segmentation")
    print(f"{'='*60}")
    per_sym = seg.per_symbol_segmentation(trades)
    for sym, sym_report in per_sym.items():
        _print_report(sym_report, f"Symbol: {sym}")

    print(f"\n{'='*60}")
    print("  Rolling Segmentation (window=5)")
    print(f"{'='*60}")
    rolling = seg.rolling_segmentation(trades, window=5)
    for r in rolling:
        print(f"  [{r['index']}] window={r['window_start']}-{r['window_end']}: "
              f"core={r['core_only']['total_pnl']:.4f} "
              f"stair={r['staircase_weighted']['total_pnl']:.4f} "
              f"amp_net={r['amplifier_net_effect']:.4f}")


if __name__ == "__main__":
    main()
