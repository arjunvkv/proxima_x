import json
import logging
import os
import statistics
from collections import defaultdict


logger = logging.getLogger("proxima_ops.monitoring.distribution_observer")

STATE_PATH = "state/distribution_observer_state.json"


AMPLIFIER_BINS = [
    ("<0.7", lambda v: v < 0.7),
    ("0.7-0.9", lambda v: 0.7 <= v < 0.9),
    ("0.9-1.1", lambda v: 0.9 <= v < 1.1),
    ("1.1-1.3", lambda v: 1.1 <= v < 1.3),
    (">1.3", lambda v: v >= 1.3),
]

FINAL_VOLUME_BINS = [
    ("<0.01", lambda v: v < 0.01),
    ("0.01", lambda v: 0.01 <= v < 0.02),
    ("0.02", lambda v: 0.02 <= v < 0.03),
    ("0.03", lambda v: 0.03 <= v < 0.04),
    ("0.04", lambda v: 0.04 <= v < 0.05),
    ("0.05", lambda v: 0.05 <= v < 0.06),
    ("0.06-0.10", lambda v: 0.06 <= v <= 0.10),
    (">0.10", lambda v: v > 0.10),
]

INTER_TRADE_BINS = [
    ("1", lambda g: g == 1),
    ("2-5", lambda g: 2 <= g <= 5),
    ("6-10", lambda g: 6 <= g <= 10),
    ("11-25", lambda g: 11 <= g <= 25),
    ("26-50", lambda g: 26 <= g <= 50),
    ("50+", lambda g: g > 50),
]

RSI_BINS = [
    ("<20", lambda v: v < 20),
    ("20-30", lambda v: 20 <= v < 30),
    ("30-40", lambda v: 30 <= v < 40),
    ("40-60", lambda v: 40 <= v < 60),
    ("60-70", lambda v: 60 <= v < 70),
    ("70-80", lambda v: 70 <= v < 80),
    (">80", lambda v: v >= 80),
]

ATR_PERCENTILE_BINS = [
    ("0-20", lambda v: 0 <= v < 20),
    ("20-40", lambda v: 20 <= v < 40),
    ("40-60", lambda v: 40 <= v < 60),
    ("60-80", lambda v: 60 <= v < 80),
    ("80-100", lambda v: 80 <= v <= 100),
]


def _bin_label(value, bins):
    for label, predicate in bins:
        if predicate(value):
            return label
    return "unknown"


def _compute_rsi(closes: list[float]) -> float:
    """14-period Wilder's RSI returning the latest value as a float."""
    if len(closes) < 15:
        return 50.0
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(diff if diff > 0 else 0.0)
        losses.append(-diff if diff < 0 else 0.0)
    avg_gain = sum(gains[:14]) / 14.0
    avg_loss = sum(losses[:14]) / 14.0
    for i in range(14, len(gains)):
        avg_gain = (avg_gain * 13.0 + gains[i]) / 14.0
        avg_loss = (avg_loss * 13.0 + losses[i]) / 14.0
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


class DistributionObserver:
    def __init__(self, state_path: str = STATE_PATH, burst_threshold: int = 10):
        self._state_path = state_path

        self.base_volume_histogram: dict[str, int] = {}
        self.amplifier_mult_histogram: dict[str, int] = {}
        self.final_volume_histogram: dict[str, int] = {}

        self.phase_counts: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        self.phase_trade_counts: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}

        self.trades_per_symbol: dict[str, int] = defaultdict(int)
        self.volume_per_symbol: dict[str, float] = defaultdict(float)

        self.inter_trade_gaps: list[int] = []
        self.inter_trade_histogram: dict[str, int] = {}

        self.amplifier_effect_trades: list[dict] = []

        self.regime_occupancy: dict[str, dict[str, int]] = {}
        self.regime_trade_counts: dict[str, dict[str, int]] = {}
        self.rsi_bins: dict[str, dict[str, int]] = {}
        self.atr_percentile_bins: dict[str, dict[str, int]] = {}

        self.trade_cycles: list[int] = []
        self._trade_cycle_symbols: dict[int, list[str]] = {}
        self.burst_threshold: int = burst_threshold
        self.bursts: list[dict] = []

        self._last_trade_cycle: int = 0
        self._total_cycles: int = 0
        self._current_phase: int = 1
        self._prev_phase: int = 1
        self._prev_trade_count: int = 0
        self._prev_total_cycles: int = 0

        self._load_state()

    def _log_state_changes(self, cycle_data: dict, trade_executed: bool) -> None:
        changes = []
        phase = self._current_phase
        if phase != self._prev_phase:
            changes.append(f"phase: {self._prev_phase} -> {phase}")
            self._prev_phase = phase
        total_trades = len(self.trade_cycles)
        if total_trades > self._prev_trade_count:
            new_trades = total_trades - self._prev_trade_count
            changes.append(f"trades: {self._prev_trade_count} -> {total_trades} (+{new_trades})")
            self._prev_trade_count = total_trades
        if trade_executed:
            changes.append("trade_executed")
        if changes:
            logger.info("DistributionObserver state change [cycle %d]: %s",
                        cycle_data.get("cycle", self._total_cycles), " | ".join(changes))

    def record_cycle(self, cycle_data: dict) -> None:
        self._total_cycles += 1

        volume_comp = cycle_data.get("volume_composition")
        if volume_comp:
            base_vol = volume_comp.get("base_volume")
            amp_mult = volume_comp.get("amplifier_multiplier")
            final_vol = volume_comp.get("final_volume")

            if base_vol is not None:
                base_key = str(base_vol)
                self.base_volume_histogram[base_key] = self.base_volume_histogram.get(base_key, 0) + 1

            if amp_mult is not None:
                amp_label = _bin_label(amp_mult, AMPLIFIER_BINS)
                self.amplifier_mult_histogram[amp_label] = self.amplifier_mult_histogram.get(amp_label, 0) + 1

            if final_vol is not None:
                fv_label = _bin_label(final_vol, FINAL_VOLUME_BINS)
                self.final_volume_histogram[fv_label] = self.final_volume_histogram.get(fv_label, 0) + 1

        staircase_desc = cycle_data.get("staircase_describe")
        if staircase_desc:
            phase = staircase_desc.get("current_phase", self._current_phase)
            self._current_phase = phase
            if phase in self.phase_counts:
                self.phase_counts[phase] += 1

        execution_result = cycle_data.get("execution_result")
        if execution_result and execution_result.get("success"):
            ticket = execution_result.get("ticket", 0)
            if ticket:
                self.phase_trade_counts[self._current_phase] = self.phase_trade_counts.get(self._current_phase, 0) + 1

        current_cycle = cycle_data.get("cycle", self._total_cycles)
        trade_executed = volume_comp is not None
        if trade_executed and self._last_trade_cycle > 0:
            gap = current_cycle - self._last_trade_cycle
            self.inter_trade_gaps.append(gap)
            gap_label = _bin_label(gap, INTER_TRADE_BINS)
            self.inter_trade_histogram[gap_label] = self.inter_trade_histogram.get(gap_label, 0) + 1

        if trade_executed:
            self._last_trade_cycle = current_cycle

        if volume_comp:
            amp_mult = volume_comp.get("amplifier_multiplier", 1.0)
            if amp_mult != 1.0:
                staircase_vol = volume_comp.get("base_volume", 0)
                final_vol = volume_comp.get("final_volume", 0)
                self.amplifier_effect_trades.append({
                    "cycle": current_cycle,
                    "base_volume": staircase_vol,
                    "amplifier_multiplier": amp_mult,
                    "final_volume": final_vol,
                    "delta": round(final_vol - staircase_vol, 6),
                })

        regime_dashboard = cycle_data.get("regime_dashboard")
        if regime_dashboard:
            for symbol, data in regime_dashboard.items():
                regime_label = data.get("regime") or data.get("regime_label", "unknown")
                if symbol not in self.regime_occupancy:
                    self.regime_occupancy[symbol] = {}
                self.regime_occupancy[symbol][regime_label] = self.regime_occupancy[symbol].get(regime_label, 0) + 1

                rsi_val = data.get("rsi")
                if rsi_val is not None:
                    if symbol not in self.rsi_bins:
                        self.rsi_bins[symbol] = {}
                    rsi_label = _bin_label(rsi_val, RSI_BINS)
                    self.rsi_bins[symbol][rsi_label] = self.rsi_bins[symbol].get(rsi_label, 0) + 1

                atr_pct = data.get("atr_percentile")
                if atr_pct is not None:
                    if symbol not in self.atr_percentile_bins:
                        self.atr_percentile_bins[symbol] = {}
                    atr_label = _bin_label(atr_pct, ATR_PERCENTILE_BINS)
                    self.atr_percentile_bins[symbol][atr_label] = self.atr_percentile_bins[symbol].get(atr_label, 0) + 1

        self._log_state_changes(cycle_data, trade_executed)
        self._save_state()

    def record_trade(self, trade_data: dict) -> None:
        if not trade_data:
            return
        symbol = trade_data.get("symbol", "")
        volume = trade_data.get("volume", 0.0)
        cycle = trade_data.get("cycle", self._total_cycles)
        regime_label = trade_data.get("regime") or trade_data.get("regime_label", "")

        if symbol:
            self.trades_per_symbol[symbol] += 1
            self.volume_per_symbol[symbol] += volume

            if regime_label:
                if symbol not in self.regime_trade_counts:
                    self.regime_trade_counts[symbol] = {}
                self.regime_trade_counts[symbol][regime_label] = self.regime_trade_counts[symbol].get(regime_label, 0) + 1

        if cycle > 0:
            self.trade_cycles.append(cycle)
            if cycle not in self._trade_cycle_symbols:
                self._trade_cycle_symbols[cycle] = []
            if symbol and symbol not in self._trade_cycle_symbols[cycle]:
                self._trade_cycle_symbols[cycle].append(symbol)
            self._detect_bursts()

        self._save_state()

    def _detect_bursts(self) -> None:
        sorted_cycles = sorted(self.trade_cycles) if self.trade_cycles else []
        if len(sorted_cycles) == 1:
            syms = sorted(self._trade_cycle_symbols.get(sorted_cycles[0], []))
            self.bursts = [{"start_cycle": sorted_cycles[0], "end_cycle": sorted_cycles[0], "trade_count": 1, "symbols_involved": syms}]
            return
        if not sorted_cycles:
            self.bursts = []
            return
        bursts = []
        start = sorted_cycles[0]
        symbols = set(self._trade_cycle_symbols.get(start, []))
        count = 1
        for i in range(1, len(sorted_cycles)):
            gap = sorted_cycles[i] - sorted_cycles[i - 1]
            if gap <= self.burst_threshold:
                count += 1
                symbols.update(self._trade_cycle_symbols.get(sorted_cycles[i], []))
            else:
                bursts.append({"start_cycle": start, "end_cycle": sorted_cycles[i - 1], "trade_count": count, "symbols_involved": sorted(symbols)})
                start = sorted_cycles[i]
                symbols = set(self._trade_cycle_symbols.get(start, []))
                count = 1
        bursts.append({"start_cycle": start, "end_cycle": sorted_cycles[-1], "trade_count": count, "symbols_involved": sorted(symbols)})
        self.bursts = bursts

    def regime_coverage_report(self) -> dict:
        total_cycles = self._total_cycles if self._total_cycles > 0 else 1
        regime_aggregate = {}
        for symbol, regimes in self.regime_occupancy.items():
            for regime_label, count in regimes.items():
                regime_aggregate[regime_label] = regime_aggregate.get(regime_label, 0) + count
        regime_occupancy_pct = {k: f"{round(v / total_cycles * 100, 1)}%" for k, v in regime_aggregate.items()}
        total_trades = sum(self.trades_per_symbol.values())
        regime_trade_aggregate = {}
        for symbol, regimes in self.regime_trade_counts.items():
            for regime_label, count in regimes.items():
                regime_trade_aggregate[regime_label] = regime_trade_aggregate.get(regime_label, 0) + count
        regime_trade_density = {}
        for regime_label, trade_count in regime_trade_aggregate.items():
            regime_cycles = regime_aggregate.get(regime_label, 1)
            density = round(trade_count / regime_cycles * 100, 1) if regime_cycles > 0 else 0.0
            regime_trade_density[regime_label] = f"{density}%"
        rsi_all_bins = {}
        for symbol, bins in self.rsi_bins.items():
            for bin_label, count in bins.items():
                rsi_all_bins[bin_label] = rsi_all_bins.get(bin_label, 0) + count
        rsi_extreme = sum(rsi_all_bins.get(b, 0) for b in ("<20", ">80"))
        rsi_total = sum(rsi_all_bins.values())
        rsi_extreme_rate = f"{round(rsi_extreme / rsi_total * 100, 1)}%" if rsi_total > 0 else "0%"
        atr_all_bins = {}
        for symbol, bins in self.atr_percentile_bins.items():
            for bin_label, count in bins.items():
                atr_all_bins[bin_label] = atr_all_bins.get(bin_label, 0) + count
        atr_spike = atr_all_bins.get("80-100", 0)
        atr_total = sum(atr_all_bins.values())
        atr_spike_rate = f"{round(atr_spike / atr_total * 100, 1)}%" if atr_total > 0 else "0%"
        per_symbol = {}
        for symbol in set(list(self.regime_occupancy.keys()) + list(self.regime_trade_counts.keys()) + list(self.rsi_bins.keys()) + list(self.atr_percentile_bins.keys())):
            entry = {}
            if symbol in self.regime_occupancy:
                entry["regime_occupancy"] = dict(self.regime_occupancy[symbol])
            if symbol in self.regime_trade_counts:
                entry["regime_trade_counts"] = dict(self.regime_trade_counts[symbol])
            if symbol in self.rsi_bins:
                entry["rsi_bins"] = dict(self.rsi_bins[symbol])
            if symbol in self.atr_percentile_bins:
                entry["atr_percentile_bins"] = dict(self.atr_percentile_bins[symbol])
            per_symbol[symbol] = entry
        return {
            "total_cycles_tracked": self._total_cycles,
            "regime_occupancy": regime_occupancy_pct,
            "regime_trade_density": regime_trade_density,
            "rsi_extreme_rate": rsi_extreme_rate,
            "atr_spike_rate": atr_spike_rate,
            "per_symbol": per_symbol,
        }

    def burst_report(self) -> dict:
        total_trades = len(self.trade_cycles)
        total_bursts = len([b for b in self.bursts if b["trade_count"] > 1])
        isolated_trades = len([b for b in self.bursts if b["trade_count"] == 1])
        burst_trades = sum(b["trade_count"] for b in self.bursts if b["trade_count"] > 1)
        burst_density = f"{round(burst_trades / total_trades * 100, 1)}%" if total_trades > 0 else "0%"
        avg_trades_per_burst = round(burst_trades / total_bursts, 1) if total_bursts > 0 else 0.0
        burst_list = []
        for b in self.bursts:
            entry = {
                "start_cycle": b["start_cycle"],
                "end_cycle": b["end_cycle"],
                "trade_count": b["trade_count"],
                "symbols_involved": b.get("symbols_involved", []),
            }
            burst_list.append(entry)
        return {
            "total_bursts": total_bursts,
            "isolated_trades": isolated_trades,
            "avg_trades_per_burst": avg_trades_per_burst,
            "burst_density": burst_density,
            "bursts": burst_list,
        }

    def inter_trade_gap_report(self) -> dict:
        if not self.inter_trade_gaps:
            return {
                "mean_gap_cycles": 0.0,
                "median_gap_cycles": 0.0,
                "min_gap": 0,
                "max_gap": 0,
                "histogram": {},
            }
        mean_gap = round(statistics.mean(self.inter_trade_gaps), 1)
        median_gap = round(statistics.median(self.inter_trade_gaps), 1)
        min_gap = min(self.inter_trade_gaps)
        max_gap = max(self.inter_trade_gaps)
        histogram = {}
        for label, _ in INTER_TRADE_BINS:
            histogram[label] = self.inter_trade_histogram.get(label, 0)
        return {
            "mean_gap_cycles": mean_gap,
            "median_gap_cycles": median_gap,
            "min_gap": min_gap,
            "max_gap": max_gap,
            "histogram": histogram,
        }

    def distribution_report(self) -> dict:
        return {
            "total_cycles_observed": self._total_cycles,
            "total_trades": sum(self.trades_per_symbol.values()),
            "base_volume_histogram": dict(self.base_volume_histogram),
            "amplifier_mult_histogram": dict(self.amplifier_mult_histogram),
            "final_volume_histogram": dict(self.final_volume_histogram),
            "phase_counts": dict(self.phase_counts),
            "phase_trade_counts": dict(self.phase_trade_counts),
            "trades_per_symbol": dict(self.trades_per_symbol),
            "volume_per_symbol": {k: round(v, 4) for k, v in self.volume_per_symbol.items()},
            "inter_trade_gaps": list(self.inter_trade_gaps),
            "inter_trade_histogram": dict(self.inter_trade_histogram),
            "amplifier_effect_trades": list(self.amplifier_effect_trades),
            "regime_occupancy": {k: dict(v) for k, v in self.regime_occupancy.items()},
            "regime_trade_counts": {k: dict(v) for k, v in self.regime_trade_counts.items()},
            "rsi_bins": {k: dict(v) for k, v in self.rsi_bins.items()},
            "atr_percentile_bins": {k: dict(v) for k, v in self.atr_percentile_bins.items()},
            "trade_cycles": list(self.trade_cycles),
            "bursts": list(self.bursts),
            "burst_threshold": self.burst_threshold,
        }

    def reset(self) -> None:
        self.base_volume_histogram.clear()
        self.amplifier_mult_histogram.clear()
        self.final_volume_histogram.clear()
        self.phase_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        self.phase_trade_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        self.trades_per_symbol.clear()
        self.volume_per_symbol.clear()
        self.inter_trade_gaps.clear()
        self.inter_trade_histogram.clear()
        self.amplifier_effect_trades.clear()
        self.regime_occupancy.clear()
        self.regime_trade_counts.clear()
        self.rsi_bins.clear()
        self.atr_percentile_bins.clear()
        self.trade_cycles.clear()
        self._trade_cycle_symbols.clear()
        self.bursts.clear()
        self._last_trade_cycle = 0
        self._total_cycles = 0
        self._current_phase = 1
        self._save_state()
        logger.debug("DistributionObserver state reset")

    def _load_state(self) -> None:
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as f:
                data = json.load(f)

            self.base_volume_histogram = data.get("base_volume_histogram", {})
            self.base_volume_histogram = {str(k): int(v) for k, v in self.base_volume_histogram.items()}

            self.amplifier_mult_histogram = data.get("amplifier_mult_histogram", {})
            self.amplifier_mult_histogram = {str(k): int(v) for k, v in self.amplifier_mult_histogram.items()}

            self.final_volume_histogram = data.get("final_volume_histogram", {})
            self.final_volume_histogram = {str(k): int(v) for k, v in self.final_volume_histogram.items()}

            raw_pc = data.get("phase_counts", {})
            self.phase_counts = {int(k): int(v) for k, v in raw_pc.items()}
            if 1 not in self.phase_counts:
                self.phase_counts[1] = 0
            if 2 not in self.phase_counts:
                self.phase_counts[2] = 0
            if 3 not in self.phase_counts:
                self.phase_counts[3] = 0
            if 4 not in self.phase_counts:
                self.phase_counts[4] = 0

            raw_ptc = data.get("phase_trade_counts", {})
            self.phase_trade_counts = {int(k): int(v) for k, v in raw_ptc.items()}
            if 1 not in self.phase_trade_counts:
                self.phase_trade_counts[1] = 0
            if 2 not in self.phase_trade_counts:
                self.phase_trade_counts[2] = 0
            if 3 not in self.phase_trade_counts:
                self.phase_trade_counts[3] = 0
            if 4 not in self.phase_trade_counts:
                self.phase_trade_counts[4] = 0

            self.trades_per_symbol = defaultdict(int, data.get("trades_per_symbol", {}))
            self.volume_per_symbol = defaultdict(float, data.get("volume_per_symbol", {}))

            self.inter_trade_gaps = data.get("inter_trade_gaps", [])
            self.inter_trade_histogram = data.get("inter_trade_histogram", {})
            self.inter_trade_histogram = {str(k): int(v) for k, v in self.inter_trade_histogram.items()}

            self.amplifier_effect_trades = data.get("amplifier_effect_trades", [])

            self.regime_occupancy = data.get("regime_occupancy", {})
            self.regime_occupancy = {str(k): {str(rk): int(rv) for rk, rv in v.items()} for k, v in self.regime_occupancy.items()}

            self.regime_trade_counts = data.get("regime_trade_counts", {})
            self.regime_trade_counts = {str(k): {str(rk): int(rv) for rk, rv in v.items()} for k, v in self.regime_trade_counts.items()}

            self.rsi_bins = data.get("rsi_bins", {})
            self.rsi_bins = {str(k): {str(bk): int(bv) for bk, bv in v.items()} for k, v in self.rsi_bins.items()}

            self.atr_percentile_bins = data.get("atr_percentile_bins", {})
            self.atr_percentile_bins = {str(k): {str(bk): int(bv) for bk, bv in v.items()} for k, v in self.atr_percentile_bins.items()}

            self.trade_cycles = data.get("trade_cycles", [])
            self.trade_cycles = [int(c) for c in self.trade_cycles]

            raw_cycle_symbols = data.get("_trade_cycle_symbols", {})
            self._trade_cycle_symbols = {int(k): list(v) for k, v in raw_cycle_symbols.items()}

            self.burst_threshold = data.get("burst_threshold", 10)
            self.bursts = data.get("bursts", [])
            self.bursts = [{str(k): v for k, v in b.items()} for b in self.bursts]

            self._last_trade_cycle = data.get("_last_trade_cycle", 0)
            self._total_cycles = data.get("_total_cycles", 0)
            self._current_phase = data.get("_current_phase", 1)

            logger.debug("DistributionObserver state loaded from %s (%d cycles)", self._state_path, self._total_cycles)
        except Exception as e:
            logger.warning("Failed to load DistributionObserver state: %s", e)

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
            data = {
                "base_volume_histogram": self.base_volume_histogram,
                "amplifier_mult_histogram": self.amplifier_mult_histogram,
                "final_volume_histogram": self.final_volume_histogram,
                "phase_counts": {str(k): v for k, v in self.phase_counts.items()},
                "phase_trade_counts": {str(k): v for k, v in self.phase_trade_counts.items()},
                "trades_per_symbol": dict(self.trades_per_symbol),
                "volume_per_symbol": dict(self.volume_per_symbol),
                "inter_trade_gaps": self.inter_trade_gaps,
                "inter_trade_histogram": self.inter_trade_histogram,
                "amplifier_effect_trades": self.amplifier_effect_trades,
                "regime_occupancy": self.regime_occupancy,
                "regime_trade_counts": self.regime_trade_counts,
                "rsi_bins": self.rsi_bins,
                "atr_percentile_bins": self.atr_percentile_bins,
                "trade_cycles": self.trade_cycles,
                "_trade_cycle_symbols": {str(k): v for k, v in self._trade_cycle_symbols.items()},
                "bursts": self.bursts,
                "burst_threshold": self.burst_threshold,
                "_last_trade_cycle": self._last_trade_cycle,
                "_total_cycles": self._total_cycles,
                "_current_phase": self._current_phase,
            }
            with open(self._state_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("DistributionObserver state saved to %s", self._state_path)
        except Exception as e:
            logger.warning("Failed to save DistributionObserver state: %s", e)
