import json, os, random, sys, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from run_proxima_shadow_replay import ShadowReplayRunner, SUCCESS_CRITERIA
from research.ucf.replay.execution_model.execution_simulator import ExecutionSimulator


class ShadowReplayRunnerStress(ShadowReplayRunner):
    def __init__(self, mode: str = "synthetic") -> None:
        super().__init__(mode=mode)
        self._adversarial_fill_flags: dict[str, bool] = {}

    def _inject_slippage_spike(self, fill: dict) -> dict:
        if random.random() < 0.15:
            fill["slippage"] = fill.get("slippage", 0.0001) * random.uniform(3.0, 5.0)
            fill["net_pnl"] = fill.get("net_pnl", 0.0) * (1.0 - random.uniform(0.1, 0.3))
        return fill

    def _inject_delayed_fill(self, fill: dict) -> dict:
        if random.random() < 0.10:
            fill["latency"] = fill.get("latency", 0.05) * random.uniform(5.0, 10.0)
            fill["net_pnl"] = fill.get("net_pnl", 0.0) * (1.0 - random.uniform(0.05, 0.15))
        return fill

    def _inject_partial_fill(self, fill: dict) -> dict:
        if random.random() < 0.08:
            fill["filled_ratio"] = random.uniform(0.3, 0.7)
            fill["net_pnl"] = fill.get("net_pnl", 0.0) * fill["filled_ratio"]
        return fill

    def _inject_atr_shock(self, symbol: str, base_price: float, base_spread: float) -> tuple[float, float]:
        if random.random() < 0.05:
            return base_price * (1.0 + random.uniform(-0.03, 0.03)), base_spread * random.uniform(3.0, 5.0)
        return base_price, base_spread

    def run_stress_test(self, test_type: str) -> dict:
        print(f"[STRESS_TEST] type={test_type}")
        self.generate_ticks(5000)
        result = self.orchestrator.run_replay(batch_size=100)
        logs = result.get("logs", [])
        summary = result.get("summary", {})

        base_prices = {"EURUSD": 1.10, "AUDUSD": 0.72, "GBPUSD": 1.25,
                       "USDJPY": 110.0, "USDCHF": 0.92, "USDCAD": 1.35, "NZDUSD": 0.68}

        replay_records: list[dict] = []
        fill_records: list[dict] = []

        for cycle_idx, log_entry in enumerate(logs):
            ranked = log_entry.get("ranked_symbols", [])
            if not ranked:
                continue
            top = ranked[0]
            symbol = top.get("symbol", "EURUSD")
            direction = top.get("direction", 0)
            ucf_score = top.get("ucf_score", 0.0)

            base_price = base_prices.get(symbol, 1.10)
            base_spread = random.uniform(0.0001, 0.0005)

            price, spread = base_price, base_spread
            if test_type == "regime_shock":
                price, spread = self._inject_atr_shock(symbol, base_price, base_spread)

            self._mra.update(symbol, price, spread)
            self._recovery.update_rv(symbol, random.uniform(0.2, 0.8))
            self._recovery.update_rc(symbol, random.uniform(0.3, 0.9))
            reg_vol = self._mra.get_regime_volatility(symbol)
            self._recovery.set_regime_volatility(symbol, reg_vol)
            dampen = reg_vol > 0.7
            gate_dec = self._recovery.resolve(symbol)
            gate_dec["symbol"] = symbol
            gate_dec["cycle"] = cycle_idx
            gate_dec["mra"] = self._mra.get_mra(symbol, dampen=dampen)
            gate_dec["emd"] = self._emd.get_emd(symbol, dampen=dampen)
            self._gate_decisions.append(gate_dec)

            replay_records.append({
                "cycle": cycle_idx, "symbol": symbol, "direction": direction,
                "confidence": ucf_score, "regime": log_entry.get("regime", "neutral"),
            })

            if direction == 0 or ucf_score < 0.01:
                continue

            entry_price = base_price + random.uniform(-0.001, 0.001)
            spread_val = random.uniform(1, 5) * 0.0001
            volatility = random.uniform(0.0001, 0.001)
            exit_price = entry_price + direction * abs(ucf_score) * volatility * 10.0

            if test_type == "correlation_break":
                exit_price = entry_price - direction * abs(ucf_score) * volatility * 5.0

            entry_r = self.execution_sim.simulate_entry(entry_price, direction, spread_val, volatility)
            exit_r = self.execution_sim.simulate_exit(entry_price, exit_price, direction, spread_val, volatility)
            fill = self.execution_sim.simulate_trade(entry_r, exit_r)
            fill["symbol"] = symbol
            fill["cycle"] = cycle_idx
            fill["direction"] = direction

            if test_type == "execution_stress":
                fill = self._inject_slippage_spike(fill)
                fill = self._inject_delayed_fill(fill)
                fill = self._inject_partial_fill(fill)

            if test_type == "position_feedback":
                base_risk = abs(fill.get("net_pnl", 0.0))
                if fill.get("net_pnl", 0) > 0:
                    pass

            if test_type == "correlation_break":
                fill["net_pnl"] = fill.get("net_pnl", 0.0) * -1.0

            fill["pnl"] = fill.get("net_pnl", 0.0)
            fill_records.append(fill)
            self.pnl_tracker.record_trade(fill)

        self.results_log = replay_records
        self.fill_log = fill_records
        report = self._build_report(result, replay_records, fill_records)
        self._save_report(report)
        self._print_summary(report, summary)
        report["test_type"] = test_type
        return report


ACCEPTANCE_THRESHOLDS = {
    "execution_stress": {"alignment_min": 0.50, "pnl_corr_min": 0.40, "gate_pass_min": 0.70},
    "regime_shock": {"alignment_min": 0.45, "pnl_corr_min": 0.35, "gate_pass_min": 0.65},
    "correlation_break": {"alignment_min": 0.30, "pnl_corr_min": 0.10, "gate_pass_min": 0.60},
    "position_feedback": {"alignment_min": 0.45, "pnl_corr_min": 0.35, "gate_pass_min": 0.70},
}


def run_acceptance_suite() -> dict:
    results: dict[str, dict] = {}
    for test in ["execution_stress", "regime_shock", "correlation_break", "position_feedback"]:
        print(f"\n{'=' * 60}")
        print(f"[ACCEPTANCE] Running test: {test}")
        print(f"{'=' * 60}")
        runner = ShadowReplayRunnerStress(mode="synthetic")
        report = runner.run_stress_test(test)
        thresholds = ACCEPTANCE_THRESHOLDS[test]
        alignment = report.get("alignment_score", 0)
        pnl_corr = report.get("pnl_correlation", 0)
        gate_pass = report.get("gate_metrics", {}).get("gate_pass_rate", 0)
        align_pass = alignment >= thresholds["alignment_min"]
        corr_pass = pnl_corr >= thresholds["pnl_corr_min"]
        gate_pass_ok = gate_pass >= thresholds["gate_pass_min"]
        overall = align_pass and corr_pass and gate_pass_ok
        results[test] = {
            "alignment": alignment,
            "pnl_correlation": pnl_corr,
            "gate_pass_rate": gate_pass,
            "alignment_pass": align_pass,
            "pnl_corr_pass": corr_pass,
            "gate_pass_ok": gate_pass_ok,
            "overall": overall,
        }
        print(f"  Alignment:    {alignment:.4f} >= {thresholds['alignment_min']} -> {'PASS' if align_pass else 'FAIL'}")
        print(f"  PnL Corr:     {pnl_corr:.4f} >= {thresholds['pnl_corr_min']} -> {'PASS' if corr_pass else 'FAIL'}")
        print(f"  Gate Pass:    {gate_pass:.2%} >= {thresholds['gate_pass_min']:.0%} -> {'PASS' if gate_pass_ok else 'FAIL'}")
        print(f"  OVERALL:      {'PASS' if overall else 'FAIL'}")

    summary = {
        "all_pass": all(r["overall"] for r in results.values()),
        "results": results,
    }
    out_path = os.path.join(os.path.dirname(__file__), "acceptance_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"[ACCEPTANCE] {'ALL PASS' if summary['all_pass'] else 'SOME FAILED'}")
    print(f"{'=' * 60}")
    return summary


if __name__ == "__main__":
    run_acceptance_suite()
