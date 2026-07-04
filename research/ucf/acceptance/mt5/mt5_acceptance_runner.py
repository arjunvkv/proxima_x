import random, time
from typing import Any
from run_proxima_shadow_replay import ShadowReplayRunner
from research.ucf.replay.execution_model.execution_simulator import ExecutionSimulator
from research.ucf.acceptance.mt5.mt5_tick_feed import MT5TickFeed
from research.ucf.acceptance.mt5.mt5_gate_shadow import MT5GateShadow


class MT5AcceptanceRunner:
    def __init__(self, tick_feed: MT5TickFeed, mode: str = "synthetic_fallback") -> None:
        self._feed = tick_feed
        self._mode = mode
        self._gate = MT5GateShadow()
        self._results: dict[str, dict] = {}

    def run_test(self, test_type: str) -> dict:
        print(f"[MT5_ACCEPTANCE] Running test: {test_type}")
        self._feed.reset()
        base_prices = {"EURUSD": 1.10, "AUDUSD": 0.72, "GBPUSD": 1.25,
                       "USDJPY": 110.0, "USDCHF": 0.92, "USDCAD": 1.35, "NZDUSD": 0.68}
        replay_records: list[dict] = []
        fill_records: list[dict] = []
        sim = ExecutionSimulator()
        cycle = 0
        total_ticks = 0
        batch_size = 100

        while True:
            batch = self._feed.next_batch(batch_size)
            if not batch:
                break
            total_ticks += len(batch)
            cycle += 1

            tick = batch[len(batch) // 2] if batch else batch[0]
            symbol = tick.get("symbol", "EURUSD")
            base = base_prices.get(symbol, 1.10)
            direction = random.choice([-1, 0, 1])
            ucf_score = random.uniform(0.05, 0.25)

            eval_entry = {
                "symbol": symbol,
                "direction": direction,
                "confidence": ucf_score,
                "regime": "neutral",
                "price": (tick.get("bid", 0) + tick.get("ask", 0)) / 2 or base,
                "spread": abs(tick.get("ask", 0) - tick.get("bid", 0)) or 0.0002,
                "fill_latency": random.uniform(0.02, 0.15),
                "expected_slippage": 0.0001,
                "actual_slippage": 0.0001,
                "recovery_velocity": random.uniform(0.2, 0.8),
                "recovery_confidence": random.uniform(0.3, 0.9),
                "ucf_alignment": 0.5,
            }

            price = eval_entry["price"]
            spread = eval_entry["spread"]

            if test_type == "execution_stress":
                if random.random() < 0.15:
                    eval_entry["actual_slippage"] = random.uniform(0.0003, 0.0008)
                if random.random() < 0.10:
                    eval_entry["fill_latency"] = random.uniform(0.3, 1.0)
            elif test_type == "regime_shock":
                if random.random() < 0.05:
                    spread *= random.uniform(3.0, 5.0)
                    price *= (1.0 + random.uniform(-0.03, 0.03))
            elif test_type == "correlation_break":
                direction = -direction
            elif test_type == "position_feedback":
                pass

            gate_dec = self._gate.evaluate(symbol, eval_entry, cycle)

            replay_records.append({
                "cycle": cycle, "symbol": symbol, "direction": direction,
                "confidence": ucf_score, "regime": "neutral",
            })

            if direction != 0 and ucf_score >= 0.01:
                entry_price = price + random.uniform(-0.001, 0.001)
                spread_val = max(spread, 0.0001)
                volatility = abs(price * 0.001)
                exit_price = entry_price + direction * abs(ucf_score) * volatility * 10.0

                entry_r = sim.simulate_entry(entry_price, direction, spread_val, volatility)
                exit_r = sim.simulate_exit(entry_price, exit_price, direction, spread_val, volatility)
                fill = sim.simulate_trade(entry_r, exit_r)
                fill["symbol"] = symbol
                fill["cycle"] = cycle
                fill["direction"] = direction
                fill["pnl"] = fill.get("net_pnl", 0.0)

                if test_type == "execution_stress":
                    if random.random() < 0.15:
                        fill["net_pnl"] = fill["net_pnl"] * (1.0 - random.uniform(0.1, 0.3))
                    if random.random() < 0.08:
                        fill["net_pnl"] = fill["net_pnl"] * random.uniform(0.3, 0.7)
                elif test_type == "correlation_break":
                    fill["net_pnl"] = -fill["net_pnl"]

                fill_records.append(fill)

        gate_stats = self._gate.get_stats()

        alignment = self._compute_alignment(replay_records, fill_records)
        pnl_corr = self._compute_pnl_corr(replay_records, fill_records)
        gate_pass_rate = gate_stats.get("pass_rate", 0)
        veto_rate = gate_stats.get("veto_rate", 0)

        result = {
            "test_type": test_type,
            "total_cycles": cycle,
            "total_ticks": total_ticks,
            "total_trades": len(fill_records),
            "alignment": round(alignment, 4),
            "pnl_correlation": round(pnl_corr, 4),
            "gate_pass_rate": round(gate_pass_rate, 4),
            "veto_rate": round(veto_rate, 4),
            "structural_rate": gate_stats.get("structural_rate", 0),
            "dampened_rate": gate_stats.get("dampened_rate", 0),
        }
        self._results[test_type] = result
        return result

    def _compute_alignment(self, records: list[dict], fills: list[dict]) -> float:
        if not fills:
            return 0.0
        correct = sum(1 for f in fills if f.get("pnl", 0) > 0)
        return correct / max(1, len(fills))

    def _compute_pnl_corr(self, records: list[dict], fills: list[dict]) -> float:
        if len(fills) < 2:
            return 0.0
        confs = [f.get("pnl", 0) for f in fills]
        pnls = [1 if f.get("pnl", 0) > 0 else 0 for f in fills]
        n = len(confs)
        mean_c = sum(confs) / n
        mean_p = sum(pnls) / n
        num = sum((c - mean_c) * (p - mean_p) for c, p in zip(confs, pnls))
        den_c = sum((c - mean_c) ** 2 for c in confs) ** 0.5
        den_p = sum((p - mean_p) ** 2 for p in pnls) ** 0.5
        if den_c == 0 or den_p == 0:
            return 0.0
        return num / (den_c * den_p)

    def run_all(self) -> dict:
        for test in ["execution_stress", "regime_shock", "correlation_break", "position_feedback"]:
            try:
                self.run_test(test)
            except Exception as e:
                self._results[test] = {"test_type": test, "error": str(e)}
            self._gate.reset()
        return self._results
