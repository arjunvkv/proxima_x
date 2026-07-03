"""Counterfactual Execution Engine — simulates trades under alternate execution physics."""
import sys; sys.path.insert(0, ".")
import random
import math
from research.cep.execution_profile import ExecutionProfile
from research.cep.session_partition import SessionPartitioner


class CounterfactualExecutionEngine:
    SIGNAL_HORIZONS = {"OSS": 5, "TrOSS_cross1": 10, "TrOSS_cross2": 10, "SAL": 5}

    def __init__(self, execution_profile: ExecutionProfile, seed: int = 42):
        self._profile = execution_profile
        self._rng = random.Random(seed)
        self._partitioner = SessionPartitioner()

    def simulate_trade(self, signal: int, entry_ts: int, entry_price: float,
                       exit_prices: list[float], exit_ts: int,
                       signal_name: str = "OSS") -> dict | None:
        # Sample latency
        if self._profile.latency_ms_mean <= 0:
            latency_ms = 0.0
        else:
            latency_ms = self._rng.lognormvariate(
                math.log(self._profile.latency_ms_mean),
                self._profile.latency_ms_std / self._profile.latency_ms_mean
            )
            latency_ms = max(1, min(latency_ms, 500))

        # Queue priority (skip check if perfect)
        if self._profile.queue_priority < 1.0:
            if self._rng.random() > self._profile.queue_priority:
                return None

        # Fill probability
        if self._profile.fill_probability < 1.0:
            if self._rng.random() > self._profile.fill_probability:
                return None

        # Reject
        if self._profile.reject_probability > 0:
            if self._rng.random() < self._profile.reject_probability:
                return None

        # Slippage
        if self._profile.slippage_bps_mean <= 0 and self._profile.slippage_bps_std <= 0:
            slippage = 0.0
        else:
            slippage_bps = self._rng.gauss(
                self._profile.slippage_bps_mean,
                self._profile.slippage_bps_std
            )
            slippage = entry_price * slippage_bps / 10000

        # Spread cost (one-way)
        spread_one_way = entry_price * self._profile.spread_bps / 10000
        spread_round_trip = spread_one_way * 2

        # Entry price with slippage
        exec_price = entry_price + (slippage * signal)

        # Exit: use the price at horizon
        exit_price = exit_prices[-1] if exit_prices else entry_price

        gross_pnl = signal * (exit_price - exec_price)
        net_pnl = gross_pnl - spread_round_trip

        if net_pnl == 0 and gross_pnl != 0:
            net_pnl = gross_pnl - spread_round_trip

        return {
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "signal": signal,
            "signal_name": signal_name,
            "entry_price": round(exec_price, 5),
            "exit_price": round(exit_price, 5),
            "latency_ms": round(latency_ms, 2),
            "slippage": round(slippage, 6),
            "spread_cost": round(spread_round_trip, 6),
            "gross_pnl": round(gross_pnl, 6),
            "net_pnl": round(net_pnl, 6),
            "session": self._partitioner.classify(entry_ts),
        }

    def run_signals(self, signals: list[dict], price_lookup: dict,
                    signal_name: str = "OSS") -> list[dict]:
        trades = []
        for sig in signals:
            ts = sig["ts"]
            horizon = self.SIGNAL_HORIZONS.get(signal_name, 5)
            exit_ts = ts + horizon
            entry_price = sig.get("price", 0)
            exit_prices = self._lookup_exit_prices(price_lookup, ts, exit_ts)
            trade = self.simulate_trade(
                signal=sig["direction"],
                entry_ts=ts,
                entry_price=entry_price,
                exit_prices=exit_prices,
                exit_ts=exit_ts,
                signal_name=signal_name,
            )
            if trade:
                trades.append(trade)
        return trades

    def _lookup_exit_prices(self, price_lookup: dict, entry_ts: int, exit_ts: int) -> list[float]:
        prices = []
        for t in range(entry_ts + 1, exit_ts + 1):
            if t in price_lookup:
                prices.append(price_lookup[t])
        return prices

    @property
    def profile(self):
        return self._profile
