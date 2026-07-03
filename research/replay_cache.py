"""ReplayCache — precompute ECDF, entropy, and raw tick data.

Caches per-tick data (price, ecdf, entropy) so analysis scripts can
run DOA + WFV for any signal mode without rebuilding the environment.
"""
import sys; sys.path.insert(0, '.')
import os
import time
import math
from typing import List, Dict, Optional
from collections import defaultdict

from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from features.ecdf_transform import PerSymbolECDF


CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


class ReplayCache:
    def __init__(self, symbols: List[str], start: str, end: str,
                 tick_limit: int = 100000, seed: int = 42):
        self.symbols = symbols
        self.start = start
        self.end = end
        self.tick_limit = tick_limit
        self.seed = seed
        self._ticks: List[Dict] = []
        sym_str = "_".join(sorted(symbols))
        self._path = os.path.join(CACHE_DIR, f"ticks_{sym_str}_{start}_{end}_{tick_limit}_{seed}.parquet")

    @property
    def is_cached(self) -> bool:
        return os.path.exists(self._path)

    def compute(self, force: bool = False) -> List[Dict]:
        if self.is_cached and not force:
            return self._load()
        return self._compute()

    def _compute(self) -> List[Dict]:
        cfg = ReplayConfig(
            symbols=self.symbols, start=self.start, end=self.end,
            speed=500000, burst=True, latency=False, slippage=False, seed=self.seed,
        )
        env = build_replay_environment(cfg)
        patch_clock(env.clock)
        symbols = list(env.replay_feed._symbols) if hasattr(env, 'replay_feed') else self.symbols

        ecdf = PerSymbolECDF(window_size=2000)
        price_bufs: Dict[str, List[float]] = {}
        ticks: List[Dict] = []
        eval_data: Dict = {}
        for s in symbols:
            eval_data[s] = {"price": 0.0, "ecdf_rank": 0.5, "entropy": 0.5}

        total_ticks = 0
        while total_ticks < self.tick_limit:
            for sym in symbols:
                tick_obj = env.tick_source.get_tick(sym)
                if tick_obj is None:
                    continue
                total_ticks += 1
                price = tick_obj.get("ask", 0) or tick_obj.get("price", 0)
                ecdf_rank = ecdf.update(sym, price)

                # Entropy
                if sym not in price_bufs:
                    price_bufs[sym] = []
                price_bufs[sym].append(price)
                if len(price_bufs[sym]) > 50:
                    price_bufs[sym] = price_bufs[sym][-50:]
                entropy = self._entropy(price_bufs[sym])

                eval_data[sym] = {"price": price, "ecdf_rank": ecdf_rank, "entropy": entropy}

                ticks.append({
                    "sym": sym,
                    "tick_num": total_ticks,
                    "price": price,
                    "ecdf": ecdf_rank,
                    "entropy": entropy,
                })
                if total_ticks >= self.tick_limit:
                    break
            if total_ticks >= self.tick_limit:
                break

        self._ticks = ticks
        self._save()
        return ticks

    def _save(self):
        _ensure_cache_dir()
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
            table = pa.Table.from_pylist(self._ticks)
            pq.write_table(table, self._path, compression="zstd")
        except ImportError:
            pass

    def _load(self) -> List[Dict]:
        import pyarrow.parquet as pq
        self._ticks = pq.read_table(self._path).to_pylist()
        return self._ticks

    def get_ticks(self) -> List[Dict]:
        return self._ticks

    @staticmethod
    def _entropy(prices: list) -> float:
        if len(prices) < 10:
            return 0.5
        mn = min(prices)
        mx = max(prices)
        if mx == mn:
            return 0.0
        nb = 10
        n = len(prices)
        hist = [0] * nb
        for p in prices:
            idx = int((p - mn) / (mx - mn) * nb)
            if idx >= nb:
                idx = nb - 1
            hist[idx] += 1
        ent = 0.0
        for h in hist:
            if h > 0:
                pv = h / n
                ent -= pv * math.log2(pv)
        return min(ent / math.log2(nb), 1.0)


def run_doa_and_wfv(ticks: List[Dict], signal_fn, doa_horizon: int = 20,
                    wfv_train: int = 5, wfv_test: int = 3) -> Dict:
    """Run DOA + WFV on precomputed ticks with a custom signal function."""
    from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
    from validation.wfv_engine import WalkForwardValidator, StatisticalEdgeTest

    doa = DelayedOutcomeEngine(horizon_ticks=doa_horizon)
    wfv_records: List[Dict] = []
    eval_data: Dict = {}

    for t in ticks:
        sym = t["sym"]
        eval_data[sym] = {
            "price": t["price"],
            "ecdf_rank": t["ecdf"],
            "entropy": t["entropy"],
            "signal": signal_fn(t),
        }

        doa.record_snapshot(eval_data)

        if doa.ready:
            current_prices = {s: eval_data[s]["price"] for s in eval_data}
            outcomes = doa.evaluate(current_prices)
            for s, outcome in outcomes.items():
                wfv_records.append({
                    "signal": eval_data[s]["signal"],
                    "outcome": outcome,
                    "ecdf": eval_data[s]["ecdf_rank"],
                    "entropy": eval_data[s]["entropy"],
                })

    wfv = WalkForwardValidator(train_size=wfv_train, test_size=wfv_test).run(wfv_records)
    edge = StatisticalEdgeTest.run(wfv)
    return {
        "accuracy": edge["accuracy"],
        "pnl_proxy": edge["pnl_proxy"],
        "edge_detected": edge["edge_detected"],
        "wfv_records": len(wfv_records),
    }
