"""CrossScaleCache — align tick states to forward M1/M5/M15 excursion."""
import sys; sys.path.insert(0, '.')
import os
import math
import time
from collections import defaultdict
from typing import List, Dict, Optional

from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from features.ecdf_transform import PerSymbolECDF

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")

SECONDS_IN_MINUTE = 60

class CrossScaleCache:
    """Tick cache with timestamps and forward bar returns.

    Each tick record includes:
      - sym, tick_num, price, ecdf, entropy (as before)
      - time_sec: UNIX timestamp
      - mid: (bid+ask)/2
      - fwd_1m, fwd_5m, fwd_15m: price change over next N minutes
      - fwd_1m_dir, fwd_5m_dir, fwd_15m_dir: direction of forward move
    """
    def __init__(self, symbols: List[str], start: str, end: str,
                 tick_limit: int = 100000, seed: int = 42):
        self.symbols = symbols
        self.start = start
        self.end = end
        self.tick_limit = tick_limit
        self.seed = seed
        self._ticks: List[Dict] = []

    def compute(self, force: bool = False) -> List[Dict]:
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
        total_ticks = 0

        while total_ticks < self.tick_limit:
            for sym in symbols:
                tick_obj = env.tick_source.get_tick(sym)
                if tick_obj is None:
                    continue
                total_ticks += 1

                time_sec = tick_obj.get("time_sec", 0)
                bid = tick_obj.get("bid", 0)
                ask = tick_obj.get("ask", 0)
                mid = (bid + ask) / 2 if bid and ask else tick_obj.get("price", mid)
                price = tick_obj.get("ask", 0) or tick_obj.get("price", 0)
                ecdf_rank = ecdf.update(sym, price)

                if sym not in price_bufs:
                    price_bufs[sym] = []
                price_bufs[sym].append(price)
                if len(price_bufs[sym]) > 50:
                    price_bufs[sym] = price_bufs[sym][-50:]
                entropy = self._entropy(price_bufs[sym])

                ticks.append({
                    "sym": sym,
                    "tick_num": total_ticks,
                    "time_sec": time_sec,
                    "price": price,
                    "mid": mid,
                    "ecdf": ecdf_rank,
                    "entropy": entropy,
                    "fwd_1m": 0.0, "fwd_5m": 0.0, "fwd_15m": 0.0,
                    "fwd_1m_dir": 0, "fwd_5m_dir": 0, "fwd_15m_dir": 0,
                })
                if total_ticks >= self.tick_limit:
                    break
            if total_ticks >= self.tick_limit:
                break

        self._compute_forward_returns(ticks)
        self._ticks = ticks
        return ticks

    def _compute_forward_returns(self, ticks: List[Dict]):
        """For each tick, compute forward price change over N minutes."""
        horizons = {"fwd_1m": 1, "fwd_5m": 5, "fwd_15m": 15}

        # Group ticks by symbol
        by_sym = defaultdict(list)
        for t in ticks:
            by_sym[t["sym"]].append(t)

        for sym, sts in by_sym.items():
            # For each horizon
            for key, minutes in horizons.items():
                delta_sec = minutes * SECONDS_IN_MINUTE
                dir_key = key.replace("fwd_", "fwd_") + "_dir"
                dir_key = key.replace("1m", "1m_dir").replace("5m", "5m_dir").replace("15m", "15m_dir")

                # For each tick, scan forward to find a tick at horizon
                j = 0
                for i in range(len(sts)):
                    base_time = sts[i]["time_sec"]
                    target_time = base_time + delta_sec

                    # Advance j until we pass target_time
                    while j < len(sts) and sts[j]["time_sec"] < target_time and sts[j]["tick_num"] <= sts[i]["tick_num"]:
                        j += 1
                    # Scan forward from max(i+1, j) to find target
                    k = max(i + 1, j)
                    while k < len(sts) and sts[k]["time_sec"] < target_time:
                        k += 1

                    if k < len(sts) and sts[k]["time_sec"] >= target_time:
                        fwd = sts[k]["price"] - sts[i]["price"]
                    elif k - 1 > i and k - 1 < len(sts):
                        fwd = sts[k - 1]["price"] - sts[i]["price"]
                    else:
                        fwd = 0.0

                    sts[i][key] = fwd
                    sts[i][dir_key] = 1 if fwd > 0 else (-1 if fwd < 0 else 0)

    @staticmethod
    def _entropy(prices: list) -> float:
        if len(prices) < 10:
            return 0.5
        mn = min(prices); mx = max(prices)
        if mx == mn:
            return 0.0
        nb = 10; n = len(prices)
        hist = [0] * nb
        for p in prices:
            idx = int((p - mn) / (mx - mn) * nb)
            if idx >= nb: idx = nb - 1
            hist[idx] += 1
        ent = 0.0
        for h in hist:
            if h > 0:
                pv = h / n
                ent -= pv * math.log2(pv)
        return min(ent / math.log2(nb), 1.0)


class StateProjectionEngine:
    """Maps tick-state to forward excursion surfaces.

    Input state vector:
      - ecdf_bucket (0-10)
      - transition_type (0=steady, 1=any_trans, 2=cross1, 3=cross2)
      - entropy_decile (0-9)
      - sal_score quantile (if available)

    Output: conditional forward-return statistics per state.
    """
    def __init__(self):
        self._surface = {}  # state_key -> {n, total_fwd, total_abs_fwd, wins, losses}

    def record(self, ecdf, prev_ecdf, entropy, fwd_return):
        ecdf_bucket = min(int(ecdf * 10), 9)
        entropy_decile = min(int(entropy * 10), 9)

        if prev_ecdf is None:
            trans_type = 0
        else:
            prev_bucket = min(int(prev_ecdf * 10), 9)
            diff = abs(ecdf_bucket - prev_bucket)
            if diff > 2: trans_type = 3
            elif diff > 1: trans_type = 2
            elif diff >= 1: trans_type = 1
            else: trans_type = 0

        state_key = (ecdf_bucket, trans_type, entropy_decile)
        if state_key not in self._surface:
            self._surface[state_key] = {"n": 0, "sum_fwd": 0.0, "sum_abs_fwd": 0.0, "wins": 0, "losses": 0}
        s = self._surface[state_key]
        s["n"] += 1
        s["sum_fwd"] += fwd_return
        s["sum_abs_fwd"] += abs(fwd_return)
        if fwd_return > 0: s["wins"] += 1
        elif fwd_return < 0: s["losses"] += 1

    def surface_summary(self, min_samples=5):
        rows = []
        for key, s in self._surface.items():
            if s["n"] < min_samples:
                continue
            bkt, tt, ed = key
            mean_fwd = s["sum_fwd"] / s["n"]
            mean_abs = s["sum_abs_fwd"] / s["n"]
            wr = s["wins"] / max(s["n"], 1) * 100
            pf = s["wins"] / max(s["losses"], 1) if s["losses"] > 0 else float('inf')
            rows.append({
                "bucket": bkt, "trans_type": tt, "entropy_dec": ed,
                "n": s["n"], "wr": wr, "pf": pf,
                "mean_fwd": mean_fwd, "mean_abs": mean_abs,
            })
        rows.sort(key=lambda r: r["n"], reverse=True)
        return rows

    def top_states(self, min_n=10, min_wr=55, n=10):
        rows = self.surface_summary(min_samples=min_n)
        filtered = [r for r in rows if r["wr"] >= min_wr]
        filtered.sort(key=lambda r: r["mean_fwd"], reverse=True)
        return filtered[:n]

    def amplitude_expansion_ratio(self, tick_avg_abs_fwd, min_n=10):
        rows = self.surface_summary(min_samples=min_n)
        if not rows:
            return 0.0
        avg_meso = sum(r["mean_abs"] for r in rows) / len(rows)
        return avg_meso / max(tick_avg_abs_fwd, 0.0001)
