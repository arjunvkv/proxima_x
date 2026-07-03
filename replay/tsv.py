import logging
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger("proxima.replay.tsv")


class TemporalShuffleValidator:
    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._chunk_size_days = 5

    def shuffle_chunks(self, ticks: list[dict]) -> list[dict]:
        if not ticks:
            return []
        chunks = self._split_chunks(ticks)
        chunk_order = list(range(len(chunks)))
        self._rng.shuffle(chunk_order)
        result = []
        for idx in chunk_order:
            result.extend(chunks[idx])
        return result

    def _split_chunks(self, ticks: list[dict]) -> list[list[dict]]:
        if not ticks:
            return []
        chunks = []
        current_chunk = []
        chunk_start = None
        for t in ticks:
            ts = t.get("time_sec", t.get("timestamp", 0))
            dt = datetime.fromtimestamp(ts)
            if chunk_start is None:
                chunk_start = dt
            if (dt - chunk_start).days >= self._chunk_size_days:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                chunk_start = dt
            current_chunk.append(t)
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def compute_stability_metrics(self, original: list[dict], shuffled: list[dict]) -> dict:
        orig_ts = [t.get("time_sec", t.get("timestamp", 0)) for t in original]
        shuff_ts = [t.get("time_sec", t.get("timestamp", 0)) for t in shuffled]
        orig_returns = self._compute_returns(original)
        shuff_returns = self._compute_returns(shuffled)
        corr = np.corrcoef(orig_returns[:len(shuff_returns)], shuff_returns[:len(orig_returns)])[0, 1] if len(orig_returns) > 1 and len(shuff_returns) > 1 else 0.0
        return {
            "n_original": len(original),
            "n_shuffled": len(shuffled),
            "return_correlation": float(corr) if not np.isnan(corr) else 0.0,
            "mean_orig_return": float(np.mean(orig_returns)) if orig_returns else 0.0,
            "mean_shuff_return": float(np.mean(shuff_returns)) if shuff_returns else 0.0,
            "chunks": len(self._split_chunks(original)),
            "chunk_size_days": self._chunk_size_days,
        }

    def _compute_returns(self, ticks: list[dict]) -> list[float]:
        prices = []
        for t in ticks[:10000]:
            bid = t.get("bid", 0.0)
            if bid > 0:
                prices.append(bid)
        if len(prices) < 2:
            return []
        arr = np.array(prices, dtype=np.float64)
        return list(np.diff(np.log(arr + 1e-10)))
