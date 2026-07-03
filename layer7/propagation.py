"""
DPL-18: Cross-Asset Propagation Engine.

Detects directional inheritance and lead-lag propagation
across the canonical 5-asset universe.

Metrics per directed pair (source -> target):
  - optimal_lag       : lag (bars) maximizing cross-correlation
  - max_corr          : cross-correlation at optimal lag
  - dir_agreement     : fraction of times target direction = source direction at lag
  - propagation_score : composite = |max_corr| * dir_agreement
  - stability         : rolling window consistency of propagation_score
"""
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

MAX_HISTORY = 300
MIN_SAMPLES = 20
MAX_LAG = 10
STABILITY_WINDOW = 60


class CrossAssetPropagationEngine:
    def __init__(self):
        self._tpi_history: Dict[str, list] = defaultdict(list)
        self._dir_history: Dict[str, list] = defaultdict(list)
        self._timestamps: Dict[str, list] = defaultdict(list)
        self._score_history: Dict[Tuple[str, str], list] = defaultdict(list)

    def update(self, symbol: str, tpi: float, direction: int, timestamp: float) -> None:
        self._tpi_history[symbol].append(tpi)
        self._dir_history[symbol].append(direction)
        self._timestamps[symbol].append(timestamp)
        if len(self._tpi_history[symbol]) > MAX_HISTORY:
            self._tpi_history[symbol].pop(0)
            self._dir_history[symbol].pop(0)
            self._timestamps[symbol].pop(0)

    def _cross_corr(self, a: np.ndarray, b: np.ndarray, max_lag: int) -> Tuple[int, float]:
        n = min(len(a), len(b))
        if n < MIN_SAMPLES:
            return 0, 0.0
        a, b = a[-n:], b[-n:]
        a_norm = (a - np.mean(a)) / (np.std(a) + 1e-10)
        b_norm = (b - np.mean(b)) / (np.std(b) + 1e-10)
        best_lag, best_corr = 0, 0.0
        for lag in range(0, min(max_lag + 1, n // 2)):
            if lag == 0:
                corr = np.corrcoef(a_norm, b_norm)[0, 1]
            else:
                corr = np.corrcoef(a_norm[:-lag], b_norm[lag:])[0, 1]
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        return best_lag, best_corr

    def _dir_agreement(self, src_dirs: list, tgt_dirs: list, lag: int) -> float:
        if lag >= len(src_dirs) or lag >= len(tgt_dirs):
            return 0.0
        aligned = 0
        total = 0
        for i in range(len(src_dirs) - lag):
            if i + lag < len(tgt_dirs):
                sd = src_dirs[i]
                td = tgt_dirs[i + lag]
                if sd != 0 and sd == td:
                    aligned += 1
                    total += 1
                elif sd != 0:
                    total += 1
        return aligned / max(total, 1)

    def compute(self, symbols: List[str]) -> dict:
        results = {}
        for src in symbols:
            for tgt in symbols:
                if src == tgt:
                    continue
                pair = (src, tgt)
                src_tpi = np.array(self._tpi_history.get(src, []), dtype=float)
                tgt_tpi = np.array(self._tpi_history.get(tgt, []), dtype=float)
                if len(src_tpi) < MIN_SAMPLES or len(tgt_tpi) < MIN_SAMPLES:
                    results[pair] = {"optimal_lag": None, "max_corr": None, "dir_agreement": None, "propagation_score": None, "stability": None}
                    continue
                lag, corr = self._cross_corr(src_tpi, tgt_tpi, MAX_LAG)
                src_dirs = self._dir_history.get(src, [])
                tgt_dirs = self._dir_history.get(tgt, [])
                dagr = self._dir_agreement(src_dirs, tgt_dirs, lag)
                ps = abs(corr) * dagr
                self._score_history[pair].append(ps)
                if len(self._score_history[pair]) > STABILITY_WINDOW:
                    self._score_history[pair].pop(0)
                stb = float(np.std(self._score_history[pair])) if len(self._score_history[pair]) >= 10 else None
                results[pair] = {
                    "optimal_lag": lag,
                    "max_corr": round(corr, 4),
                    "dir_agreement": round(dagr, 3),
                    "propagation_score": round(ps, 4),
                    "stability": round(stb, 4) if stb is not None else None,
                }
        return results

    def compute_leaders(self, symbols: List[str]) -> List[dict]:
        matrix = self.compute(symbols)
        leader_scores = defaultdict(float)
        for (src, tgt), m in matrix.items():
            if m["propagation_score"] is not None:
                leader_scores[src] += m["propagation_score"]
        ranked = sorted(leader_scores.items(), key=lambda x: -x[1])
        return [{"symbol": sym, "score": round(score, 4)} for sym, score in ranked]

    def compute_followers(self, symbols: List[str]) -> List[dict]:
        matrix = self.compute(symbols)
        follower_scores = defaultdict(float)
        for (src, tgt), m in matrix.items():
            if m["propagation_score"] is not None:
                follower_scores[tgt] += m["propagation_score"]
        ranked = sorted(follower_scores.items(), key=lambda x: -x[1])
        return [{"symbol": sym, "score": round(score, 4)} for sym, score in ranked]

    def summary(self, symbols: List[str]) -> str:
        matrix = self.compute(symbols)
        leaders = self.compute_leaders(symbols)
        followers = self.compute_followers(symbols)
        lines = []
        lines.append("  DPL-18: CROSS-ASSET PROPAGATION")
        lines.append("-" * 52)
        lines.append(f"  {'Source':<8s} {'Target':<8s} {'Lag':<5s} {'Corr':<8s} {'DirAgr':<8s} {'PScore':<8s} {'Stab':<8s}")
        for (src, tgt), m in sorted(matrix.items()):
            lag = str(m["optimal_lag"]) if m["optimal_lag"] is not None else "?"
            corr = f"{m['max_corr']:.3f}" if m["max_corr"] is not None else "?"
            dagr = f"{m['dir_agreement']:.2f}" if m["dir_agreement"] is not None else "?"
            ps = f"{m['propagation_score']:.3f}" if m["propagation_score"] is not None else "?"
            stb = f"{m['stability']:.3f}" if m["stability"] is not None else "?"
            lines.append(f"  {src:<8s} {tgt:<8s} {lag:<5s} {corr:<8s} {dagr:<8s} {ps:<8s} {stb:<8s}")
        lines.append("")
        lines.append("  LEADERS (most influential):")
        for i, l in enumerate(leaders):
            lines.append(f"    {i+1}. {l['symbol']}  score={l['score']:.4f}")
        lines.append("  FOLLOWERS (most influenced):")
        for i, f in enumerate(followers):
            lines.append(f"    {i+1}. {f['symbol']}  score={f['score']:.4f}")
        return "\n".join(lines)
