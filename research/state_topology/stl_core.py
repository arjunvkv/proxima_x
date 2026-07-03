"""STL Core: State Topology Lab — builds state space from ES, AT, Regime, Memory."""
import sys, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZON_LABELS
from research.directional_state.dsr_core import WalkForwardValidator

# State space dimensions
ES_QUINTILES = 5
AT_QUINTILES = 5
REGIME_STATES = 3
MEMORY_QUINTILES = 5
MAX_STATES = ES_QUINTILES * AT_QUINTILES * REGIME_STATES * MEMORY_QUINTILES


def _quintile(x):
    """Discretize into 0-4 quintiles."""
    n = len(x)
    result = np.full(n, -1, dtype=np.int64)
    valid = ~np.isnan(x)
    if np.sum(valid) < 10:
        return result
    sorted_vals = np.sort(x[valid])
    q = [np.percentile(sorted_vals, p) for p in [20, 40, 60, 80]]
    result[valid & (x <= q[0])] = 0
    result[valid & (x > q[0]) & (x <= q[1])] = 1
    result[valid & (x > q[1]) & (x <= q[2])] = 2
    result[valid & (x > q[2]) & (x <= q[3])] = 3
    result[valid & (x > q[3])] = 4
    return result


def state_id(es_q, at_q, regime, mem_q):
    """Encode 4D state into a single integer ID."""
    if any(v < 0 for v in [es_q, at_q, regime, mem_q]):
        return -1
    return int(es_q * (AT_QUINTILES * REGIME_STATES * MEMORY_QUINTILES)
               + at_q * (REGIME_STATES * MEMORY_QUINTILES)
               + regime * MEMORY_QUINTILES
               + mem_q)


def decode_state(sid):
    """Decode state ID into components."""
    mem_q = sid % MEMORY_QUINTILES
    regime = (sid // MEMORY_QUINTILES) % REGIME_STATES
    at_q = (sid // (MEMORY_QUINTILES * REGIME_STATES)) % AT_QUINTILES
    es_q = sid // (MEMORY_QUINTILES * REGIME_STATES * AT_QUINTILES)
    return es_q, at_q, regime, mem_q


class STLCore:
    """Builds and analyzes the 4D state space: ES×AT×Regime×Memory."""

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._states = {}
        self._data = {}

    def load_symbol(self, symbol, force_reload=False):
        cache_path = self.cache_dir / f"{symbol}_stl.npz"
        if cache_path.exists() and not force_reload:
            arr = np.load(cache_path)
            self._data[symbol] = {k: arr[k] for k in arr.files}
            return self._data[symbol]

        # Load fresh from DPLData (has AT)
        dpl = DPLData(symbol)
        es = dpl.es
        at = dpl.adaptive_time
        regime = dpl.states  # already discrete 0,1,2
        mem = dpl.memory_density
        fut_ret = dpl.fut_ret
        n = len(es)

        # Discretize
        es_q = _quintile(es)
        at_q = _quintile(at)
        mem_q = _quintile(mem)

        # Build state IDs
        sids = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            if es_q[i] < 0 or at_q[i] < 0 or regime[i] < 0 or mem_q[i] < 0:
                continue
            sids[i] = state_id(es_q[i], at_q[i], int(regime[i]), mem_q[i])

        result = {
            "state_id": sids,
            "es_q": es_q,
            "at_q": at_q,
            "regime": regime.astype(np.int64),
            "mem_q": mem_q,
            "fut_ret": fut_ret.astype(np.float32),
            "es": es.astype(np.float32),
            "at": at.astype(np.float32),
            "memory_density": mem.astype(np.float32),
        }
        np.savez_compressed(cache_path, **result)
        self._data[symbol] = result
        return result

    def load_all(self, force_reload=False):
        for sym in SYMBOLS:
            self.load_symbol(sym, force_reload=force_reload)
        return self._data

    def state_summary(self, symbol):
        """Count occurrences and compute P(up) per state."""
        d = self._data.get(symbol)
        if d is None:
            d = self.load_symbol(symbol)
        sids = d["state_id"]
        fut_ret = d["fut_ret"]
        n = len(sids)

        states_info = {}
        for i in range(n):
            sid = sids[i]
            if sid < 0:
                continue
            if sid not in states_info:
                states_info[sid] = {"count": 0, "es_q": None, "at_q": None, "regime": None, "mem_q": None,
                                    "up": {h: 0 for h in [5, 20, 50, 100]},
                                    "down": {h: 0 for h in [5, 20, 50, 100]},
                                    "total": {h: 0 for h in [5, 20, 50, 100]}}
            info = states_info[sid]
            info["count"] += 1
            if info["es_q"] is None:
                es_q, at_q, regime, mem_q = decode_state(sid)
                info["es_q"] = es_q
                info["at_q"] = at_q
                info["regime"] = regime
                info["mem_q"] = mem_q

            for hi, h in enumerate([5, 20, 50, 100]):
                ret = fut_ret[i, hi]
                if np.isnan(ret):
                    continue
                info["total"][h] += 1
                if ret > 0:
                    info["up"][h] += 1

        results = {}
        for sid, info in states_info.items():
            p_ups = {}
            for h in [5, 20, 50, 100]:
                t = info["total"][h]
                p_ups[h] = round(info["up"][h] / t, 4) if t > 0 else None
            results[int(sid)] = {
                "state_id": int(sid),
                "es_q": info["es_q"],
                "at_q": info["at_q"],
                "regime": info["regime"],
                "mem_q": info["mem_q"],
                "count": info["count"],
                "p_up": p_ups,
                "n_up": {h: info["up"][h] for h in [5, 20, 50, 100]},
                "n_total": {h: info["total"][h] for h in [5, 20, 50, 100]},
            }
        return results

    def directional_states(self, symbol, threshold=0.70, min_count=5):
        """Find states where P(up) > threshold at any horizon."""
        summary = self.state_summary(symbol)
        directional = []
        for sid, info in summary.items():
            if info["count"] < min_count:
                continue
            for h in [5, 20, 50, 100]:
                p = info["p_up"].get(h)
                if p is not None and p >= threshold:
                    directional.append({
                        "state_id": sid,
                        "es_q": info["es_q"],
                        "at_q": info["at_q"],
                        "regime": info["regime"],
                        "mem_q": info["mem_q"],
                        "count": info["count"],
                        "p_up": p,
                        "horizon": h,
                        "symbol": symbol,
                    })
                    break
        return directional

    def null_states(self, symbol, lower=0.45, upper=0.55, min_count=10):
        """Find states where P(up) is near 0.5 (no directional edge)."""
        summary = self.state_summary(symbol)
        nulls = []
        for sid, info in summary.items():
            if info["count"] < min_count:
                continue
            for h in [5, 20, 50, 100]:
                p = info["p_up"].get(h)
                if p is not None and lower <= p <= upper:
                    nulls.append({
                        "state_id": sid,
                        "es_q": info["es_q"],
                        "at_q": info["at_q"],
                        "regime": info["regime"],
                        "mem_q": info["mem_q"],
                        "count": info["count"],
                        "p_up": p,
                        "horizon": h,
                        "symbol": symbol,
                    })
                    break
        return nulls


def save_stl_report(report, name):
    path = Path(__file__).parent / "reports" / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=True)
    print(f"Saved {path}")
    return path


if __name__ == "__main__":
    stl = STLCore()
    stl.load_all()
    for sym in SYMBOLS:
        d = stl._data[sym]
        sids = d["state_id"]
        unique_sids = np.unique(sids[sids >= 0])
        n_unique = len(unique_sids)
        pct_valid = np.sum(sids >= 0) / len(sids) * 100
        dir_states = stl.directional_states(sym, threshold=0.70)
        null_states = stl.null_states(sym)
        print(f"{sym}: n={len(sids)}, unique_states={n_unique}/{MAX_STATES}, valid={pct_valid:.1f}%, "
              f"directional(>70%)={len(dir_states)}, null(45-55%)={len(null_states)}")
    print("STL core ready.")
