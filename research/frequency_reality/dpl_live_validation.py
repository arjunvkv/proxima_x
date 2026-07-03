import logging
import json
import os
import time
import numpy as np

logger = logging.getLogger("proxima_x.dpl_live")

CANDIDATES = ["es_rank", "at_rank", "es_x_at", "blocked_vs_executed"]


class DPLLiveValidation:
    def __init__(self, save_path: str = None):
        self._save_path = save_path or os.path.join(
            os.path.dirname(__file__), "data", "dpl_live_features.json")
        self._features = {}
        self._load()

    def record_features(self, signal_id: str, symbol: str,
                        es_rank: float, at_rank: float,
                        entry_price: float, is_blocked: bool = False,
                        block_reason: str = None,
                        frequency_band: str = None,
                        wavelet: str = None,
                        energy_regime: int = None,
                        time_regime: int = None,
                        combined_regime: int = None):
        entry = {
            "dpl_version": "live_v1",
            "signal_id": signal_id,
            "symbol": symbol,
            "es_rank": round(es_rank, 4) if es_rank else None,
            "at_rank": round(at_rank, 4) if at_rank else None,
            "entry_price": entry_price,
            "is_blocked": is_blocked,
            "block_reason": block_reason,
            "frequency_band": frequency_band,
            "wavelet": wavelet,
            "energy_regime": energy_regime,
            "time_regime": time_regime,
            "combined_regime": combined_regime,
            "residual_sign": None,
            "memory_position": None,
            "timestamp": int(time.time()),
            "return_h20": None,
            "return_h50": None,
            "return_h100": None,
            "return_short_h20": None,
            "return_short_h50": None,
            "return_short_h100": None,
            "direction_h20": None,
            "direction_h50": None,
            "direction_h100": None,
            "short_direction_h20": None,
            "short_direction_h50": None,
            "short_direction_h100": None,
            "resolved": False,
        }
        key = f"{signal_id}_features"
        if key not in self._features:
            self._features[key] = entry
            self._save()

    def attach_outcome(self, signal_id: str, returns: dict):
        key = f"{signal_id}_features"
        entry = self._features.get(key)
        if not entry:
            return
        if entry.get("resolved"):
            return
        has_any = False
        for horizon_tag in ["h20", "h50", "h100"]:
            ret = returns.get(f"return_{horizon_tag}")
            if ret is not None:
                long_ret = round(float(ret), 6)
                entry[f"return_{horizon_tag}"] = long_ret
                entry[f"direction_{horizon_tag}"] = 1 if long_ret > 0 else (0 if long_ret < 0 else None)
                short_ret = round(-float(ret), 6)
                entry[f"return_short_{horizon_tag}"] = short_ret
                entry[f"short_direction_{horizon_tag}"] = 1 if short_ret > 0 else (0 if short_ret < 0 else None)
                has_any = True
        if has_any:
            entry["resolved"] = True
            self._save()

    def accuracy(self, feature: str, horizon: str = "h20",
                 threshold: float = None) -> dict:
        resolved = [e for e in self._features.values()
                    if e.get("resolved") and e.get(f"direction_{horizon}") is not None]
        if not resolved:
            return {"n": 0, "accuracy": 0.5, "info_gain": 0.0, "p_up": 0.5}
        values = np.array([e.get(feature, 0) or 0 for e in resolved])
        directions = np.array([e[f"direction_{horizon}"] for e in resolved])
        base_p_up = float(np.mean(directions))
        if threshold is not None:
            high = values > threshold
            low = values <= threshold
        else:
            median = float(np.median(values))
            high = values > median
            low = values <= median
        if np.sum(high) < 3 or np.sum(low) < 3:
            return {"n": len(resolved), "accuracy": base_p_up, "info_gain": 0.0, "p_up": base_p_up}
        p_up_high = float(np.mean(directions[high]))
        p_up_low = float(np.mean(directions[low]))
        n_high = int(np.sum(high))
        n_low = int(np.sum(low))
        pred = np.where(high, 1, 0)
        actual = directions
        acc = float(np.mean(pred == actual))
        base_entropy = self._entropy(base_p_up)
        cond_entropy = (n_high / len(resolved)) * self._entropy(p_up_high) + \
                       (n_low / len(resolved)) * self._entropy(p_up_low)
        ig = base_entropy - cond_entropy
        return {
            "n": len(resolved),
            "n_high": n_high,
            "n_low": n_low,
            "p_up_high": round(p_up_high, 4),
            "p_up_low": round(p_up_low, 4),
            "accuracy": round(acc, 4),
            "info_gain": round(ig, 6),
            "base_p_up": round(base_p_up, 4),
        }

    def regime_accuracy(self, horizon: str = "h20") -> dict:
        resolved = [e for e in self._features.values()
                    if e.get("resolved") and e.get(f"direction_{horizon}") is not None
                    and e.get("energy_regime") is not None]
        if len(resolved) < 10:
            return {"n": len(resolved), "regimes": {}}
        regimes = {}
        for e in resolved:
            r = e["energy_regime"]
            regimes.setdefault(r, []).append(e[f"direction_{horizon}"])
        result = {}
        for r, dirs in regimes.items():
            p_up = float(np.mean(dirs))
            n = len(dirs)
            se = float(np.std(dirs) / np.sqrt(n))
            z = (p_up - 0.5) / max(se, 1e-6)
            result[int(r)] = {"n": n, "p_up": round(p_up, 4), "se": round(se, 4), "z_score": round(z, 3)}
        return {"n": len(resolved), "regimes": result}

    def tournament(self) -> dict:
        results = {}
        for candidate in ["es_rank", "at_rank"]:
            for horizon in ["h20", "h50", "h100"]:
                key = f"{candidate}_{horizon}"
                results[key] = self.accuracy(candidate, horizon)
        es_at = self.accuracy_combined(horizon="h20")
        results[f"es_x_at_h20"] = es_at
        regime = self.regime_accuracy("h20")
        results["regime_h20"] = regime
        for r_val, r_info in regime.get("regimes", {}).items():
            results[f"regime_{r_val}_h20"] = r_info
        short = self.accuracy("es_rank", "short_h20")
        results["short_es_rank_h20"] = short
        return results

    def accuracy_combined(self, horizon: str = "h20") -> dict:
        resolved = [e for e in self._features.values()
                    if e.get("resolved") and e.get(f"direction_{horizon}") is not None]
        if len(resolved) < 10:
            return {"n": len(resolved), "accuracy": 0.5, "info_gain": 0.0}
        es = np.array([e.get("es_rank", 0) or 0 for e in resolved])
        at = np.array([e.get("at_rank", 0) or 0 for e in resolved])
        directions = np.array([e[f"direction_{horizon}"] for e in resolved])
        combined = es + at
        median = float(np.median(combined))
        high = combined > median
        if np.sum(high) < 3:
            return {"n": len(resolved), "accuracy": 0.5, "info_gain": 0.0}
        p_up_high = float(np.mean(directions[high]))
        base_p_up = float(np.mean(directions))
        pred = np.where(high, 1, 0)
        acc = float(np.mean(pred == directions))
        base_entropy = self._entropy(base_p_up)
        cond_entropy = (np.sum(high) / len(resolved)) * self._entropy(p_up_high) + \
                       ((len(resolved) - np.sum(high)) / len(resolved)) * self._entropy(1 - p_up_high)
        ig = base_entropy - cond_entropy
        return {
            "n": len(resolved),
            "n_high": int(np.sum(high)),
            "p_up_high": round(p_up_high, 4),
            "accuracy": round(acc, 4),
            "info_gain": round(ig, 6),
            "base_p_up": round(base_p_up, 4),
        }

    def summary(self) -> dict:
        total = len(self._features)
        resolved = sum(1 for e in self._features.values() if e.get("resolved"))
        regime_counts = {}
        for e in self._features.values():
            r = e.get("energy_regime")
            if r is not None:
                regime_counts[r] = regime_counts.get(r, 0) + 1
        return {
            "total_snapshots": total,
            "resolved": resolved,
            "pct_resolved": round(100 * resolved / max(total, 1), 1),
            "symbols": list(set(e["symbol"] for e in self._features.values())),
            "regime_distribution": regime_counts,
            "has_short_outcomes": any(e.get("return_short_h20") is not None for e in self._features.values()),
            "tournament": self.tournament(),
        }

    @staticmethod
    def _entropy(p: float) -> float:
        p = max(min(p, 1 - 1e-12), 1e-12)
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    def _save(self):
        try:
            parent = os.path.dirname(self._save_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            with open(self._save_path, "w", encoding="utf-8") as f:
                json.dump(self._features, f, indent=2)
        except Exception as e:
            logger.warning(f"DPLLiveValidation save error: {e}")

    def _load(self):
        try:
            if os.path.exists(self._save_path):
                with open(self._save_path, "r", encoding="utf-8") as f:
                    self._features = json.load(f)
                logger.info(f"Loaded {len(self._features)} feature snapshots")
        except Exception as e:
            logger.warning(f"DPLLiveValidation load error: {e}")

    def purge(self):
        self._features = {}
        self._save()
