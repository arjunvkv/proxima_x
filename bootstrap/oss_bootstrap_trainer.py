"""OSSBootstrapTrainer — quick-train OSS from bootstrap OHLC data at engine start.

Uses prefix-only ECDF ranks (no lookahead leakage) and forward returns
over multiple horizons. Drift-conditioned: P(up | ECDF_bucket, drift_state).
"""
import logging
import os
import pickle
import numpy as np
from signals.outcome_surface_signal import OutcomeSurfaceSignal

logger = logging.getLogger("proxima_ops.bootstrap.oss")

HORIZONS = [3, 10, 20]
MIN_SAMPLES = 200
EMA_SPAN = 20
CACHE_DIR = os.path.join(os.path.dirname(__file__), "oss_cache")


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


DRIFT_Z_THRESHOLD = 0.5

def _z_drift_state(price: float, ema: float, local_std: float) -> int:
    if local_std < 1e-10:
        return 0
    z = (price - ema) / local_std
    if z > DRIFT_Z_THRESHOLD:
        return 1
    elif z < -DRIFT_Z_THRESHOLD:
        return -1
    return 0


def _train_from_closes(closes: list[float], horizon: int) -> list[dict]:
    """Generate training records: prefix-only ECDF rank × z-score drift × forward outcome."""
    arr = np.array(closes, dtype=float)
    n = len(arr)
    records = []

    # Pre-compute prefix-only EMAs and diffs (no lookahead)
    emas = np.zeros(n)
    diffs = np.zeros(n)
    for i in range(n):
        ema_val = _ema(arr[:i+1], min(EMA_SPAN, i+1))[-1]
        emas[i] = ema_val
        diffs[i] = arr[i] - ema_val

    for i in range(n - horizon):
        rank = float(np.mean(arr[:i+1] <= arr[i]))
        start = max(0, i - EMA_SPAN + 1)
        local_std = float(np.std(diffs[start:i+1]))
        drift = _z_drift_state(arr[i], emas[i], local_std)
        forward_ret = arr[i + horizon] - arr[i]
        outcome = 1 if forward_ret > 0 else (-1 if forward_ret < 0 else 0)
        records.append({"ecdf": rank, "drift": drift, "outcome": outcome, "abs_move": abs(forward_ret)})
    _d0 = sum(1 for r in records if r["drift"] == 0)
    _dp = sum(1 for r in records if r["drift"] == 1)
    _dn = sum(1 for r in records if r["drift"] == -1)
    logger.info(f"[OSS TRAIN DRIFT] horizon={horizon} total={len(records)} drift0={_d0} drift+1={_dp} drift-1={_dn}")
    return records


class OSSBootstrapTrainer:
    def __init__(self):
        self._models: dict[str, dict[int, OutcomeSurfaceSignal]] = {}
        self._trained = False

    def train_symbol(self, symbol: str, closes: list[float], horizons: list[int] = None) -> dict:
        if horizons is None:
            horizons = HORIZONS
        if len(closes) < min(horizons) + MIN_SAMPLES:
            logger.warning(f"[BOOTSTRAP OSS] {symbol}: insufficient bars ({len(closes)})")
            return {"symbol": symbol, "trained": False, "samples": 0, "reason": "insufficient data"}
        sym_models = {}
        total_samples = 0
        for horizon in horizons:
            records = _train_from_closes(closes, horizon=horizon)
            if len(records) < MIN_SAMPLES:
                continue
            sym_models[horizon] = OutcomeSurfaceSignal.from_pipeline_records(records, ev_threshold=0.05)
            total_samples = len(records)
        if not sym_models:
            return {"symbol": symbol, "trained": False, "samples": 0, "reason": "no horizon produced enough samples"}
        self._models[symbol] = sym_models
        density = list(sym_models.values())[0].signal_density()
        n_buckets = list(sym_models.values())[0].bucket_count()
        logger.info(f"[BOOTSTRAP OSS] {symbol}: trained={total_samples} samples, {n_buckets} buckets, density={density:.3f}, horizons={list(sym_models.keys())}")
        return {
            "symbol": symbol,
            "trained": True,
            "samples": total_samples,
            "buckets": n_buckets,
            "density": round(density, 3),
            "horizons": list(sym_models.keys()),
        }

    def _cache_path(self, symbol: str) -> str:
        os.makedirs(CACHE_DIR, exist_ok=True)
        return os.path.join(CACHE_DIR, f"{symbol}.pkl")

    def _load_cache(self, symbols: list[str]) -> tuple[bool, set[str]]:
        loaded = set()
        for sym in symbols:
            path = self._cache_path(sym)
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        self._models[sym] = pickle.load(f)
                    loaded.add(sym)
                except Exception:
                    pass
        return len(loaded) == len(symbols), loaded

    def _save_cache(self):
        for sym, models in self._models.items():
            path = self._cache_path(sym)
            try:
                with open(path, "wb") as f:
                    pickle.dump(models, f)
            except Exception as e:
                logger.warning(f"[OSS CACHE] failed to save {sym}: {e}")

    def train_all(self, bootstrap_data: dict[str, dict], symbols: list[str] = None) -> dict:
        if symbols is None:
            symbols = list(bootstrap_data.keys())
        all_loaded, loaded_set = self._load_cache(symbols)
        if all_loaded:
            self._trained = True
            logger.info(f"[BOOTSTRAP OSS] full cache hit — skipped training ({len(symbols)} symbols)")
            return {"trained": True, "symbols": {s: {"trained": True} for s in symbols}, "cache": "hit"}
        trained_any = False
        results = {}
        for sym in symbols:
            if sym in loaded_set:
                results[sym] = {"trained": True, "cache": True}
                trained_any = True
                continue
            seed = bootstrap_data.get(sym, {})
            closes = seed.get("closes", [])
            res = self.train_symbol(sym, closes)
            results[sym] = res
            if res.get("trained"):
                trained_any = True
        self._trained = trained_any
        self._save_cache()
        trained_syms = [s for s, r in results.items() if r.get("trained")]
        logger.info(f"[BOOTSTRAP OSS] train_all: trained={trained_any} symbols={trained_syms}")
        return {"trained": trained_any, "symbols": results}

    def has_surface(self, symbol: str) -> bool:
        return symbol in self._models

    def get_model(self, symbol: str, horizon: int = 10) -> OutcomeSurfaceSignal | None:
        sym_models = self._models.get(symbol)
        if sym_models is None:
            return None
        return sym_models.get(horizon)

    def predict(self, symbol: str, ecdf: float, horizon: int = 10, drift: int = 0) -> int:
        model = self.get_model(symbol, horizon)
        if model is None:
            return 0
        return model.predict(ecdf, drift_state=drift)

    def predict_with_info(self, symbol: str, ecdf: float, horizon: int = 10, drift: int = 0) -> dict:
        model = self.get_model(symbol, horizon)
        if model is None:
            return {"signal": 0, "bucket": "N/A", "ev": 0.0, "diagnostics": {"fallback_reason": "no_model"}}
        return model.predict_with_info(ecdf, drift_state=drift)

    def get_oss(self) -> OutcomeSurfaceSignal | None:
        for sym, models in self._models.items():
            for h, m in models.items():
                return m
        return None