"""
DPL-22: Entropy Compression Engine.

Detects coherence collapse in TPI state before directional release.

Modules:
  A. Rolling Shannon entropy over multiple windows
  B. Entropy slope (dH/dt)
  C. Compression regime detection
  D. Pre-release validation
  E. Directional lift measurement
  F. Fusion with Meta-State
"""
import inspect
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

N_TPI_BINS = 10
ENTROPY_WINDOWS = [20, 50, 100]
MIN_SAMPLES = 10
WARMUP_REQUIRED = 100


H_PMAX_ALPHA = 0.3
H_HIGH_THRESHOLD = 0.65
PMAX_HIGH_THRESHOLD = 0.15


class EntropyCompressionEngine:
    def __init__(self):
        self._tpi_history: Dict[str, list] = defaultdict(list)
        self._entropy_history: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
        self._compression_events: List[dict] = []
        self._outcomes: List[dict] = []
        self._max_history = 500
        self._warmup_done: set = set()
        self._warmup_prior: Dict[str, float] = {}
        # Topology tracking
        self._ema_h: Dict[str, float] = {}
        self._ema_pmax: Dict[str, float] = {}
        self._prev_h: Dict[str, float] = {}
        self._last_pmax: Dict[str, float] = {}
        self._dp_hist: Dict[str, list] = {}
        import logging
        logging.getLogger("proxima_demo").info(
            "[TOPOLOGY_INIT] loaded from %s", inspect.getfile(self.__class__)
        )

    def warmup_from_bars(self, symbol: str, opens: list, highs: list, lows: list, closes: list) -> None:
        """Pre-populate entropy history from M5 bar data (unsupervised)."""
        n = min(len(closes), 500)
        if n < WARMUP_REQUIRED:
            return
        # Generate pseudo-TPI from range-normalized directional efficiency
        pseudo_tpi = []
        for i in range(1, n):
            rng = highs[i] - lows[i]
            if rng > 1e-10:
                eff = (closes[i] - opens[i]) / rng  # -1 to +1
            else:
                eff = 0.0
            pseudo_tpi.append(eff)
        # Seed the engine
        for val in pseudo_tpi:
            self.update(symbol, val)
        # Compute and store prior entropy
        state = self.compute_state(symbol)
        self._warmup_prior[symbol] = state.get("normalized_entropy", 0.5) or 0.5
        self._warmup_done.add(symbol)

    def is_warmed(self, symbol: str) -> bool:
        return symbol in self._warmup_done

    def warmup_prior(self, symbol: str) -> float:
        return self._warmup_prior.get(symbol, 0.5)

    @property
    def symbols_warmed(self) -> int:
        return len(self._warmup_done)

    def decompose(self, symbol: str) -> dict:
        """Return entropy components: occupied/total bins, dominant prob, raw/norm entropy, effective N."""
        vals = self._tpi_history.get(symbol, [])
        if len(vals) < MIN_SAMPLES:
            prior = self._warmup_prior.get(symbol)
            if prior is not None:
                return {"status": "PRIOR", "normalized_entropy": prior}
            return {"status": "INSUFFICIENT", "n": len(vals)}
        w = 50  # default decomposition window
        window = vals[-min(w, len(vals)):]
        arr = np.array(window)
        lo, hi = np.min(arr), np.max(arr)
        result = {"status": "ACTIVE", "window": len(window), "n": len(vals)}
        if hi - lo < 1e-10:
            result.update({"occupied_bins": 1, "total_bins": N_TPI_BINS, "dominant_prob": 1.0,
                           "raw_entropy": 0.0, "max_entropy": np.log2(N_TPI_BINS),
                           "normalized_entropy": 0.0})
            return result
        bins = np.linspace(lo, hi, N_TPI_BINS + 1)
        counts, _ = np.histogram(arr, bins=bins)
        probs = counts / len(arr)
        occupied = int(np.sum(counts > 0))
        dominant = float(np.max(probs))
        probs_pos = probs[probs > 0]
        raw_h = float(-np.sum(probs_pos * np.log2(probs_pos)))
        max_h = np.log2(N_TPI_BINS)
        result.update({"occupied_bins": occupied, "total_bins": N_TPI_BINS,
                       "dominant_prob": round(dominant, 4),
                       "raw_entropy": round(raw_h, 4),
                       "max_entropy": round(max_h, 4),
                       "normalized_entropy": round(raw_h / max_h, 4)})
        return result

    def window_audit(self, symbol: str, windows: list = None) -> dict:
        """Compute entropy over multiple window sizes for sensitivity analysis."""
        if windows is None:
            windows = [32, 64, 128, 256]
        vals = self._tpi_history.get(symbol, [])
        result = {"symbol": symbol, "n": len(vals)}
        for w in windows:
            if len(vals) < min(w, MIN_SAMPLES):
                result[f"H{w}"] = None
                continue
            window = vals[-min(w, len(vals)):]
            h = self._compute_entropy(window)
            max_h = np.log2(N_TPI_BINS) if h is not None else 1.0
            if h is not None:
                result[f"H{w}"] = round(h, 4)
                result[f"nH{w}"] = round(h / max_h, 4)
            else:
                result[f"H{w}"] = None
                result[f"nH{w}"] = None
        return result

    def update(self, symbol: str, tpi: float) -> None:
        self._tpi_history[symbol].append(tpi)
        if len(self._tpi_history[symbol]) > self._max_history:
            self._tpi_history[symbol].pop(0)

    def _compute_entropy(self, values: list) -> Optional[float]:
        if len(values) < MIN_SAMPLES:
            return None
        arr = np.array(values)
        lo, hi = np.min(arr), np.max(arr)
        if hi - lo < 1e-10:
            return 0.0
        bins = np.linspace(lo, hi, N_TPI_BINS + 1)
        counts, _ = np.histogram(arr, bins=bins)
        probs = counts / len(arr)
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    def topology(self, symbol: str, all_symbols: list = None) -> dict:
        """Return (H, pmax, dH, dpmax, rank, pctl, topology) for a symbol.
        Uses relative (median-based) thresholds when all_symbols is provided,
        falling back to absolute thresholds."""
        d = self.decompose(symbol)
        import logging
        logging.getLogger("proxima_demo").info(
            "[TOPOLOGY_DBG] %s decompose=%s", symbol, d
        )
        status = d.get("status") or "INSUFFICIENT"
        if status != "ACTIVE":
            _r = {"status": status, "n": d.get("n", 0)}
            assert "status" in _r and _r["status"] is not None, "status must be set"
            return _r
        h = d["normalized_entropy"]
        pmax = d["dominant_prob"]
        prev_h = self._prev_h.get(symbol, h)
        dh = h - prev_h
        self._prev_h[symbol] = h
        prev_pmax = self._last_pmax.get(symbol, pmax)
        dp_inst = pmax - prev_pmax
        hist = self._dp_hist.setdefault(symbol, [])
        hist.append(dp_inst)
        if len(hist) > 5:
            hist.pop(0)
        dp = round(sum(hist), 4)
        self._last_pmax[symbol] = pmax
        if symbol not in self._ema_h:
            self._ema_h[symbol] = h
            self._ema_pmax[symbol] = pmax
        else:
            self._ema_h[symbol] = H_PMAX_ALPHA * h + (1 - H_PMAX_ALPHA) * self._ema_h[symbol]
            self._ema_pmax[symbol] = H_PMAX_ALPHA * pmax + (1 - H_PMAX_ALPHA) * self._ema_pmax[symbol]
        # Cross-sectional rank (if all_symbols provided)
        rank = 0
        percentile = 0.0
        h_threshold = H_HIGH_THRESHOLD
        pmax_threshold = PMAX_HIGH_THRESHOLD
        if all_symbols and len(all_symbols) > 1:
            active_syms = []
            for s in all_symbols:
                sd = self.decompose(s)
                if sd.get("status") == "ACTIVE":
                    active_syms.append((sd["normalized_entropy"], sd["dominant_prob"], s))
            if active_syms:
                all_h = [x[0] for x in active_syms]
                all_pmax = [x[1] for x in active_syms]
                h_sorted = sorted(all_h)
                h_threshold = h_sorted[len(h_sorted) // 2]
                pmax_sorted = sorted(all_pmax)
                pmax_threshold = pmax_sorted[len(pmax_sorted) // 2]
                # Rank by entropy (descending)
                active_syms.sort(key=lambda x: -x[0])
                for i, (_, _, s) in enumerate(active_syms):
                    if s == symbol:
                        rank = i + 1
                        percentile = (len(active_syms) - i) / len(active_syms)
                        break
        h_high = h > h_threshold
        pmax_high = pmax > pmax_threshold
        if h_high and not pmax_high:
            topology_name = "DIFFUSE_CHAOS"
        elif h_high and pmax_high:
            topology_name = "DIRECTED_TURBULENCE"
        elif not h_high and pmax_high:
            topology_name = "STRUCTURED_DOMINANCE"
        else:
            topology_name = "TRANSITIONAL_COMPRESSION"
        _r = {
            "entropy": round(h, 4),
            "dominant_prob": round(pmax, 4),
            "d_entropy": round(dh, 4),
            "d_pmax": round(dp, 4),
            "ema_h": round(self._ema_h[symbol], 4),
            "ema_pmax": round(self._ema_pmax[symbol], 4),
            "rank": rank,
            "percentile": round(percentile, 4),
            "topology": topology_name,
            "status": "ACTIVE",
        }
        assert "status" in _r and _r["status"] is not None, "status must be set"
        return _r

    def compute_state(self, symbol: str) -> dict:
        vals = self._tpi_history.get(symbol, [])
        if len(vals) < MIN_SAMPLES:
            prior = self._warmup_prior.get(symbol)
            if prior is not None:
                return {"state": "ACTIVE", "n": len(vals),
                        "entropy_20": prior, "entropy_50": prior, "entropy_100": prior,
                        "slope_20": 0.0, "slope_50": 0.0,
                        "normalized_entropy": prior, "low_entropy": prior < 0.5,
                        "compression": False, "compression_strength": 0.0,
                        "sustained_compression": False,
                        "warmup_prior": True}
            return {"state": "INSUFFICIENT", "n": len(vals)}

        entropies = {}
        slopes = {}
        for w in ENTROPY_WINDOWS:
            window = vals[-min(w, len(vals)):]
            h = self._compute_entropy(window)
            entropies[w] = h
            prev = vals[-min(w * 2, len(vals)):-min(w, len(vals))]
            if len(prev) >= MIN_SAMPLES:
                h_prev = self._compute_entropy(prev)
                slopes[w] = (h - h_prev) if h is not None and h_prev is not None else None
            else:
                slopes[w] = None
            if h is not None:
                self._entropy_history[symbol][w].append(h)
                if len(self._entropy_history[symbol][w]) > self._max_history:
                    self._entropy_history[symbol][w].pop(0)

        # Compression detection: negative slope on short window
        compression = False
        compression_strength = 0.0
        if slopes.get(20) is not None and slopes[20] < 0:
            compression = True
            compression_strength = abs(slopes[20])
        sustained_compression = False
        if slopes.get(50) is not None and slopes[50] < -0.1:
            sustained_compression = True

        # Normalized entropy (current / max possible for this symbol)
        max_h = np.log2(N_TPI_BINS) if len(vals) >= MIN_SAMPLES else 1.0
        norm_entropy = (entropies.get(20, 0) or 0) / max_h if max_h > 0 else 0.0
        low_entropy = norm_entropy < 0.5

        result = {
            "state": "ACTIVE",
            "n": len(vals),
            "entropy_20": entropies.get(20),
            "entropy_50": entropies.get(50),
            "entropy_100": entropies.get(100),
            "slope_20": slopes.get(20),
            "slope_50": slopes.get(50),
            "normalized_entropy": round(norm_entropy, 4),
            "low_entropy": low_entropy,
            "compression": compression,
            "compression_strength": round(compression_strength, 4),
            "sustained_compression": sustained_compression,
        }

        if compression:
            self._compression_events.append({
                "symbol": symbol,
                "strength": compression_strength,
                "norm_entropy": norm_entropy,
            })
            if len(self._compression_events) > self._max_history:
                self._compression_events.pop(0)

        return result

    def record_outcome(self, symbol: str, compression: bool, norm_entropy: float,
                       compression_strength: float, predicted_dir: int, actual_dir: int,
                       meta_score: float = 0.0) -> None:
        correct = (predicted_dir != 0 and predicted_dir == actual_dir)
        self._outcomes.append({
            "symbol": symbol,
            "compression": compression,
            "norm_entropy": norm_entropy,
            "strength": compression_strength,
            "correct": correct,
            "predicted": predicted_dir,
            "actual": actual_dir,
            "meta_score": meta_score,
        })
        if len(self._outcomes) > self._max_history:
            self._outcomes.pop(0)

    def directional_lift(self) -> dict:
        if len(self._outcomes) < 20:
            return {"n_total": len(self._outcomes), "status": "INSUFFICIENT"}
        total_correct = sum(1 for o in self._outcomes if o["correct"])
        total_acc = total_correct / len(self._outcomes)
        comp_obs = [o for o in self._outcomes if o["compression"]]
        low_ent_obs = [o for o in self._outcomes if o["norm_entropy"] < 0.5]
        reports = {}
        if len(comp_obs) >= 10:
            comp_acc = sum(1 for o in comp_obs if o["correct"]) / len(comp_obs)
            reports["compression"] = {
                "n": len(comp_obs),
                "accuracy": round(comp_acc, 4),
                "lift": round((comp_acc - total_acc) * 100, 2),
            }
        if len(low_ent_obs) >= 10:
            le_acc = sum(1 for o in low_ent_obs if o["correct"]) / len(low_ent_obs)
            reports["low_entropy"] = {
                "n": len(low_ent_obs),
                "accuracy": round(le_acc, 4),
                "lift": round((le_acc - total_acc) * 100, 2),
            }
        return {
            "n_total": len(self._outcomes),
            "baseline_accuracy": round(total_acc, 4),
            "subsets": reports,
            "status": "ACTIVE",
        }

    def fusion_values(self, symbols: List[str]) -> Dict[str, float]:
        """Return entropy compression score for each symbol (for Meta-State fusion)."""
        result = {}
        for sym in symbols:
            s = self.compute_state(sym)
            if s.get("compression"):
                result[sym] = min(s["compression_strength"] * 2.0, 1.0)
            else:
                result[sym] = max(1.0 - s.get("normalized_entropy", 0.5), 0.1)
        return result

    def summary(self, symbols: List[str]) -> str:
        lines = []
        lines.append("  DPL-22: ENTROPY COMPRESSION")
        lines.append("-" * 52)
        total = len(symbols)
        warmed = self.symbols_warmed
        if warmed > 0:
            lines.append(f"  Warmup: {warmed}/{total} symbols ready")
        lines.append(f"  {'Symbol':<8s} {'H20':<7s} {'H50':<7s} {'H100':<8s} {'Slope':<8s} {'NormH':<7s} {'Cmp':<5s} {'SCmp':<6s} {'State'}")
        for sym in symbols:
            s = self.compute_state(sym)
            if s.get("state") != "ACTIVE":
                warmup_tag = " (prior)" if s.get("warmup_prior") else ""
                lines.append(f"  {sym:<8s} {s.get('state', '?')}{warmup_tag:<20s}")
                continue
            h20 = f"{s['entropy_20']:.3f}" if s["entropy_20"] is not None else "?"
            h50 = f"{s['entropy_50']:.3f}" if s["entropy_50"] is not None else "?"
            h100 = f"{s['entropy_100']:.3f}" if s["entropy_100"] is not None else "?"
            sl = f"{s['slope_20']:+.3f}" if s["slope_20"] is not None else "?"
            nh = f"{s['normalized_entropy']:.3f}"
            cmp = "YES" if s["compression"] else "no"
            scmp = "YES" if s["sustained_compression"] else "no"
            st = "LOW" if s["low_entropy"] else "NORMAL"
            lines.append(f"  {sym:<8s} {h20:<7s} {h50:<7s} {h100:<8s} {sl:<8s} {nh:<7s} {cmp:<5s} {scmp:<6s} {st}")

        lines.append("")
        dl = self.directional_lift()
        if dl.get("status") == "ACTIVE":
            lines.append(f"  Baseline accuracy: {dl['baseline_accuracy']:.1%} (n={dl['n_total']})")
            for subset_name, subset in dl.get("subsets", {}).items():
                lines.append(f"  {subset_name:15s}: {subset['accuracy']:.1%} lift={subset['lift']:+.2f}% (n={subset['n']})")
        else:
            lines.append(f"  Collecting outcomes ({dl['n_total']}/20 minimum)...")
        lines.append("")
        lines.append(f"  Compression events: {len(self._compression_events)} total")
        return "\n".join(lines)
