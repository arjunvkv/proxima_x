"""OMS core: Observable Market State infrastructure — investigates what market condition appears when residual sign exists."""
import sys, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.residual_origin.rol_core import ROLCore, SYMBOLS
from research.directional_state.dsr_core import WalkForwardValidator, DSRCore, HORIZON_KEYS

class OMSCore:
    """Observable Market State — investigates latent state behind residual marker."""

    def __init__(self):
        self.rol = ROLCore()
        self.dsr = DSRCore()
        self._data = {}
        self._residual_runs = {}
        self._marker_presence = {}

    def load_all(self, force_reload=False):
        self._data = self.rol.load_all(force_reload)
        for sym in SYMBOLS:
            self._residual_runs[sym] = self.rol.residual_run_lengths(sym)
            self._compute_marker_presence(sym)
        return self._data

    def _compute_marker_presence(self, symbol: str):
        """Residual marker = 1 when |residual| > threshold (non-zero residual present)."""
        res = self.rol.get_residuals(symbol)
        marker = np.zeros(len(res), dtype=np.int64)
        marker[~np.isnan(res) & (np.abs(res) > np.nanpercentile(np.abs(res), 10))] = 1
        self._marker_presence[symbol] = marker
        return marker

    def marker_present(self, symbol: str) -> np.ndarray:
        """1 when residual marker is active, 0 otherwise."""
        return self._marker_presence.get(symbol, self._compute_marker_presence(symbol))

    def get_residual(self, symbol: str):
        return self.rol.get_residuals(symbol)

    def get_regime(self, symbol: str):
        return self.rol.get_regime(symbol)

    def get_future_returns(self, symbol: str):
        return self.rol.get_future_returns(symbol)

    def get_es(self, symbol: str):
        return self.rol.get_es(symbol)

    def get_memory_density(self, symbol: str):
        return self.rol.get_memory_density(symbol)

    def volatility(self, symbol: str, window: int = 20):
        """Rolling realized volatility from ES (or from residuals)."""
        es = self.get_es(symbol)
        vol = np.full(len(es), np.nan)
        for i in range(window, len(es)):
            vol[i] = np.std(es[i-window:i])
        return vol

    def range_expansion(self, symbol: str, window: int = 20):
        """Rolling high-low range relative to recent average."""
        d = self._data.get(symbol)
        if d is None:
            self.load_all()
            d = self._data.get(symbol)
        if d is None:
            return np.full(1822, np.nan)
        # Use ES as proxy for activity range
        es = self.get_es(symbol)
        range_ratio = np.full(len(es), np.nan)
        for i in range(window, len(es)):
            recent_range = np.max(es[i-window:i]) - np.min(es[i-window:i])
            older_range = np.max(es[max(0,i-2*window):i-window]) - np.min(es[max(0,i-2*window):i-window])
            range_ratio[i] = recent_range / max(older_range, 1e-12)
        return range_ratio

    def entropy_estimate(self, x: np.ndarray, window: int = 20):
        """Rolling entropy of sign changes (permutation entropy style)."""
        entropy = np.full(len(x), np.nan)
        for i in range(window, len(x)):
            chunk = x[i-window:i]
            chunk = chunk[~np.isnan(chunk)]
            if len(chunk) < 10:
                continue
            signs = np.sign(chunk)
            n_pos = np.sum(signs > 0)
            n_neg = np.sum(signs < 0)
            total = n_pos + n_neg
            if total == 0:
                continue
            p_pos = n_pos / total
            p_neg = n_neg / total
            if p_pos > 0 and p_neg > 0:
                entropy[i] = -(p_pos * np.log2(p_pos) + p_neg * np.log2(p_neg))
            elif p_pos > 0:
                entropy[i] = 0.0
            else:
                entropy[i] = 0.0
        return entropy

    def cross_asset_sync_index(self):
        """For each bar, count how many assets have marker present."""
        # Pre-load all markers
        markers = {}
        for sym in SYMBOLS:
            markers[sym] = self.marker_present(sym)
        min_len = min(len(m) for m in markers.values())
        sync = np.zeros(min_len, dtype=np.int64)
        for i in range(min_len):
            sync[i] = sum(markers[sym][i] for sym in SYMBOLS)
        return sync

    def synchronization_count(self, symbol: str):
        """Count of OTHER assets showing marker presence for each bar of given symbol."""
        markers = {}
        for sym in SYMBOLS:
            markers[sym] = self.marker_present(sym)
        ref = markers[symbol]
        n = len(ref)
        sync = np.zeros(n, dtype=np.int64)
        for i in range(min(n, min(len(m) for m in markers.values()))):
            sync[i] = sum(markers[s][i] for s in SYMBOLS if s != symbol)
        return sync


def save_oms_report(report: dict, name: str):
    path = Path(__file__).parent / "reports" / f"{name}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved {path}")
    return path


if __name__ == "__main__":
    oms = OMSCore()
    oms.load_all()
    print("OMS core ready.")
    sync = oms.cross_asset_sync_index()
    print(f"Cross-asset sync index: min={sync.min()}, max={sync.max()}, mean={sync.mean():.2f}, std={sync.std():.2f}")
