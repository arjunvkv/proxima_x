"""ROL shared infrastructure: residual origin analysis on cached DSR data."""
import sys, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.directional_state.dsr_core import DSRCore, SYMBOLS, HORIZON_KEYS, save_report

class ROLCore:
    """Residual Origin Lab — investigates what creates residual sign."""

    def __init__(self):
        self.dsr = DSRCore()
        self._data = {}

    def load_all(self, force_reload=False):
        for sym in SYMBOLS:
            self._data[sym] = self.dsr.load_symbol(sym, force_reload=force_reload)
        return self._data

    def get_residuals(self, symbol: str, kind: str = "linear"):
        d = self._data.get(symbol)
        if d is None:
            d = self.dsr.load_symbol(symbol)
            self._data[symbol] = d
        if kind == "linear":
            return d["residual"]
        elif kind == "sign":
            return d["residual_sign"]
        elif kind == "pressure":
            return d["residual_pressure"]
        elif kind == "acceleration":
            return d["residual_acceleration"]
        return d["residual"]

    def get_regime(self, symbol: str):
        d = self._data.get(symbol) or self.dsr.load_symbol(symbol)
        return d["regime"]

    def get_future_returns(self, symbol: str):
        d = self._data.get(symbol) or self.dsr.load_symbol(symbol)
        return d["fut_ret"]

    def get_es(self, symbol: str):
        d = self._data.get(symbol) or self.dsr.load_symbol(symbol)
        return d["es"]

    def get_memory_density(self, symbol: str):
        d = self._data.get(symbol) or self.dsr.load_symbol(symbol)
        return d["memory_density"]

    def get_memory_imbalance(self, symbol: str):
        d = self._data.get(symbol) or self.dsr.load_symbol(symbol)
        return d["memory_imbalance"]

    def sign_flips(self, symbol: str):
        """Find all points where residual sign changes."""
        res = self.get_residuals(symbol)
        sign = np.sign(res)
        sign[np.isnan(sign)] = 0
        flip_idx = np.where((sign[1:] != 0) & (sign[:-1] != 0) & (sign[1:] != sign[:-1]))[0] + 1
        flip_from = sign[flip_idx - 1]
        flip_to_idx = flip_idx + 1
        flip_to_idx = flip_to_idx[flip_to_idx < len(sign)]
        flip_to = sign[flip_to_idx] if len(flip_to_idx) > 0 else np.array([])
        return flip_idx, flip_from, flip_to

    def residual_run_lengths(self, symbol: str):
        """Compute consecutive bars with same residual sign."""
        sign = self.get_residuals(symbol, "sign")
        runs = []
        current_sign = 0
        current_len = 0
        for i, s in enumerate(sign):
            if np.isnan(s) or s == 0:
                if current_len > 0:
                    runs.append((current_sign, current_len, i - current_len))
                    current_len = 0
                continue
            if s == current_sign:
                current_len += 1
            else:
                if current_len > 0:
                    runs.append((current_sign, current_len, i - current_len))
                current_sign = s
                current_len = 1
        if current_len > 0:
            runs.append((current_sign, current_len, len(sign) - current_len))
        return runs


def save_rol_report(report: dict, name: str):
    path = Path(__file__).parent / "reports" / f"{name}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved {path}")
    return path


if __name__ == "__main__":
    rol = ROLCore()
    rol.load_all()
    for sym in SYMBOLS:
        n = len(rol.get_residuals(sym))
        flips, _, _ = rol.sign_flips(sym)
        runs = rol.residual_run_lengths(sym)
        print(f"{sym}: n={n}, sign_flips={len(flips)}, runs={len(runs)}, avg_run_len={np.mean([r[1] for r in runs]) if runs else 0:.1f}")
    print("ROL core ready.")
