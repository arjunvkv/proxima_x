"""LSV core: Latent State Verification — determines whether residual sign is market phenomenon or measurement artifact."""
import sys, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.residual_origin.rol_core import ROLCore, SYMBOLS
from research.directional_state.dsr_core import WalkForwardValidator

class LSVCore:
    """Latent State Verification — tests whether residual sign is market-linked or model-linked."""

    def __init__(self):
        self.rol = ROLCore()
        self._data = {}
        self._residuals = {}
        self._markers = {}

    def load_all(self, force_reload=False):
        self._data = self.rol.load_all(force_reload)
        for sym in SYMBOLS:
            self._residuals[sym] = self.rol.get_residuals(sym)
            self._markers[sym] = self._compute_marker(sym, self._residuals[sym])
        return self._data

    @staticmethod
    def _compute_marker(sym, res, threshold_pct=10):
        marker = np.zeros(len(res), dtype=np.int64)
        valid = ~np.isnan(res)
        thr = np.nanpercentile(np.abs(res[valid]), threshold_pct) if np.any(valid) else 0
        marker[valid & (np.abs(res) > thr)] = 1
        return marker

    def residual(self, sym):
        return self._residuals.get(sym, np.array([]))

    def residual_sign(self, sym):
        res = self.residual(sym)
        s = np.sign(res)
        s[np.isnan(s)] = 0
        return s.astype(np.int64)

    def marker(self, sym):
        return self._markers.get(sym, np.array([]))

    def future_returns(self, sym):
        return self.rol.get_future_returns(sym)

    def regime(self, sym):
        return self.rol.get_regime(sym)

    def es(self, sym):
        return self.rol.get_es(sym)

    def memory_density(self, sym):
        return self.rol.get_memory_density(sym)

    # --- SYNTHETIC RESIDUAL GENERATORS ---

    @staticmethod
    def shuffled_residual(original):
        """Randomly shuffle residual values (destroys temporal structure)."""
        valid = ~np.isnan(original)
        shuffled = original.copy()
        shuffled[valid] = np.random.permutation(shuffled[valid])
        return shuffled

    @staticmethod
    def lagged_residual(original, lag=5):
        """Shift residuals forward by lag bars."""
        lagged = np.full_like(original, np.nan)
        if lag > 0:
            lagged[lag:] = original[:-lag]
        return lagged

    @staticmethod
    def random_sign_with_persistence(n, p_pos=0.5, p_flip=0.1):
        """Generate random sign sequence with controllable persistence."""
        sign = np.zeros(n, dtype=np.int64)
        current = 1 if np.random.random() > 0.5 else -1
        for i in range(n):
            if np.random.random() < p_flip:
                current = -current
            sign[i] = current
        return sign

    @staticmethod
    def markov_sign(original):
        """Fit a 2-state Markov chain to residual sign, then generate synthetic sequence."""
        sign = np.sign(original)
        sign = sign[~np.isnan(sign)]
        sign = sign[sign != 0]
        if len(sign) < 10:
            return np.full_like(original, 0, dtype=np.int64)
        n_pos = np.sum(sign > 0)
        n_neg = np.sum(sign < 0)
        transitions = np.zeros((2, 2))
        for i in range(1, len(sign)):
            prev = 0 if sign[i-1] > 0 else 1
            curr = 0 if sign[i] > 0 else 1
            transitions[prev, curr] += 1
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_prob = transitions / row_sums
        synthetic = np.zeros(len(original), dtype=np.int64)
        current = 0 if np.random.random() < n_pos / (n_pos + n_neg) else 1
        for i in range(len(original)):
            if np.isnan(original[i]):
                synthetic[i] = 0
                continue
            if np.random.random() < trans_prob[current, 0]:
                current = 0
            else:
                current = 1
            synthetic[i] = 1 if current == 0 else -1
        return synthetic

    @staticmethod
    def sign_from_hurst(original, H=0.86):
        """Generate synthetic sign with target Hurst using spectral method for fGn."""
        n = len(original)
        if n < 3:
            return np.zeros(n, dtype=np.int64)
        M = 2 * n
        # Power spectrum of fGn: S(f) = sin(pi*f)^2 * sum_{k=-inf}^{inf} |f+k|^{-(2H+1)}
        # Approximation: S(f) ~ f^{-(2H-1)} for low frequencies, use Fourier method
        freqs = np.fft.fftfreq(M)[1:n]  # positive frequencies only
        if len(freqs) == 0:
            return np.zeros(n, dtype=np.int64)
        pow_spec = freqs ** (-(2 * H - 1) / 2)
        pow_spec = np.clip(pow_spec, 0, 1e6)
        phases = np.random.uniform(0, 2 * np.pi, len(freqs))
        # Build symmetric spectrum
        X = np.zeros(M, dtype=np.complex128)
        X[1:n] = pow_spec * np.exp(1j * phases)
        X[M-1:M-n:-1] = np.conj(X[1:n])
        noise = np.fft.ifft(X).real[:n]
        noise = (noise - np.mean(noise)) / max(np.std(noise), 1e-12)
        sign = np.sign(noise).astype(np.int64)
        sign[sign == 0] = 1
        return sign


def save_lsv_report(report: dict, name: str):
    path = Path(__file__).parent / "reports" / f"{name}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved {path}")
    return path


if __name__ == "__main__":
    lsv = LSVCore()
    lsv.load_all()
    for sym in SYMBOLS:
        res = lsv.residual(sym)
        m = lsv.marker(sym)
        pct = np.mean(m) * 100
        print(f"{sym}: marker prevalence={pct:.1f}%, n={len(res)}")
    print("LSV core ready.")
