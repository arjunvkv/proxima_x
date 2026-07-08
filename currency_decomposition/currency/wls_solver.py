import numpy as np
from typing import Optional
from config.settings import CURRENCY_LIST, BASE_CURRENCY_MAP, WLS_REGULARIZATION


class WLSSolver:
    CURRENCIES = CURRENCY_LIST

    def __init__(self):
        self._build_design_matrix()

    def _build_design_matrix(self):
        pairs = list(BASE_CURRENCY_MAP.keys())
        n = len(self.CURRENCIES)
        A = np.zeros((len(pairs), n))
        b_labels = []
        for i, (sym, (base, quote)) in enumerate(BASE_CURRENCY_MAP.items()):
            A[i, self.CURRENCIES.index(base)] = 1.0
            A[i, self.CURRENCIES.index(quote)] = -1.0
            b_labels.append(sym)
        self.A = A
        self.pair_labels = b_labels

    def solve(self, pair_returns: dict[str, float], weights: Optional[dict[str, float]] = None,
              prior: Optional[dict[str, float]] = None, lam: float = WLS_REGULARIZATION) -> dict[str, float]:
        b = np.array([pair_returns.get(sym, 0.0) for sym in self.pair_labels])
        if weights is None:
            W = np.eye(len(self.pair_labels))
        else:
            w_vals = np.array([weights.get(sym, 1.0) for sym in self.pair_labels])
            W = np.diag(w_vals)
        if prior is None:
            prior_vec = np.zeros(len(self.CURRENCIES))
        else:
            prior_vec = np.array([prior.get(c, 0.0) for c in self.CURRENCIES])
        AtW = self.A.T @ W
        lhs = AtW @ self.A + lam * np.eye(len(self.CURRENCIES))
        rhs = AtW @ b + lam * prior_vec
        x = np.linalg.solve(lhs, rhs)
        x = x - np.mean(x)
        return {c: float(x[i]) for i, c in enumerate(self.CURRENCIES)}

    def compute_residuals(self, pair_returns: dict[str, float],
                          strengths: dict[str, float]) -> dict[str, float]:
        residuals = {}
        for sym, (base, quote) in BASE_CURRENCY_MAP.items():
            predicted = strengths.get(base, 0.0) - strengths.get(quote, 0.0)
            actual = pair_returns.get(sym, 0.0)
            residuals[sym] = actual - predicted
        return residuals

