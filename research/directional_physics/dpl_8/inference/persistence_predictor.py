import numpy as np
from ..model.survival_head import SurvivalHead


class PersistenceSignalEngine:
    def __init__(self, model, survival_head=None):
        self.model = model
        self.survival = survival_head or SurvivalHead()

    def generate_signal(self, X_t, z_t):
        if X_t.ndim == 1:
            X_t = X_t.reshape(1, -1)
        tau_pred = float(self.model.predict(X_t)[0])
        survival = self.survival.tau_to_survival(tau_pred)
        size = self.survival.to_position_size(survival)
        horizon = self.survival.to_horizon(tau_pred)
        direction = float(np.sign(np.sum(z_t)))
        return {
            "direction": direction,
            "tau": int(horizon),
            "confidence": round(survival, 4),
            "position_size": round(size, 4)
        }
