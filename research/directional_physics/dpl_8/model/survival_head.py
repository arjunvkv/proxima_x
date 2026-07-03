import numpy as np


class SurvivalHead:
    def __init__(self, decay=18.0):
        self.decay = decay

    def tau_to_survival(self, tau_pred):
        return float(np.exp(-tau_pred / self.decay))

    def to_position_size(self, survival_prob, max_risk=0.02):
        return float(max_risk * survival_prob)

    def to_horizon(self, tau_pred):
        return int(np.clip(tau_pred, 3, 100))
