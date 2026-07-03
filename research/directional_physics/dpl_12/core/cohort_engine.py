import numpy as np


class CohortBase:
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight

    def vote(self, t, z_seq, features_list, extra=None):
        raise NotImplementedError


class MomentumCohort(CohortBase):
    def __init__(self, window=15, weight=1.0):
        super().__init__("momentum", weight)
        self.window = window

    def vote(self, t, z_seq, features_list, extra=None):
        if t < self.window:
            return 0.0, 0.0
        balance_vals = [features_list[i].get("energy_balance", 0.0) for i in range(t - self.window, t)]
        if len(balance_vals) < 2:
            return 0.0, 0.0
        slope = balance_vals[-1] - balance_vals[0]
        direction = float(np.sign(slope))
        confidence = float(min(abs(slope) * 5.0, 1.0))
        return direction, confidence


class ReversionCohort(CohortBase):
    def __init__(self, window=10, weight=0.8):
        super().__init__("reversion", weight)
        self.window = window

    def vote(self, t, z_seq, features_list, extra=None):
        if t < self.window:
            return 0.0, 0.0
        release_vals = [features_list[i].get("energy_release", 0.0) for i in range(t - self.window, t)]
        creation_vals = [features_list[i].get("energy_creation", 0.0) for i in range(t - self.window, t)]
        release_delta = release_vals[-1] - release_vals[0]
        creation_delta = creation_vals[-1] - creation_vals[0]
        exhaustion = release_delta - creation_delta
        direction = float(-np.sign(exhaustion))
        confidence = float(min(abs(exhaustion) * 3.0, 1.0))
        return direction, confidence


class CrossAssetCohort(CohortBase):
    def __init__(self, weight=1.2):
        super().__init__("cross_asset", weight)

    def vote(self, t, z_seq, features_list, extra=None):
        if extra is None:
            return 0.0, 0.0
        anchor_signal = extra.get("anchor_signal", 0.0)
        coupling_strength = extra.get("coupling_strength", 0.0)
        if abs(anchor_signal) < 0.01 or coupling_strength < 0.1:
            return 0.0, 0.0
        direction = float(np.sign(anchor_signal))
        confidence = float(np.clip(coupling_strength * 2.0, 0.0, 1.0))
        return direction, confidence


class TransitionCohort(CohortBase):
    def __init__(self, transition_field=None, weight=0.7):
        super().__init__("transition", weight)
        self.tfield = transition_field

    def vote(self, t, z_seq, features_list, extra=None):
        if self.tfield is None or t < 1:
            return 0.0, 0.0
        z_t = z_seq[t]
        v_pred = self.tfield.predict(z_t)
        if v_pred is None:
            return 0.0, 0.0
        v_n = v_pred / (np.linalg.norm(v_pred) + 1e-8)
        if extra is None or "flow_w" not in extra:
            return 0.0, 0.0
        w = extra["flow_w"]
        score = float(np.dot(v_n, w))
        direction = float(np.sign(score))
        confidence = float(np.clip(abs(score) * 2.0, 0.0, 1.0))
        return direction, confidence


class TCMAPriorCohort(CohortBase):
    def __init__(self, projector=None, weight=1.0):
        super().__init__("tcma_prior", weight)
        self.projector = projector

    def vote(self, t, z_seq, features_list, extra=None):
        if self.projector is None:
            return 0.0, 0.0
        z_t = z_seq[t]
        score = self.projector.raw_signal(z_t)
        direction = float(np.sign(score))
        confidence = float(np.clip(abs(score) * 1.5, 0.0, 1.0))
        return direction, confidence


class CohortEnsemble:
    def __init__(self):
        self.cohorts = []

    def add(self, cohort):
        self.cohorts.append(cohort)

    def votes(self, t, z_seq, features_list, extra=None):
        results = {}
        for c in self.cohorts:
            d, conf = c.vote(t, z_seq, features_list, extra)
            if abs(d) > 0:
                results[c.name] = {"direction": d, "confidence": conf, "weight": c.weight}
        return results
