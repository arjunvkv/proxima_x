import numpy as np
from ..core.edge_activation_engine import EdgeActivationEngine


class TradeGate:
    def __init__(self):
        self.engine = EdgeActivationEngine()

    def initialize(self, z_history):
        self.engine.fit_thresholds(z_history)

    def generate_signal(self, z_seq, risk=0.01):
        gate = self.engine.compute_gate(z_seq)
        t = len(z_seq) - 1
        if gate[t] == 0.0:
            return {"trade": False, "reason": "gate_closed", "direction": 0.0, "confidence": 0.0, "position_size": 0.0}
        direction = float(np.sign(np.sum(z_seq[t])))
        confidence = self.engine.edge_strength(z_seq[t])
        size = float(risk * (0.5 + 0.5 * confidence) ** 2)
        return {"trade": True, "reason": "gate_open", "direction": direction, "confidence": confidence, "position_size": size}
