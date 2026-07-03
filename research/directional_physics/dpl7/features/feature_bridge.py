import numpy as np
from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.temporal_topology import TemporalTopology


class FeatureBridge:
    def __init__(self, ed: EnergyDynamics = None, tt: TemporalTopology = None):
        self.ed = ed or EnergyDynamics()
        self.tt = tt or TemporalTopology()

    def extract(self, data: dict, idx: int) -> dict:
        data_dict = {k: v[:idx + 1] for k, v in data.items()
                     if k not in ("symbol", "timestamps", "n", "raw")}
        result_ed = self.ed.compute(data_dict)
        result_tt = self.tt.compute(data_dict)
        es_arr = np.nan_to_num(result_ed.get("energy_storage", np.zeros(idx + 1)), nan=0.0)
        current_es = es_arr[-1]
        es_rank = float(np.sum(es_arr <= current_es)) / len(es_arr) if len(es_arr) > 0 else 0.5
        energy_balance = np.nan_to_num(result_ed.get("energy_balance", np.zeros(idx + 1)), nan=0.0)
        energy_creation = np.nan_to_num(result_ed.get("energy_creation", np.zeros(idx + 1)), nan=0.0)
        energy_release = np.nan_to_num(result_ed.get("energy_release", np.zeros(idx + 1)), nan=0.0)
        energy_efficiency = np.nan_to_num(result_ed.get("energy_efficiency", np.zeros(idx + 1)), nan=0.0)
        time_density = np.nan_to_num(result_tt.get("time_density", np.zeros(idx + 1)), nan=0.0)
        event_density = np.nan_to_num(result_tt.get("event_density", np.zeros(idx + 1)), nan=0.0)
        information_density = np.nan_to_num(result_tt.get("information_density", np.zeros(idx + 1)), nan=0.0)
        adaptive_time = np.nan_to_num(result_tt.get("adaptive_time_coordinate", np.zeros(idx + 1)), nan=0.0)
        returns = data.get("returns", np.zeros(idx + 1))
        returns_vol = float(np.std(returns[-min(50, idx + 1):]) + 1e-8)
        es_slope = (es_arr[-1] - es_arr[-min(20, idx + 1)]) / max(min(20, idx + 1), 1)
        energy_balance_slope = (energy_balance[-1] - energy_balance[-min(20, idx + 1)]) / max(min(20, idx + 1), 1)
        features = {
            "es_value": float(current_es),
            "es_rank": es_rank,
            "es_slope": float(es_slope),
            "energy_balance": float(energy_balance[-1]),
            "energy_balance_slope": float(energy_balance_slope),
            "energy_creation": float(energy_creation[-1]),
            "energy_release": float(energy_release[-1]),
            "energy_efficiency": float(energy_efficiency[-1]),
            "time_density": float(time_density[-1]),
            "event_density": float(event_density[-1]),
            "information_density": float(information_density[-1]),
            "adaptive_time": float(adaptive_time[-1]),
            "returns_vol": returns_vol,
            "last_return": float(returns[-1]),
        }
        return features

    def reset(self):
        self.ed.reset()
        self.tt.reset()
