"""Execution profiles for counterfactual simulation."""
from dataclasses import dataclass
import yaml
import os


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    spread_bps: float
    latency_ms_mean: float
    latency_ms_std: float
    fill_probability: float
    reject_probability: float
    slippage_bps_mean: float
    slippage_bps_std: float
    queue_priority: float


_PROFILES_DIR = os.path.dirname(os.path.abspath(__file__))
_YAML_PATH = os.path.join(_PROFILES_DIR, "execution_profiles.yaml")


def load_profiles(path: str = None) -> dict[str, ExecutionProfile]:
    path = path or _YAML_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)
    profiles = {}
    for name, data in raw.items():
        profiles[name] = ExecutionProfile(name=name, **data)
    return profiles
