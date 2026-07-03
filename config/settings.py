from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(os.getenv("PROXIMA_X_ROOT", Path(__file__).resolve().parent.parent))
    market_data: Path = root / "data" / "market"
    feature_store: Path = root / "data" / "features"
    research_cache: Path = root / "data" / "research_cache"
    metadata: Path = root / "data" / "metadata"
    states: Path = root / "research" / "states"


@dataclass(frozen=True)
class PerformanceConfig:
    lazy_execution: bool = True
    parallel_features: bool = True
    numba_parallel: bool = True
    cache_intermediates: bool = True
    memory_mapped: bool = True


@dataclass(frozen=True)
class StateConfig:
    vector_dtype: str = "float32"
    vector_dim: int = 64
    faiss_index_type: str = "IVF100,Flat"
    similarity_top_k: int = 100
    min_cluster_size: int = 50
    umap_n_components: int = 8


@dataclass(frozen=True)
class ResearchConfig:
    min_memory_samples: int = 100
    dna_sequence_length: int = 48
    pressure_window: int = 20
    liquidity_window: int = 50
    entanglement_timeframes: tuple = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")
    echo_max_chain: int = 5
    echo_decay_half_life: int = 10


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    state: StateConfig = field(default_factory=StateConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)

    @classmethod
    def from_env(cls) -> Settings:
        return cls()


settings = Settings.from_env()
