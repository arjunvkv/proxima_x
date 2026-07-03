import time
import json
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl

from research.temporal_reality.temporal_validator import TemporalRealityValidator, TemporalValidationReport
from research.temporal_reality.universality import UniversalityAnalyzer, UniversalityReport
from research.mechanism_discovery.temporal_topology import TemporalTopology
from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.memory_landscape import MemoryLandscape


class TemporalRealityPipeline:
    """
    End-to-end pipeline for Reality Phase 4.
    
    Workflow:
    1. Load data for asset(s)
    2. Compute adaptive_time_coordinate via TemporalTopology
    3. Compute supporting signals (energy, memory components)
    4. Run all analyses via TemporalRealityValidator
    5. Run universality analysis if multiple assets
    6. Generate final verdict
    """
    
    def __init__(self, data_dir: str = "data/market"):
        self.data_dir = Path(data_dir)
        self.tt = TemporalTopology()
        self.ed = EnergyDynamics()
        self.ml = MemoryLandscape()
    
    def run(self, asset: str = "EURJPY", 
            cross_assets: Optional[list[str]] = None) -> dict:
        """
        Run the full pipeline for a single asset.
        """
        wall = time.time()
        timing = {}
        
        # 1. Load data
        t0 = time.time()
        data = self._load_data(asset)
        timing["load_data"] = time.time() - t0
        print(f"[{asset}] Loaded {len(data['price'])} rows")
        
        # 2. Compute adaptive_time_coordinate and supporting signals
        t0 = time.time()
        analysis_data = self._prepare_analysis_data(data)
        timing["compute_signals"] = time.time() - t0
        print(f"[{asset}] Signals computed in {timing['compute_signals']:.2f}s")
        
        # 3. Run validator
        t0 = time.time()
        validator = TemporalRealityValidator(asset=asset)
        report = validator.validate(analysis_data)
        timing["validate"] = time.time() - t0
        print(f"[{asset}] Validation complete in {timing['validate']:.2f}s")
        
        # 4. Run universality if cross-assets specified
        universality_report = None
        if cross_assets:
            t0 = time.time()
            universality_report = self._run_universality(asset, cross_assets)
            timing["universality"] = time.time() - t0
            print(f"[{asset}] Universality: {universality_report.verdict}")
        
        timing["total"] = time.time() - wall
        
        return self._build_output(asset, report, universality_report, timing)
    
    def run_multi(self, assets: list[str]) -> dict:
        """Run pipeline on multiple assets and universality analysis."""
        results = {}
        all_data = {}
        
        for asset in assets:
            data = self._load_data(asset)
            all_data[asset] = data
            results[asset] = self.run(asset, cross_assets=None)
        
        # Run universality across all
        uni_analyzer = UniversalityAnalyzer(assets)
        uni_data = {}
        for asset in assets:
            dt = all_data[asset]
            ad = self._prepare_analysis_data(dt)
            uni_data[asset] = {"adaptive_time": ad["adaptive_time"], "states": ad["states"]}
        
        uni_report = uni_analyzer.compute(uni_data)
        
        return {
            "assets": results,
            "universality": {
                "verdict": uni_report.verdict,
                "pairwise": str(uni_report.distribution_similarity),
            }
        }
    
    def _load_data(self, asset: str) -> dict:
        """Load parquet data for an asset."""
        path = self.data_dir / f"{asset}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Data not found: {path}")
        
        df = pl.read_parquet(str(path))
        
        price = df["close"].to_numpy().astype(np.float64)
        returns = (df["log_return"].to_numpy().astype(np.float64) 
                  if "log_return" in df.columns 
                  else np.diff(np.log(price), prepend=np.log(price[0])))
        volume = (df["volume"].to_numpy().astype(np.float64)
                  if "volume" in df.columns
                  else np.ones(len(price), dtype=np.float64))
        high = df["high"].to_numpy().astype(np.float64) if "high" in df.columns else price.copy()
        low = df["low"].to_numpy().astype(np.float64) if "low" in df.columns else price.copy()
        
        return {
            "price": price,
            "returns": returns,
            "volume": volume,
            "high": high,
            "low": low,
        }
    
    def _prepare_analysis_data(self, data: dict) -> dict:
        """Compute all signals needed for analysis."""
        # Temporal topology -> adaptive_time
        result_tt = self.tt.compute(data)
        
        # Energy dynamics -> energy components for regime classification
        result_ed = self.ed.compute(data)
        
        # Compute volatility and entropy
        returns = data["returns"]
        n = len(returns)
        
        volatility = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            volatility[i] = np.std(returns[i-20:i])
        
        # Entropy via rolling window using numba helpers
        from research.information_discovery.mi_estimator import _fast_percentile, _fast_digitize, _fast_entropy_digitized
        window = 20
        n_bins = 10
        entropy = np.zeros(n, dtype=np.float64)
        q = np.linspace(0.0, 1.0, n_bins + 1)
        for i in range(window, n):
            segment = returns[i-window:i]
            valid = segment[~np.isnan(segment)]
            if len(valid) < 2:
                continue
            bins = _fast_percentile(valid, q)
            if len(np.unique(bins)) < 2:
                continue
            dig = _fast_digitize(valid, bins)
            entropy[i] = _fast_entropy_digitized(dig, n_bins)
        
        # Use time_regime as state proxy (0=quiet, 1=normal, 2=active)
        states = result_tt.get("time_regime", np.zeros(n, dtype=np.int64)).astype(np.int64)
        
        # Compute future_state_mutation for conditional info analysis
        from research.information_discovery.mi_estimator import _fast_percentile, _fast_digitize, _fast_entropy_digitized
        future_states = np.roll(states, -1)  # state at t+1
        state_transition_rate = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            state_transition_rate[i] = float(states[i] != states[i-1])
        
        analysis = {
            "adaptive_time": result_tt["adaptive_time_coordinate"],
            "states": states,
            "future_states": future_states,
            "state_transition_rate": state_transition_rate,
            "returns": returns,
            "volume": data.get("volume", np.ones(n, dtype=np.float64)),
            "energy_creation": result_ed.get("energy_creation", np.zeros(n)),
            "energy_storage": result_ed.get("energy_storage", np.zeros(n)),
            "energy_dissipation": result_ed.get("energy_dissipation", np.zeros(n)),
            "volatility": volatility,
            "entropy": entropy,
            "event_density": result_tt.get("event_density", np.zeros(n)),
        }
        
        # Compute state mutation rate
        analysis["state_mutation_rate"] = self._compute_mutation_rate(
            analysis["states"])
        analysis["regime_change_events"] = self._compute_regime_changes(
            analysis["states"])
        
        return analysis
    
    @staticmethod
    def _compute_mutation_rate(states: np.ndarray, window: int = 20) -> np.ndarray:
        """Rolling state mutation rate."""
        from research.temporal_reality.evolution_clock import EvolutionClockAnalyzer
        rates, _, _, _ = EvolutionClockAnalyzer._compute_state_rates(
            states.astype(np.float64), window)
        return rates
    
    @staticmethod
    def _compute_regime_changes(states: np.ndarray) -> np.ndarray:
        """Binary regime change indicator."""
        changes = np.zeros(len(states), dtype=np.float64)
        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                changes[i] = 1.0
        return changes
    
    def _run_universality(self, primary: str, cross_assets: list[str]) -> UniversalityReport:
        """Run universality analysis."""
        uni = UniversalityAnalyzer([primary] + cross_assets)
        asset_data = {}
        
        for asset in [primary] + cross_assets:
            data = self._load_data(asset)
            ad = self._prepare_analysis_data(data)
            asset_data[asset] = {"adaptive_time": ad["adaptive_time"], "states": ad["states"]}
        
        return uni.compute(asset_data)
    
    def _build_output(self, asset: str, report, uni_report, timing: dict) -> dict:
        return {
            "asset": asset,
            "final_verdict": report.final_verdict,
            "timing": timing,
            "conditional_info_survival": report.conditional_info.get("information_survival_ratio", 0),
            "evolution_clock_verdict": report.evolution_clock.verdict if hasattr(report.evolution_clock, 'verdict') else "",
            "causality_lead": report.causality.get("lead_or_follow", ""),
            "null_model_verdict": report.null_models.verdict,
            "dependency_verdict": report.dependency_graph.verdict if hasattr(report.dependency_graph, 'verdict') else "",
            "universality_verdict": uni_report.verdict if uni_report else None,
        }


def run_temporal_reality(asset: str = "EURJPY", 
                         cross_assets: list[str] = None,
                         save: bool = True):
    """Convenience function to run the pipeline."""
    pipeline = TemporalRealityPipeline()
    result = pipeline.run(asset, cross_assets)
    
    print(f"\n=== Reality Phase 4: {asset} ===")
    print(f"Final Verdict: {result['final_verdict']}")
    print(f"Timing: {result['timing']}")
    
    if save:
        out_path = Path(f"reality/phase4_{asset}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved to {out_path}")
    
    return result
