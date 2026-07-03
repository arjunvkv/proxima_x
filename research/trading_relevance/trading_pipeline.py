import time
import json
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl

from research.trading_relevance.trading_validator import TradingRelevanceValidator, TradingValidationReport
from research.trading_relevance.cross_asset import CrossAssetRelevanceAnalyzer
from research.mechanism_discovery.temporal_topology import TemporalTopology
from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.memory_landscape import MemoryLandscape


class TradingRelevancePipeline:
    """
    End-to-end pipeline for Reality Phase 5 (TRI-2).
    
    Workflow:
    1. Load data for asset(s)
    2. Compute adaptive_time_coordinate + existing mechanism components
    3. Run outcome distribution analysis (RQ1, RQ4, RQ7, RQ8)
    4. Run trade survivability analysis (RQ2)
    5. Run risk profile analysis (RQ3)
    6. Run mechanism interaction analysis (RQ5, RQ6)
    7. Run economic value analysis (RQ10)
    8. Run cross-asset analysis (RQ9) if multiple assets
    9. Generate final verdict
    """
    
    def __init__(self, data_dir: str = "data/market"):
        self.data_dir = Path(data_dir)
        self.tt = TemporalTopology()
        self.ed = EnergyDynamics()
        self.ml = MemoryLandscape()
    
    def run(self, asset: str = "EURJPY",
            cross_assets: Optional[list] = None) -> dict:
        """Run the full pipeline for a single asset."""
        wall = time.time()
        timing = {}
        
        # 1. Load data
        t0 = time.time()
        data = self._load_data(asset)
        timing["load_data"] = time.time() - t0
        print(f"[{asset}] Loaded {len(data['price'])} rows ({timing['load_data']:.2f}s)")
        
        # 2. Compute signals
        t0 = time.time()
        analysis_data = self._prepare_analysis_data(data)
        timing["compute_signals"] = time.time() - t0
        print(f"[{asset}] Signals computed in {timing['compute_signals']:.2f}s")
        
        # 3. Validate
        t0 = time.time()
        validator = TradingRelevanceValidator(asset=asset)
        
        if cross_assets:
            all_data = {asset: analysis_data}
            for ca in cross_assets:
                ca_data = self._load_data(ca)
                ca_analysis = self._prepare_analysis_data(ca_data)
                all_data[ca] = ca_analysis
            report = validator.validate_multi_asset(asset, all_data)
            timing["cross_asset"] = time.time() - t0
        else:
            report = validator.validate(analysis_data)
            timing["validate"] = time.time() - t0
        
        print(f"[{asset}] Validation: {report.final_verdict} ({timing.get('validate', timing.get('cross_asset', 0)):.2f}s)")
        
        timing["total"] = time.time() - wall
        print(f"[{asset}] Total: {timing['total']:.2f}s")
        
        return self._build_output(asset, report, timing)
    
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
        
        return {"price": price, "returns": returns, "volume": volume, "high": high, "low": low}
    
    def _prepare_analysis_data(self, data: dict) -> dict:
        """Compute all signals needed for trading relevance analysis."""
        price = data["price"]
        returns = data["returns"]
        vol = data["volume"]
        n = len(price)
        
        # Temporal topology -> adaptive_time
        result_tt = self.tt.compute(data)
        # Energy dynamics -> energy components
        result_ed = self.ed.compute(data)
        # Memory landscape -> memory components
        result_ml = self.ml.compute(data)
        
        # States from time_regime
        states = result_tt.get("time_regime", np.zeros(n, dtype=np.int64)).astype(np.int64)
        
        return {
            "adaptive_time": result_tt["adaptive_time_coordinate"],
            "returns": returns,
            "states": states,
            "price": price,
            "energy_storage": result_ed.get("energy_storage", np.zeros(n)),
            "memory_density": result_ml.get("memory_density", np.zeros(n)),
            "memory_gradient": result_ml.get("memory_gradient", np.zeros(n)),
        }
    
    def _build_output(self, asset: str, report, timing: dict) -> dict:
        od = report.outcome_distribution
        sv = report.trade_survivability
        rp = report.risk_profile
        mi = report.mechanism_interaction
        ev = report.economic_value
        
        return {
            "asset": asset,
            "final_verdict": report.final_verdict,
            "timing": timing,
            "outcome_separation_avg": getattr(od, 'outcome_separation_avg', 0.0),
            "outcome_verdict": od.verdict if hasattr(od, 'verdict') else "",
            "survivability_verdict": sv.verdict if hasattr(sv, 'verdict') else "",
            "risk_verdict": rp.verdict if hasattr(rp, 'verdict') else "",
            "mechanism_improvement": getattr(mi, 'adaptive_time_improvement', 0.0),
            "uncertainty_reduction": getattr(ev, 'uncertainty_reduction', 0.0),
            "economic_verdict": ev.verdict if hasattr(ev, 'verdict') else "",
            "cross_asset_verdict": report.cross_asset.verdict if report.cross_asset else None,
        }
