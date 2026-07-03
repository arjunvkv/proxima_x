import time
import json
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl

from research.tidv.experiments import (
    experiment_a, experiment_b, experiment_c, experiment_d,
    experiment_e, experiment_f, experiment_g, experiment_h,
)
from research.tidv.adjudicator import TIDVAdjudicator
from research.mechanism_discovery.temporal_topology import TemporalTopology
from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.memory_landscape import MemoryLandscape


class TIDVPipeline:
    def __init__(self, data_dir: str = "data/market"):
        self.data_dir = Path(data_dir)
        self.tt = TemporalTopology()
        self.ed = EnergyDynamics()
        self.ml = MemoryLandscape()
        self.adjudicator = TIDVAdjudicator()

    def run(self, asset: str = "EURJPY",
            cross_assets: Optional[list] = None) -> dict:
        wall = time.time()
        timing = {}

        t0 = time.time()
        data = self._load_data(asset)
        timing["load_data"] = time.time() - t0
        print(f"[{asset}] Loaded {len(data['price'])} rows ({timing['load_data']:.2f}s)")

        t0 = time.time()
        analysis_data = self._prepare_analysis_data(data)
        timing["compute_signals"] = time.time() - t0
        print(f"[{asset}] Signals computed in {timing['compute_signals']:.2f}s")

        at = analysis_data["adaptive_time"]
        ret = analysis_data["returns"]
        states = analysis_data["states"]
        price = analysis_data["price"]
        es = analysis_data["energy_storage"]
        md = analysis_data["memory_density"]
        mg = analysis_data["memory_gradient"]

        results = {}

        t0 = time.time()
        results["experiment_a"] = experiment_a(at, ret, es, md, mg, states)
        timing["experiment_a"] = time.time() - t0
        print(f"[{asset}] A(RegimeFilter): {results['experiment_a']['verdict']} ({timing['experiment_a']:.2f}s)")

        t0 = time.time()
        results["experiment_b"] = experiment_b(at, ret)
        timing["experiment_b"] = time.time() - t0
        print(f"[{asset}] B(DecisionQuality): ur={results['experiment_b']['uncertainty_reduction']:.4f} ({timing['experiment_b']:.2f}s)")

        t0 = time.time()
        results["experiment_c"] = experiment_c(at, ret, states)
        timing["experiment_c"] = time.time() - t0
        print(f"[{asset}] C(RiskConditioning): {results['experiment_c']['verdict']} ({timing['experiment_c']:.2f}s)")

        t0 = time.time()
        results["experiment_d"] = experiment_d(at, ret)
        timing["experiment_d"] = time.time() - t0
        print(f"[{asset}] D(PositionSizing): sharpe_improv={results['experiment_d']['avg_sharpe_improvement']:.4f} ({timing['experiment_d']:.2f}s)")

        t0 = time.time()
        results["experiment_e"] = experiment_e(at, ret, states, price)
        timing["experiment_e"] = time.time() - t0
        print(f"[{asset}] E(HoldingPeriod): ({timing['experiment_e']:.2f}s)")

        t0 = time.time()
        results["experiment_f"] = experiment_f(at, ret)
        timing["experiment_f"] = time.time() - t0
        print(f"[{asset}] F(TradeSurvivability): {results['experiment_f']['verdict']} ({timing['experiment_f']:.2f}s)")

        t0 = time.time()
        results["experiment_h"] = experiment_h(at, ret)
        timing["experiment_h"] = time.time() - t0
        print(f"[{asset}] H(EconomicValue): {results['experiment_h']['economic_verdict']} ({timing['experiment_h']:.2f}s)")

        if cross_assets:
            t0 = time.time()
            all_data = {asset: analysis_data}
            for ca in cross_assets:
                ca_data = self._load_data(ca)
                ca_analysis = self._prepare_analysis_data(ca_data)
                all_data[ca] = ca_analysis
            results["experiment_g"] = experiment_g(all_data)
            timing["experiment_g"] = time.time() - t0
            print(f"[{asset}] G(CrossAsset): {results['experiment_g']['verdict']} ({timing['experiment_g']:.2f}s)")
        else:
            results["experiment_g"] = None
            timing["experiment_g"] = 0.0

        t0 = time.time()
        verdict = self.adjudicator.adjudicate(results)
        timing["adjudicate"] = time.time() - t0

        timing["total"] = time.time() - wall

        output = {
            "asset": asset,
            "classification": verdict.classification,
            "integration_recommendation": verdict.integration_recommendation,
            "scores": verdict.scores,
            "evidence": verdict.evidence,
            "experiments": {
                "a": results["experiment_a"],
                "b": results["experiment_b"],
                "c": results["experiment_c"],
                "d": results["experiment_d"],
                "e": results["experiment_e"],
                "f": results["experiment_f"],
                "g": results["experiment_g"],
                "h": results["experiment_h"],
            },
            "timing": timing,
        }
        return output

    def _load_data(self, asset: str) -> dict:
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
        price = data["price"]
        returns = data["returns"]
        n = len(price)
        result_tt = self.tt.compute(data)
        result_ed = self.ed.compute(data)
        result_ml = self.ml.compute(data)
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
