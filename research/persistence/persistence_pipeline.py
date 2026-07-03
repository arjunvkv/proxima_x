import json
import time
from research.persistence.rq1_persistence_drivers import RQ1PersistenceDrivers
from research.persistence.rq2_survival_curves import RQ2SurvivalCurves
from research.persistence.rq3_threshold_mapping import RQ3ThresholdMapping
from research.persistence.rq4_residual_lifespan import RQ4ResidualLifespan
from research.persistence.rq5_delayed_alpha import RQ5DelayedAlpha
from research.persistence.rq6_cross_asset import RQ6CrossAsset
from research.persistence.rq7_walk_forward import RQ7WalkForward
from research.persistence.rq8_threshold_drift_order import RQ8ThresholdDriftOrder
from research.persistence.rq9_regime_classifier import RQ9RegimeClassifier
from research.persistence.rq10_adjudication import RQ10Adjudication


class PersistencePipeline:

    def __init__(self, assets: list[str] | None = None):
        self.assets = assets or ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD"]
        self.results: dict = {}

    def run_rq1(self) -> dict:
        rq = RQ1PersistenceDrivers(self.assets[0])
        result = rq.run()
        self.results["rq1_persistence_drivers"] = result
        return result

    def run_rq2(self) -> dict:
        rq = RQ2SurvivalCurves(self.assets[0])
        result = rq.run()
        self.results["rq2_survival_curves"] = result
        return result

    def run_rq3(self) -> dict:
        rq = RQ3ThresholdMapping(self.assets[0])
        result = rq.run()
        self.results["rq3_threshold_mapping"] = result
        return result

    def run_rq4(self) -> dict:
        rq = RQ4ResidualLifespan(self.assets[0])
        result = rq.run()
        self.results["rq4_residual_lifespan"] = result
        return result

    def run_rq5(self) -> dict:
        rq = RQ5DelayedAlpha(self.assets[0])
        result = rq.run()
        self.results["rq5_delayed_alpha"] = result
        return result

    def run_rq6(self) -> dict:
        rq = RQ6CrossAsset(self.assets)
        result = rq.run()
        self.results["rq6_cross_asset"] = result
        return result

    def run_rq7(self) -> dict:
        rq = RQ7WalkForward(self.assets[0])
        result = rq.run()
        self.results["rq7_walk_forward"] = result
        return result

    def run_rq8(self) -> dict:
        rq = RQ8ThresholdDriftOrder(self.assets[0])
        result = rq.run()
        self.results["rq8_threshold_drift_order"] = result
        return result

    def run_rq9(self) -> dict:
        rq = RQ9RegimeClassifier(self.assets[0])
        result = rq.run()
        self.results["rq9_regime_classifier"] = result
        return result

    def run_rq10(self) -> dict:
        rq = RQ10Adjudication(self.results)
        result = rq.run()
        self.results["rq10_adjudication"] = result
        return result

    def run_all(self) -> dict:
        t0 = time.time()
        self.run_rq1()
        self.run_rq2()
        self.run_rq3()
        self.run_rq4()
        self.run_rq5()
        self.run_rq6()
        self.run_rq7()
        self.run_rq8()
        self.run_rq9()
        self.run_rq10()
        self.results["_runtime_sec"] = time.time() - t0
        return self.results

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
