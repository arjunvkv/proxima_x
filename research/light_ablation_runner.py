"""Lightweight ablation runner — processes ticks from replay directly, no demo overhead."""
import sys; sys.path.insert(0, '.')
import time
import math
import logging
logging.basicConfig(level=logging.WARNING)

from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from features.ecdf_transform import PerSymbolECDF
from engine.ranking_engine import RankingEngine
from engine.topk_rotation_engine import TopKRotationEngine
from risk.h20_cap_engine import H20CapEngine
from execution.execution_mapper import ExecutionMapper
from fusion_kernel.fusion_kernel import SignalFusionKernel
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from learning.afl_engine import AFLFeedbackEngine
from causal.cal_engine import CALEngine
from learning.fwo_engine import FeatureWeightOptimizer
from learning.rsl_engine import RegimeSegmentedLearning
from regime.rtd_engine import RegimeTransitionDetector
from learning.tca_engine import TemporalCreditAssignment
from learning.cwf_engine import CausalWeightFusion
from monitoring.cdm_engine import ConsensusDriftMonitor
from learning.drl_engine import DriftResolutionLayer
from stability.mso_engine import MetaStabilityOptimizer
from stability.lct_engine import LongHorizonConvergenceTracker
from meta.ssol_engine import SystemSelfOptimizationLoop
from research.layer_config import LayerConfig


class LightAblationRunner:
    def __init__(self, layer_config: LayerConfig, tick_limit: int = 100000,
                 doa_horizon: int = 20, invert_signals: bool = False,
                 track_signals: bool = False,
                 signal_mode: str = "fusion"):
        self.config = layer_config
        self.tick_limit = tick_limit
        self._doa_horizon = doa_horizon
        self._invert_signals = invert_signals
        self._track_signals = track_signals
        self._signal_mode = signal_mode  # "fusion", "ecdf", "ecdf_inverted", "entropy_only"

        self._ecdf = PerSymbolECDF(window_size=2000)
        self._ranking = RankingEngine(ecdf_weight=0.8, entropy_weight=0.2)
        self._rotation = TopKRotationEngine(top_k=2, min_margin=0.03, persistence=3)
        self._h20 = H20CapEngine(max_cap_per_symbol=0.60, min_cap_per_symbol=0.10)
        self._exec_mapper = ExecutionMapper(base_lot=1.0, max_lot=5.0, min_lot=0.01, risk_per_unit=1000.0)
        self._fusion = SignalFusionKernel(entropy_flip_threshold=0.65, coherence_penalty=0.15)
        self._doa = DelayedOutcomeEngine(horizon_ticks=self._doa_horizon)
        self._afl = AFLFeedbackEngine(learning_rate=0.05, entropy_sensitivity=1.0, rotation_sensitivity=1.0)
        self._cal = CALEngine()
        self._fwo = FeatureWeightOptimizer(lr=0.05)
        self._rsl = RegimeSegmentedLearning(base_weights={"ecdf": 0.40, "entropy": 0.35, "spread": 0.15, "signal": 0.10})
        self._rtd = RegimeTransitionDetector(enter_threshold=0.65, exit_threshold=0.55, min_persistence=3)
        self._tca = TemporalCreditAssignment(decay=0.85, max_history=50)
        self._cwf = CausalWeightFusion(cal_weight=0.5, tca_weight=0.5)
        self._cdm = ConsensusDriftMonitor(drift_threshold=0.75)
        self._drl = DriftResolutionLayer(base_cal_weight=0.5, base_tca_weight=0.5)
        self._mso = MetaStabilityOptimizer(window=10, oscillation_threshold=0.25)
        self._lct = LongHorizonConvergenceTracker(window=50)
        self._ssol = SystemSelfOptimizationLoop()

        self._eval_data: dict = {}
        self._allocations: dict = {}
        self._execution_plan: dict = {}
        self._wfv_records: list = []
        self._price_buffers: dict = {}

    def run(self, env) -> dict:
        symbols = env.replay_feed._symbols if hasattr(env, 'replay_feed') and env.replay_feed else []
        symbols = list(symbols)
        if not symbols:
            return {"error": "No symbols in replay feed"}

        for sym in symbols:
            self._eval_data[sym] = {
                "price": 0.0,
                "spread": 0.0,
                "ecdf_rank": 0.5,
                "entropy": 0.5,
                "signal": 0,
            }

        tick_count = {sym: 0 for sym in symbols}
        doa_count = 0
        afl_count = 0
        cal_count = 0
        fwo_count = 0
        rsl_count = 0
        rtd_count = 0
        lct_count = 0
        ssol_count = 0
        rotation_changes = 0
        total_ticks = 0
        _wall_start = time.perf_counter()

        prev_selected = []
        signal_flip_count = 0
        last_signals = {}
        allocation_change_count = 0
        last_allocs = {}
        param_history = {
            "fusion_threshold": [],
            "rotation_persistence": [],
            "h20_max_cap": [],
            "afl_lr": [],
        }

        while total_ticks < self.tick_limit:
            for sym in symbols:
                tick = env.tick_source.get_tick(sym)
                if tick is None:
                    continue
                tick_count[sym] += 1
                total_ticks += 1

                price = tick.get("ask", 0)
                spread = tick.get("spread", 0)

                self._eval_data[sym]["price"] = price
                self._eval_data[sym]["spread"] = spread
                self._eval_data[sym]["ecdf_rank"] = self._ecdf.update(sym, price)

                # Compute streaming entropy from price buffer
                if sym not in self._price_buffers:
                    self._price_buffers[sym] = []
                self._price_buffers[sym].append(price)
                if len(self._price_buffers[sym]) > 50:
                    self._price_buffers[sym] = self._price_buffers[sym][-50:]
                self._eval_data[sym]["entropy"] = self._compute_entropy(self._price_buffers[sym])

                # Ranking + Rotation
                if self.config.ranking:
                    ranked = self._ranking.rank_all(self._eval_data)
                else:
                    ranked = [(s, 0.5) for s in symbols]

                if self.config.rotation:
                    selected = self._rotation.select(ranked)
                else:
                    selected = [s for s, _ in ranked]

                if set(selected) != set(prev_selected):
                    rotation_changes += 1
                prev_selected = list(selected)

                # Fusion
                if self.config.fusion_kernel:
                    signals = self._fusion.generate(self._eval_data)
                else:
                    signals = {}
                # Signal mode override (bypass fusion kernel)
                if self._signal_mode == "ecdf":
                    signals = {}
                    for s in symbols:
                        e = self._eval_data[s].get("ecdf_rank", 0.5)
                        d = e - 0.5
                        signals[s] = 1 if d > 0.05 else (-1 if d < -0.05 else 0)
                elif self._signal_mode == "ecdf_inverted":
                    signals = {}
                    for s in symbols:
                        e = self._eval_data[s].get("ecdf_rank", 0.5)
                        d = 0.5 - e
                        signals[s] = 1 if d > 0.05 else (-1 if d < -0.05 else 0)
                elif self._signal_mode == "entropy_only":
                    signals = {}
                    for s in symbols:
                        e = self._eval_data[s].get("entropy", 0.5)
                        d = e - 0.5
                        signals[s] = 1 if d > 0.05 else (-1 if d < -0.05 else 0)

                if self._invert_signals:
                    signals = {s: -sig for s, sig in signals.items()}
                for s, sig in signals.items():
                    self._eval_data[s]["signal"] = sig
                if self._track_signals:
                    for s, sig in signals.items():
                        if s in last_signals and last_signals[s] != 0 and sig != 0 and (sig > 0) != (last_signals[s] > 0):
                            signal_flip_count += 1
                        last_signals[s] = sig

                # H20
                if self.config.h20:
                    allocs = self._h20.allocate(selected, self._eval_data)
                else:
                    allocs = {s: 1.0 / max(1, len(selected)) for s in selected}
                self._allocations = allocs

                # Execution
                if self.config.execution:
                    self._execution_plan = self._exec_mapper.map(self._allocations, self._eval_data)
                else:
                    self._execution_plan = {}

                # Track allocation changes
                if self._track_signals and last_allocs:
                    for s, w in allocs.items():
                        if s in last_allocs and abs(w - last_allocs[s]) > 0.01:
                            allocation_change_count += 1
                last_allocs = dict(allocs)

                # DOA + TCA record
                if self.config.doa:
                    self._doa.record_snapshot(self._eval_data)
                if self.config.tca:
                    self._tca.record(self._eval_data)

                if self.config.doa and self._doa.ready:
                    current_prices = {s: self._eval_data[s].get("price", 0) for s in symbols}
                    doa_results = self._doa.evaluate(current_prices)
                    doa_count += 1

                    # WFV record
                    for s, outcome in doa_results.items():
                        ed = self._eval_data.get(s, {})
                        ecdf = float(ed.get("ecdf_rank", 0.5))
                        ent = float(ed.get("entropy", 0.5))
                        self._wfv_records.append({
                            "signal": signals.get(s, 0),
                            "outcome": outcome,
                            "sym": s,
                            "ecdf": ecdf,
                            "entropy": ent,
                            "rotation_stability": 1.0 if s in selected else 0.0,
                            "allocation_weight": self._allocations.get(s, 0.0),
                            "signal_strength": abs(ecdf - ent),
                            "regime_confidence": abs(ecdf - ent),
                        })

                    # LCT
                    if self.config.lct:
                        self._lct.record(doa_results)
                    lct_score = self._lct.convergence_score() if self.config.lct else 0.5
                    lct_count += 1

                    # TCA assign
                    if self.config.tca:
                        tca_report = self._tca.assign_credit(current_prices, doa_results)
                    else:
                        tca_report = {}

                    # AFL
                    if self.config.afl:
                        afl_state = self._afl.update(doa_results)
                        afl_count += 1
                        self._fusion.entropy_flip_threshold = 0.65 * afl_state["entropy_sensitivity"]
                        self._rotation.persistence = max(1, int(3 * afl_state["rotation_sensitivity"]))
                        self._h20.max_cap = min(0.8, 0.6 * afl_state["rotation_sensitivity"])
                    else:
                        afl_state = {"entropy_sensitivity": 1.0, "rotation_sensitivity": 1.0}

                    # CAL
                    if self.config.cal:
                        cal_report = self._cal.attribute(self._eval_data, doa_results)
                        cal_count += 1
                    else:
                        cal_report = {}

                    # FWO
                    if self.config.fwo:
                        fwo_weights = self._fwo.update(cal_report)
                        fwo_count += 1
                    else:
                        fwo_weights = {}

                    # CDM
                    if self.config.cdm:
                        drift_scores = self._cdm.compute_drift(cal_report, tca_report)
                    else:
                        drift_scores = {s: 0.0 for s in doa_results}

                    # DRL
                    if self.config.drl:
                        drift_state = self._drl.adapt(drift_scores)
                    else:
                        drift_state = {"cal_weight": 0.5, "tca_weight": 0.5, "regularization": 1.0}

                    # MSO
                    if self.config.mso:
                        self._mso.record(drift_state)
                        stable_state = self._mso.stabilize(drift_state)
                        if stable_state != drift_state:
                            drift_state = stable_state

                    # CWF
                    if self.config.cwf:
                        cwf_report = self._cwf.fuse_with_weights(cal_report, tca_report,
                            drift_state["cal_weight"], drift_state["tca_weight"])
                    else:
                        cwf_report = cal_report if cal_report else tca_report if tca_report else {}

                    # RTD
                    if self.config.rtd:
                        regime = self._rtd.detect(self._eval_data)
                        rtd_count += 1
                    else:
                        regime = "STABLE_DEFAULT"

                    # RSL
                    if self.config.rsl and regime != "TRANSITION":
                        clean_regime = regime.replace("STABLE_", "")
                        self._rsl.update(clean_regime, cwf_report)
                        rsl_count += 1

                    # SSOL
                    if self.config.ssol:
                        ssol_state = self._ssol.update(
                            lct_score=lct_score,
                            drift_scores=drift_scores,
                            mso_state=drift_state,
                            drl_state=drift_state,
                        )
                        ssol_count += 1
                        self._fusion.entropy_flip_threshold = 0.65 * ssol_state["stability"]
                        self._rotation.persistence = max(1, int(3 * ssol_state["stability"]))
                        self._h20.max_cap = 0.6 * ssol_state["stability"]
                        self._afl.lr = ssol_state["learning_rate"]
                    # Track final parameter values (AFL or SSOL source)
                    param_history["fusion_threshold"].append(self._fusion.entropy_flip_threshold)
                    param_history["rotation_persistence"].append(self._rotation.persistence)
                    param_history["h20_max_cap"].append(self._h20.max_cap)
                    param_history["afl_lr"].append(self._afl.lr)

        elapsed = time.perf_counter() - _wall_start

        # Compute WFV on collected records
        from validation.wfv_engine import WalkForwardValidator, StatisticalEdgeTest
        wfv_results = WalkForwardValidator(train_size=5, test_size=3).run(self._wfv_records) if self._wfv_records else {}
        edge = StatisticalEdgeTest.run(wfv_results) if wfv_results else {"accuracy": 0.5, "pnl_proxy": 0.0, "edge_detected": False}

        allocation_entropy = self._entropy(self._allocations)

        # Parameter drift summary
        def _drift(hist):
            if len(hist) < 2:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0}
            return {
                "mean": round(sum(hist) / len(hist), 4),
                "std": round((sum((x - sum(hist)/len(hist))**2 for x in hist) / len(hist))**0.5, 4),
                "min": round(min(hist), 4),
                "max": round(max(hist), 4),
                "range": round(max(hist) - min(hist), 4),
            }
        param_drift = {k: _drift(v) for k, v in param_history.items()}

        return {
            "total_ticks": total_ticks,
            "wall_runtime_sec": round(elapsed, 1),
            "ticks_per_second": round(total_ticks / elapsed, 1) if elapsed > 0 else 0,
            "doa_evaluations": doa_count,
            "afl_updates": afl_count,
            "cal_updates": cal_count,
            "fwo_updates": fwo_count,
            "rsl_updates": rsl_count,
            "rtd_detections": rtd_count,
            "lct_updates": lct_count,
            "ssol_updates": ssol_count,
            "rotation_changes": rotation_changes,
            "signal_flip_count": signal_flip_count,
            "allocation_change_count": allocation_change_count,
            "active_symbol_count": len(symbols),
            "allocation_entropy": round(allocation_entropy, 4),
            "wfv_records": len(self._wfv_records),
            "wf_accuracy": edge["accuracy"],
            "wf_pnl_proxy": edge["pnl_proxy"],
            "wf_edge_detected": edge["edge_detected"],
            "param_drift": param_drift,
        }

    @staticmethod
    def _compute_entropy(prices: list) -> float:
        if len(prices) < 10:
            return 0.5
        min_p = min(prices)
        max_p = max(prices)
        if max_p == min_p:
            return 0.0
        bins = 10
        n = len(prices)
        hist = [0] * bins
        for p in prices:
            idx = int((p - min_p) / (max_p - min_p) * bins)
            if idx >= bins:
                idx = bins - 1
            hist[idx] += 1
        entropy = 0.0
        for h in hist:
            if h > 0:
                p = h / n
                entropy -= p * math.log2(p)
        return min(entropy / math.log2(bins), 1.0)

    @staticmethod
    def _entropy(weights: dict) -> float:
        vals = [abs(w) for w in weights.values()]
        total = sum(vals)
        if total == 0:
            return 0.0
        probs = [v / total for v in vals]
        return -sum(p * math.log(p) for p in probs if p > 0)


if __name__ == "__main__":
    from research.layer_config import LayerConfig
    from research.experiment_config import HMS24_MINIMAL

    cfg = ReplayConfig(
        symbols=["EURJPY", "USDJPY"],
        start="2026-03-12",
        end="2026-03-14",
        speed=500000,
        burst=True,
        latency=False,
        slippage=False,
        seed=42,
    )

    print("=== FULL_V4 ===")
    env = build_replay_environment(cfg)
    patch_clock(env.clock)
    runner = LightAblationRunner(LayerConfig(), tick_limit=100000)
    r_full = runner.run(env)
    for k, v in r_full.items():
        print(f"  {k}: {v}")

    print("\n=== HMS24_MINIMAL ===")
    env2 = build_replay_environment(cfg)
    patch_clock(env2.clock)
    runner2 = LightAblationRunner(HMS24_MINIMAL, tick_limit=100000)
    r_hms = runner2.run(env2)
    for k, v in r_hms.items():
        print(f"  {k}: {v}")
