import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research.live_validation.signal_tracker import SignalTracker, SignalRecord
from research.live_validation.threshold_monitor import ThresholdMonitor
from research.live_validation.drift_detector import DriftDetector
from research.live_validation.persistence_monitor import PersistenceMonitor
from research.live_validation.residual_monitor import ResidualMonitor
from research.live_validation.frequency_monitor import FrequencyMonitor
from research.live_validation.performance_monitor import PerformanceMonitor
from research.live_validation.anomaly_detector import AnomalyDetector
from research.live_validation.deployment_score import DeploymentScore
from proxima_v2.live.paper_engine_v2 import PaperEngineV2
from proxima_v2.strategies.es_v2_strategy import ESV2Strategy, SignalState

ASSETS = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD"]
THRESHOLD = 0.80
STEP = 20
CAPITAL = 100000.0


def _rolling_pct_rank(arr: np.ndarray, window: int = 504) -> np.ndarray:
    n = len(arr)
    result = np.full(n, 0.5)
    for i in range(window - 1, n):
        chunk = arr[i - window + 1 : i + 1]
        result[i] = float(np.sum(chunk <= arr[i])) / float(window)
    return result


class LivePipeline:
    def __init__(self):
        self.signal_tracker = SignalTracker()
        self.threshold_monitor = ThresholdMonitor()
        self.drift_detector = DriftDetector()
        self.persistence_monitor = PersistenceMonitor()
        self.residual_monitor = ResidualMonitor()
        self.frequency_monitor = FrequencyMonitor(target=30)
        self.performance_monitor = PerformanceMonitor()
        self.anomaly_detector = AnomalyDetector()
        self.deployment_score = DeploymentScore()
        self.paper = PaperEngineV2(capital=CAPITAL)
        self._daily_signal_count = 0
        self._current_day = 0

    def load_and_run(self, assets: list[str] | None = None):
        targets = assets or ASSETS
        from proxima_v1.core.signal_engine import SignalEngine
        from proxima_v2.core.regime_engine import RegimeEngine
        from proxima_v2.core.persistence_forecaster import PersistenceForecaster
        from proxima_v2.core.signal_allocator import SignalAllocator

        all_data = {}
        for asset in targets:
            eng = SignalEngine(asset)
            eng.precompute_full()
            n = min(len(eng._full_es), len(eng._full_residual), len(eng._full_at))
            es_r = _rolling_pct_rank(eng._full_es[:n])
            at_r = _rolling_pct_rank(eng._full_at[:n])
            composite = np.clip(0.70 * es_r + 0.30 * at_r, 0.0, 1.0)
            persistence_dur = np.zeros(n)
            i = 504
            while i < n:
                if composite[i] > 0.6:
                    j = i
                    while j < n and composite[j] > 0.6:
                        j += 1
                    persistence_dur[i:j] = j - i
                    i = j
                else:
                    i += 1
            all_data[asset] = {
                "eng": eng, "n": n,
                "es": eng._full_es[:n], "res": eng._full_residual[:n],
                "at": eng._full_at[:n], "price": eng._data["price"][:n],
                "es_r": es_r, "at_r": at_r,
                "composite": composite, "persistence_dur": persistence_dur}

        re = RegimeEngine()
        all_es = np.concatenate([all_data[a]["es"] for a in targets])
        all_at = np.concatenate([all_data[a]["at"] for a in targets])
        all_pdur = np.concatenate([all_data[a]["persistence_dur"] for a in targets])
        re.train(all_es, all_es, all_at, all_pdur)

        sa = SignalAllocator()
        threshold = THRESHOLD

        # Train PersistenceForecaster using signal starts from all assets
        all_signal_starts = []
        all_signal_durs = []
        for asset in targets:
            d = all_data[asset]
            comp = d["composite"]
            for idx in range(504, d["n"]):
                if comp[idx] > 0.6 and comp[idx - 1] <= 0.6:
                    all_signal_starts.append(idx)
                    all_signal_durs.append(int(d["persistence_dur"][idx]))
        pf = PersistenceForecaster()
        if len(all_signal_starts) >= 10:
            pf.train(d["es"], np.zeros(d["n"]), d["at"],
                     np.array(all_signal_durs), all_signal_starts)

        min_n = min(all_data[a]["n"] for a in targets)
        trade_sig_values: dict[str, tuple[float, float]] = {}
        for i in range(504, min_n, STEP):
            day = i // 24
            if day != self._current_day:
                self._daily_signal_count = 0
                self._current_day = day

            paper_signals_this_step = []

            for asset in targets:
                d = all_data[asset]
                if i >= d["n"]:
                    continue

                regime_snap = re.predict(
                    d["es"], d["es"], d["at"],
                    d["persistence_dur"][:d["n"]], i)

                es_pct = float(np.sum(
                    d["es"][max(0, i-504):i+1] <= d["es"][i]
                )) / 505.0

                pf_snap = pf.predict(d["es"], np.zeros(d["n"]), d["at"], i)

                # Record signal (all types)
                self.signal_tracker.record(
                    i, asset, es_pct, float(d["res"][i]),
                    float(d["at_r"][i]), threshold,
                    regime_snap.regime.value,
                    pf_snap.duration_class.value, "CHECK", 0.0)

                self.drift_detector.record_batch({
                    "es_percentile": es_pct,
                    "at_percentile": float(d["at_r"][i]),
                    "threshold": threshold})

                strat = ESV2Strategy(asset)
                signal = strat.generate(
                    float(d["es"][i]), es_pct, threshold,
                    pf_snap.duration_class.value, regime_snap.regime.value)

                # Update last recorded signal with actual state
                self.signal_tracker._signals[-1] = SignalRecord(
                    self.signal_tracker._signals[-1].timestamp,
                    self.signal_tracker._signals[-1].asset,
                    self.signal_tracker._signals[-1].es_percentile,
                    self.signal_tracker._signals[-1].residual_value,
                    self.signal_tracker._signals[-1].at_percentile,
                    self.signal_tracker._signals[-1].threshold_used,
                    self.signal_tracker._signals[-1].regime,
                    self.signal_tracker._signals[-1].persistence_forecast,
                    signal.state.value, signal.score)

                # Persistence: record predicted vs actual duration
                actual_bars = int(d["persistence_dur"][i])
                pred_class = pf_snap.duration_class.value
                if actual_bars < 5:
                    actual_class = "SHORT"
                elif actual_bars < 15:
                    actual_class = "MEDIUM"
                else:
                    actual_class = "LONG"
                self.persistence_monitor.record(pred_class, actual_class, 10, actual_bars, i)

                # Trade execution
                if signal.state in (SignalState.LONG, SignalState.STRONG_LONG):
                    self._daily_signal_count += 1
                    self.frequency_monitor.record_signal(i)

                    alloc = sa.decide(pf_snap.duration_class.value,
                                      signal.score, regime_snap.regime.value)
                    price = float(d["price"][min(i, d["n"] - 1)])
                    pos_size = self.paper.capital * 0.02 / max(price, 1)
                    self.paper.process_signal(asset, signal.score, signal.state.value,
                                              alloc.entry_delay, price, i, pos_size)
                    trade_sig_values[asset] = (es_pct, float(d["res"][i]))
                    paper_signals_this_step.append(asset)

            prices = {a: float(all_data[a]["price"][min(i, all_data[a]["n"] - 1)])
                      for a in targets if a in all_data}
            self.paper.tick(prices, i)
            self.threshold_monitor.record(threshold, i)

            # Record closed trades in residual + performance monitors
            for trade in self.paper._trades:
                if trade.closed and not hasattr(trade, "_recorded"):
                    pnl_pct = trade.pnl / max(self.paper.capital, 1)
                    self.performance_monitor.record_trade(trade.exit_time or i, pnl_pct)
                    # Track signal values: find the entry signal for this asset
                    sig_val = trade_sig_values.get(trade.asset, (0.5, 0.0))
                    es_val, res_val = sig_val
                    self.residual_monitor.record_trade(trade.exit_time or i, es_val, res_val, pnl_pct)
                    trade._recorded = True

            # Daily checks
            if (i - 504) % max(STEP * 5, 24) == 0 and i > 504:
                self.anomaly_detector.check_signal_drought(
                    self._daily_signal_count, i, threshold=1)
                self.anomaly_detector.check_threshold_explosion(
                    self.threshold_monitor.current, i)
                pm_s = self.persistence_monitor.summary()
                self.anomaly_detector.check_persistence_collapse(
                    pm_s["directional_accuracy"], i)

                # Check frequency every few steps
                fm_s = self.frequency_monitor.summary()
                self.anomaly_detector.check_frequency(
                    fm_s["frequency_cv"], fm_s["actual_frequency"], i)

        self.frequency_monitor.finalize()

        # Compute deployment score from monitors
        st = self.signal_tracker.summary()
        fm = self.frequency_monitor.summary()
        rm = self.residual_monitor.summary()
        pm = self.persistence_monitor.summary()
        tm = self.threshold_monitor.summary()

        # Count actual trades (LONG + STRONG_LONG)
        state_dist = st.get("state_distribution", {})
        total_trades = state_dist.get("LONG", 0) + state_dist.get("STRONG_LONG", 0)

        sig_health = min(total_trades / 300.0, 1.0) if total_trades > 0 else 0.0
        freq_stab = max(1.0 - fm["frequency_cv"], 0.0) if fm["frequency_cv"] > 0 else 1.0
        res_strength = float(np.clip(rm.get("ratio", 0.5), 0.0, 1.0))
        persist_acc = float(np.clip(pm["directional_accuracy"], 0.0, 1.0))
        thresh_stab = float(np.clip(1.0 - tm["deviation"], 0.0, 1.0))

        self.deployment_score.compute(sig_health, freq_stab, res_strength, persist_acc, thresh_stab)

        return self._build_report()

    def _build_report(self) -> dict:
        return {
            "signal_tracker": self.signal_tracker.summary(),
            "threshold_monitor": self.threshold_monitor.summary(),
            "drift_detector": self.drift_detector.summary(),
            "persistence_monitor": self.persistence_monitor.summary(),
            "residual_monitor": self.residual_monitor.summary(),
            "frequency_monitor": self.frequency_monitor.summary(),
            "performance_monitor": self.performance_monitor.summary(),
            "anomaly_detector": self.anomaly_detector.summary(),
            "deployment_score": self.deployment_score.summary()}


if __name__ == "__main__":
    pipe = LivePipeline()
    report = pipe.load_and_run()
    for k, v in report.items():
        print(f"\n=== {k.upper()} ===")
        for sk, sv in v.items():
            print(f"  {sk}: {sv}")
