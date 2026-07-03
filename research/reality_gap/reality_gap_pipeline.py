import json
import numpy as np
from research.reality_gap.signal_decay import SignalDecay
from research.reality_gap.execution_sensitivity import ExecutionSensitivity
from research.reality_gap.trade_clustering import TradeClustering
from research.reality_gap.portfolio_correlation import PortfolioCorrelation
from research.reality_gap.signal_latency import SignalLatency
from research.reality_gap.regime_failure_detector import RegimeFailureDetector
from research.reality_gap.capital_efficiency import CapitalEfficiency
from research.reality_gap.signal_quality_monitor import SignalQualityMonitor
from proxima_v1.core.signal_engine import SignalEngine, SignalResult
from proxima_v1.core.risk_engine import RiskEngine
from proxima_v1.core.execution_engine import ExecutionEngine
from proxima_v1.core.position_manager import PositionManager
from proxima_v1.core.trade_lifecycle import TradeLifecycle
from proxima_v1.core.portfolio_engine import PortfolioEngine
from proxima_v1.core.metrics import MetricsTracker, TradeRecord


class RealityGapPipeline:

    def __init__(self, assets: list[str] | None = None):
        self.assets = assets or ["EURJPY", "USDJPY", "GBPJPY"]
        self.results: dict[str, dict] = {}

    def run_signal_decay(self) -> dict:
        dec = SignalDecay()
        result = dec.run()
        self.results["signal_decay"] = result
        return result

    def run_execution_sensitivity(self) -> dict:
        es = ExecutionSensitivity(self.assets)
        result = es.run()
        self.results["execution_sensitivity"] = result
        return result

    def run_trade_clustering(self) -> dict:
        tc = TradeClustering(self.assets)
        result = tc.run()
        self.results["trade_clustering"] = result
        return result

    def run_portfolio_correlation(self) -> dict:
        pc = PortfolioCorrelation(self.assets)
        result = pc.run()
        self.results["portfolio_correlation"] = result
        return result

    def run_signal_latency(self) -> dict:
        sl = SignalLatency()
        result = sl.run()
        self.results["signal_latency"] = result
        return result

    def run_regime_failure(self) -> dict:
        rfd = RegimeFailureDetector()
        result = rfd.run()
        self.results["regime_failure"] = result
        return result

    def run_capital_efficiency(self) -> dict:
        ce = CapitalEfficiency(self.assets)
        result = ce.run()
        self.results["capital_efficiency"] = result
        return result

    def run_monte_carlo(self, n_simulations: int = 1000) -> dict:
        np.random.seed(42)
        capital = 100000.0
        load_data = {}
        engines = {}
        signals = {}
        for asset in self.assets:
            eng = SignalEngine(asset)
            eng.precompute_full()
            engines[asset] = eng
            load_data[asset] = eng._data
            full_es = np.nan_to_num(eng._full_es, nan=0.0)
            full_res = np.nan_to_num(eng._full_residual, nan=0.0)
            n = len(full_es)
            rolling_res = np.full(n, 0.5)
            rolling_es = np.full(n, 0.5)
            rolling_at = np.full(n, 0.5)
            for i in range(503, n):
                r_slice = full_res[max(0, i - 503):i + 1]
                e_slice = full_es[max(0, i - 503):i + 1]
                a_slice = np.nan_to_num(eng._full_at[max(0, i - 503):i + 1], nan=0.0)
                rolling_res[i] = float(np.sum(r_slice <= full_res[i])) / len(r_slice)
                rolling_es[i] = float(np.sum(e_slice <= full_es[i])) / len(e_slice)
                rolling_at[i] = float(np.sum(a_slice <= np.nan_to_num(eng._full_at[i], nan=0.0))) / len(a_slice) if len(a_slice) > 0 else 0.5
            signals[asset] = {
                "residual_rank": rolling_res,
                "es_rank": rolling_es,
                "at_rank": rolling_at,
                "composite": np.clip(0.60 * rolling_res + 0.30 * rolling_es + 0.10 * rolling_at, 0.0, 1.0),
            }
        min_len = min(len(signals[a]["composite"]) for a in self.assets)
        step = 20

        def run_one_sim(signal_miss_rate: float, exec_delay: int, spread_mult: float, slippage_pips: float) -> dict:
            rng = np.random.RandomState()
            metrics = MetricsTracker()
            portfolio = PortfolioEngine(capital)
            risk = RiskEngine()
            exec_eng = ExecutionEngine()
            exec_eng.spread = 0.00005 * spread_mult
            exec_eng.slippage = slippage_pips * 0.00001
            pm = PositionManager(3)
            tl = TradeLifecycle(risk, exec_eng, pm)
            prev_equity = float(capital)
            pending_signals: dict[str, list[dict]] = {a: [] for a in self.assets}
            for i in range(504, min_len, step):
                timestamp = i
                for asset in self.assets:
                    new_pending = []
                    for ps in pending_signals[asset]:
                        ps["delay_remaining"] -= 1
                        if ps["delay_remaining"] <= 0:
                            pass
                        else:
                            new_pending.append(ps)
                    pending_signals[asset] = new_pending
                sig_results = {}
                for asset in self.assets:
                    s = signals[asset]
                    composite = float(s["composite"][i])
                    if composite > 0.9:
                        state = "STRONG_LONG"
                    elif composite > 0.7:
                        state = "LONG"
                    elif composite > 0.5:
                        state = "WATCH"
                    else:
                        state = "NONE"
                    sig_results[asset] = SignalResult(asset, "composite", composite, composite, state, composite, timestamp)
                    if state in ("LONG", "STRONG_LONG") and rng.uniform() > signal_miss_rate:
                        pending_signals[asset].append({
                            "signal": sig_results[asset],
                            "delay_remaining": exec_delay,
                        })
                allocations = portfolio.allocate(sig_results)
                prices = {}
                for asset in self.assets:
                    arr = load_data[asset]["price"]
                    idx = min(i, len(arr) - 1)
                    prices[asset] = float(arr[idx])
                for asset in self.assets:
                    ready = [ps for ps in pending_signals[asset] if ps["delay_remaining"] <= 0]
                    pending_signals[asset] = [ps for ps in pending_signals[asset] if ps["delay_remaining"] > 0]
                    for ps in ready:
                        sig = ps["signal"]
                        if sig.state not in ("LONG", "STRONG_LONG"):
                            continue
                        at_arr = np.nan_to_num(engines[asset]._full_at, nan=0.0)
                        at_slice = at_arr[max(0, i - 100):i + 1] if len(at_arr) > 0 else np.zeros(10)
                        if len(at_slice) > 0:
                            at_bucket = min(int(np.searchsorted(np.nanpercentile(at_slice, [20, 40, 60, 80]), at_slice[-1])), 4)
                        else:
                            at_bucket = 2
                        tl.process_signal(asset, sig, prices.get(asset, 0.0), timestamp, at_bucket, capital / max(len(self.assets), 1))
                closed = tl.tick(prices, timestamp)
                for pos in closed:
                    entry_p = pos.entry_price
                    exit_p = pos.current_price
                    pnl = pos.pnl
                    pnl_pct = pnl / (pos.size * entry_p) if entry_p > 0.0 else 0.0
                    sig_res = sig_results.get(pos.asset)
                    sig_score = sig_res.score if sig_res else 0.0
                    holding = timestamp - pos.entry_time if pos.entry_time > 0 else 0
                    metrics.record_trade(TradeRecord(
                        asset=pos.asset, side=pos.side,
                        entry_time=pos.entry_time, exit_time=timestamp,
                        entry_price=entry_p, exit_price=exit_p,
                        size=pos.size, pnl=pnl, pnl_pct=pnl_pct,
                        holding_period=holding, signal_score=sig_score,
                    ))
                open_pnl = sum(p.pnl for p in pm.open_positions)
                closed_pnl = sum(t.pnl for t in metrics.trades) if metrics.trades else 0.0
                total_equity = capital + open_pnl + closed_pnl
                period_pnl = total_equity - prev_equity
                metrics.record_daily_pnl(period_pnl, total_equity)
                prev_equity = total_equity
            return metrics.compute_all_metrics()

        sharpes = np.zeros(n_simulations, dtype=np.float64)
        pps = np.zeros(n_simulations, dtype=np.float64)
        max_dds = np.zeros(n_simulations, dtype=np.float64)
        returns = np.zeros(n_simulations, dtype=np.float64)

        for sim in range(n_simulations):
            seed = 42 + sim
            np.random.seed(seed)
            signal_miss_rate = np.random.uniform(0.0, 0.5)
            exec_delay = int(np.random.uniform(0, 3))
            spread_mult = np.random.uniform(1.0, 3.0)
            slippage_pips = np.random.uniform(0.0, 3.0)
            m = run_one_sim(signal_miss_rate, exec_delay, spread_mult, slippage_pips)
            sharpes[sim] = float(m.get("sharpe", 0.0))
            pps[sim] = float(m.get("pp", 0.5))
            max_dds[sim] = float(m.get("max_dd", 0.0))
            returns[sim] = float(m.get("total_return", 0.0))

        def percentiles(arr: np.ndarray) -> dict:
            return {
                "p5": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            }

        survival_mask = (sharpes > 0.75) & (pps > 0.65) & (max_dds < 0.15)
        survival_rate = float(np.mean(survival_mask))

        result = {
            "n_simulations": n_simulations,
            "seed": 42,
            "sharpe": {
                **percentiles(sharpes),
                "mean": float(np.mean(sharpes)),
                "std": float(np.std(sharpes)),
            },
            "pp": {
                **percentiles(pps),
                "mean": float(np.mean(pps)),
                "std": float(np.std(pps)),
            },
            "max_dd": {
                **percentiles(max_dds),
                "mean": float(np.mean(max_dds)),
                "std": float(np.std(max_dds)),
            },
            "total_return": {
                **percentiles(returns),
                "mean": float(np.mean(returns)),
                "std": float(np.std(returns)),
            },
            "survival_rate": survival_rate,
        }
        self.results["monte_carlo"] = result
        return result

    def deployment_decision(self) -> dict:
        sd = self.results.get("signal_decay", {})
        ex = self.results.get("execution_sensitivity", {})
        tc = self.results.get("trade_clustering", {})
        pc = self.results.get("portfolio_correlation", {})
        sl = self.results.get("signal_latency", {})
        rf = self.results.get("regime_failure", {})
        ce = self.results.get("capital_efficiency", {})
        mc = self.results.get("monte_carlo", {})

        sizing = ce.get("sizing_comparison", {})
        best_method = ce.get("best_method", "FIXED")
        best_sizing = sizing.get(best_method, {})
        best_sharpe = float(best_sizing.get("sharpe", 0.0))
        best_pp = float(best_sizing.get("pp", 0.5))
        best_dd = float(best_sizing.get("max_dd", 0.0))

        survival_rate = float(mc.get("survival_rate", 0.0))

        baseline_pp = float(sd.get(0, {}).get("pp", 0.5))
        max_delay_pp = max(
            float(sd.get(d, {}).get("pp", 0.0))
            for d in (0, 1, 2, 5, 10, 20)
            if isinstance(sd.get(d), dict)
        ) if any(isinstance(sd.get(d), dict) for d in (0, 1, 2, 5, 10, 20)) else 0.0
        alpha_improves_with_delay = max_delay_pp > baseline_pp + 0.05

        spread_results = ex.get("spread_results", {})
        baseline_sharpe = float(spread_results.get("1x", {}).get("sharpe", 0.0))
        sharpe_at_10x = float(spread_results.get("10x", {}).get("sharpe", 0.0))
        execution_robust = baseline_sharpe > 0 and (sharpe_at_10x / baseline_sharpe) > 0.95
        max_tolerable_spread = ex.get("max_tolerable_spread", 0)
        max_tolerable_slippage = ex.get("max_tolerable_slippage", 0)

        period_keys = [k for k in sl if isinstance(sl[k], dict) and "signal_frequency" in sl[k]]
        freqs = [sl[k]["signal_frequency"] for k in period_keys]
        durations = [sl[k].get("signal_duration_mean", 0) for k in period_keys]
        freq_collapse = len(freqs) >= 2 and freqs[-1] < freqs[0] * 0.5
        duration_collapse = len(durations) >= 2 and durations[-1] < durations[0] * 0.5

        regime_periods = rf.get("period_metrics", {})
        regimes_detected = len(regime_periods) >= 2
        pp_power = rf.get("predictive_power", {})
        regime_predictive = pp_power.get("f1", 0) > 0.3

        base_methods = list(sizing.keys()) if isinstance(sizing, dict) else []
        better_sizing_available = best_method != "AT_SIZE"

        trade_cluster_ratio = float(tc.get("cluster_ratio", 0)) if tc else 0
        eff_positions = float(pc.get("effective_positions", 1)) if pc else 1

        findings = {
            "alpha_improves_with_delay": alpha_improves_with_delay,
            "execution_robust_to_10x_spread": execution_robust,
            "max_tolerable_spread_multiple": max_tolerable_spread,
            "max_tolerable_slippage_pips": max_tolerable_slippage,
            "signal_frequency_collapse": freq_collapse,
            "signal_duration_collapse": duration_collapse,
            "regimes_detected": regimes_detected,
            "regime_predictive_power": regime_predictive,
            "best_sizing_method": best_method,
            "better_sizing_available": better_sizing_available,
            "trade_cluster_ratio": trade_cluster_ratio,
            "effective_positions": eff_positions,
        }

        positives = sum([
            alpha_improves_with_delay,
            execution_robust,
            better_sizing_available,
            regimes_detected,
        ])
        negatives = sum([
            freq_collapse,
            duration_collapse,
            not regime_predictive if regimes_detected else False,
        ])

        root_cause_found = freq_collapse or duration_collapse
        costs_not_cause = execution_robust
        sizing_can_improve = better_sizing_available

        if not costs_not_cause:
            classification = "RESEARCH_ONLY"
            recommendations = ["Cannot deploy — execution costs destroy edge"]
        elif root_cause_found and not regimes_detected:
            classification = "EXTENDED_RESEARCH"
            recommendations = [
                "Root cause identified (signal frequency collapse) but regimes not detectable",
                "Research: improve regime detection before deployment",
            ]
        elif root_cause_found and regimes_detected and not regime_predictive:
            classification = "CONDITIONAL_PAPER"
            recommendations = [
                f"Switch sizing from AT_SIZE to {best_method}",
                "Add regime-aware signal frequency filter for calm periods",
                "Keep current execution params (costs exonerated)",
                "Note: regime detector needs predictive power improvement (F1=0)",
            ]
        elif root_cause_found and regimes_detected and regime_predictive:
            classification = "CONDITIONAL_LIVE_TEST"
            recommendations = [
                f"Switch sizing from AT_SIZE to {best_method}",
                "Deploy regime-aware signal frequency filter",
                "Add Kalman filter for bar-level noise reduction",
                "Start with 0.25x capital allocation",
            ]
        elif positives >= 3:
            classification = "PAPER_EDGE"
            recommendations = [
                f"Switch sizing from AT_SIZE to {best_method}",
                "Deploy with current execution params",
                "Monitor for signal frequency degradation",
            ]
        else:
            classification = "EXTENDED_RESEARCH"
            recommendations = [
                "Insufficient positive findings for deployment",
                "Further research required",
            ]

        criteria = {
            "backtest_best_sizing": {
                "sharpe": best_sharpe,
                "pp": best_pp,
                "max_dd": best_dd,
            },
            "monte_carlo": {
                "survival_rate": survival_rate,
            },
            "research_findings": findings,
        }

        result = {
            "classification": classification,
            "criteria": criteria,
            "mitigation_plan": recommendations,
        }
        self.results["deployment_decision"] = result
        return result

    def run_all(self) -> dict:
        self.run_signal_decay()
        self.run_execution_sensitivity()
        self.run_trade_clustering()
        self.run_portfolio_correlation()
        self.run_signal_latency()
        self.run_regime_failure()
        self.run_capital_efficiency()
        self.run_monte_carlo()
        self.deployment_decision()
        return self.results

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
