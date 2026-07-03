import json
import numpy as np
from proxima_v1.core.signal_engine import SignalEngine, SignalResult
from proxima_v1.core.risk_engine import RiskEngine
from proxima_v1.core.execution_engine import ExecutionEngine
from proxima_v1.core.position_manager import PositionManager
from proxima_v1.core.trade_lifecycle import TradeLifecycle
from proxima_v1.core.portfolio_engine import PortfolioEngine
from proxima_v1.core.metrics import MetricsTracker, TradeRecord


class ExecutionSensitivity:
    SPREAD_MULTIPLIERS = [1, 2, 5, 10]
    SLIPPAGE_PIPS = [0, 1, 2, 5, 10]

    def __init__(self, assets: list[str] | None = None):
        self.assets = assets or ["EURJPY", "USDJPY", "GBPJPY"]

    def _run_backtest(self, spread_mult: float, slippage_pips: float) -> dict:
        capital = 100000.0
        metrics = MetricsTracker()
        portfolio = PortfolioEngine(capital)
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
        risk = RiskEngine()
        exec_eng = ExecutionEngine()
        exec_eng.spread = 0.00005 * spread_mult
        exec_eng.slippage = slippage_pips * 0.00001
        pm = PositionManager(3)
        tl = TradeLifecycle(risk, exec_eng, pm)
        prev_equity = float(capital)
        for i in range(504, min_len, step):
            timestamp = i
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
            allocations = portfolio.allocate(sig_results)
            prices = {}
            for asset in self.assets:
                arr = load_data[asset]["price"]
                idx = min(i, len(arr) - 1)
                prices[asset] = float(arr[idx])
            for alloc in allocations:
                sig = sig_results.get(alloc.asset)
                if sig is None or sig.state not in ("LONG", "STRONG_LONG"):
                    continue
                at_arr = np.nan_to_num(engines[alloc.asset]._full_at, nan=0.0)
                at_slice = at_arr[max(0, i - 100):i + 1] if len(at_arr) > 0 else np.zeros(10)
                if len(at_slice) > 0:
                    at_bucket = min(int(np.searchsorted(np.nanpercentile(at_slice, [20, 40, 60, 80]), at_slice[-1])), 4)
                else:
                    at_bucket = 2
                tl.process_signal(alloc.asset, sig, prices.get(alloc.asset, 0.0), timestamp, at_bucket, capital / max(len(self.assets), 1))
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

    def run(self) -> dict:
        spread_results = {}
        for mult in self.SPREAD_MULTIPLIERS:
            spread_results[f"{mult}x"] = self._run_backtest(float(mult), 0.0)
        slippage_results = {}
        for pips in self.SLIPPAGE_PIPS:
            slippage_results[str(pips)] = self._run_backtest(1.0, float(pips))
        max_tolerable_spread = 1.0
        for mult in sorted(self.SPREAD_MULTIPLIERS):
            if spread_results[f"{mult}x"].get("sharpe", 0.0) > 0.5:
                max_tolerable_spread = float(mult)
        max_tolerable_slippage = 0.0
        for pips in sorted(self.SLIPPAGE_PIPS):
            if slippage_results[str(pips)].get("sharpe", 0.0) > 0.5:
                max_tolerable_slippage = float(pips)
        return {
            "spread_results": spread_results,
            "slippage_results": slippage_results,
            "max_tolerable_spread": max_tolerable_spread,
            "max_tolerable_slippage": max_tolerable_slippage,
        }

    def save(self, path: str):
        results = self.run()
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
