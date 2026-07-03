import json
import numpy as np
from proxima_v1.core.signal_engine import SignalEngine, SignalResult
from proxima_v1.core.risk_engine import RiskEngine
from proxima_v1.core.execution_engine import ExecutionEngine
from proxima_v1.core.position_manager import PositionManager
from proxima_v1.core.trade_lifecycle import TradeLifecycle
from proxima_v1.core.portfolio_engine import PortfolioEngine
from proxima_v1.core.metrics import MetricsTracker, TradeRecord


class PortfolioCorrelation:
    def __init__(self, assets: list[str] | None = None):
        self.assets = assets or ["EURJPY", "USDJPY", "GBPJPY"]

    def run(self) -> dict:
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
        pm = PositionManager(3)
        tl = TradeLifecycle(risk, exec_eng, pm)
        prev_asset_pnl: dict[str, float] = {a: 0.0 for a in self.assets}
        prev_equity = float(capital)
        asset_pnl_series: dict[str, list[float]] = {a: [] for a in self.assets}

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

            period_asset_pnl: dict[str, float] = {}
            for asset in self.assets:
                pos = pm.get_position(asset)
                if pos is not None and pos.is_open:
                    mtm = float(pos.size * (prices.get(asset, pos.entry_price) - pos.entry_price))
                else:
                    mtm = 0.0
                closed_asset_pnl = sum(t.pnl for t in metrics.trades if t.asset == asset)
                period_asset_pnl[asset] = mtm + closed_asset_pnl - prev_asset_pnl[asset]
                asset_pnl_series[asset].append(period_asset_pnl[asset])

            total_equity = capital + sum(p.pnl for p in pm.open_positions) + sum(t.pnl for t in metrics.trades)
            period_pnl = total_equity - prev_equity
            metrics.record_daily_pnl(period_pnl, total_equity)
            prev_equity = total_equity
            for asset in self.assets:
                prev_asset_pnl[asset] = sum(t.pnl for t in metrics.trades if t.asset == asset) + (pm.get_position(asset).pnl if pm.get_position(asset) is not None and pm.get_position(asset).is_open else 0.0)

        pnl_matrix = np.column_stack([np.array(asset_pnl_series[a], dtype=np.float64) for a in self.assets])
        if pnl_matrix.shape[0] < 10:
            return {"pairwise_correlations": {}, "mean_correlation": 0.0, "effective_independent_positions": 1.0, "stress_correlation": 0.0, "normal_correlation": 0.0, "positions_during_stress": 1.0}

        pnl_matrix = np.nan_to_num(pnl_matrix, nan=0.0)
        corr_matrix = np.corrcoef(pnl_matrix.T)
        pairs = {}
        for i in range(len(self.assets)):
            for j in range(i + 1, len(self.assets)):
                key = f"{self.assets[i]}_{self.assets[j]}"
                pairs[key] = float(np.clip(corr_matrix[i, j], -1.0, 1.0))

        tri_upper = []
        for i in range(len(self.assets)):
            for j in range(i + 1, len(self.assets)):
                tri_upper.append(float(corr_matrix[i, j]))
        mean_corr = float(np.mean(tri_upper)) if tri_upper else 0.0
        mean_corr = float(np.clip(mean_corr, -1.0, 1.0))
        eff_pos = 1.0 / max(abs(mean_corr), 0.01)

        portfolio_returns = np.mean(pnl_matrix, axis=1)
        vol_threshold = float(np.nanpercentile(np.abs(portfolio_returns), 80))
        stress_mask = np.abs(portfolio_returns) >= vol_threshold
        normal_mask = ~stress_mask

        if np.sum(stress_mask) < 5 or np.sum(normal_mask) < 5:
            return {"pairwise_correlations": pairs, "mean_correlation": mean_corr, "effective_independent_positions": eff_pos, "stress_correlation": 0.0, "normal_correlation": 0.0, "positions_during_stress": 1.0}

        stress_corr_matrix = np.corrcoef(pnl_matrix[stress_mask].T)
        stress_corr = float(np.mean([stress_corr_matrix[i, j] for i in range(len(self.assets)) for j in range(i + 1, len(self.assets))]))
        stress_corr = float(np.clip(stress_corr, -1.0, 1.0))

        normal_corr_matrix = np.corrcoef(pnl_matrix[normal_mask].T)
        normal_corr = float(np.mean([normal_corr_matrix[i, j] for i in range(len(self.assets)) for j in range(i + 1, len(self.assets))]))
        normal_corr = float(np.clip(normal_corr, -1.0, 1.0))

        stress_eff = 1.0 / max(abs(stress_corr), 0.01)

        return {
            "pairwise_correlations": pairs,
            "mean_correlation": mean_corr,
            "effective_independent_positions": eff_pos,
            "stress_correlation": stress_corr,
            "normal_correlation": normal_corr,
            "positions_during_stress": stress_eff,
        }

    def save(self, path: str):
        results = self.run()
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
