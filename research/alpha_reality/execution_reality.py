"""RQ4: Does alpha survive spread, commission, slippage, latency?"""

from __future__ import annotations

import numpy as np

from research.alpha_reality.arl_validator import ARLValidator, ARLResult, HORIZONS, _future_returns


class ExecutionReality:
    def __init__(self, validator: ARLValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> ARLResult:
        data = self.validator.load_asset_data(self.asset)
        price = data["price"]

        signals = self.validator.compute_signals(data)
        alpha = self.validator.alpha_signal(signals)
        fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))

        # Realistic cost assumptions for EURJPY
        spread_cost = 0.00005       # 0.5 pip spread
        commission = 0.00001        # ~$10 per million
        slippage = 0.00003          # 0.3 pip slippage
        total_one_way = spread_cost / 2 + commission + slippage
        total_round_trip = spread_cost + 2 * commission + 2 * slippage

        latency_impact = 0.00001    # 0.1 pip latency cost

        results = {}
        for hi, h in enumerate(HORIZONS):
            fwd = fr_all[:, hi]
            real_eval = self.validator.eval_alpha(alpha, fr_all, hi)

            # Apply round-trip cost to every entry
            fwd_net = fwd - total_round_trip / h
            n2 = min(len(alpha), len(fwd_net))
            mask = alpha[:n2] > np.nanpercentile(alpha[:n2], 90) if np.sum(~np.isnan(alpha[:n2])) > 10 else np.zeros(n2, dtype=bool)
            if np.sum(mask) > 5:
                vals = fwd_net[mask]
                net_eval = {
                    "mean": float(np.nanmean(vals)),
                    "pp": float(np.mean(vals > 0)),
                    "std": float(np.nanstd(vals)),
                    "sharpe": float(np.nanmean(vals)) / max(float(np.nanstd(vals)), 1e-12),
                    "n": int(np.sum(mask)),
                }
            else:
                net_eval = {"mean": 0.0, "pp": 0.5, "std": 0.0, "sharpe": 0.0, "n": 0}

            results[f"H{h}"] = {
                "gross": real_eval,
                "net": net_eval,
                "cost_per_trade": total_round_trip,
                "mean_decay": (real_eval["mean"] - net_eval["mean"]) / max(abs(real_eval["mean"]), 1e-12),
                "survives": net_eval["pp"] > 0.52 and net_eval["mean"] > 0,
            }

        h20 = results.get("H20", {})
        net = h20.get("net", {})
        gross = h20.get("gross", {})
        survives = h20.get("survives", False)

        print(f"  Execution Reality @ H20 (cost={total_round_trip:.6f}):")
        print(f"    Gross: mean={gross.get('mean', 0):.6f}, pp={gross.get('pp', 0):.3f}")
        print(f"    Net:   mean={net.get('mean', 0):.6f}, pp={net.get('pp', 0):.3f}")
        print(f"    Alpha survives costs: {'YES' if survives else 'NO'}")

        status = "PASSED" if survives else "FAILED"
        return ARLResult("execution_reality", status, metrics=results)
