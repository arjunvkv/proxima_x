from __future__ import annotations

from typing import Any

import numpy as np

from research.interaction_asymmetry.interaction_validator import InteractionValidator, IAEResult
from research.adaptive_alpha_engine.aae_validator import HORIZONS, TARGET_ASSETS, TIME_WINDOWS, _zscore, _future_returns


class TensionSurface:
    def __init__(self, validator: InteractionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> IAEResult:
        self.validator.load(self.asset)
        md_z = self.validator.md_z()
        es_z = self.validator.es_z()
        at_z = self.validator.at_z()

        tension = self.validator.tension_index(md_z, es_z, at_z, window=20)
        tension_z = _zscore(tension.copy())

        n = len(tension)
        p33 = float(np.nanpercentile(tension, 33))
        p67 = float(np.nanpercentile(tension, 67))
        p90 = float(np.nanpercentile(tension, 90))

        regimes = np.full(n, "NORMAL", dtype=object)
        regimes[tension < p33] = "COMPRESSION"
        regimes[(tension >= p33) & (tension < p67)] = "NORMAL"
        regimes[tension >= p67] = "RELEASE"
        regimes[tension > p90] = "INSTABILITY"

        fut_ret = self.validator.fut_ret
        fwd_h20 = fut_ret[:, 2]
        lim = min(len(regimes), len(fwd_h20))
        regimes = regimes[:lim]
        fwd_h20 = fwd_h20[:lim]

        regime_names = ["COMPRESSION", "NORMAL", "RELEASE", "INSTABILITY"]
        tension_regime_metrics: dict[str, Any] = {}
        for rname in regime_names:
            mask = regimes == rname
            count = int(np.sum(mask))
            if count >= 5:
                vals = fwd_h20[mask]
                mean = float(np.nanmean(vals))
                std = float(np.nanstd(vals))
                pp = float(np.mean(vals > 0))
                sharpe = mean / max(std, 1e-12)
            else:
                mean = 0.0
                std = 0.0
                pp = 0.5
                sharpe = 0.0
            tension_regime_metrics[rname] = {
                "mean": mean, "pp": pp, "sharpe": sharpe, "std": std, "n": count, "count": count,
            }

        tension_clean = np.nan_to_num(tension_z, nan=0.0)
        if len(tension_clean) >= 10 and np.sum(tension_clean != 0) >= 5:
            tension_alpha = self.validator.eval_alpha(tension_clean, 2)
        else:
            tension_alpha = {"mean": 0.0, "pp": 0.5, "std": 0.0, "sharpe": 0.0, "n": 0}

        state_mutation = self.validator.signals["state_mutation_rate"]
        mutation_bars = np.where(state_mutation > 0)[0]
        if len(mutation_bars) > 0:
            pre_tens = []
            for mb in mutation_bars:
                start = max(0, int(mb) - 10)
                wt = tension[start:int(mb)]
                wt = wt[~np.isnan(wt)]
                if len(wt) > 0:
                    pre_tens.append(float(np.nanmean(wt)))
            tension_before_mutation = float(np.mean(pre_tens)) if pre_tens else 0.0
        else:
            tension_before_mutation = 0.0

        tail_th = float(np.nanpercentile(tension, 90))
        tail_mask = tension > tail_th
        tlen = min(len(tail_mask), len(fwd_h20))
        tail_mask = tail_mask[:tlen]
        tfwd = fwd_h20[:tlen]
        if np.sum(tail_mask) >= 5:
            tv = tfwd[tail_mask]
            tm = float(np.nanmean(tv))
            ts = float(np.nanstd(tv))
            tp = float(np.mean(tv > 0))
            tsh = tm / max(ts, 1e-12)
        else:
            tm = 0.0
            ts = 0.0
            tp = 0.5
            tsh = 0.0
        tension_tail_alpha = {"mean": tm, "pp": tp, "sharpe": tsh, "std": ts, "n": int(np.sum(tail_mask))}

        benchmark_es = self.validator.benchmark_es_alpha()
        es_pp = benchmark_es.get("pp", 0.5)
        es_sharpe = benchmark_es.get("sharpe", 0.0)
        tension_beats_es = tp > es_pp or tsh > es_sharpe * 1.1

        print("=== Tension Surface (RQ6) ===")
        print(f"\nTension Percentiles: p33={p33:.6f} p67={p67:.6f} p90={p90:.6f}")
        print("\nTension Regime Metrics:")
        for rname in regime_names:
            m = tension_regime_metrics[rname]
            print(f"  {rname}: count={m['count']} mean={m['mean']:.6f} pp={m['pp']:.4f} sharpe={m['sharpe']:.4f}")
        print(f"\nTension Alpha: mean={tension_alpha.get('mean', 0.0):.6f} pp={tension_alpha.get('pp', 0.5):.4f} sharpe={tension_alpha.get('sharpe', 0.0):.4f}")
        print(f"Tension Before Mutation: {tension_before_mutation:.6f}")
        print(f"Tension Tail Alpha (top 10%): mean={tm:.6f} pp={tp:.4f} sharpe={tsh:.4f}")
        print(f"Benchmark ES Alpha:       mean={benchmark_es.get('mean', 0.0):.6f} pp={es_pp:.4f} sharpe={es_sharpe:.4f}")
        print(f"Tension Beats ES: {tension_beats_es}")

        metrics: dict[str, Any] = {
            "tension_regime_metrics": tension_regime_metrics,
            "tension_alpha": tension_alpha,
            "tension_before_mutation": tension_before_mutation,
            "tension_tail_alpha": tension_tail_alpha,
            "benchmark_es_alpha": benchmark_es,
            "tension_beats_es": tension_beats_es,
        }

        return IAEResult(rq_name="RQ6_Tension_Surface", status="COMPLETE", metrics=metrics)
