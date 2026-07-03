import numpy as np
from numpy.typing import NDArray

from research.trading_relevance.mechanism_interaction import MechanismInteractionAnalyzer
from research.trading_relevance.economic_value import EconomicValueAnalyzer
from research.trading_relevance.risk_profile import RiskProfileAnalyzer
from research.trading_relevance.trade_survivability import TradeSurvivabilityAnalyzer
from research.trading_relevance.cross_asset import CrossAssetRelevanceAnalyzer
from research.trading_relevance.outcome_distribution import (
    OutcomeDistributionAnalyzer, _time_to_return_threshold_batch,
    _time_to_drawdown_threshold_batch, _time_to_state_change_batch,
    _volatility_batch, _drawdowns_batch, _runups_batch,
)

BUCKET_LABELS = ["very_low", "low", "medium", "high", "extreme"]

def _quantile_buckets(arr: NDArray, n_buckets: int) -> NDArray[np.int32]:
    n = len(arr)
    out = np.zeros(n, dtype=np.int32)
    order = np.argsort(arr)
    for i in range(n):
        out[order[i]] = min(int(i * n_buckets / n), n_buckets - 1)
    return out

def experiment_a(adaptive_time, returns, energy_storage, memory_density, memory_gradient, states):
    """Regime Filter — does adaptive_time improve mechanism quality per bucket?"""
    ma = MechanismInteractionAnalyzer()
    report = ma.compute(adaptive_time, returns, energy_storage, memory_density, memory_gradient, states)
    return {
        "adaptive_time_only_ig": report.adaptive_time_only_ig,
        "energy_only_ig": report.energy_only_ig,
        "memory_only_ig": report.memory_only_ig,
        "combined_ig": report.combined_ig,
        "adaptive_time_improvement": report.adaptive_time_improvement,
        "ig_difference": report.ig_difference,
        "sid_difference": report.sid_difference,
        "sir_difference": report.sir_difference,
        "verdict": report.verdict,
    }

def experiment_b(adaptive_time, returns):
    """Decision Quality — does adaptive_time reduce uncertainty?"""
    ea = EconomicValueAnalyzer()
    report = ea.compute(adaptive_time, returns)
    return {
        "unconditioned_entropy": report.unconditioned_entropy,
        "conditioned_entropy": report.conditioned_entropy,
        "information_gain": report.information_gain,
        "uncertainty_reduction": report.uncertainty_reduction,
        "distribution_separation": report.distribution_separation,
        "verdict": report.verdict,
    }

def experiment_c(adaptive_time, returns, states):
    """Risk Conditioning — does adaptive_time improve risk estimation?"""
    ra = RiskProfileAnalyzer()
    report = ra.compute(adaptive_time, returns, states)
    buckets_out = {}
    for label, bkt in report.buckets.items():
        buckets_out[label] = {
            "count": bkt.count,
            "risk_score": bkt.risk_score,
            "future_volatility": bkt.future_volatility,
            "future_entropy": bkt.future_entropy,
            "future_state_mutation": bkt.future_state_mutation,
            "future_regime_change": bkt.future_regime_change,
        }
    return {"buckets": buckets_out, "verdict": report.verdict}

def experiment_d(adaptive_time, returns):
    """Position Sizing — equal_weight vs adaptive_time_weighted."""
    n = len(adaptive_time)
    buckets = _quantile_buckets(adaptive_time, 5)
    results = {}
    for b in range(5):
        label = BUCKET_LABELS[b]
        mask = buckets == b
        at_b = adaptive_time[mask]
        ret_b = returns[mask]
        cnt = mask.sum()
        if cnt < 3:
            results[label] = {"count": cnt, "eq_sharpe": 0, "at_sharpe": 0, "sharpe_improvement": 0, "eq_max_dd": 0, "at_max_dd": 0, "dd_improvement": 0, "eq_variance": 0, "at_variance": 0}
            continue
        at_min, at_max = float(at_b.min()), float(at_b.max())
        span = at_max - at_min if at_max > at_min else 1.0
        weights = 0.5 + (at_b - at_min) / span
        w_ret = weights * ret_b
        eq_sharpe = float(np.mean(ret_b) / np.std(ret_b) * np.sqrt(252)) if np.std(ret_b) > 1e-15 else 0.0
        at_sharpe = float(np.mean(w_ret) / np.std(w_ret) * np.sqrt(252)) if np.std(w_ret) > 1e-15 else 0.0
        eq_eq = np.cumprod(1 + ret_b)
        at_eq = np.cumprod(1 + w_ret)
        eq_mdd = float(_max_drawdown(eq_eq))
        at_mdd = float(_max_drawdown(at_eq))
        results[label] = {
            "count": cnt, "eq_sharpe": eq_sharpe, "at_sharpe": at_sharpe,
            "sharpe_improvement": at_sharpe - eq_sharpe,
            "eq_max_dd": eq_mdd, "at_max_dd": at_mdd,
            "dd_improvement": at_mdd - eq_mdd,
            "eq_variance": float(np.var(ret_b)), "at_variance": float(np.var(w_ret)),
        }
    all_sharpes = [r["sharpe_improvement"] for r in results.values() if r["count"] >= 3]
    avg_improvement = float(np.mean(all_sharpes)) if all_sharpes else 0.0
    return {"buckets": results, "avg_sharpe_improvement": avg_improvement}

def _max_drawdown(equity: NDArray) -> float:
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < mdd:
            mdd = dd
    return mdd

def experiment_e(adaptive_time, returns, states, price):
    """Holding Period — does adaptive_time imply different holding horizons?"""
    n = len(adaptive_time)
    buckets = _quantile_buckets(adaptive_time, 5)
    max_h = 500
    results = {}
    for b in range(5):
        label = BUCKET_LABELS[b]
        mask = buckets == b
        cnt = int(mask.sum())
        if cnt < 2:
            results[label] = {"count": cnt, "time_to_profit": max_h, "time_to_loss": max_h, "time_to_state_change": max_h}
            continue
        starts = np.where(mask)[0].astype(np.int64)
        ttp = float(np.mean(_time_to_return_threshold_batch(returns, starts, 0.01, max_h)))
        ttl = float(np.mean(_time_to_drawdown_threshold_batch(price, starts, -0.01, max_h)))
        tts = float(np.mean(_time_to_state_change_batch(states, starts, max_h)))
        results[label] = {"count": cnt, "time_to_profit": ttp, "time_to_loss": ttl, "time_to_state_change": tts}
    return {"buckets": results}

def experiment_f(adaptive_time, returns):
    """Trade Survivability — does adaptive_time affect trade durability?"""
    sa = TradeSurvivabilityAnalyzer()
    report = sa.compute(adaptive_time, returns)
    buckets_out = {}
    for label, bkt in report.buckets.items():
        buckets_out[label] = {
            "count": bkt.count,
            "probability_positive": bkt.probability_positive,
            "probability_negative": bkt.probability_negative,
            "average_time_to_profit": bkt.average_time_to_profit,
            "average_time_to_loss": bkt.average_time_to_loss,
            "survival_ratio": bkt.survival_ratio,
        }
    return {"buckets": buckets_out, "verdict": report.verdict}

def experiment_g(asset_data: dict):
    """Cross-Asset — does operational value transfer?"""
    ca = CrossAssetRelevanceAnalyzer(list(asset_data.keys()))
    report = ca.compute(asset_data)
    per_asset_out = {a: {
        "outcome_separation": m.outcome_separation,
        "risk_score": m.risk_score,
        "survivability_ratio": m.survivability_ratio,
        "information_gain": m.information_gain,
    } for a, m in report.per_asset.items()}
    return {
        "per_asset": per_asset_out,
        "outcome_separation_consistency": report.outcome_separation_consistency,
        "risk_consistency": report.risk_consistency,
        "survivability_consistency": report.survivability_consistency,
        "verdict": report.verdict,
    }

def experiment_h(adaptive_time, returns):
    """Economic Value — mechanism vs mechanism + adaptive_time conditioning."""
    od = OutcomeDistributionAnalyzer()
    ea = EconomicValueAnalyzer()
    od_report = od.compute(adaptive_time, returns, np.zeros(len(adaptive_time), dtype=np.int64), np.ones(len(adaptive_time)))
    ev_report = ea.compute(adaptive_time, returns)
    return {
        "outcome_separation_avg": od_report.outcome_separation_avg,
        "information_gain_avg": od_report.information_gain_avg,
        "uncertainty_reduction": ev_report.uncertainty_reduction,
        "distribution_separation": ev_report.distribution_separation,
        "economic_verdict": ev_report.verdict,
        "outcome_verdict": od_report.verdict,
    }
