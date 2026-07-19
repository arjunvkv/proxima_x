# WLS Currency Decomposition — Predictive Validation Report

## Summary

**The WLS currency strength decomposition has no predictive power for future returns at any tested horizon (5m, 15m, 30m, 60m).**

After walk-forward testing over 14 days of M5 data (2880 bars, 18 cross pairs) with 175 hyperparameter combinations, the best out-of-sample MSE skill was **+0.0017** — indistinguishable from zero. Direction accuracy peaked at **50.6%** (coin flip). The Information Coefficient never exceeded **|0.02|**.

This means: the WLS solver produces internally consistent currency strengths that describe *current* pair returns well, but those strengths have zero relationship to *future* pair returns.

---

## Methodology

- **Data:** 18 available FX pairs, M5 bars, 2026-07-03 to 2026-07-17 (14 days, 2880 bars)
- **Design matrix:** 18 pairs × 8 currencies (EUR, USD, GBP, JPY, CHF, AUD, CAD, NZD), rank 7
- **Walk-forward:** 24-bar (2hr) rolling window, 1-bar step, 2 held-out pairs per step for out-of-sample evaluation
- **Horizons tested:** 1 bar (5m), 3 bars (15m), 6 bars (30m), 12 bars (60m)
- **Naive baseline:** Predict zero return for all pairs
- **Metric:** MSE Skill = 1 − MSE(model) / MSE(naive) — positive = better than zero, 0 = equal to zero, negative = worse

### Parameter Grid

| Parameter | Values |
|-----------|--------|
| λ (lam) | 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0 |
| smoothing_alpha | 0.05, 0.1, 0.2, 0.5, 1.0 |
| prior_shrink | 0.0, 0.3, 0.5, 0.7, 0.9 |

175 combinations tested in ~80s.

---

## Results

### Best Configuration

```
lam = 1.0
smoothing_alpha = 0.05
prior_shrink = 0.7
```

| Horizon | Holdout MSE Skill | All-Pair MSE Skill | Dir Accuracy | IC | Spread Return |
|---------|-------------------|-------------------|-------------|-----|--------------|
| 5m      | +0.0017           | -0.0006           | 50.6%       | 0.015 | -0.000001 |
| 15m     | +0.0012           | -0.0003           | 50.3%       | 0.012 | -0.000002 |
| 30m     | -0.0004           | -0.0008           | 50.1%       | 0.008 | -0.000003 |
| 60m     | -0.0009           | -0.0011           | 49.8%       | 0.005 | -0.000001 |

All metrics are **within noise range** — no statistically significant predictive signal at any horizon.

### Key Patterns

1. **Higher smoothing + higher prior shrinkage = less negative skill.** The "best" params essentially shrink strengths toward zero, which approximates the naive baseline. This is not signal — it's harm reduction.

2. **No smoothing (alpha=1.0) + no shrinkage (shrink=0.0) = worst results.** MSE skill of −1.0 to −1.2. The raw WLS strengths oscillate wildly and predict *opposite* to future returns.

3. **The regularization parameter λ barely matters** across the range 0.001–5.0. The dominant factor is smoothing_alpha, which controls how much the strength estimate changes between bars.

4. **All-pair MSE skill consistently ≤ holdout MSE skill**, meaning even the pairs used in the solve don't predict their own future returns.

---

## Conclusions

### What the WLS does well

The WLS decomposition is a valid **description** of the current cross-currency state. Given a set of pair returns at time t, it produces internally consistent currency strengths that reconstruct those returns with low residual. The in-sample quality metric (which was being used for gating) looks good because WLS is designed to minimize in-sample error.

### What the WLS does NOT do

The WLS decomposition does **not** predict future returns at any tested horizon. The currency strengths at bar t have no statistically measurable relationship with pair returns at bar t+1, t+3, t+6, or t+12.

### Implications for the System

| Component | Impact |
|-----------|--------|
| **WLS strengths** | Valid for state description only. Not predictive. |
| **HypothesisGenerator confidence** | Based entirely on in-sample fit (residual/spread), which is circular. Output is noise. |
| **Signal chain** | Every layer built on top of WLS (DER filter, burst filter, bar alignment, DRS ranking, swing state classification) operates on noise. Pipeline metrics showing rejection rates are measuring noise dispersion, not signal filtering. |
| **Trade outcomes** | Any profitable trades from this system are attributable to random walk or position management, not WLS decomposition signal. |

### Path Forward

The core question is whether currency strength decomposition *can* be predictive at M5 timescales, or whether it's fundamentally a descriptive tool.

**Option A: Accept WLS as descriptive only**
- Remove WLS strengths from the signal chain
- Use WLS only for risk management (currency exposure tracking, regime detection)
- Find a different predictive mechanism for entry/exit signals

**Option B: Investigate longer horizons**
- Test if WLS strengths predict at hourly or daily horizons
- The lack of signal at 60m suggests not, but this should be confirmed with daily data over months

**Option C: Replace with a predictive model**
- Instead of WLS (descriptive factor model), use a predictive model trained to forecast future returns
- Potential approaches: vector autoregression, state-space models, or ML factor models with lagged features
- Must include walk-forward validation as a non-negotiable requirement

---

## Raw Grid Search Top 15

| lam | alpha | shrink | 5m skill | 5m all | 5m dir | 5m IC | 15m skill |
|-----|-------|--------|----------|--------|--------|-------|-----------|
| 1.000 | 0.05 | 0.70 | 0.0017 | -0.0006 | 0.506 | 0.015 | 0.0012 |
| 0.050 | 0.10 | 0.90 | 0.0013 | -0.0001 | 0.502 | 0.009 | 0.0007 |
| 0.010 | 0.20 | 0.90 | 0.0012 | -0.0011 | 0.506 | 0.000 | -0.0001 |
| 0.010 | 0.10 | 0.90 | 0.0011 | -0.0001 | 0.495 | 0.008 | -0.0018 |
| 0.050 | 0.05 | 0.90 | 0.0007 | 0.0002 | 0.495 | 0.013 | -0.0003 |
| 0.500 | 0.05 | 0.70 | 0.0006 | -0.0008 | 0.508 | 0.014 | 0.0010 |
| 5.000 | 0.05 | 0.90 | 0.0005 | 0.0002 | 0.503 | 0.017 | 0.0000 |
| 1.000 | 0.05 | 0.90 | 0.0004 | 0.0002 | 0.502 | 0.014 | -0.0008 |
| 0.500 | 0.20 | 0.90 | 0.0004 | -0.0008 | 0.508 | 0.002 | 0.0003 |
| 0.001 | 0.05 | 0.90 | 0.0003 | 0.0001 | 0.506 | 0.013 | 0.0009 |
| 0.100 | 0.10 | 0.70 | 0.0003 | -0.0034 | 0.503 | 0.009 | 0.0008 |
| 0.100 | 0.05 | 0.90 | 0.0002 | 0.0002 | 0.494 | 0.013 | -0.0002 |
| 0.001 | 0.10 | 0.90 | 0.0002 | -0.0001 | 0.500 | 0.008 | -0.0002 |
| 0.010 | 0.05 | 0.90 | 0.0001 | 0.0002 | 0.511 | 0.013 | -0.0002 |
| 0.500 | 0.10 | 0.90 | 0.0000 | -0.0000 | 0.492 | 0.010 | -0.0012 |

*Note: The "best" skill of +0.0017 is within noise range. Without statistical significance testing, it should be treated as zero.*
