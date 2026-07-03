# Directional Physics Lab (DPL) — Final Report

## Adjudication: REGIME_DEPENDENT_DIRECTION

## 1. Executive Summary

Energy Storage (ES) is fundamentally a **magnitude predictor**, not a directional predictor. Across 5 assets and 6 horizons, **ES correlates more strongly with |future return| than signed future return** in 23/30 (76.7%) of cases. Directional signal exists but is **weak (~55-60% accuracy)** and emerges primarily from **ES prediction residuals** modulated by **regime and memory geometry**.

| Question | Answer |
|----------|--------|
| Is ES directional? | **NO** — magnitude-only for 3/5 assets |
| Is ES magnitude-only? | **YES** — at trading horizons (H1-H20), purely magnitude |
| What determines long vs short? | Residual sign + regime interaction + memory positioning (weak ensemble) |
| Strongest directional layer? | **Residual sign** (55.1% accuracy, -0.0036 info gain) |
| Can direction survive walk-forward? | **UNLIKELY** — all layers below 57% |
| Can long-short Proxima be built? | **NOT CURRENTLY** — long-only validated |

> **Bottom Line**: Proxima's ES measures *how much* the market will move, not *which direction*. Directional resolution requires a separate mechanism not yet discovered. Long-only deployment is scientifically validated; short-side remains unproven.

---

## 2. DPL-1: ES Magnitude vs Direction

**Core finding**: ES is primarily a magnitude predictor.

| Asset | Classification | Abs Wins / Total | Key Finding |
|-------|---------------|-----------------|-------------|
| EURJPY | MIXED | 3/6 | Sign beats abs at H50+ (directional at long horizons) |
| USDJPY | MAGNITUDE_ONLY | 5/6 | Abs dominates at all horizons |
| GBPJPY | MIXED | 4/6 | Weak directional at H500 |
| XAUUSD | MAGNITUDE_ONLY | 6/6 | **Strongest magnitude-only** (100%) |
| NAS100 | MAGNITUDE_ONLY | 5/6 | Abs dominates; sign near zero |

**At trading horizons (H1-H50)**: ABS dominates in 14/15 asset-horizon pairs (93.3%). ES = magnitude predictor.

---

## 3. DPL-2: Residual Direction Hypothesis

**Residual sign carries weak directional signal** (~60.5% cross-asset accuracy).

| Residual Type | EURJPY | USDJPY | GBPJPY | XAUUSD | NAS100 |
|--------------|--------|--------|--------|--------|--------|
| XGBoost | 68.2% | 61.0% | 62.5% | 53.8% | 52.9% |
| Random Forest | 65.5% | 58.8% | 61.1% | 54.7% | 58.1% |
| Linear | 67.0% | 65.1% | 59.8% | 54.9% | 64.2% |

Key insight: **Linear residuals perform as well as XGBoost** — the directional signal is simple, not complex. Positive residual → up (55-60% accuracy). Negative residual → down (55-60% accuracy).

---

## 4. DPL-3: Memory Positioning Hypothesis

**Price location relative to memory clusters affects direction**, but pattern is asset-dependent.

| Asset | Memory Below → | Memory Above → | Strength |
|-------|---------------|---------------|----------|
| EURJPY | Higher p_up at H100 (85.5%) | Moderate p_up | Moderate |
| USDJPY | Higher p_up at H100 (68.5%) | Lower p_up | Weak |
| **GBPJPY** | **Strong p_up (72.7% at H20, 88.2% at H100)** | Weaker p_up | **Strongest** |
| XAUUSD | Higher p_up at H100 (84.4%) | Moderate p_up | Moderate |
| NAS100 | Moderate p_up | Higher p_up (68.8% at H20) | Opposite pattern |

GBPJPY shows strongest effect: price below dominant memory cluster → 72.7% probability of up at H20. NAS100 shows **opposite** pattern — above memory center → more likely up.

Memory geometry improves ES baseline in 14/30 (46.7%) of evaluations.

---

## 5. DPL-4: Energy Gradient Theory

**FAILS.** Energy gradient (change in ES) does NOT predict direction.

| Metric | Finding |
|--------|---------|
| Rising gradient → up? | **No** — falling gradient often has higher p_up |
| Mean corr(gradient, direction) | ~0.0 (near zero) |
| Gradient beats ES? | Only 12/38 (31.6%) of evaluations |
| Acceleration predicts? | No |
| Curvature predicts? | No |

Physics analogy disproved: Energy gradient does NOT determine release direction.

---

## 6. DPL-5: State Transition Directionality

Only 3 states consistently detected. Some transitions show directional bias:

| Asset | Best Transition Up Prob | Best N |
|-------|------------------------|--------|
| EURJPY | 83.0% at H100 (n=88) | Low |
| USDJPY | 97.5% at H500 (n=319) | Moderate |
| GBPJPY | 70.6% at H50 (n=517) | Moderate |
| XAUUSD | 86.7% at H100 (n=451) | Moderate |
| NAS100 | 83.1% at H500 (n=307) | Moderate |

Limited by small state space (3 states). Transition matrices are sparse.

---

## 7. DPL-6: Regime Sign Inversion — CONFIRMED

**Same ES state produces opposite directional outcomes in different regimes.**

| Asset | Horizons with Sign Inversion |
|-------|------------------------------|
| EURJPY | H50 |
| USDJPY | H20, H100 |
| **GBPJPY** | **H20, H50, H100** (strongest) |
| **XAUUSD** | **H20, H50, H100, H500** (most inverting) |
| NAS100 | H500 |

**This is the most important finding**: Direction depends on regime. Same ES level → long in regime A, short in regime B. Regime interaction is the strongest candidate for directional resolution.

---

## 8. DPL-7: Information Flow Layer

**Cross-asset information pressure exists but concentrated at long horizons (H500).**

| Edge | Correlation |
|------|-------------|
| EURJPY ← NAS100 @ H500 | 0.470 |
| GBPJPY ← NAS100 @ H500 | 0.432 |
| USDJPY ← NAS100 @ H500 | 0.399 |
| XAUUSD ← GBPJPY @ H500 | 0.381 |
| XAUUSD ← USDJPY @ H500 | 0.369 |

NAS100 is the strongest information source (affects 3/4 other assets at H500). Short-horizon cross-asset flow is negligible — direction is predominantly asset-specific at trading horizons.

---

## 9. DPL-8: Directional Survivorship Tournament

| Rank | Candidate | Accuracy | Info Gain | Score |
|------|-----------|----------|-----------|-------|
| **#1** | **residual_sign** | **0.551** | **-0.0036** | **0.6055** |
| #2 | memory_distance | 0.563 | 0.0080 | 0.5862 |
| #3 | regime_interaction | 0.557 | 0.0020 | 0.5679 |
| #4 | energy_gradient | 0.492 | -0.0632 | 0.5548 |
| #5 | state_transition | 0.500 | -0.0547 | 0.5391 |
| #6 | information_pressure | 0.445 | -0.1095 | 0.4814 |

**All candidates score near chance (50%).** No directional layer breaks 57% accuracy. Information gain is near zero for all candidates.

---

## 10. Final Answer: DPL Questions

### Q1: Is Energy Storage directional?
**NO.** ES is overwhelmingly a magnitude predictor. `corr(ES, |return|)` > `corr(ES, return)` in 76.7% of tests.

### Q2: Is Energy Storage magnitude-only?
**YES** for 3/5 assets. MIXED for EURJPY and GBPJPY at very long horizons (H50+). At trading-relevant horizons, purely magnitude-only.

### Q3: What determines long vs short?
**Residual sign** (+60.5%), **memory positioning** (+47% of evaluations), and **regime interaction** (sign inversion confirmed). No single mechanism dominates.

### Q4: What is the strongest directional layer?
**Residual sign** from ES prediction against volatility metrics. Accuracy: 55.1% (tournament winner).

### Q5: Can direction survive walk-forward testing?
**UNLIKELY.** 55-60% accuracy with near-zero information gain → would not survive out-of-sample testing.

### Q6: Can a long-short Proxima system be built?
**NOT CURRENTLY.** Directional signal is too weak for short-side deployment. Long-only is validated.

---

## 11. Proposed Directional Hierarchy

```
Energy Storage (ES)
  │
  ├── Role: MAGNITUDE PREDICTOR
  │   └── Measures tension, instability, potential energy
  │
  ├── Directional Layer: RESIDUAL SIGN (~60%)
  │   ├── ES - predicted_ES(vol metrics) → sign predicts direction
  │   ├── Best at H20-H100 for JPY crosses
  │   └── Weak but real
  │
  ├── Modulating Layer: REGIME INTERACTION
  │   ├── Same ES → opposite direction in different regimes
  │   ├── Confirmed for all 5 assets
  │   └── Provides context-dependent directional resolution
  │
  ├── Secondary Layer: MEMORY GEOMETRY
  │   ├── Price relative to dominant memory clusters
  │   ├── Asset-specific patterns
  │   └── Modest contribution
  │
  └── Conclusion: No robust directional mechanism found
      └── Long-only deployment justified; short-side requires new physics
```

---

## 12. Implications for Deployment

| Aspect | Implication |
|--------|-------------|
| Long-only current approach | **Validated** — ES magnitude signal is real and strong |
| Short-side implementation | **Not justified** — directional accuracy too weak |
| Current BUY-only constraint | **Scientifically correct** — no short-bias evidence |
| Research priority | Directional mechanism needs fundamental discovery, not optimization |
| Next experiments | Investigate regime-memory interaction; test directional ensemble at H50+ |
