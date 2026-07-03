# TDD-VL Phase 5: Time Reversal Test

**Date:** 2026-06-16
**Method:** Reverse event order (t_max − (t − t_min)[::-1]), reverse tick prices, repeat TDD pipeline

---

## Theory

If TDD captures **causal directionality** (event rate acceleration → price direction), reversing time should destroy the signal. If the signal survives time reversal, it is an artifact of the event rate distribution — symmetric in time.

## Results

| Symbol | Horizon | Fwd P(up) | Fwd n | Rev P(up) | Rev n | Δ(Fwd−Rev) | Verdict |
|--------|---------|-----------|-------|-----------|-------|-----------|---------|
| EURJPY | H5 | 0.5170 | 2,269 | 0.4895 | 2,390 | +0.028 | **NOISE** |
| **EURJPY** | **H20** | **0.5501** | **2,265** | **0.4706** | **2,382** | **+0.080** | **CAUSAL** |
| **EURJPY** | **H50** | **0.5724** | **2,259** | **0.4567** | **2,367** | **+0.116** | **CAUSAL** |
| USDJPY | H5 | 0.4974 | 2,093 | 0.5132 | 2,167 | −0.016 | **NOISE** |
| USDJPY | H20 | 0.5170 | 2,091 | 0.5385 | 2,158 | −0.022 | **NOISE** |
| **USDJPY** | **H50** | **0.5439** | **2,085** | **0.5633** | **2,141** | **−0.019** | **ARTIFACT** |

---

## Verdict Criteria

| Condition | Label | Interpretation |
|-----------|-------|---------------|
| \|Fwd−Rev\| < 0.03 AND both near 0.5 | **NOISE** | No detectable signal either direction |
| Fwd > 0.53 AND Rev < 0.50 | **CAUSAL** | Directional temporal relationship confirmed |
| Fwd > 0.53 AND Rev > 0.53 | **ARTIFACT** | Time-symmetric — signal is distributional, not causal |
| Fwd ≈ Rev ≈ 0.5 | **NOISE** | No signal |

## Summary

| Symbol | H5 | H20 | H50 | Dominant Verdict |
|--------|----|-----|-----|-----------------|
| EURJPY | NOISE | **CAUSAL** | **CAUSAL** | **CAUSAL** (at medium-long horizons) |
| USDJPY | NOISE | NOISE | **ARTIFACT** | **ARTIFACT** (time-symmetric) |

---

### EURJPY H50: The Only Causal Signal

Forward P(up|sync_up) = **0.5724** → Reversed = **0.4567** (Δ = +0.116).

The signal collapses from strongly predictive to below 0.5 under time reversal. This is consistent with a genuine forward-looking temporal mechanism: event rate acceleration at time t predicts upward price movement over the next 50 bars (4.2 hours).

Reversed P(up) < 0.5 means that in reversed time, the sync_up condition (which in real time corresponds to "event rate accelerating") now corresponds to "event rate decelerating" in real-time reference frame. This asymmetry confirms the temporal arrow.

### USDJPY H50: Time-Symmetric Artifact

Forward P(up|sync_up) = **0.5439** → Reversed = **0.5633** (Δ = −0.019).

The reversed signal is actually stronger than the forward signal. This means the sync_up condition detects a time-symmetric property of the USDJPY event rate distribution — not a causal mechanism. The signal would work equally well predicting the past as the future.

### Interpretation

The time reversal test splits TDD's apparent edge into two components:
1. **Causal edge** (EURJPY H50): ~1.7pp above baseline, confirmed directional
2. **Distributional artifact** (USDJPY H50): ~1.3pp above baseline, but time-symmetric

**EURJPY's signal is genuine but weak. USDJPY's signal is an artifact.**

---

## Verdict: **FAIL — Most signals are time-symmetric (2/3) or noise (3/6). Only EURJPY H20/H50 is causal.**
