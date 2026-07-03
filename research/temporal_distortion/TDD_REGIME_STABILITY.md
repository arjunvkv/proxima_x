# TDD-VL Phase 1: Regime Stability Audit

**Date:** 2026-06-16
**Asset Universe:** EURJPY, USDJPY (10.4M + 6.3M ticks)
**Method:** 60s event rate, 5-min bar grid, sync_up at H50

---

## Method

Regimes classified per bar using 500-bar rolling statistics:

| Regime | Condition |
|--------|-----------|
| BULL | Rolling P(up) > 0.55 |
| BEAR | Rolling P(up) < 0.45 |
| RANGE | 0.45 ≤ Rolling P(up) ≤ 0.55 |
| VOL_EXPAND | Rolling ATR > 80th percentile of all ATR |
| VOL_CONTRACT | Rolling ATR < 20th percentile of all ATR |
| Combined | Regime + VOL state (e.g., BEAR_VOL_EXPAND) |

---

## Results: EURJPY

Unconditional H50 P(up) = 0.404

| Regime | Bars | P(up\|regime) | SyncUp_n | P(up\|sync_up) | Edge | Verdict |
|--------|------|--------------|----------|----------------|------|---------|
| BEAR | 6,165 | 0.3212 | 460 | **0.6000** | +0.279 | PASS |
| BEAR_VOL_CONTRACT | 5,084 | 0.3017 | 140 | **0.5000** | +0.198 | **BORDERLINE** |
| BEAR_VOL_EXPAND | 1,249 | 0.2466 | 114 | **0.5614** | +0.315 | PASS |
| BULL | 4,813 | 0.5340 | 442 | **0.5294** | -0.005 | PASS |
| BULL_VOL_EXPAND | 2,368 | 0.5059 | 430 | **0.6442** | +0.138 | PASS |
| RANGE | 4,274 | 0.5098 | 311 | **0.7395** | +0.230 | PASS |
| RANGE_VOL_EXPAND | 1,467 | 0.4315 | 208 | **0.5481** | +0.117 | PASS |

**EURJPY: 6/7 regimes pass** (only BEAR_VOL_CONTRACT borderline at 0.5000)

## Results: USDJPY

Unconditional H50 P(up) = 0.412

| Regime | Bars | P(up\|regime) | SyncUp_n | P(up\|sync_up) | Edge | Verdict |
|--------|------|--------------|----------|----------------|------|---------|
| BEAR | 6,283 | 0.2984 | 459 | **0.4706** | +0.172 | **FAIL** |
| BEAR_VOL_CONTRACT | 4,977 | 0.2636 | 151 | **0.2781** | +0.014 | **FAIL** |
| BEAR_VOL_EXPAND | 1,490 | 0.3732 | 203 | **0.5172** | +0.144 | PASS |
| BULL | 6,161 | 0.5884 | 336 | **0.6488** | +0.060 | PASS |
| BULL_VOL_EXPAND | 1,957 | 0.5416 | 317 | **0.6877** | +0.146 | PASS |
| RANGE | 2,808 | 0.4003 | 188 | **0.4734** | +0.073 | **FAIL** |
| RANGE_VOL_EXPAND | 1,637 | 0.5174 | 269 | **0.5651** | +0.048 | PASS |

**USDJPY: 4/7 regimes pass** — BEAR (0.471), BEAR_VOL_CONTRACT (0.278), RANGE (0.473) all fail.

---

## Verdict: **FAIL — Regime-Dependent**

| Symbol | Regimes Passed | Regimes Failed | Robustness |
|--------|---------------|---------------|------------|
| EURJPY | 6/7 | 1 (borderline) | HIGH |
| USDJPY | 4/7 | 3 (including 0.278) | **LOW** |

The sync_up signal is **regime-dependent**:
- Strong in BULL, BULL_VOL_EXPAND, and RANGE_VOL_EXPAND regimes across both symbols
- Weak or flipped in BEAR and RANGE regimes (especially USDJPY)
- Worst case: USDJPY BEAR_VOL_CONTRACT gives P(up|sync_up) = **0.2781** — actively harmful
- Best case: EURJPY RANGE gives P(up|sync_up) = **0.7395** — strongly predictive

Any deployment would require regime-aware gating. The signal cannot be used unconditionally.
