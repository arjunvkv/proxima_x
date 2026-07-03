# Residual Physics Research Program — Formal Closure

**Date:** 2026-06-16
**Program Span:** DPL → CDER → DSR → ROL → OMS → LSV
**Final Classification:** STRUCTURAL_ARTIFACT

---

## 1. Original Hypothesis

```
Direction = f(Energy Storage)
```

Energy Storage (ES) was hypothesized to measure directional movement potential. The program goal was to discover the modulating function that converts ES magnitude into directional resolution.

## 2. The Research Chain

### Phase 1: Directional Physics Lab (DPL)
| Experiment | Finding | Verdict |
|------------|---------|---------|
| DPL-1: Magnitude vs Direction | ES alone → P(up|high ES)=0.478 | **Zero directional signal** |
| DPL-2: Residual Direction | Residual sign → directional influence | Survived |
| DPL-3: Memory Positioning | Memory weakly modulates | Weak signal |
| DPL-4: Energy Gradient | Gradient theory → corr≈0.0 | **Failed** |
| DPL-5: State Transitions | Transitions matter | Confirmed |
| DPL-6: Regime Sign Inversion | XAUUSD S0=0.67 vs S1=0.45 | **Confirmed inversion** |
| DPL-7: Information Flow | EURJPY→GBPJPY at 89% | Survived |
| DPL-8: Tournament | ES alone worthless, context matters | **Context confirmed** |

**Classification:** REGIME_DEPENDENT_DIRECTION

### Phase 2: Context-Dependent Energy Release (CDER)
6-layer framework discovered:
- Regime Control (primary gate)
- Residual Physics (directional signal, H=0.86)
- Memory Geometry (modulator)
- ES×Memory Interaction (cross-term)
- Multi-Timeframe Context (bias correction)
- Information Propagation (cross-asset)

**Estimated ceiling:** ~71%

### Phase 3: Directional State Reconstruction (DSR)
| Phase | Experiment | Key Finding |
|-------|------------|-------------|
| 1 | State Reconstruction | 27.1% stable directional states |
| 2 | Regime×Residual Surface | XAUUSD P(up)=0.943 in regime1×Q0 |
| 3 | Memory Gate | Imbalance improves separation +0.073 |
| 4 | Transition Physics | Transitions 2× more predictive than states |
| 5 | State Persistence | Half-life ~10.6 bars |
| 6 | Cross-Asset Cascade | EURJPY→GBPJPY→USDJPY at 66-69% |
| **7** | **Walk-Forward** | **residual_only → 74.21% H50 OOS** |
| 8 | Architecture Compression | Every extra feature degrades |

**Classification upgraded:** DEPLOYABLE_DIRECTIONAL_ENGINE

### Phase 4: Residual Origin Lab (ROL)
| Question | Finding | Verdict |
|----------|---------|---------|
| ROL-1: Sign Flip Causes | Both flip types → ~68% up | **Polarity irrelevant** |
| ROL-2: Persistence Physics | H>1 across all symbols | Non-stationary residual space |
| ROL-3: Pressure Surface | Sign (66.4%) beats mag×dur (57.6%) | **Magnitude irrelevant** |
| ROL-4: Cross-Asset Residual | Price beats residual 68.8% | Residual doesn't propagate |
| ROL-5: Memory Coupling OOS | Memory degrades -5.3% | **Zero gain OOS** |

### Phase 5: Observable Market State (OMS)
| Question | Finding | Verdict |
|----------|---------|---------|
| OMS-1: Drift Interaction | Δ ≈ 0 at aggregate | **Not drift amplification** |
| OMS-2: Volatility Expansion | Δ < 4pp | **Not volatility detection** |
| OMS-3: Cohort Sync | Sync ≈ sign, doesn't replace | **Not synchronization** |

### Phase 6: Latent State Verification (LSV) — THE KILLER
| Experiment | Finding | Verdict |
|------------|---------|---------|
| LSV-1: Synthetic Null Models | **All 5 variants preserve full edge** | **STRUCTURAL ARTIFACT** |
| LSV-2: Generator Sensitivity | Polarity flips with residual definition | **Measurement artifact** |
| LSV-3: Minority State | Marker=0=10.9% of edge | Not the driver |
| LSV-4: Global Field | Sync=0.75, local beats global | Real but irrelevant |

## 3. The Final Explanation

The residual sign edge is a **structural artifact** of:

1. **Persistence Geometry**: Residual sign sequences have high autocorrelation (H=0.86, mean run length 15-18 bars)
2. **Class Imbalance**: The market has an upward drift (P(up) ≈ 55-60% at H50 for JPY crosses)
3. **Horizon Overlap**: At H50, the prediction horizon is 50 bars, but sign sequences persist 15-18 bars — meaning the sign at time t contains information about the next 50 bars simply because it persists into that window

Under these conditions:
```
persistent state process
+ directional base rate
= apparent predictive signal
```

The residual sign is a **convenient carrier** of this persistence structure, not a cause.

### Proof: The Synthetic Null Test
Shuffled, lagged, Markov, fGn, and random-persistence sign sequences ALL reproduce the same 74% H50 accuracy (edge collapse ratio ≈ 1.0). The market information content of the real residual sign is indistinguishable from zero.

### Proof: The Generator Test
Changing the residual construction method (linear vs RF vs XGBoost vs return-based vs MAE) flips the edge polarity. A genuine latent market state would survive changes in reasonable measurement operators. This is classic evidence of a measurement artifact — a **coordinate-system effect**, not a physical field.

## 4. What Was Actually Discovered

Not a directional engine. Not a hidden market state. Not a deployable signal.

What was discovered:

| Discovery | Value |
|-----------|-------|
| ES is a magnitude predictor, NOT directional | Confirmed across 5 assets, 300M ticks, 8 years |
| Sign persistence in residual space is extreme (H=0.86) | Novel measurement of residual geometry |
| Synthetic counterfactuals can distinguish market phenomena from structural artifacts | **New validation principle** |
| The entire CDER framework (6 layers) is unnecessary — the edge comes from persistence alone | Architecture compression proof |
| Feature ablation OOS shows every extra variable degrades performance | Occam's razor in practice |

## 5. Retired Branches

The following are formally retired as alpha-generation candidates:

- ✅ DPL (Directional Physics Lab)
- ✅ CDER (Context-Dependent Energy Release)
- ✅ DSR (Directional State Reconstruction)
- ✅ ROL (Residual Origin Lab)
- ✅ OMS (Observable Market State)
- ✅ LSV (Latent State Verification)

Not because they are wrong. Because they have been **explained**.

## 6. The Validation Principle

> Any proposed market phenomenon must outperform synthetic processes with equivalent persistence structure before it can be considered market-linked.

This is the strongest legacy of the research program. Future discoveries should pass through this gate in Phase X.

### Implementation:
```
Phase X — Synthetic Counterfactual Test:
1. Identify the statistical structure of the candidate signal
   (autocorrelation, run lengths, transition probabilities)
2. Generate synthetic variants that preserve this structure
   (shuffled, Markov, fGn, random persistence)
3. If any synthetic variant reproduces the edge → STRUCTURAL ARTIFACT
4. If ALL synthetic variants fail → genuine market phenomenon
```

## 7. Outstanding Items

### POSITION_EXISTS Dataset (Priority 4)
- 479 blocked signals identified (all from 2026-06-16 live demo)
- Cannot be backfilled with future returns (market data ends Dec 2025)
- **Recommendation:** Add forward-return tracking to SignalFunnel for future signals
- Review at 50+ executed trades milestone

### Deployment (V2.5 Freeze)
- Continue paper trading UNINTERRUPTED
- No deployment changes
- No direction filters added to Proxima
- Re-evaluate at 100 trades, 300 trades, 500 trades

## 8. Key Files

| File | Content |
|------|---------|
| `research/directional_physics/reports/` | DPL (8 reports) + CDER (7 reports) |
| `research/directional_state/reports/` | DSR (8 reports) + Final Adjudication |
| `research/residual_origin/reports/` | ROL (5 reports) |
| `research/observable_market_state/reports/` | OMS (3 reports) |
| `research/latent_state_verification/reports/` | LSV (4 reports) |
| `proxima_ops/reports/OQ7_ObservabilityTestReport.md` | Observability test |
| `proxima_ops/reports/OQ8_LiveObservabilityVerification.md` | Live verification |

---

*The most valuable result of this program is not a trading signal. It is the demonstration that a statistically significant, walk-forward-validated, cross-asset-robust signal can be a structural artifact — and the methodology to prove it.*
