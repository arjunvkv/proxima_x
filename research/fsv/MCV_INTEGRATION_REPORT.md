# MacroContextVector Integration Strategy Report

## 1. Current Direction Pipeline (D)

```
D = PhaseA( OSS(drift, ecdf, p_cont) , Shadow(ecdf, entropy) )
```

D is `{-1, 0, +1}` — a discrete ternary signal produced by a 5-state agreement filter between OSS persistence and Shadow entropy-ecdf fusion. Four probes exist (OSS, Shadow, TPI, Alpha) but only OSS and Shadow currently participate in D. TPI and Alpha are informational only.

Confidence U is `p_cont` from OSS (blended across 3/10/20 horizons), clamped to [0,1]. No formal uncertainty propagation exists.

## 2. Existing Fundamental Infrastructure

| Component | File | Capability |
|---|---|---|
| **EconomicIntelligence** | `layer5/__init__.py` | FRED API, ForexFactory; computes z-score surprise, impact-weighted currency scores |
| **NewsIntelligence** | `layer6/__init__.py` | RSS feeds (5 tiers); sentiment, impact, topic classification |
| **FSV Schema** | `research/fsv/core/fsv_schema.py` | 5-field FundamentalStateVector: bias_alignment, macro_pressure, sentiment_gradient, event_risk, regime_stability |
| **FSV Engine** | `research/fsv/core/fsv_engine.py` | Event ingestion, decay (λ=0.01), state merge |
| **FSVModulator** | `research/fsv/integration/fsv_modulator.py` | Conviction modulation ±15% via 5 fields; `post_signal_authority_pre_uesl` integration point |
| **MacroAlignmentEngine** | `research/fsv/phase2/core/macro_alignment.py` | Central bank bias, risk environment, direction alignment |
| **RegimeContextClassifier** | `research/fsv/phase2/core/regime_context.py` | 4-regime classification (risk_on/risk_off/neutral/transition) with regime-aware weights |
| **FundamentalRanker** | `research/fsv/phase2/core/fundamental_ranker.py` | Symbol-level scoring from FSV fields |
| **MacroSnapshotEngine** | `research/fsv/phase2/ingestion/macro_snapshot.py` | Aggregate snapshot, environment detection, alert flags |

### Current Integration Points
- UCF field computed per-cycle (line 2584) but **hardcoded to neutral regime** (line 2588)
- FSVModulator not wired into demo execution path — `_current_fsv_states` exists (line 2587) but no call to `FSVModulator.modulate()` found in the direction pipeline
- Existing `FSVIntegrationPoint` dataclass specifies `post_signal_authority_pre_uesl` but no actual integration code exists

## 3. MCV Field-by-Field Integration Map

### a. risk_sentiment (-1 risk-off, +1 risk-on)
| Impact | Selection | Rationale |
|---|---|---|
| Regime gating | **PRIMARY** | Directly maps to risk_off/risk_on regime — feeds `RegimeContextClassifier` |
| Uncertainty adj | Affects U | Extreme values (±1.0) → increase U by 0.1-0.2 (market pricing instability) |
| Risk filter | Affects sizing | risk_off → reduce position sizing by 0.5× via `_phase6_current_mult` |
| Weight modulation | Affects w1-w4 | risk_on → boost Shadow (w1) +0.1, reduce OSS (w2) -0.05; risk_off → invert |
| NOT D bias | No direct +D | Too high-level for directional correction |

**Source**: MacroAlignmentEngine.evaluate_risk_environment() already computes this from aggregate sentiment_gradient. Needs to be promoted from log-only to active gate.

### b. news_shock_intensity (0-1)
| Impact | Selection | Rationale |
|---|---|---|
| Uncertainty adj | **PRIMARY** | High news intensity → signal-to-noise collapses → U += 0.3×intensity |
| Weight modulation | Affects w3 | TPI is tick-level — breaks down during news events → w3 × (1 - intensity) |
| Weight modulation | Affects w1, w2 | Shift weight from Shadow (entropy-sensitive, volatile during news) to OSS (persistence-based, robust) |
| NOT D bias | No direct +D | News sentiment direction is unreliable (±0.3 accuracy from `_estimate_sentiment`) |
| NOT Risk filter | No sizing cut | News events are normal market activity, not risk events |

**Source**: NewsIntelligence (layer6) → impact-weighted clustering, but shock_intensity requires aggregation across clusters. Currently missing — needs `max(cluster.avg_impact)` or `sum(impacts × magnitude)` across all clusters in a window.

### c. spread_regime (z-score vs baseline)
| Impact | Selection | Rationale |
|---|---|---|
| Risk filter | **PRIMARY** | Spread > 2σ → liquidity stress → block entry (position sizing = 0) |
| Uncertainty adj | Affects U | High spread → execution uncertainty → U += 0.2 × z-score |
| NOT D bias | No direct +D | Spread has no directional information |
| NOT Regime gating | No | Spread is too fast for regime classification |

**Source**: MT5 tick data or price stream. Currently not computed — needs new ingestion from bid/ask spread history.

### d. tick_velocity_anomaly (z-score)
| Impact | Selection | Rationale |
|---|---|---|
| Uncertainty adj | **PRIMARY** | Anomalous tick velocity → microstructure instability → U += 0.25 × z-score |
| Weight modulation | Affects w3 | TPI is tick-imbalance — high anomaly → TPI unreliable → w3 × clamp(1 - abs(z)/5, 0, 1) |
| NOT D bias | No direct +D | Velocity has no directional content |
| NOT Regime gating | No | Too fast, microstructural |

**Source**: Currently exists as entropy topology signal in `_entropy_compression`. Already partially computed. Needs explicit z-scoring against rolling baseline.

### e. carry_bps (annualized carry)
| Impact | Selection | Rationale |
|---|---|---|
| Bias correction | **PRIMARY** | Positive carry → structural long bias: D += sign(carry) × min(abs(carry)/500, 0.15). Slow, persistent, directionally reliable. |
| Regime gating | Supports | Carry trade dominance is a risk_on signal. Large positive carry + risk_sentiment > 0 → confirm risk_on |
| Weight modulation | Affects w2 | OSS persistence may underperform in carry-driven trends → reduce w2 by 0.05-0.1, increase w1 (Shadow) |
| Uncertainty adj | Minor | High carry can mean higher volatility in exotic pairs → U += 0.05 |
| NOT Risk filter | No | Carry is a fundamental, not a risk metric |

**Source**: Interest rate differential from central bank rates (layer5) or broker swap rates. Not currently computed. Needs `get_carry(symbol)` using broker swap long/short values or central bank rate differential.

### f. cpi_surprise (z-score)
| Impact | Selection | Rationale |
|---|---|---|
| Bias correction | **PRIMARY** | +Z CPI → rate hike expectations → long currency: D += sign(z) × min(abs(z)/3, 0.12). Decays over 5-10 cycles. |
| Regime gating | Supports | Positive CPI surprise → shift toward hawkish regime if persistent |
| Uncertainty adj | Affects U | Release-time volatility spike → U += 0.15 × impact_weight for that cycle |
| Weight modulation | Affects w1, w2 | CPI release → shadow entropy spikes → temporarily reduce w1 (Shadow), increase w2 (OSS persistence) |
| NOT Risk filter | No | CPI is directional, not a risk event |

**Source**: EconomicIntelligence.process_event() in layer5 — already computes z-score, impact. CPI surprise = z_score from line 36: `surprise / std`.

### g. employment_surprise (z-score)
| Impact | Selection | Rationale |
|---|---|---|
| Bias correction | **PRIMARY** | +Z employment → economic strength → long currency: D += sign(z) × min(abs(z)/3, 0.10). Slower decay than CPI (10-20 cycles). |
| Uncertainty adj | Affects U | NFP release → volatility → U += 0.1 × impact_weight |
| Regime gating | Supports | Sustained employment surprises change labor market regime |
| NOT Risk filter | No | Employment is directional |
| Weight modulation | Minor | Less disruptive to microstructure than CPI; minimal w1-w4 adjustment |

**Source**: EconomicIntelligence layer5 — NFP/PAYEMS events. Same z-score computation as CPI.

### h. rate_decision_surprise (z-score)
| Impact | Selection | Rationale |
|---|---|---|
| Bias correction | **PRIMARY** | +Z rate decision → immediate policy shift → D += sign(z) × min(abs(z)/2, 0.20). Strongest directional magnitude. |
| Regime gating | **PRIMARY** | Surprise rate decision changes monetary policy regime. Override macro_environment if z > 2. |
| Uncertainty adj | Affects U | Market repricing → U += 0.25 × impact_weight (highest of all surprises) |
| Weight modulation | Affects w1-w4 | Repricing period (next 3-5 cycles): reduce all probe weights, increase U dominance |
| NOT Risk filter | No | Rate decisions are the most directional macro input |

**Source**: EconomicIntelligence layer5 — RATE_DECISION events, impact=1.0 (highest).

### i. gdp_surprise (z-score)
| Impact | Selection | Rationale |
|---|---|---|
| Regime gating | **PRIMARY** | GDP surprise changes expansion/contraction regime — most regime-relevant of all surprises |
| Bias correction | Affects D | +Z GDP → growth → currency strength: D += sign(z) × min(abs(z)/3, 0.08). Slowest decay (20-40 cycles). |
| Uncertainty adj | Minor | GDP releases are quarterly, well-anticipated — minimal U impact |
| NOT Risk filter | No | GDP is macro-directional |
| Weight modulation | Minor | Too slow for probe weight changes |

**Source**: EconomicIntelligence layer5 — GDP/GDPC1 FRED series.

### j. yield_2y10y_spread_bps
| Impact | Selection | Rationale |
|---|---|---|
| Regime gating | **PRIMARY** | < 0 bps → inverted curve → risk_off/recession regime. > 200 bps → steep → risk_on/growth. Most regime-relevant single field. |
| Bias correction | Affects D | Steepening → growth → long positions in pro-cyclical currencies: D += sign(spread) × min(abs(spread)/200, 0.10) |
| Uncertainty adj | Affects U | Inversion period → high macro uncertainty → U += 0.2 during inversion |
| Risk filter | Affects sizing | Inversion → reduce risk exposure → sizing × 0.7 |
| NOT Weight modulation | No | Yield curve is too slow for probe mixing |

**Source**: FRED series T10Y2Y or broker rates. Not currently computed — needs new ingestion from FRED API or market data.

### k. central_bank_bias (-1 dovish, +1 hawkish)
| Impact | Selection | Rationale |
|---|---|---|
| Regime gating | **PRIMARY** | Defines monetary policy regime — dovish → accommodative, hawkish → restrictive |
| Bias correction | **PRIMARY** | Hawkish → currency strength: D += bias × 0.15. Most important persistent directional bias. |
| Uncertainty adj | Affects U | Policy pivot period → U += 0.15 |
| Weight modulation | Affects w2 | Hawkish regime → OSS trend-following more reliable → w2 += 0.1 |
| NOT Risk filter | No | Central bank bias is macro-directional |

**Source**: MacroAlignmentEngine.evaluate_central_bank_bias() already computes this from FSV (regime_stability + macro_pressure). Currently `full_evaluation()` captures it but not wired into D.

### l. macro_environment (risk_on/risk_off/mixed/neutral)
| Impact | Selection | Rationale |
|---|---|---|
| Regime gating | **PRIMARY** | This IS the regime. Determines regime_adjustment multiplier in D equation. |
| Risk filter | Affects sizing | risk_off → sizing × 0.5; risk_on → sizing × 1.0; mixed → sizing × 0.75; neutral → sizing × 0.85 |
| Uncertainty adj | Affects U | mixed → U += 0.1 (regime uncertainty) |
| NOT D bias | No | This is an aggregate — no independent directional content |
| NOT Weight modulation | No | Macro environment acts as top-level gate, not per-probe |

**Source**: RegimeContextClassifier.classify() already computes this from FSV data. Currently used in `get_regime_parameters()` but not wired into demo execution path.

## 4. Integration Dependencies Between Fields

Some MCV fields are NOT orthogonal — they share information or one can be derived from others:

| Primary Field | Depends On | Collapsible? |
|---|---|---|
| macro_environment | risk_sentiment, central_bank_bias, yield_2y10y | Can be collapsed into risk_sentiment + regime gating |
| news_shock_intensity | NewsIntelligence (layer6) | Independent — cannot derive |
| rate_decision_surprise | EconomicIntelligence (layer5) | Independent |
| yield_2y10y_spread_bps | FRED data | Independent |
| central_bank_bias | rate_decision_surprise, cpi_surprise, employment_surprise | Partially derivable from surprises + spread |

**Recommendation**: Keep all 12 fields as the MCV. They are observables, not features — dimensional reduction should happen in the fusion layer, not the field definition.

## 5. Proposed D Equation with MCV Integration

```
w1 = Shadow_weight × w_risk_sentiment × w_news_shock × w_cpi_surprise
w2 = OSS_weight × w_risk_sentiment × w_central_bank_bias
w3 = TPI_weight × w_news_shock × w_tick_velocity
w4 = Alpha_weight × w_central_bank_bias × w_carry

regime_adjustment = switch(macro_environment):
    risk_on:     1.2
    risk_off:    0.7
    neutral:     0.9
    mixed:       0.5

macro_bias = (
    carry_bias(carry_bps) +
    surprise_bias(cpi_surprise,  decay=5) +
    surprise_bias(employment_surprise, decay=10) +
    surprise_bias(rate_decision_surprise, decay=3) +
    surprise_bias(gdp_surprise, decay=20) +
    central_bank_bias(central_bank_bias) +
    yield_curve_bias(yield_2y10y)
)

D = (Shadow×w1 + OSS×w2 + TPI×w3 + Alpha×w4) × regime_adjustment + macro_bias

U = base_uncertainty × U_spread × U_tick_velocity × U_news_shock × U_regime
```

Where:
- All w_i sum to 1.0 after modulation
- `surprise_bias(z, decay)` = sign(z) × clamp(abs(z)/3, 0, max_val) × exp(-cycles_since_event/decay)
- `U_*` factors are ≥ 1.0 (they increase U)

## 6. Implementation Priority

| Tier | Fields | Effort | Impact | Risk |
|---|---|---|---|---|
| **P0** | macro_environment, central_bank_bias | Low | High | Low — already computed, just needs wiring |
| **P0** | carry_bps | Medium | High | Low — slow, persistent, reliable |
| **P1** | rate_decision_surprise, cpi_surprise | Low | High | Low — already in layer5/FSV |
| **P1** | risk_sentiment (regime gating) | Low | High | Low — FSV RegimeContextClassifier exists |
| **P2** | employment_surprise, gdp_surprise | Low | Medium | Low — already in layer5 |
| **P2** | news_shock_intensity | Medium | Medium | Medium — needs aggregation layer |
| **P2** | yield_2y10y_spread_bps | Medium | High | Low — new FRED series, straightforward |
| **P3** | spread_regime | High | Medium | Medium — needs bid/ask data pipeline |
| **P3** | tick_velocity_anomaly | High | Low | Medium — needs tick history window |

## 7. Key Architectural Changes Required

1. **Wire FSVModulator into demo** — currently `_current_fsv_states` is populated (line 2587) but `FSVModulator.modulate()` is never called. Add call at `post_signal_authority_pre_uesl` integration point.

2. **Replace hardcoded regime_state** at line 2588 with live RegimeContextClassifier.classify() using current FSV states.

3. **Add macro_bias accumulator** in direction pipeline — compute `macro_bias` per symbol and add to D before quantization.

4. **Add probe weight modulation** — replace static equal weights with MCV-modulated weights computed from current MCV field values.

5. **Add uncertainty channels** — propagate U through the pipeline via multiplication (not addition), so multiple sources of uncertainty compound correctly.

6. **Add surprise decay tracking** — each surprise event needs `cycles_since_event` counter for exponential decay in macro_bias.

## 8. Fields Excluded from Direct D Integration

| Field | Exclusion Reason |
|---|---|
| spread_regime | No directional content — execution filter only |
| tick_velocity_anomaly | No directional content — U + TPI weight only |
| news_shock_intensity | Direction too unreliable (±0.3 accuracy) — U + weight modulation only |
| macro_environment | Aggregate — regime gate only, no independent directional signal |
| risk_sentiment | Too high-level — regime gate + U + sizing, not directional |
