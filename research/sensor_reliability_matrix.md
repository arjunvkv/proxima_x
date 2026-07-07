# Sensor Reliability Matrix

## Scoring Dimensions (0–10)

| Dim | Label | Definition |
|-----|-------|------------|
| DP | Directional Prediction | Does the sensor's sign predict future price direction? |
| SD | State Detection | Does the sensor detect market regime changes? |
| ET | Entry Timing | Does the sensor identify good entry timing? |
| UD | Uncertainty Detection | Does the sensor tell us when it's wrong? |
| RB | Robustness | Is the sensor stable across symbols, regimes, timeframes? |

---

## 1. OSS (Outcome Surface Signal)

**Files**: `signals/outcome_surface_signal.py`, `bootstrap/oss_bootstrap_trainer.py`, `run_proxima_demo.py:2422-2474`

**Computation**: Pre-computed frozen lookup table mapping (ECDF_bucket, drift_state) → P(continuation | evidence from training window). Three horizons (3/10/20) blended by entropy-weighted triangular weights. Drift is EMA(diff,span=3) / EMA(|diff|,span=3) thresholded at ±0.5. Signal fires when p_cont ≥ 0.60 (drift persists) or EV exceeds threshold.

**Architecture**: *Directional* (produces LONG/SHORT signal)

**Known biases**: p_cont dead zone [0.40, 0.60) returns 0 always. EV-based signal is telemetry-only (not used for execution). Horizon blend uses triangular entropy weight: at entropy=0.5, w3=0.0, w20=0.0, w10=1.0 (degenerate to single horizon). Requires drift_state ≠ 0 to fire.

**Log evidence**: `[OSS SURFACE] EURJPY ecdf=0.8177 exec_drift=0.9806 signal=1 p_cont=0.81` — signal fires when drift is strong AND p_cont high. At p_cont=0.81, accuracy historically ~81% in training. But system also shows `oss=False` frequently (symbols with no trained surface or p_cont < 0.60).

**Scoring**:
| DP | SD | ET | UD | RB |
|----|----|----|----|----|
| 6 | 2 | 4 | 5 | 5 |

- **DP=6**: Frozen lookup from training window — not adaptive. p_cont=0.81 in training ≠ live 81% accuracy. No online updating. However, the persistence-conditioned approach (drift persists → follow drift) is theoretically sound and avoids phantom reversals since the sign-inversion fix.
- **SD=2**: Detects nothing about regime. ECDF is rank within recent window, not a regime classifier. Has no concept of volatility regime, session, or cross-asset state.
- **ET=4**: Entry timing is entirely drift-dependent (needs exec_drift > 0.5). Fast EMA(3) converges in ~3 ticks, but the threshold is arbitrary. OSS cannot identify pullbacks or reversals — only persistence.
- **UD=5**: p_cont is an explicit uncertainty measure. p_cont ∈ [0.40, 0.60) suppresses all signals (this is correct behavior). However, the discontinuity at boundaries (0.60 fires, 0.599 does not) creates edge instability.
- **RB=5**: Trained per-symbol on available OHLC data. Symbols without sufficient training data (MIN_SAMPLES=200) get no surface. Signal density varies (some symbols have sparse bucket coverage). Restart-dependent (cache or retrain). Cross-symbol consistency depends on data quality per symbol.

**Primary classification**: Directional

---

## 2. Shadow/Fusion Kernel

**Files**: `fusion_kernel/fusion_kernel.py`, `run_proxima_demo.py:2709-2757`

**Computation**: `base_signal = sign(ecdf_rank - entropy)` — a simple difference score thresholded at ±0.05. Then `_regime()` classifies as CHAOTIC (avg_entropy > 0.65), STRUCTURED (< 0.4), or TRANSITION. `_apply_flip_suppression()` blocks flips when entropy > 0.65 and a signal already exists (prevents churn in chaos). Coherence filter is currently a no-op (Phase A). Exhaustion detection overrides shadow signal if ecdf ≥ 0.80 AND entropy ≥ 0.88 AND d_entropy ≥ 0.0 AND d_pmax ≤ -0.010.

**Architecture**: *Directional* (produces LONG/SHORT signal) with *Regime* overlay (entropy classification)

**Known biases**: ecdf - entropy as signal is arbitrary (±0.05 threshold has no theoretical basis). Flip suppression in chaos means shadow "sticks" to prior direction during high entropy — this creates directional persistence bias when market is actually random. SELL exhaustion (ecdf≥0.80) triggers more often than BUY exhaustion (ecdf≤0.20) because ECDF is right-skewed in trending markets.

**Log evidence**: `[SHADOW FUSION] {EURJPY: 1, USDJPY: -1, ...}` shows discrete signals. `[SHADOW_RAW] AUDCHF ecdf=0.6154 entropy=0.4938 score=+0.1216 raw=+1 final=1 flip_suppress=False` — typical output. `[POLARITY_AUDIT] SHADOW: buy=13 sell=13` — roughly balanced in recent window. `[SHADOW_MICRO] lkg_sim=0.9990` — high similarity to production, meaning shadow rarely disagrees.

**Scoring**:
| DP | SD | ET | UD | RB |
|----|----|----|----|----|
| 4 | 6 | 3 | 3 | 6 |

- **DP=4**: ecdf - entropy is a heuristic, not a predictive model. No training, no validation. Signal direction depends on relative rank vs. entropy, which has no demonstrated predictive relationship with future returns. The ±0.05 threshold is arbitrary.
- **SD=6**: Entropy-based regime classification (CHAOTIC/STRUCTURED/TRANSITION) is a legitimate state detection mechanism. Exhaustion detection (4-condition gate on ecdf, entropy, dH, dp) is a genuinely creative regime indicator. This is Shadow's strongest dimension.
- **ET=3**: Shadow has no timing mechanism. Entry decisions are binary (signal exists or doesn't) with no entry quality scoring. Exhaustion detection acts as a counter-trend timing signal but fires rarely.
- **UD=3**: No explicit confidence or uncertainty measure. The flip suppression during chaos is a crude uncertainty heuristic (high entropy = don't flip), but it's not calibrated. Exhaustion score is min(1.0, distance_from_threshold) — linear, not probabilistic.
- **RB=6**: Operates on eval_data fields (ecdf, entropy) that are computed per-symbol consistently. No per-symbol training required. Works identically across all symbols. Exhaustion thresholds are fixed (not adaptive) which limits robustness to regime shifts, but the core computation is stable.

**Primary classification**: Directional (primary) / Regime (secondary)

---

## 3. TPI (Tick Pressure Index)

**Files**: `layer7/get_tpi_signal.py`, `data/tick_buffer.py:56-88`

**Computation**: Winsorized magnitude-weighted TPI over last 200 ticks: (sum_up − sum_down) / total_magnitude. Deltas clipped at P5/P95 to prevent outlier dominance. Direction = sign(TPI). Confidence = |TPI|. Eligibility requires |TPI| ≥ P90 of historical distribution AND session hours. Limited to 5 symbols (EURJPY, EURUSD, GBPJPY, USDJPY, XAUUSD).

**Architecture**: *Directional* (produces LONG/SHORT signal) with *Timing* characteristics (tick-level)

**Known biases**: Confidence = |TPI| — a value in [0,1] but not a probability. Session-based eligibility means no signals outside liquid hours. Winsorization at P5/P95 removes true outlier events (flash crashes). 200-tick window is fixed — no adaptive window sizing. Only 5 symbols supported.

**Log evidence**: (No explicit TPI log lines in recent tail — may be unused in current demo mode based on dashboard import only). `tpi_dashboard` and `cache_tpi_signal` are imported but the main demo does not log TPI values in the visible window. It appears TPI is a research/layer7 probe that is captured into the TPI dashboard but doesn't directly affect production signals in the current demo.

**Scoring**:
| DP | SD | ET | UD | RB |
|----|----|----|----|----|
| 5 | 3 | 7 | 4 | 3 |

- **DP=5**: TPI measures tick-level flow imbalance — buying vs selling pressure at micro timescale. This is directionally informative in the very short term (next few ticks). However, it measures current flow, not future flow. Correlation with next-bar direction is moderate.
- **SD=3**: No regime detection. Only detects micro-flow pressure. The percentile filter (P90) is an eligibility gate, not a state detector.
- **ET=7**: TPI's strength. Tick-level flow imbalance is a legitimate entry timing signal. When buy pressure dominates (TPI > 0.5) at P90+ percentile during liquid session, it's a high-quality short-term entry signal. The session filter adds practical timing value.
- **UD=4**: |TPI| as confidence is reasonable but primitive. The percentile ranking (|TPI| ≥ P90) adds statistical grounding. No calibration or backtest of confidence reliability.
- **RB=3**: Only 5 supported symbols. 200-tick fixed window doesn't adapt to volatility regime. Session filtering helps but means zero signal outside active hours. Winsorization discards true tails. No cross-symbol generalization.

**Primary classification**: Directional (microstructure) / Timing

---

## 4. Alpha Strategies

**Files**: `proxima_v5/layer2/__init__.py`

**Computation**: 9 independent alpha models (A–I), each producing direction + score + confidence. Compute_all runs all 9, filters by |score| > 10. Alphas cover: liquidity sweep, compression boom, trend pullback, regime shift, relative strength, flow divergence, cross-asset, volatility rotation, institutional momentum.

**Architecture**: *Directional* (each produces LONG/SHORT signal) — diverse sources

**Known biases**: Each alpha has its own bias profile. Alpha-C (trend pullback) requires trend strength ≥ 30 and retracement ≥ 0.5% — only fires in strong trends. Alpha-D (regime shift) only fires in volatility ratio ≥ 1.5 and ATR percentile 40–80 — narrow activation band. Alpha-H (vol rotation) only in volatility EXPANSION regime — counter-trend signal during vol expansion. Alpha-G (cross-asset) blends DXY/SPX/VIX — this is a multi-asset inference but directions are hardcoded (DXY > 0 → SHORT USD).

**Log evidence**: Not visible in demo log (research-only / V5 engine). The alpha strategies appear to be under `proxima_v5` which is a separate pipeline/engine. In the demo pipeline, they are not called directly.

**Scoring**:
| DP | SD | ET | UD | RB |
|----|----|----|----|----|
| 5 | 5 | 5 | 3 | 4 |

- **DP=5**: Each alpha has a specific thesis — some are likely predictive (regime shift, relative strength), others are heuristic (cross-asset hardcoded rules). The diversity is a strength (different world models), but individual alpha quality is untested here. Without backtest per alpha, directional accuracy is unknown.
- **SD=5**: Several alphas detect specific market states: compression (Alpha-B), expansion (Alpha-H), regime shift (Alpha-D). This is state detection within each alpha's domain.
- **ET=5**: Alpha-C (trend pullback) and Alpha-F (flow divergence) are inherently entry-timing strategies. Others are less timing-specific.
- **UD=3**: Each alpha provides a confidence score (bounded 0-100), but these are engineered heuristics (score * 0.7 + trend_strength * 0.3), not calibrated probabilities. No uncertainty framework.
- **RB=4**: Dependent on `MarketState` infrastructure — requires properly populated liquidity, volatility, trend, currency matrix, cross-asset states. This is a heavy dependency chain. If any dependency is missing, alpha returns None silently. Cross-symbol consistency depends on infrastructure quality.

**Primary classification**: Directional (diverse sources)

---

## 5. MacroContextVector / FSV

**Files**: `research/fsv/core/fsv_engine.py`, `research/fsv/core/fsv_schema.py`

**Computation**: Fundamental State Vector per symbol with 5 fields: bias_alignment, macro_pressure, sentiment_gradient, event_risk, regime_stability. Updated by NormalizedEvent (CPI, NEWS, RATE, GDP, SENTIMENT) with impact_weight blending. Exponential decay (λ=0.01) toward neutral over time. State = merge of event vector with current state: `new = current * (1-w) + incoming * w`.

**Architecture**: *Regime* (fundamental factor state) with *Risk* characteristics

**Known biases**: Requires external events to update — no event → vector decays to neutral. `decay_lambda=0.01` is fixed per engine (not per-symbol or per-field). Blending weight = event.impact_weight means events with higher impact_weight dominate regardless of their accuracy. The macro_pressure field uses `surprise_score * impact_weight` — this double-counts impact.

**Log evidence**: `_current_fsv_states` is referenced in `run_proxima_demo.py:2587` but only as `{}` (empty dict passed to UCF). The FSV engine is NOT initialized or updated in the main demo pipeline. It is used only in shadow replay and UCF research modules. As noted in AGENTS.md: FSV code is **dead code** — never imported by demo.

**Scoring**:
| DP | SD | ET | UD | RB |
|----|----|----|----|----|
| 1 | 7 | 2 | 6 | 2 |

- **DP=1**: No directional signal. bias_alignment ranges [-1,1] but is used as conviction weight, not price prediction. The FSV is not designed as a directional model.
- **SD=7**: The FSV's purpose is regime detection via fundamental factors. regime_stability, macro_pressure, and event_risk together describe the macro regime. This is the most structured state detection system in Proxima. However, it requires a constant stream of macro event data to stay current.
- **ET=2**: No entry timing logic. The FSV decays toward neutral continuously — it provides context, not timing signals.
- **UD=6**: event_risk is an explicit uncertainty measure. regime_stability directly measures confidence in the current state assessment. Decay toward neutral over time correctly expresses increasing uncertainty when no new events arrive.
- **RB=2**: Dead code in main demo. Only used in research replay. Requires an external event ingestion pipeline that doesn't exist in production. Without events, states are always neutral. Per-symbol vector but no cross-symbol consistency. The design is theoretically sound but unimplemented in the active pipeline.

**Primary classification**: Regime / Risk

---

## 6. exec_drift / exec_momentum

**Files**: `run_proxima_demo.py:2428-2446`

**Computation**: Real-time EMA of price changes: `_diff_ema[sym] = diff * α + prev * (1-α)` where α = 2/4 (span=3). Similar EMA of |diff| for normalization. Raw momentum = diff_ema / |diff_ema|, clamped to [-1, 1]. Quantized to {-1,0,1} at ±0.5 threshold. Used as `oss_drift` for OSS bucket lookup.

**Architecture**: *Directional* (drift direction) with *Timing* characteristics (fast convergence)

**Known biases**: Double EMA structure (diff EMA + |diff| EMA) means momentum converges slowly to +1/-1. Threshold at ±0.5 means exec_drift spends significant time at 0 (no direction) when momentum is weak. Single-alpha (span=3) only — no multi-timescale drift. Converges in ~3 ticks but the quantization threshold creates a "dead zone" where drift is measured but not used.

**Log evidence**: `[OSS SURFACE] EURJPY exec_drift=0.9806` — near-maximum drift, suggests strong directional momentum. `[DRIFT_DIST] runtime: 0=9 +1=11 -1=8` — distribution varies, indicating genuine responsiveness to market conditions.

**Scoring**:
| DP | SD | ET | UD | RB |
|----|----|----|----|----|
| 5 | 1 | 6 | 3 | 8 |

- **DP=5**: Measures directional momentum over ~3 ticks. EMA(diff)/EMA(|diff|) is a robust momentum estimator (bounded [-1,1], unitless, comparable across symbols). However, it measures recent price direction, not future direction. Predictive value depends on momentum persistence.
- **SD=1**: No regime detection. Pure fast momentum. Single timescale.
- **ET=6**: Fast convergence (α=2/4) means exec_drift adapts within ticks. When exec_drift > 0.5 or < -0.5, it signals strong recent directional persistence — a reasonable entry condition. The quantization at ±0.5 prevents noise entry during weak moves.
- **UD=3**: The |diff| EMA denominator provides implicit normalization (momentum strength relative to recent volatility), but there's no explicit uncertainty measure. exec_drift = 0.5 means "confident up" but 0.5/1.0 is not a probability.
- **RB=8**: Most robust sensor. Computed identically for all symbols. No per-symbol training. No configuration. No external dependencies. Works on any price series with any tick frequency. The EMA structure is numerically stable. Only failure mode is zero-price edge case (handled).

**Primary classification**: Directional (momentum)

---

## Summary Matrix

| Sensor | DP | SD | ET | UD | RB | Σ | Primary Type |
|--------|----|----|----|----|----|---|-------------|
| OSS | 6 | 2 | 4 | 5 | 5 | 22 | Directional |
| Shadow/Fusion | 4 | 6 | 3 | 3 | 6 | 22 | Directional/Regime |
| TPI | 5 | 3 | 7 | 4 | 3 | 22 | Directional/Timing |
| Alpha Strategies | 5 | 5 | 5 | 3 | 4 | 22 | Directional |
| FSV/MacroContext | 1 | 7 | 2 | 6 | 2 | 18 | Regime/Risk |
| exec_drift/momentum | 5 | 1 | 6 | 3 | 8 | 23 | Directional |

### Key Findings

1. **No sensor scores >7 in any dimension** — every sensor has significant limitations
2. **exec_drift is highest total (23) due to robustness** — the most reliable sensor, but only measures momentum, not future direction
3. **FSV has highest state detection (7) but is dead code** — the best regime sensor never runs in production
4. **TPI has highest entry timing (7) but narrowest symbol coverage** — useful for 5 symbols only
5. **Shadow/Fusion has highest robustness among regime sensors (6)** but its direction signal is arbitrary (ecdf - entropy)
6. **Uncertainty detection is universally weak** — max is FSV at 6 (dead code). OSS p_cont at 5 is the best active uncertainty measure
7. **The system lacks a single sensor that scores >6 on both DP and UD** — meaning no active sensor can simultaneously predict direction AND know when it's wrong
8. **Sensors are uncorrelated** — OSS and Shadow agree via arbitration, but they measure fundamentally different things (frozen statistics vs heuristic ecdf-entropy gap)

### Critical Gap

The arbitration system (Phase A) requires OSS and Shadow to agree for a signal to fire. But:
- OSS scores DP=6, UD=5 (moderate directional, moderate uncertainty)
- Shadow scores DP=4, UD=3 (weak directional, weak uncertainty)

**When they agree, the combined signal inherits both sensors' weaknesses**: Shadow's arbitrary direction lowers accuracy, and neither provides good uncertainty calibration. The system conflates "two sensors agree" with "the signal is correct" — which is fallacious when both sensors have the same blind spots (both use ECDF, both lack cross-asset awareness).
