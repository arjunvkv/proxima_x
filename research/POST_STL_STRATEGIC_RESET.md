# Post-STL Strategic Reset — New Research Directions

**Date:** 2026-06-16
**Status:** Design Phase — Not for Execution
**Gate Condition:** Each direction must include counterfactual design BEFORE implementation

---

## What Has Been Falsified

Two independent programs conclusively demonstrate:

| Framework | Hypothesis | Result | Killer Evidence |
|-----------|------------|--------|-----------------|
| Residual Physics | Direction = f(residual persistence) | STRUCTURAL_ARTIFACT | Synthetic sign preserves edge (ratio≈1.0) |
| State Topology | Direction = f(ES×AT×Regime×Memory) | STRUCTURAL_ARTIFACT | Shuffled states preserve directional IDs (ratio≈0.92) |

The common failure mode: **static decomposition of persistent variables in a drift-biased market** produces apparent directional structure that survives all conventional validation but collapses under counterfactual testing.

## The New Research Principle

> The mechanism must be destroyed by counterfactual generation.
> If shuffling preserves the effect, the effect is structural, not informational.

This rules out:
- State binning (shuffled bins preserve structure)
- Sign persistence (shuffled signs preserve edge)
- Regime conditioning (shuffled regimes produce same splits)
- Memory topology (shuffled memory preserves directional pockets)

---

## Research Direction 1: Temporal Distortion Dynamics (TDD)

### Concept
Market behavior depends on **internal event time**, not calendar time. When markets process information rapidly (high event density), time is "compressed." When markets drift with low information flow, time is "expanded." Directional moves are preceded by systematic changes in the rate of market time.

### Why This Is Different from State Classification
State classification bins variables at calendar-time intervals. Temporal distortion measures the **rate of change** of information arrival. This is a second-order property (derivative of activity), not a first-order property (level of activity).

### Core Theory
Markets alternate between:
- **Information-dominant phases**: High event density, rapid price discovery, compressed market time
- **Liquidity-dominant phases**: Low event density, drift, expanded market time

The TRANSITION between these phases — not the phase itself — contains directional information. When market time decelerates after acceleration, direction emerges.

### Mathematical Representation
Let `τ` be market time (cumulative event count normalized to calendar time):

```
τ(t) = ∫₀ᵗ λ(s) ds
```

where `λ(s)` is the instantaneous event rate (tick arrivals, volatility clusters, volume).

Define **temporal distortion** `δ(t)`:

```
δ(t) = dτ/dt = λ(t)  (the "speed" of market time)
```

Define **temporal acceleration** `α(t)`:

```
α(t) = d²τ/dt² = dλ/dt
```

**Hypothesis:** Directional moves occur when `α(t)` crosses zero after a period of sustained positive or negative acceleration — i.e., at the inflection points of market time.

### Why This Might Not Be Arbitraged Away
Temporal acceleration is a second-order property that emerges from the collective behavior of heterogeneous participants. No single agent can control the market's event rate. Arbitrage requires predicting the aggregate activity pattern, not just the price level.

### Data Requirements
- Tick-level data (already have: 300M ticks across 5 assets)
- Timestamp resolution: second or millisecond
- Event definition: price changes, volume clusters, spread changes
- Current data is daily OHLC — may need to go back to raw tick data

### Measurement Procedure
1. Compute rolling event rate `λ(t)` = number of price changes in a rolling window
2. Compute `α(t)` = change in `λ(t)` (acceleration)
3. Compute `δ(t)` = current `λ(t)` relative to baseline
4. Test: P(up | α(t) > threshold AND δ(t) > threshold) at H5, H20, H50
5. Test: P(up | inflection point detected) where α(t) crosses 0

### Multi-Timeframe Framework
- H1: Micro-temporal distortion (tick-level event clustering)
- H5: Intraday distortion (session-level activity)
- H20: Daily distortion (regime of activity)
- H50: Weekly distortion (macro activity regime)

Each horizon should have its own `λ`, `δ`, `α` computed at the corresponding timescale.

### Counterfactual Gate Design
**The critical test:** Resample events uniformly in calendar time (Poisson process with same mean rate but no clustering).

1. Compute the observed event sequence `E = {t₁, t₂, ..., tₙ}`
2. Generate synthetic `E'` with same number of events but uniformly distributed in time
3. Compute `δ'(t)`, `α'(t)` from synthetic data
4. Test if directional predictions survive

**Prediction:** Temporal distortion effects should be DESTROYED by uniform resampling because the information is in the clustering of events, not their count. If P(up | α > threshold) survives in synthetic data, the effect is artifact.

**Second counterfactual:** Shuffle ONLY the inter-event intervals (preserve count, destroy clustering structure). If effect survives, it's not temporal distortion.

### Failure Modes
1. **Tick data not available**: Current data is daily OHLC — temporal distortion requires sub-daily resolution
2. **Event definition sensitivity**: Results may depend on how "event" is defined (tick, volume, spread)
3. **Non-stationary event rates**: Market hours, holidays, news events create exogenous rate changes

### Historical Validation Plan
1. If tick data is available: compute δ(t) and α(t) for each asset over 2018-2026
2. Split into Train (2018-2022) and Test (2023-2025)
3. Validate directional predictions OOS
4. Apply counterfactual gate

### Relationship to Prior Work
The AT (Adaptive Time) measure used in STL is a smoothed version of market time. STL binned AT into quintiles (static state). TDD would use the DERIVATIVE and ACCELERATION of AT — a fundamentally different quantity.

---

## Research Direction 2: Forecast Revision Dynamics (FRD)

### Concept
Price moves are driven by CHANGES in collective expectations, not by the current state of expectations. Direction emerges when market participants revise their probability estimates — a revision process that is inherently sequential and cannot be captured by static state variables.

### Why This Is Different from State Classification
State classification measures "where are we" (level). Forecast revision measures "how are we changing our minds" (delta). Two identical states can produce opposite directions depending on whether expectations are being revised up or down.

### Core Theory
Financial markets are Bayesian updating machines. Each new piece of information causes a revision of expected future prices. The revision process has persistence (revisions cluster) and asymmetry (upward revisions cluster differently from downward revisions).

Define the **revision signal** `R(t)`:

```
R(t) = E[P(t+H) | I(t)] − E[P(t+H) | I(t−1)]
```

where `I(t)` is the information set at time `t`.

We cannot observe expectations directly. But we can proxy revisions through:

```
R_ES(t) = ES(t) − ES(t−1)    (revision of energy storage)
R_AT(t) = AT(t) − AT(t−1)    (revision of market time)
R_regime(t) = regime_change    (revision of regime classification)
```

### Mathematical Representation
Define the **forecast revision vector**:

```
FR(t) = [ΔES(t), ΔAT(t), Δregime(t), Δmemory(t), Δresidual(t)]
```

where `Δx(t) = x(t) − x(t−k)` for some lookback `k`.

Define **revision momentum**:

```
M_rev(t) = sign(FR(t) · FR(t−1))
```

Positive momentum = revisions are accelerating in the same direction (conviction building)
Negative momentum = revisions are decelerating / reversing (uncertainty)

**Hypothesis:** Direction emerges when forecast revision vector aligns across multiple dimensions (ES, AT, regime, memory all revising in the same direction) AND revision momentum is positive.

### Why This Might Not Be Arbitraged Away
Revisions require new information to enter the market. No single participant controls when information arrives or how the collective interprets it. Revision clustering emerges from the social dynamics of belief updating, which is inherently unpredictable in timing.

### Data Requirements
- ES, AT, regime, memory, residuals (all available from DSR core)
- These are already computed for every bar per asset

### Measurement Procedure
1. Compute `Δx(t)` for each variable at lookbacks [1, 3, 5, 10] bars
2. Compute revision alignment: how many of the 5 revision signals agree in direction?
3. Compute revision momentum: signed product of consecutive revisions
4. Test: P(up | alignment ≥ 3 of 5 AND momentum > 0) at H5, H20, H50
5. Test: P(up | alignment increases from t-1 to t) — does growing consensus predict direction?

### Multi-Timeframe Framework
- Fast revisions: Δx at lookback 1-3 bars (micro consensus building)
- Medium revisions: Δx at lookback 5-10 bars (routine expectation changes)
- Slow revisions: Δx at lookback 20+ bars (structural expectation shifts)

Consensus across timeframes: when fast, medium, and slow revisions all agree, direction is most predictable.

### Counterfactual Gate Design
**The critical test:** Randomize the ORDER of revisions while preserving their distribution.

1. Compute the sequence of revision vectors `{FR(1), FR(2), ..., FR(n)}`
2. Generate synthetic sequence by randomly permuting the order of FR vectors
3. This preserves the marginal distribution of revisions but destroys the sequential structure
4. Test if directional predictions survive

**Prediction:** FRD effects should be DESTROYED because the information is in the SEQUENCE of revisions (momentum, alignment changes), not in their marginal distribution. If shuffled revisions produce the same directional accuracy, the effect is structural.

**Second counterfactual:** Randomize the sign of each revision component independently (destroy alignment). If alignment-based predictions survive, they're structural.

### Failure Modes
1. **Revision proxy quality**: Our ΔES, ΔAT may not capture true expectation revisions
2. **Lookback sensitivity**: Results may depend on chosen lookback window
3. **Alignment threshold sensitivity**: Threshold for "consensus" is a parameter

### Historical Validation Plan
1. Compute FR(t) for all symbols over 2018-2026
2. Define state: (alignment count, momentum sign, revision direction)
3. Train P(up | state) on 2018-2022, test on 2023-2025
4. Apply counterfactual gate

### Relationship to Prior Work
DSR Phase 4 (Transition Physics) studied regime transitions — a specific case of revision (regime change). FRD generalizes this to ALL variables (ES, AT, regime, memory, residual) and adds revision momentum and alignment. The critical difference: DSR Phase 4 studied static transitions (state → state), while FRD studies the revision process itself (Δstate → direction).

---

## Research Direction 3: Information Propagation Topology (IPT)

### Concept
Direction emerges from how information TRAVELS between assets, not from any single asset's state. The topology of the information network — which assets lead, which follow, how fast information propagates — contains directional information that no individual asset's state can capture.

### Why This Is Different from State Classification
State classification studies each asset independently. IPT studies the RELATIONSHIPS between assets. The network structure (who leads, who follows, how quickly) evolves over time. Changes in network topology may precede directional moves.

### Core Theory
Markets form an information propagation network. Assets are nodes; information flows along edges. The network topology changes over time:
- **High-coherence periods**: Information flows quickly and consistently from leaders to followers
- **Low-coherence periods**: Information flow breaks down, assets decouple

Directional moves at the market level emerge when coherence increases (multiple assets start telling the same story).

### Mathematical Representation
Define the **information adjacency matrix** `A(t)`:

```
A_ij(t) = 1 if asset i Granger-causes asset j at time t (significant at p < 0.05)
```

Define **network coherence** `C(t)`:

```
C(t) = (number of significant edges) / (total possible edges)
```

Define **network entropy** `H_net(t)`:

```
H_net(t) = -Σ p_out(i) log p_out(i)
```

where `p_out(i)` is the fraction of outgoing edges from node i.

Define **leadership concentration** `L(t)`:

```
L(t) = max_i (incoming edges to i) / (total edges)
```

**Hypothesis:** Directional market moves are preceded by:
1. Decreasing network entropy (information concentrates on fewer leaders)
2. Increasing coherence (more edges become significant)
3. Increasing leadership concentration (a single asset becomes the dominant driver)

### Why This Might Not Be Arbitraged Away
Network topology emerges from the collective behavior of all market participants across multiple assets. No single participant can control the cross-asset information flow. The network structure is an emergent property that cannot be directly traded.

### Data Requirements
- Daily OHLC data for all 5 assets (already have: 2018-2026)
- Rolling Granger causality tests across all pairs
- Minimum 50 bars per window for statistical significance

### Measurement Procedure
1. For each rolling window (100 bars), compute Granger causality for each asset pair
2. Build adjacency matrix A(t) for each window
3. Compute C(t), H_net(t), L(t) for each window
4. Test: P(up | decreasing H_net) at H5, H20, H50
5. Test: P(up | increasing C(t)) at H5, H20, H50
6. Test: P(up | leadership changes — new asset becomes dominant driver)
7. Combine: P(up | decreasing H_net AND increasing C AND leadership change)

### Multi-Timeframe Framework
- Short horizon (H5): Micro-topology (hourly information flow)
- Medium horizon (H20): Daily topology (end-of-day information structure)
- Long horizon (H50): Weekly topology (macro information regimes)

Network topology at each timescale may evolve at different rates.

### Counterfactual Gate Design
**The critical test:** Randomize the cross-asset lag structure.

1. Compute the observed lead-lag relationships
2. Generate synthetic data where cross-asset lags are randomly permuted
3. Preserve each asset's individual time series (autocorrelation, volatility)
4. Destroy only the cross-asset TEMPORAL ORDER
5. Compute network topology metrics on synthetic data
6. Test if directional predictions survive

**Prediction:** IPT effects should be DESTROYED because the information is in WHICH asset leads WHICH and WHEN — the temporal ordering of cross-asset information flow. If randomizing lags preserves the effect, it's structural.

**Second counterfactual:** Compute network metrics on shuffled residuals (preserve cross-asset marginals, destroy sequential dependencies).

### Failure Modes
1. **Granger causality sensitivity**: Results depend on lag order, significance threshold
2. **Rolling window size**: Too small → noisy; too large → slow to detect changes
3. **Only 5 assets**: May not be enough for meaningful network topology
4. **Static vs dynamic**: Need to distinguish between fixed structural relationships and evolving ones

### Historical Validation Plan
1. Compute 100-bar rolling windows every 20 bars (avoid overlap)
2. For each window: compute adjacency matrix, coherence, entropy, concentration
3. Train directional model on 2018-2022
4. Test on 2023-2025
5. Apply counterfactual gate

### Relationship to Prior Work
DSR Phase 6 (Cross-Asset Cascade) and CDER Layer 6 (Information Propagation) studied static lead-lag relationships. IPT extends this to DYNAMIC network topology — studying how the network structure ITSELF evolves and whether changes in topology predict direction. Prior work asked "does A lead B?" IPT asks "does the network structure changing predict direction?"

---

## Research Direction 4: Adaptation Rate Asymmetry (ARA)

### Concept
Different market participants adapt to new information at different speeds. The GAP between fast and slow adaptation rates creates directional pressure. When fast participants (HFT, algos) adapt before slow participants (institutions, hedgers), the resulting order flow imbalance drives price direction until slow participants catch up.

### Why This Is Different from State Classification
State classification measures the AGGREGATE market state. ARA measures the DISAGREEMENT between participant classes — a relational property that cannot be captured by any single aggregated state variable.

### Core Theory
Markets contain multiple participant classes with different adaptation rates:
- **Fast participants**: Adapt within seconds/minutes (HFT, market makers, algo funds)
- **Medium participants**: Adapt within hours/days (momentum traders, swing funds)
- **Slow participants**: Adapt within days/weeks (institutional rebalancers, hedgers)

Information enters the market and propagates from fast → medium → slow participants. The price moves DURING this propagation, not after.

We cannot observe participant identity directly. But we can proxy adaptation rates through the RESPONSE TIME of different market segments:
- Fast: Tick-level price changes (microstructure noise)
- Medium: 1-5 bar return momentum
- Slow: 20-50 bar trend

### Mathematical Representation
Define **adaptation rate** for participant class k:

```
AR_k(t) = corr(ΔP(t), information_arrival(t−τ_k))
```

where `τ_k` is the characteristic lag of class k.

Define **adaptation gap**:

```
G(t) = AR_fast(t) − AR_slow(t)
```

When `G(t)` is large and positive, fast participants have already adapted but slow participants haven't → upward pressure persists until slow participants catch up.

Define **adaptation convergence**:

```
dG/dt = rate of gap closure
```

**Hypothesis:** Direction emerges when:
1. Adaptation gap widens (G increases) — fast participants lead
2. Adaptation gap begins to close (dG/dt < 0) — slow participants start to follow
3. The convergence phase — when slow participants adapt — coincides with the directional move

### Why This Might Not Be Arbitraged Away
Arbitraging adaptation rates requires knowing the participant composition of the market and their response functions. Both are unobservable and time-varying. The adaptation gap is an emergent property of heterogeneous agent populations, not a tradable quantity.

### Data Requirements
- Tick-level data for microstructure noise estimation (already have)
- Or daily OHLC with volume for slower adaptation estimation
- Alternative: use volatility regime as proxy for participant composition

### Measurement Procedure (Proxy Method)
Since we cannot observe participant identity directly, use microstructure proxies:

1. **Fast adaptation proxy**: Tick-level return autocorrelation (microstructure noise)
   - High noise = fast participants dominate (many small trades)
   - Low noise = slow participants dominate (fewer but larger trades)

2. **Medium adaptation proxy**: 5-bar momentum strength
   - Strong momentum = medium participants adapting

3. **Slow adaptation proxy**: 50-bar trend strength
   - Strong trend = slow participants adapting

4. **Adaptation gap**: Difference between fast and slow proxy
   - When fast >> slow: gap is wide → directional pressure building
   - When fast ≈ slow: gap is closed → pressure released

5. Test: P(up | gap widens over N bars) at H5, H20, H50
6. Test: P(up | gap begins to close after widening) — the convergence phase

### Multi-Timeframe Framework
The adaptation rates themselves are multi-timeframe:
- τ_fast: 1-5 ticks (seconds to minutes)
- τ_medium: 5-50 ticks (minutes to hours)
- τ_slow: 50-500 ticks (hours to days)

Compute gap at each timescale and their interactions.

### Counterfactual Gate Design
**The critical test:** Randomize the ORDERING of fast and slow participant signals.

1. Compute fast-time proxy (microstructure noise) and slow-time proxy (trend)
2. Shuffle the temporal relationship between them
   - Preserve each proxy's marginal distribution
   - Destroy only the SEQUENCE of fast→slow propagation
3. Compute adaptation gap G(t) from shuffled data
4. Test if directional predictions survive

**Prediction:** ARA effects should be DESTROYED because the information is in the SEQUENCE (fast adapts first, then slow follows). If randomizing the fast-slow order preserves the effect, it's structural.

**Second counterfactual:** Reverse the adaptation order (assume slow→fast instead of fast→slow). If the reversed model also "works," the direction of causality is meaningless.

### Failure Modes
1. **Proxy validity**: Our proxies for adaptation rates may not reflect actual participant behavior
2. **Stable participant composition**: During regime changes, participant composition may shift
3. **Multi-asset complexity**: Adaptation rates differ across assets, making cross-asset application difficult
4. **Tick data dependence**: Fast adaptation requires tick-level data

### Historical Validation Plan
1. Compute fast, medium, slow adaptation proxies for each asset
2. Compute adaptation gap and gap changes
3. Train: P(up | gap widening AND gap beginning to close) on 2018-2022
4. Test on 2023-2025
5. Apply counterfactual gate

### Relationship to Prior Work
ARA is entirely new. It does not reuse any state variables from STL or residual variables from DSR/ROL. The input data (tick microstructure noise, trend strength) is available but was not used in earlier programs.

---

## Research Direction 5: Belief Transition Cascades (BTC)

### Concept
Markets do not transition between states gradually. They transition through CASCADES of belief revisions — a chain reaction where one participant's revision triggers another's, which triggers another's, creating a directional cascade. The cascade structure itself — not the pre- or post-cascade state — determines direction.

### Why This Is Different from State Classification
State classification studies the endpoints (before and after). BTC studies the PROCESS between endpoints. Two identical pre→post transitions can have different directions depending on the cascade structure (fast cascade = strong direction, slow cascade = weak direction).

### Core Theory
A belief cascade occurs when:
1. Initial revision: A participant revises expectations (trigger)
2. Amplification: Other participants observe the revision and revise further
3. Propagation: The revision spreads through the participant network
4. Exhaustion: All participants have revised; the cascade ends

**Key insight:** Cascade VELOCITY (revisions per unit time) and CASCADE SIZE (total revisions) contain directional information independent of the start and end states.

### Mathematical Representation
Define **cascade intensity** `I(t)`:

```
I(t) = (number of revision events in window [t-w, t]) / w
```

where a "revision event" is a significant change in any variable (ES, AT, regime, residual) across any asset.

Define **cascade acceleration** `A(t)`:

```
A(t) = I(t) − I(t−1)
```

Define **cascade breadth** `B(t)`:

```
B(t) = (number of variables/assets showing revision at time t) / (total variables × assets)
```

**Hypothesis:** Directional cascades follow a characteristic pattern:
1. Trigger: Single variable/asset revises (low breadth, low intensity)
2. Amplification: Breadth and intensity increase (cascade growing)
3. Peak: Maximum breadth and intensity → direction determined
4. Decay: Intensity decreases, breadth remains → direction executes
5. Resolution: All variables/assets have revised → cascade ends

### Why This Might Not Be Arbitraged Away
Cascades are emergent phenomena from heterogeneous agent interactions. No single agent can control the cascade structure. Predicting cascade evolution requires modeling the agent interaction network, which is unobservable and time-varying.

### Data Requirements
- ES, AT, regime, memory, residual for all 5 assets (already computed)
- Revision event detection: significant change thresholds per variable
- Rolling window for intensity computation

### Measurement Procedure
1. Define revision event per variable per asset:
   - ES_revision: |ΔES| > 1 std of ES changes
   - AT_revision: |ΔAT| > 1 std
   - Regime_revision: regime changes
   - Residual_revision: |Δresidual| > 1 std
2. Compute I(t): events per bar, rolling window 10 bars
3. Compute B(t): assets with any revision per bar
4. Compute cascade phase:
   - BUILD: I increasing, B constant or increasing
   - PEAK: I maximum, B maximum
   - DECAY: I decreasing, B still elevated
   - RESOLUTION: I low, B low
5. Test: P(up | PEAK phase detected) at H5, H20, H50
6. Test: P(up | BUILD phase → direction at PEAK)
7. Test: P(down | symmetrical for negative cascades)

### Multi-Timeframe Framework
- Micro-cascades: revisions within 1-5 bars (fast information propagation)
- Meso-cascades: revisions within 5-20 bars (normal information diffusion)
- Macro-cascades: revisions within 20-50 bars (structural shifts)

Cascades can be nested — a macro-cascade may contain multiple meso- and micro-cascades.

### Counterfactual Gate Design
**The critical test:** Randomize the TEMPORAL ORDER of revision events while preserving their number and type.

1. Extract the sequence of revision events across all variables and assets
2. Randomly permute the sequence (preserve count of each revision type, destroy clustering)
3. Compute I(t), B(t), cascade phases from synthetic data
4. Test if cascade phase predictions survive

**Prediction:** BTC effects should be DESTROYED because the information is in the CLUSTERING of revisions — the cascading structure. If uniformly distributed revisions preserve directional predictions, the effect is structural.

**Second counterfactual:** Randomize which ASSET each revision event belongs to (preserve timing, destroy cross-asset cascade relationships).

### Failure Modes
1. **Event definition sensitivity**: Results depend on threshold for "revision event"
2. **Variable count**: Only 4 variables × 5 assets = 20 event types — may not be enough for cascade detection
3. **Cascade phase definition**: Phase boundaries are arbitrary

### Historical Validation Plan
1. Compute revision events for all variables and assets
2. Compute I(t), B(t), cascade phases
3. Train: P(up | cascade phase) on 2018-2022
4. Test on 2023-2025
5. Apply counterfactual gate

### Relationship to Prior Work
DSR Phase 4 (Transition Physics) studied single-variable transitions (regime → regime). BTC studies MULTI-VARIABLE, MULTI-ASSET cascades. The cascade structure — intensity, breadth, phase — is a fundamentally different quantity from the transition endpoints.

---

## Comparison and Prioritization

| Criterion | TDD | FRD | IPT | ARA | BTC |
|-----------|-----|-----|-----|-----|-----|
| Data available now | ❌ (needs ticks) | ✅ | ✅ | ❌ (needs ticks) | ✅ |
| Novel (not recycled) | ✅ | ✅ | ⚠️ (extends prior) | ✅ | ✅ |
| Clear counterfactual | ✅ | ✅ | ✅ | ✅ | ✅ |
| Non-arbitrage argument | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| Mathematical specificity | ✅ | ✅ | ✅ | ⚠️ | ✅ |

**Top recommendation: FRD (Forecast Revision Dynamics)**
- All data immediately available from DSR core
- Most novel relative to prior work
- Clearest counterfactual gate
- Strongest non-arbitrage argument
- Directly addresses the failure mode of static state classification

**Second: BTC (Belief Transition Cascades)**
- Also uses available data
- Entirely new framework
- Cascade structure is fundamentally different from state classification

**TDD and ARA depend on tick data availability.** If tick data is accessible, they should be prioritized.

---

## Decision Point

This document defines 5 research directions with full theoretical foundations, measurement procedures, and (most importantly) counterfactual gate designs designed BEFORE implementation.

The key advance over prior research:

1. Every direction is designed to FAIL the counterfactual test if it's structural (no more post-hoc discovery of artifacts)
2. Every direction studies PROCESS, not STATE (avoiding the failure mode of STL and Residual Physics)
3. Every direction uses data that is either already computed or can be computed from existing infrastructure

The next step is to select 1-2 directions for implementation, or design alternative directions if none of these match the criteria.
