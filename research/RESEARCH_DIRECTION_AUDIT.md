# Research Direction Selection Audit (RDSA)

**Date:** 2026-06-16
**Motto:** Assume every idea is wrong until it survives attack.

---

## Phase 1 — Hidden Assumption Extraction

### FRD (Forecast Revision Dynamics)

| Assumption Type | Statement | Verdict |
|----------------|-----------|---------|
| Explicit | Revisions of ES, AT, regime, memory, residual are independent signals | **FALSE** — all 5 are derived from the same underlying price series |
| Explicit | Revision alignment contains information beyond trend direction | **UNVERIFIED** — alignment is definitional in trending markets |
| Implicit | ΔES, ΔAT, Δregime, Δmemory, Δresidual capture different aspects of expectation change | **FALSE** — they're all transformations of price × volatility |
| Implicit | The 5 revision signals have different time dynamics | **WEAK** — ES, memory, residual are all functions of the same rolling windows |
| Unobservable | These revisions proxy for actual expectation changes | **UNVERIFIABLE** — we have no expectation data |
| Circular | "Alignment" = majority of revisions move together = market trending | **CIRCULAR** — this is a definition of trend, not a predictor of it |

**What must be true for FRD to exist?**
The 5 revision signals must be at least partially independent AND their co-movement must precede directional moves rather than coincide with them. Both are unlikely given they're all transformations of a single underlying price series.

### BTC (Belief Transition Cascades)

| Assumption Type | Statement | Verdict |
|----------------|-----------|---------|
| Explicit | Revision events can be meaningfully thresholded | **ARBITRARY** — threshold choice determines cascade detection |
| Explicit | Clustered events represent belief cascades | **FALSE** — they represent autocorrelation in threshold crossings |
| Implicit | Cascade intensity is not just rolling volatility | **FALSE** — in any persistent process, threshold crossings cluster |
| Implicit | Multi-asset cascades differ from single-asset cascades | **PLAUSIBLE** but depends on threshold synchronization across assets |
| Unobservable | These cascades correspond to actual belief revision chains | **UNVERIFIABLE** |
| Circular | Cascade phase definitions (BUILD→PEAK→DECAY) are properties of the thresholded event process | **CIRCULAR** — any persistent thresholded process has these phases |

**What must be true for BTC to exist?**
The clustering of revision events must exceed what would be expected from persistence alone. In other words, the cascade structure must contain information BEYOND the autocorrelation of individual variables. This is testable but very unlikely given the high persistence found in all prior work (H=0.86 for residuals, high autocorrelation for ES).

### TDD (Temporal Distortion Dynamics)

| Assumption Type | Statement | Verdict |
|----------------|-----------|---------|
| Explicit | Tick-level event data is available | **UNCONFIRMED** — daily data confirmed, tick data not verified |
| Explicit | Event rate acceleration precedes directional moves | **PLAUSIBLE** — consistent with microstructure theory |
| Implicit | Event rate is measurable with sufficient precision | **WEAK** — event rate estimation is noisy, especially at inflection points |
| Implicit | "Event" definition is stable across market conditions | **FALSE** — tick frequency varies enormously (10x+ between active and quiet periods) |
| Implicit | Inflection points can be detected in real time | **FALSE** — zero-crossing of acceleration requires look-ahead by definition |
| Unobservable | Event rate changes reflect information arrival | **PLAUSIBLE** but confounded with other factors (liquidity, session, news) |

**What must be true for TDD to exist?**
Tick data must be available, event rate acceleration must be detectable without look-ahead, AND the acceleration must precede directional moves (not coincide with them). The real-time detection requirement is the hardest — measuring acceleration requires smoothing, which introduces lag.

### IPT (Information Propagation Topology)

| Assumption Type | Statement | Verdict |
|----------------|-----------|---------|
| Explicit | 5 assets are sufficient for network topology | **FALSE** — 5 nodes = 20 possible directed edges. Network metrics (entropy, concentration) are meaningless at this scale |
| Explicit | Granger causality reliably identifies lead-lag | **FALSE** — highly sensitive to window choice, lag choice, significance threshold |
| Implicit | Network topology evolves on timescales relevant to trading | **UNKNOWN** — 100-bar rolling windows may be too slow |
| Implicit | Coherence changes are not just volatility changes | **FALSE** — during high volatility, Granger causality significance increases mechanically |
| Circular | "Leadership concentration" identifies a dominant driver | **CIRCULAR** — in trending markets, one asset appears to lead by chance |

**What must be true for IPT to exist?**
At least 15-20 assets with reliable data AND network topology changes must precede directional moves. With only 5 assets, meaningful topology cannot be distinguished from sampling noise.

### ARA (Adaptation Rate Asymmetry)

| Assumption Type | Statement | Verdict |
|----------------|-----------|---------|
| Explicit | Microstructure noise proxies HFT activity | **FALSE** — microstructure noise is a property of the price process, not participant identity |
| Explicit | Trend strength proxies institutional activity | **FALSE** — trend is a property of the price process, not participant identity |
| Implicit | Fast/slow proxy gap measures participant disagreement | **FALSE** — it measures the difference between short-term and long-term price dynamics |
| Implicit | Different participant classes have stable response functions | **UNVERIFIABLE** |
| Circular | "Fast adapts before slow" = short-term properties lead long-term properties | **CIRCULAR** — this is definitionally true in any trending market |

**What must be true for ARA to exist?**
The proxies must correspond to actual participant classes AND the gap between them must contain information beyond what each proxy individually contains. Both are highly unlikely — the "gap" is just a transformation of volatility and trend, which are already well-studied.

---

## Phase 2 — Artifact Risk Assessment

### Scoring Methodology
For each of 8 artifact sources, score 0-10. Total = 0-80. Then estimate additional project-specific risks.

| Artifact Source | FRD | BTC | TDD | IPT | ARA |
|----------------|-----|-----|-----|-----|-----|
| Drift | 10 | 8 | 3 | 6 | 9 |
| Autocorrelation | 9 | 9 | 4 | 5 | 8 |
| Persistence | 9 | 9 | 5 | 5 | 8 |
| Sampling bias | 5 | 5 | 7 | 9 | 5 |
| Binning | 7 | 8 | 2 | 3 | 7 |
| Windowing | 6 | 7 | 8 | 9 | 7 |
| Regime partitioning | 4 | 4 | 4 | 6 | 6 |
| Cross-validation leakage | 3 | 3 | 3 | 5 | 3 |
| **Total Artifact Risk** | **53** | **53** | **36** | **48** | **53** |

### Specific Artifact Scenarios

**FRD:** In a drifted random walk:
- ΔES is positive when drift is up → all revision signals align
- Alignment = 5/5 whenever market has trended for N bars
- This is definitional, not informational
- **Artifact probability: HIGH (~85%)**

**BTC:** In any autocorrelated process with threshold crossings:
- Threshold crossings cluster automatically (autocorrelation → clustered exceedances)
- Cascade intensity = rolling cluster count = just rolling autocorrelation
- This is a mathematical property, not a market phenomenon
- **Artifact probability: HIGH (~90%)**

**TDD:** In a Hawkes process (self-exciting point process):
- Event rates accelerate before large moves (this is a property of Hawkes, not markets)
- BUT: if price changes also cluster (they do), event rate acceleration may be a consequence rather than a cause
- Requires careful disentanglement
- **Artifact probability: MODERATE (~50%)**

**IPT:** With 5 random walks:
- Spurious Granger causality appears in ~20% of pairs by chance
- Network topology metrics are dominated by sampling noise
- "Coherence changes" are just volatility changes
- **Artifact probability: HIGH (~80%)**

**ARA:** In a drifted random walk with stochastic volatility:
- Short-term variance > long-term variance during trending periods (naturally)
- "Gap" is just saying "volatility is elevated" which is already well-known
- **Artifact probability: HIGH (~85%)**

---

## Phase 3 — Counterfactual Strength Audit

### FRD Counterfactual
**Proposed:** Randomize the order of revision vectors.

**Weakness:** Randomizing order preserves the INSTANTANEOUS alignment. For any given bar, the 5 revision signals are determined by the bar's price change. Randomizing bar order doesn't change this — it just rearranges which bars are adjacent. The alignment at each individual bar is preserved.

**Better counterfactual:** Randomize the SIGN of each revision independently while preserving the magnitude distribution. This destroys alignment entirely. If alignment-based predictions survive, they're structural.

**Score: 30/100 — Weak. Proposed counterfactual doesn't destroy the signal.**

### BTC Counterfactual
**Proposed:** Randomize the temporal order of revision events.

**Weakness:** Randomizing events destroys cascade structure, but a Markov process with the same transition probabilities would AUTOMATICALLY produce cascade-like clusters. The counterfactual must compare to a Markov null, not a uniform null.

**Better counterfactual:** Generate synthetic cascade from Markov model with same transition probabilities. Test if cascade phase predictions survive.

**Score: 40/100 — Weak. The null model is a Markov process, not uniform shuffling.**

### TDD Counterfactual
**Proposed:** Resample events uniformly in calendar time.

**Strength:** This genuinely destroys temporal clustering. A Poisson process has NO clustering structure by definition. If acceleration predicts direction in Poisson data, the effect is purely structural.

**Weakness:** Need to ensure the uniform resampling preserves the SAME NUMBER of events. If the total event count differs, the test is invalid.

**Better alternative:** Use a Cox process (doubly stochastic Poisson) with the same intensity function but randomized event times within each intensity bucket. This preserves the smooth trend in event rates but destroys within-bucket clustering.

**Score: 70/100 — Strong. The Poisson null is a meaningful baseline.**

### IPT Counterfactual
**Proposed:** Randomize cross-asset lag relationships.

**Weakness:** With only 5 assets, randomizing lags produces a random network from 5 nodes. The chance of spurious "structure" is ~80% because the adjacency matrix has only 20 entries.

**Better counterfactual:** Generate 5 independent synthetic price series (each a drifted random walk with same persistence as real assets). Compute network topology. If the real network differs significantly from the random network, the effect might be real.

**Score: 20/100 — Very Weak. 5 assets is fundamentally insufficient.**

### ARA Counterfactual
**Proposed:** Randomize the ordering of fast and slow participant signals.

**Weakness:** The "fast" and "slow" signals ARE INHERENTLY ORDERED. Short-term variance is always higher than long-term variance. The gap always exists. Randomizing doesn't fix this — the gap will persist.

**Better counterfactual:** Generate a synthetic price series where participant composition is constant (e.g., a simple GARCH process). Compute the adaptation gap. If the gap has the same properties in the synthetic series as in real data, the effect is structural.

**Score: 25/100 — Weak. The counterfactual doesn't address the fundamental confounding.**

---

## Phase 4 — Synthetic Reproducibility Analysis

### FRD

| Synthetic Process | Would FRD "Work"? | Why |
|------------------|-------------------|-----|
| Random walk | YES | Even zero drift: when price moves up by chance, all 5 revision signals align. Alignment is a retrospective description, not a predictor. |
| Drifted random walk | YES | Drift guarantees alignment in the drift direction. P(alignment > threshold | drift) ≈ 1. |
| Fractional Brownian motion | YES | Long memory makes alignment persist longer, "improving" the apparent signal. |
| **Verdict** | **Would reproduce in ALL synthetic processes.** | **REJECT.** |

### BTC

| Synthetic Process | Would BTC "Work"? | Why |
|------------------|-------------------|-----|
| Random walk with threshold | YES | Threshold crossings cluster automatically in any persistent process. |
| Markov process | YES | Two-state Markov chain produces clusters that look like "cascades." |
| GARCH process | YES | Volatility clustering → threshold crossings cluster → cascade phases appear. |
| **Verdict** | **Would reproduce in ANY autocorrelated process.** | **REJECT.** |

### TDD

| Synthetic Process | Would TDD "Work"? | Why |
|------------------|-------------------|-----|
| Poisson process | NO | Uniform event rate → no acceleration → no inflection points → TDD has nothing to measure. |
| Hawkes process | PARTIALLY | Hawkes has self-exciting clusters. TDD might "work" but the mechanism is the Hawkes excitation kernel, not market structure. |
| Drifted random walk | NO | No events to measure rate from → TDD requires point process data. |
| **Verdict** | **Fails in Poisson null. Partially works in Hawkes.** | **INVESTIGATE with Hawkes control.** |

### IPT

| Synthetic Process | Would IPT "Work"? | Why |
|------------------|-------------------|-----|
| 5 independent random walks | YES | ~20% of pairs show spurious Granger causality. Network metrics will appear to "evolve" as the spurious relationships change over rolling windows. |
| 5 correlated drifted walks | YES | Correlation creates apparent leadership. The "leader" is whichever asset happened to move first in the current window. |
| **Verdict** | **Would reproduce in ANY set of correlated series.** | **REJECT.** |

### ARA

| Synthetic Process | Would ARA "Work"? | Why |
|------------------|-------------------|-----|
| GARCH(1,1) process | YES | Short-term variance > long-term variance during volatile periods. The "gap" always exists. The "fast before slow" narrative is just volatility clustering. |
| Drifted random walk | YES | Even with constant volatility, the gap exists as an artifact of estimation windows. |
| **Verdict** | **Would reproduce in ANY volatility process.** | **REJECT.** |

---

## Phase 5 — Information Test

### FRD
**Contains:** Transformation of existing information.
- ΔES = ES(t) − ES(t−1) — linear transformation of existing ES
- ΔAT = AT(t) − AT(t−1) — linear transformation of existing AT
- Alignment = majority of transformations agree — aggregate of transformations
- The revision vector contains NO information not already present in the levels
- **Verdict: REJECT. Pure transformation.**

### BTC
**Contains:** Transformation of existing information.
- Revision events = threshold crossings of Δ variables
- Cascade intensity = count of threshold crossings in rolling window
- All inputs are derived from ES, AT, regime, memory, residual — which are themselves derived from price
- **Verdict: REJECT. Double transformation of price-derived quantities.**

### TDD
**Contains:** NEW information (contingent on tick data access).
- Event rate = number of price changes per unit calendar time
- This quantity CANNOT be computed from daily OHLC bars
- It requires raw tick data, which has never been used in the prior programs
- If tick data is accessible, this is genuinely new information
- **Verdict: KEEP. Potentially new information if tick data exists.**

### IPT
**Contains:** Transformation of existing information.
- Pairwise Granger causality = function of two price series
- Network metrics = functions of pairwise relations
- All inputs are transformations of the price series already studied
- **Verdict: REJECT. Transformation of price data with too few assets.**

### ARA
**Contains:** Transformation of existing information.
- Microstructure noise = function of tick-level returns  
- Momentum = function of intermediate returns
- Trend = function of long-term returns
- Gap = difference between noise and trend
- All inputs are transformations of return series at different horizons
- **Verdict: REJECT. Pure transformation at different timescales.**

---

## Phase 6 — Market Causality Test

### FRD
**Hypothesized mechanism:** Expectation revision alignment
**Actual causal process:** In a trending market, all momentum-based variables align. There is no mechanism that generates alignment BEFORE direction — alignment IS the direction.
**Do real markets generate this?** Yes — but only as a description of an existing trend, not as a precursor.
**Verdict: EXPLANATORY, not predictive. Describes existing trends.**

### BTC
**Hypothesized mechanism:** Belief cascades
**Actual causal process:** Information cascades DO exist in real markets (Bikhchandani, Hirshleifer, Welch 1992). But our proxy (thresholded Δ variables) does not capture them.
**Do real markets generate this?** Yes, information cascades exist. But our measurement method detects autocorrelation, not cascades.
**Verdict: CORRECT PHENOMENON, WRONG MEASUREMENT. Cascade detection requires order-level data.**

### TDD
**Hypothesized mechanism:** Market time acceleration before directional moves
**Actual causal process:** Information arrives in clusters (macro news, earnings, economic releases). Each cluster increases the event rate. The increased event rate means more participants are processing information, which means more orders, which moves price.
**Do real markets generate this?** YES — this is well-documented in market microstructure (Easley, O'Hara, Engle). Event rates demonstrably increase before volatility events.
**Verdict: CREDIBLE CAUSAL MECHANISM with microstructure theory support.**

### IPT
**Hypothesized mechanism:** Information propagation between assets
**Actual causal process:** Information does propagate between assets. But the topology of a 5-asset network is too sparse to capture meaningful propagation patterns.
**Do real markets generate this?** Yes, with enough assets. Not with 5.
**Verdict: CORRECT PHENOMENON, INSUFFICIENT DATA. Requires 15-20+ assets.**

### ARA
**Hypothesized mechanism:** Fast vs slow participant adaptation
**Actual causal process:** Different participants DO have different response times. But our proxies (microstructure noise, momentum, trend) do NOT correspond to participant types.
**Do real markets generate this?** Probably, but we cannot measure it with available data.
**Verdict: UNVERIFIABLE. Proxies are confounded with price dynamics.**

---

## Phase 7 — Research Priority Ranking

| Criterion (weight) | FRD | BTC | TDD | IPT | ARA |
|--------------------|-----|-----|-----|-----|-----|
| Novelty (20%) | 2 | 4 | 9 | 3 | 5 |
| Causal plausibility (20%) | 2 | 5 | 8 | 5 | 4 |
| Artifact resistance (20%) | 2 | 2 | 7 | 2 | 2 |
| Counterfactual robustness (15%) | 3 | 4 | 8 | 2 | 3 |
| Data availability (15%) | 9 | 9 | 2 | 9 | 3 |
| Implementation cost (5%) | 8 | 7 | 3 | 5 | 4 |
| Potential informational value (5%) | 1 | 3 | 9 | 2 | 3 |
| **Weighted Score** | **3.1** | **4.2** | **6.9** | **3.7** | **3.4** |

### Final Ranking
1. **TDD — 6.9/10** (Only direction that passes basic scrutiny)
2. BTC — 4.2/10 (Interesting phenomenon, wrong measurement)
3. IPT — 3.7/10 (Insufficient assets)
4. ARA — 3.4/10 (Unverifiable proxies)
5. FRD — 3.1/10 (Alignment = trend by definition)

---

## Phase 8 — Final Recommendation

### Classification

| Direction | Classification | Rationale |
|-----------|---------------|-----------|
| FRD | **REJECT** | Revision alignment is definitional in trending markets. Survives ALL synthetic processes. No new information content. |
| BTC | **HOLD** | Real phenomenon (belief cascades) but our measurement method (thresholded Δ variables) detects autocorrelation, not cascades. Requires order-level data. |
| TDD | **INVESTIGATE** | Only direction with plausible causality, genuine new information, and a credible counterfactual gate. Major caveat: tick data availability unconfirmed. |
| IPT | **REJECT** | 5 assets are fundamentally insufficient for network topology. Spurious relationships dominate at this scale. |
| ARA | **REJECT** | Proxies are confounded with price dynamics. "Adaptation gap" = volatility at different windows. No new information. |

### The One Program

**If we spend 3 months on ONE program: Temporal Distortion Dynamics (TDD).**

**Justification:**

1. **Only direction with new information.** Every other direction transforms existing variables. TDD requires tick-level event data that has never been used in Proxima's research.

2. **Only direction with a credible causal mechanism.** Information arrival → event rate acceleration → price discovery is well-documented in market microstructure literature (Engle & Russell 1998, Easley et al. 2012). The other directions have circular or unverifiable causal stories.

3. **Only direction that fails the Poisson counterfactual.** A Poisson process has no temporal clustering. If inflection points predict direction in Poisson data, the effect is structural. This is the only counterfactual among the 5 that would actually be informative.

4. **Only direction with strong artifact resistance (score 36/80).** All others score 48-53. TDD's artifact risk is meaningfully lower because it studies a second-derivative property that drift and autocorrelation alone cannot reproduce.

5. **Directly addresses the discovered failure mode.** The prior programs failed because they decomposed persistent variables into static states. TDD studies the DYNAMICS of event rates — a fundamentally different quantity that cannot be binned or decomposed into states.

### The Critical Condition

TDD is ONLY viable if tick-level event data exists. If only daily bars are available, TDD cannot be implemented.

**Verification step before committing:**
```
Check: Does the existing tick data (300M ticks) have sufficient timestamp resolution
to compute meaningful event rates (≥ 1 event per minute on average)?
```

If yes → TDD is the only direction worth pursuing.
If no → Close the research program entirely. No viable directions remain.

### Closing Statement

Four of five proposed directions are rejected. They would produce the same outcome as STL and Residual Physics: statistically significant in-sample results that fail the synthetic counterfactual gate.

TDD is the sole survivor — not because it's proven, but because it's the only direction that:
1. Studies a genuinely different phenomenon (event rate dynamics)
2. Has a meaningful counterfactual gate (Poisson null)
3. Has a credible causal mechanism (information arrival)
4. Cannot be reproduced by simple drift or autocorrelation

The honest conclusion: **the market may not have a discoverable directional state.** If TDD also fails, the appropriate response is to formally conclude that direction is not a reconstructable market property from the available data — and redirect research entirely toward execution, risk management, and capital allocation rather than directional prediction.
