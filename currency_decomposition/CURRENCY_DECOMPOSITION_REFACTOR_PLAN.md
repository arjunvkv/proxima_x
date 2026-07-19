# Currency Decomposition — Refactor Plan

## Problem Diagnosis

The currency decomposition system uses a Weighted Least Squares (WLS) solver to decompose cross-currency pair returns into latent "currency strengths." The core math is sound (regularized linear factor model), but the surrounding pipeline has accumulated layers of untuned heuristics, redundant features, and circular validation that together constitute overfitting.

---

## Item 1 — REMOVE: Product-of-6 Confidence Formula

**File:** `direction/hypothesis.py:51`

```python
confidence = signal_conf * graph_quality * stability * observability_factor * stability_factor * (0.5 + 0.5 * recency)
```

**Problem:** Six arbitrary factors multiplied. Any single factor's noise propagates through the entire product. Each factor has no calibrated relationship to actual predictive power.

**Action:** Replace with a single signal-to-noise ratio (SNR):

```python
mean_residual = mean(|residual| across active pairs)
spread_mag = abs(base_strength - quote_strength)
confidence = spread_mag / (mean_residual + 1e-10)
confidence = clamp(confidence, 0, 1)
```

This measures: "How large is the currency strength difference relative to the model's typical misfit?" — a grounded statistical interpretation.

---

## Item 2 — REMOVE: DER as a Confidence Multiplier

**File:** `runtime/manager.py:462-478`

```python
der_factor = 0.8 + 0.4 * effective_der   # range [0, 1.2]
h.confidence = min(1.0, h.confidence * der_factor)
```

**Problem:** Directional Efficiency Ratio (DER) measures how directional the raw price path was over ~30 seconds of ticks. This has no causal relationship to WLS strength signal quality. Using it as a multiplicative confidence scaler adds noise.

**Action:** Use DER as a binary gate only (skip pairs with DER < 0.10 as already done). Remove the multiplicative scaling.

---

## Item 3 — REMOVE: Recency Factor

**File:** `direction/hypothesis.py:41`

```python
recency = min(len(spread_history[symbol]), 60) / 60.0
# later used in confidence formula as:
(0.5 + 0.5 * recency)
```

**Problem:** This gives a 2x confidence range (0.5–1.0) based purely on how many ticks this symbol has been observed. It has no signal content and biases against new symbols.

**Action:** Remove from confidence formula. Freshness is already handled by the WLS weight matrix.

---

## Item 4 — REMOVE: Redundant Observability & Stability Factors

**File:** `direction/hypothesis.py:43-50`

```python
observability_factor = (base_obs + quote_obs) / 2.0
stability_factor = (base_stab + quote_stab) / 2.0
# multiplied into confidence
```

**Problem:** `observability_factor` is a graph-theoretic measure already baked into `graph.quality` and `graph.execution_allowed()`. `stability_factor` is `1/(1+std(strength_history))` already accessible via `graph.strength_stability()`. Both are redundant when `graph.quality` is already in the confidence formula.

**Action:** Remove both. Quality already captures system confidence at the graph level.

---

## Item 5 — REMOVE: DRS Combinatorial Selection

**Files:** `portfolio/drs.py:96-176`, `config/settings.py:72-76`

**Problem:** The DRS `select()` method runs `itertools.combinations(pool, slots_needed)` over `DRS_CANDIDATE_POOL_SIZE=10` candidates, testing every combination against:
- Mixed-sign currency vector constraints
- `MAX_CURRENCY_FACTOR_EXPOSURE=2` limit
- A weighted score with 4 arbitrary weights (0.35, 0.25, 0.20, 0.20)
- Slot inertia `[1.0, 0.85, 0.7]`
- Lambda decay `DRS_LAMBDA_DECAY = 0.05`
- Replacement margin `DRS_REPLACE_MARGIN = 0.10`

None of these parameters have been empirically calibrated. The combinatorial selection adds complexity with zero validation.

**Action:** Replace `select()` with simple top-N by `drs_score` while respecting the currency exposure limit. Remove slot inertia, lambda decay, replacement margin, and combinatorial search.

---

## Item 6 — REMOVE: health_report Hardcoded Thresholds

**File:** `currency/graph.py:220-225`

```python
if conn >= 0.7 and avg_stability >= 0.5: confidence_level = "HIGH"
elif conn >= 0.45 and avg_stability >= 0.3: confidence_level = "MEDIUM"
else: confidence_level = "LOW"
```

**Problem:** These thresholds (0.7, 0.5, 0.45, 0.3) are completely arbitrary with no empirical basis. They create false confidence when conditions happen to be met and false alarm when not.

**Action:** Remove `confidence_level` from health report. Return raw metrics only.

---

## Item 7 — FIX: Regularization Strength

**File:** `config/settings.py:36`

```python
WLS_REGULARIZATION: float = 0.01
```

**Problem:** λ=0.01 on tick return magnitudes (~1e-4) provides near-zero shrinkage. The design matrix has rank ~7 (8 currencies minus 1 due to mean-zero constraint), so the L2 regularization is the only thing preventing the solver from fitting noise.

**Action:**
- Implement a cross-validation routine: hold out 2-3 pairs, solve on the rest, predict held-out returns, measure MSE. Sweep λ across [0.001, 0.01, 0.1, 0.5, 1.0, 5.0].
- Set the default λ to the value that minimizes held-out MSE.
- Expected result: λ will likely be in the range **0.1–1.0** for tick data.

---

## Item 8 — FIX: Recursive Prior

**File:** `currency/graph.py:57`

```python
self.state.prior = strengths
```

**Problem:** The prior for each solve is set to the previous solution. On noisy tick data, this creates a random-walk trajectory in strength estimates — the strengths drift over time with no restoring force. If the true currency strength is stationary over short horizons, the prior should shrink toward zero.

**Action:** Replace with a shrunk prior:

```python
shrink = 0.3  # tune this
self.state.prior = {c: (1 - shrink) * strengths[c] for c in strengths}
```

This pulls each strength toward zero, preventing drift. The `shrink` parameter can be cross-validated alongside λ.

---

## Item 9 — FIX: Quality Metric

**File:** `currency/graph.py:86-96`

```python
fit_quality = 1.0 - min(residual_norm / return_norm, 1.0)
```

**Problem:** This is a purely in-sample metric. WLS is designed to minimize the residual, so high quality does not mean the strengths have predictive power. It's circular.

**Action:** Replace with an out-of-sample predictive quality metric:
- Hold out 2-3 pairs from the solve.
- Solve WLS on remaining 25-26 pairs.
- Predict returns for held-out pairs using the solved strengths.
- Compute `quality = 1 - MAE(held_out_prediction) / MAE(held_out_actual)`.
- Track this alongside the in-sample fit metric but use it for gating decisions.

---

## Item 10 — FIX: Solve Frequency

**Files:** `currency/graph.py:43-74`, `runtime/manager.py:413-420`

**Problem:** The WLS solve runs every ~5 seconds on tick log-returns. Tick data is dominated by spread bounce (bid-ask noise). The 5-second returns have an extremely low signal-to-noise ratio.

**Action:**
- Make the **M5 bar-level WLS** (already implemented in `bar_state.py`) the primary strength signal.
- Keep tick-level WLS as a **delta** (change since last bar close) for intra-bar refinement.
- The tick-level strengths should be reported as `bar_strength + tick_delta`, where `tick_delta` is solved with a higher λ (more regularization) since it's computed on fewer data points per solve cycle.

---

## Item 11 — FIX: BarState Preload Window

**File:** `features/bar_state.py:36-55`

```python
rates = self.mt5.get_rates_from(symbol, _M5, 1, 30)
```

**Problem:** The preload fetches 30 M5 bars. If the most recent "completed" bar is actually still forming, its close price would leak future information. The `_compute_from_cache()` processes bars from `start` to `min_bars-1`, which could include a bar that hasn't finished forming.

**Action:** Verify that `get_rates_from(symbol, M5, 1, 30)` returns only fully-closed bars (MT5 typically does, but confirm). Add an explicit check:
- Compare the timestamp of the last bar against current server time.
- If the last bar's close time is within the last 5 minutes, exclude it from processing until confirmed closed.

---

## Item 12 — FIX: Add Unit Tests

**Missing files:** `tests/test_wls_solver.py`, `tests/test_graph.py`, `tests/test_hypothesis.py`

**Problem:** Zero tests exist for the core solver, graph state machine, or hypothesis generator. Any refactoring is blind without a regression harness.

**Action:** Create a `currency_decomposition/tests/` directory with:

| Test File | What It Tests |
|-----------|---------------|
| `test_wls_solver.py` | Design matrix structure, exact reconstruction of synthetic data, regularization effect, prior effect, zero-input stability |
| `test_graph.py` | State machine (BOOTSTRAP/PARTIAL/READY), quality decay, connectivity score, stress test |
| `test_hypothesis.py` | Confidence boundaries, direction logic, spread gating, missing data handling |
| `test_drs.py` | Currency vector correctness, exposure limit, rank ordering |
| `test_bar_state.py` | Preload, forming bar logic, alignment function (with mocked data) |

---

## Implementation Order

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | #1 Fix confidence formula (SNR) | Small | High |
| P0 | #7 Cross-validate λ | Medium | High |
| P0 | #9 Out-of-sample quality metric | Medium | High |
| P1 | #8 Fix recursive prior | Small | High |
| P1 | #12 Unit tests | Medium | High |
| P2 | #2 Remove DER multiplier | Small | Medium |
| P2 | #10 Make M5 bar WLS primary | Large | Medium |
| P3 | #3 Remove recency factor | Small | Low |
| P3 | #4 Remove redundant factors | Small | Low |
| P3 | #5 Simplify DRS selection | Medium | Low |
| P3 | #6 Remove health thresholds | Small | Low |
| P3 | #11 Verify bar preload window | Small | Low |
