# TDD-VL Phase 3: Event Definition Audit

**Date:** 2026-06-16
**Asset:** EURJPY (10.4M ticks)
**Method:** 5 event definitions, 60s event rate, 5-min bar grid, sync_up at H50

---

## Event Definitions Tested

| # | Definition | Logic |
|---|-----------|-------|
| 1 | Bid changes | Event when bid price changes (diff > 0) |
| 2 | Ask changes | Event when ask price changes (diff > 0) |
| 3 | Mid-price changes | Event when (bid+ask)/2 changes (current default) |
| 4 | Spread changes | Event when spread changes by > 0.0001 |
| 5 | Volume events | Event when volume > 0 |
| 6 | High-low range | Event when 60-tick bid range > 75th percentile threshold |

---

## Results

| Event Definition | n_events | n_bars | n_sync_up | P(up\|sync_up) | P(up\|uncond) | Edge |
|-----------------|----------|--------|-----------|----------------|--------------|------|
| Bid changes | 8,598,860 | 25,919 | 2,344 | **0.5742** | 0.4040 | +0.1702 |
| Ask changes | 8,961,584 | 25,919 | 2,311 | **0.5716** | 0.4040 | +0.1676 |
| Mid-price changes | 10,302,956 | 25,919 | 2,259 | **0.5724** | 0.4040 | +0.1684 |
| Spread changes | 5,267,816 | 25,919 | 1,772 | **0.5705** | 0.4040 | +0.1665 |
| **Volume events** | **0** | — | — | **N/A** | — | — |
| High-low range | 2,597,597 | 25,919 | 1,302 | **0.5538** | 0.4041 | +0.1497 |

---

## Stability Statistics

| Metric | Value |
|--------|-------|
| Mean P(up\|sync_up) across 5 definitions | **0.5685** |
| Standard deviation | **0.0082** |
| Min | 0.5538 (HL range) |
| Max | 0.5742 (bid changes) |
| Range | 0.0204 |

**Coefficient of variation: 1.4%** — extremely stable

---

## Volume Events

Volume column is all zeros in the EURJPY tick dataset. The feed provides tick-level bid/ask/spread but no trade volume. Volume-based event detection is impossible with current data.

## High-Low Range Detail

The HL range method detects events when the 60-tick bid range exceeds its 75th percentile. It captures 2.6M events (fewer than mid-price changes) but produces a slightly lower sync_up P(up) of 0.554. The condition picks up coarser volatility events rather than individual price changes, resulting in fewer sync_up activations (1,302 vs ~2,250) and slightly weaker directional edge.

---

## Verdict: **PASS** — TDD is robust across event definitions. Mean P(up|sync_up) = 0.569 ± 0.008. No definition produces P(up) < 0.55. The signal is not an artifact of a specific event type.
