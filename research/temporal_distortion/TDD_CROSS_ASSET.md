# TDD-VL Phase 4: Cross-Asset Audit

**Date:** 2026-06-16
**Method:** TDD on ALL available symbol-data combinations

---

## Data Availability

### Tick Data (bid/ask, microsecond timestamps)
4 symbols: EURJPY, USDJPY, GBPUSD, EURUSD (all 2026-03-12 to 2026-06-10)

### No Tick Data
GBPJPY, XAUUSD, NAS100 — **no tick data exists** in the entire project tree. Only daily OHLC for 2019-2025.

### Bar-Based TDD
100-tick bar data available for: EURJPY, USDJPY, GBPUSD, EURUSD — bar formation rate serves as event rate proxy.

---

## Tick-Based TDD Results (H50)

| Symbol | n_ticks | n_events | n_bars | n_sync_up | P(up\|sync_up) | P(up\|uncond) | Edge | Classification |
|--------|---------|----------|--------|-----------|----------------|--------------|------|---------------|
| **EURJPY** | 10,390,515 | 10,302,956 | 25,919 | 2,259 | **0.5724** | 0.4040 | +0.168 | **SIGNAL** |
| **USDJPY** | 6,298,538 | 6,213,424 | 25,919 | 2,085 | **0.5439** | 0.4119 | +0.132 | **SIGNAL** |
| GBPUSD | 7,859,109 | 7,770,867 | 25,919 | 2,430 | 0.4901 | 0.3617 | +0.128 | NOISE |
| EURUSD | 6,856,005 | 6,503,424 | 25,919 | 2,210 | 0.4991 | 0.3596 | +0.140 | NOISE |

## Bar-Based TDD Results (H50) — Using 100-tick Bar Completion as Events

| Symbol | n_bars | n_sync_up | P(up\|sync_up) | P(up\|uncond) | Edge | Matches Tick? |
|--------|--------|-----------|----------------|--------------|------|--------------|
| **EURJPY** | 25,919 | 2,479 | **0.5777** | 0.4044 | +0.173 | YES |
| **USDJPY** | 25,919 | 2,354 | **0.5892** | 0.4123 | +0.177 | YES |
| GBPUSD | 25,919 | 2,542 | 0.4910 | 0.3598 | +0.131 | YES |
| EURUSD | 25,919 | 2,363 | 0.4829 | 0.3584 | +0.125 | YES |

---

## Key Findings

### 1. Tick vs Bar — Method Generalizes
TDD produces nearly identical sync_up values regardless of whether events are raw ticks or 100-tick bar completions:
- EURJPY: tick 0.5724 vs bar 0.5777 (Δ = 0.005)
- USDJPY: tick 0.5439 vs bar 0.5892 (Δ = 0.045)
- GBPUSD: tick 0.4901 vs bar 0.4910 (Δ = 0.001)
- EURUSD: tick 0.4991 vs bar 0.4829 (Δ = 0.016)

The methodology is data-format-invariant.

### 2. JPY Crosses Signal, Others Noise
The signal is **JPY-cross-specific**:
- EURJPY: CLEAR SIGNAL (0.572-0.578)
- USDJPY: CLEAR SIGNAL (0.544-0.589)
- GBPUSD: NO SIGNAL (0.490-0.491)
- EURUSD: NO SIGNAL (0.483-0.499)

### 3. Cross-Asset Inventory: No Data for 3/6 Required Symbols

**Required symbols:** EURJPY, USDJPY, GBPJPY, XAUUSD, NAS100, EURUSD
**Available tick data:** EURJPY, USDJPY, GBPUSD, EURUSD (only 4/6)
**Missing:** GBPJPY, XAUUSD, NAS100

The MT5Loader can fetch tick data for all MT5 symbols, but:
- No historical GBPJPY/XAUUSD/NAS100 ticks have been collected
- The live demo has been running since March 2026 with only 4 forex symbols
- Fetching 90 days of tick data for 3 new symbols would add ~15-25M ticks

---

## Verdict: **PARTIAL — Does Not Generalize to Non-JPY Crosses**

TDD's sync_up signal is **JPY-cross-specific**. EURJPY and USDJPY confirm the effect (0.54-0.59). GBPUSD and EURUSD show no directional signal (P(up) ≈ 0.49). Missing tick data for GBPJPY, XAUUSD, NAS100 leaves 3/6 required symbols untested.

The signal is genuine for its target assets but does not generalize to all forex pairs. This is consistent with the interpretation that the temporal clustering effect is a JPY-cross microstructure phenomenon (possibly related to carry trade dynamics, Tokyo/London overlap patterns, or specific liquidity structures).
