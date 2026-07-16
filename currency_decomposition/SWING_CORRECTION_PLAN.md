# Swing Correction Plan — Loop #4C Complete (Ready to Implement)

## Current State

TP/SL uses remaining-room fractions of M5 bar swing stats:
- `rem_up_price = max(0, (open + avg_up) - current)` — room to upper bound
- `rem_dn_price = max(0, current - (open + avg_dn))` — room to lower bound
- BUY: TP = current + rem_up × 0.60, SL = current - max(rem_dn × 0.40, 4p)
- SELL: TP = current - rem_dn × 0.60, SL = current + max(rem_up × 0.40, 4p)

## Problem

System enters trades at exhaustion points — when price has already consumed most of the swing. Example: GBPAUD BUY entry=1.93117, TP=0.5p (rem_up was tiny). Trade is dead on arrival; TP within spread/noise range.

## Root Cause (ChatGPT Loop #1)

The swing is treated as a "remaining distance calculator" when it should answer "what is the current location inside a probabilistic movement cycle." Direction is alpha — swing should provide timing context.

## Loop #2 — Dual-Layer Model

The forming-bar-only SPI was insufficient. Bar can open at structural extremes (previous bar's high), giving SPI=0 when price is actually exhausted. Two layers needed:
- **MSP** (Micro Swing Position): forming candle energy consumption
- **SSP** (Structural Swing Position): multi-bar range location

## Loop #3 — Implementation Architecture

SSP lives in `bar_state.py` with `current_price` injected by caller. SSP lookback = 20 bars (not 10 — avoids feedback loop). MSP uses a confidence score (bar_age_sec/60, cap at 1.0) to handle bar-open noise. SSP > 1 means breakout (not capped). Dashboard gets a new `swing_analysis` field per symbol, separate from `swing_reach`.

## Loop #4A — Historical Validation Design

### Logging Schema
Capture four layers per candidate: market context, WLS hypothesis, swing state, future outcome. Complete schema:
- **Identity**: timestamp, symbol, direction, candidate_id
- **Price context**: entry_price, bar_open, forming_return, spread
- **Swing state**: ssp, msp, swing_low, swing_high, avg_up_pips, avg_down_pips, swing_state, position_state
- **WLS context**: wls_score, currency_strength_base, currency_strength_quote
- **Bar context**: bar_age_seconds, completed_bar_count, recent_volatility
- **Outcome** (nullable, post-processed): mfe_10m, mae_10m, return_10m, mfe_30m, mae_30m, return_30m

### Storage
**DuckDB** (already in use) — table `swing_candidates` in `research/swing_validation.duckdb`. Not CSV (too chatty), not stderr (logs are for humans, research data is for machines).

### Deduplication
Key = `(symbol, direction, forming_bar_open_time)`. Log once per unique candidate. Allow re-log when direction changes or swing_state changes (captures transitions).

### Sampling
All unique candidates — need control group (HEALTHY, LATE, EXHAUSTED, BREAKOUT). Expected volume: ~216 observations/hour (18 symbols × 12 M5 bars/hr).

### Forward Tracking
**Post-processing only.** Log candidate with timestamp + entry_price. Offline script queries MT5 historical data to compute forward return, MFE, MAE at 10m and 30m horizons. Do not add runtime tracking to the engine.

### Validation Metrics
1. Forward return by swing_state (median EXHAUSTED < HEALTHY)
2. MFE by swing_state (EXHAUSTED should have lower extension)
3. MAE by swing_state (EXHAUSTED should have larger adverse movement)
4. MFE/MAE ratio by swing_state
5. Survival curve: probability of reaching +2/+5/+10 pips before adverse movement
6. SSP bucket analysis: forward return by 0.1-wide SSP bins to find the real turning point (not assumed 0.85)

## Loop #4B — Edge Case Hardening

### Issue 1: Units Mismatch (BUG)
MSP was `forming_return / avg_up` where `forming_return = log(current/open)` and `avg_up = high - open`. Log return ÷ price difference = meaningless.

**Fix**: Use price displacement for MSP:
- BUY: `MSP = (current_price - forming_open) / avg_up`
- SELL: `MSP = abs(current_price - forming_open) / abs(avg_dn)`

No log returns anywhere in swing. WLS can keep log returns — swing is geometric.

### Issue 2: MSP > 1.0
Valid and valuable. Means candle has exceeded normal swing energy. Add MSP state categories:
- < 0.5: EARLY
- 0.5-1.0: DEVELOPING
- 1.0-2.0: EXTENDED
- > 2.0: EXTREME

### Issue 3: Stale SSP (insufficient bars)
Do NOT default to HEALTHY (hidden bias — "unknown is safe" is wrong).

**Fix**: Use `INSUFFICIENT_DATA` state. For observation mode: allow. For production blocking: require minimum 20 completed bars.

### Issue 4: Tight Ranges
SSP is still meaningful in tight ranges; the problem is economic significance.

**Fix**: Add `range_quality = range_width / median_20bar_range`. If < 0.5, state = `COMPRESSED_RANGE`. Do not disable SSP — add the quality label.

### Issue 5: Volatility Override
News events (NFP) can have SSP=0.95, MSP=2.67 — mathematically exhausted, but momentum may continue for another 40 pips.

**Fix**: Add `vol_expansion = current_bar_movement / median_movement_last_20_bars`. If > 3.0, state = `HIGH_VOL_EXPANSION`. Suppress exhaustion classification (log as `EXHAUSTION_SUPPRESSED_BY_VOLATILITY`).

### Issue 6: Same-Bar Direction Flips
Not a swing problem — an alpha stability issue. Do not mix. Add a separate direction persistence layer later (require confirmation if same-bar flip).

### Issue 7: SELL SSP
Compute independently, not via inversion:
- `buy_ssp = (current - swing_low) / (swing_high - swing_low)`
- `sell_ssp = (swing_high - current) / (swing_high - swing_low)`

### Revised Classification Model

```
                    DATA CHECK
                       |
        +--------------+--------------+
        |                             |
   insufficient                   valid
        |                             |
   UNKNOWN                    volatility check
                                      |
                         +------------+-------------+
                         |                          |
                    expansion                  normal
                         |                          |
                SUPPRESS EXHAUSTION        classify SSP/MSP
```

## Revised Data Model (SwingAnalysis)
```python
{
    "buy_ssp": float,       # 0-1, can exceed 1 for breakout
    "sell_ssp": float,      # 0-1, can exceed 1 for breakout
    "msp": float,           # energy consumption ratio, can exceed 1
    "msp_state": str,       # EARLY / DEVELOPING / EXTENDED / EXTREME
    "range_pips": float,    # 20-bar range in pips
    "range_quality": float, # range_width / median_20bar_range
    "vol_expansion": float, # current movement / median movement
    "vol_state": str,       # NORMAL / HIGH_VOL_EXPANSION
    "position_state": str,  # INSIDE_RANGE / BREAKOUT_UP / BREAKOUT_DOWN / COMPRESSED_RANGE
    "swing_state": str,     # HEALTHY / LATE / EXHAUSTED / INSUFFICIENT_DATA
    "decision": str         # ALLOW / CAUTION / NO_CLASSIFICATION
}
```

## Implementation Stages

1. **Stage 1A (observation mode)**: Add SSP/MSP/classifier to bar_state.py + dashboard overlay. No blocking. No logging to DuckDB yet.
2. **Stage 1B (validation logger)**: Add research/swing_logger.py + DuckDB table `swing_candidates`. Collect 1-2 weeks of data before enabling blocking.
3. **Stage 2 (entry filtering)**: Enable EXHAUSTED block only (with volatility suppression). No TP/SL changes.
4. **Stage 3 (TP/SL changes)**: Replace remaining-room TP with expected-remaining-swings TP. Replace blanket 4p SL with spread-adaptive minimum.

## Pipeline Order (revised)
```
WLS → SSP Structural Location → Volatility Check → Range Quality → Bar Alignment → MSP Micro Check → TP/SL → Risk → Execute
```

## Loop #4C — Code-Level Review Final

### Blocking Issues Fixed

1. **MSP log return bug**: `forming_return_from_open()` returns log returns. Must NOT use for MSP. Added `forming_price_displacement()` returning price units.
2. **Pip conversion in bar_state**: REMOVED. `bar_state.py` returns raw price values only. Manager handles pip display.
3. **Directionless dashboard**: Store buy/sell states separately. No WLS direction selection in observation layer.
4. **MSP directionless**: `get_micro_swing_positions()` returns both `buy_msp` and `sell_msp`.
5. **Confidence < 0.5** → `UNCONFIRMED` state, not partial SSP-only classification.
6. **Vol expansion**: Bar-age-aware — early bars not classified as low vol.
7. **Range quality → range_expansion**: Renamed for clarity.

### Architecture Principle (critical)
> Market state exists first. Trade hypothesis queries it afterward.

Keeps SSP/MSP reusable for future engines (NME, regime detection, execution timing).

## Final Data Model

### bar_state.py additions

**`forming_price_displacement(symbol)`**
Returns `current_price - forming_open` in price units. Distinct from `forming_return_from_open()` which continues returning log returns.

**`get_structural_swing_position(symbol, current_price, lookback=20)`**
Returns `{ buy_ssp, sell_ssp, swing_low, swing_high, range_price, range_expansion, vol_expansion }`. No pip conversion.

**`get_micro_swing_positions(symbol, avg_up, avg_dn)`**
Returns `{ buy_msp, sell_msp, bar_age_seconds, confidence }`. No direction dependency.

**`classify_swing_state(direction, ssp_data, msp_data)`**
Direction-dependent decision layer. Returns `{ swing_state, position_state, decision }`.

### Dashboard structure (directionless)
```json
"swing_analysis": {
  "buy": { "ssp": 0.82, "msp": 0.55, "state": "LATE" },
  "sell": { "ssp": 0.18, "msp": 0.20, "state": "HEALTHY" },
  "range_expansion": 1.4,
  "vol_expansion": 0.9,
  "position_state": "INSIDE_RANGE"
}
```

## Next Steps
- **IMPLEMENT Stage 1A**: bar_state.py (4 new methods) + manager.py (dashboard wiring)
- **Verify**: Run engine, confirm `swing_analysis` appears in output, no execution changes
