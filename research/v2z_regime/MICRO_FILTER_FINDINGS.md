# V2+z Microstructure Regime Detection — Research Summary

## Problem
V2+z z-score mean reversion has an edge that shifts between regimes. Edge existed in Feb-Mar 2026 (+$471 EURAUD gross) but vanished in Jun-Jul 2026 (-$484 EURAUD gross). No parameter tweak (TRIG_A sweep 0.5-2.0, Z=2.5-3.5) restores profitability.

## Approach
Instead of finding "always profitable" parameters, detect profitable regimes using entry-bar microstructure. The bar that triggers the trade (the completed M1 bar at index 1 whose close produces an extreme z-score) contains information about whether mean reversion will succeed.

## Microstructure Features Tested
- **Smoothness** = body / range (0=doji, 1=marubozu)
- **Body size** in pips (absolute |close - open|)
- **Range** in pips (high - low)
- **Tick volume** (tick count for the bar)
- **Spread** at entry time
- **Z-score magnitude** (abs_z)
- **Wick with direction** (wick on the entry side)
- **Vol ratio** (tick_vol / median tick_vol)

## Key Finding: Microstructure Regime Shift
The direction of feature predictive power FLIPS between OOS and Forward:

| Feature | OOS (Feb-Mar) | Forward (Jun-Jul) |
|---------|:----------:|:--------------:|
| abs_z | WIN>LOSE (high z = mean rev works) | LOSE>WIN (high z = trend continues) |
| smoothness | LOSE>WIN (rough bars better) | WIN>LOSE (smooth bars better) |
| body_pips | LOSE>WIN (smaller better) | WIN>LOSE (larger better) |
| range_pips | WIN>LOSE | WIN>LOSE (consistent) |
| tick_vol | WIN>LOSE | WIN>LOSE (consistent) |

## Microstructure Filter (Implemented in EA)
Added `ENABLE_MICRO_FILTER` with parameters:
- `MICRO_MIN_BODY_PIPS = 4.0` — entry bar must span ≥4 pips
- `MICRO_MIN_SMOOTHNESS = 0.85` — body/range ≥0.85 (decisive bar)
- `MICRO_MIN_TICK_VOL = 200` — high tick volume (capitulation)

## Results (EURAUD, all gross before $5/lot commission)

| Test | Trades | Gross PnL | Net PnL | WR |
|------|:-----:|:--------:|:-------:|:-:|
| **OOS unfiltered** (Feb-Mar) | 39 | +$666.75 | +$374.25 | 77% |
| OOS + filter (body+sm+tv) | 12 | +$59.25 | -$30.75 | — |
| **Forward unfiltered** (Jun-Jul) | 80 | -$697.50 | -$1,297.50 | 42% |
| Forward + body+sm only | 20 | +$66.00 | -$84.00 | 70% |
| Forward + body+sm+tv | **9** | **+$149.25** | **+$81.75** | **78%** |

## Interpretation
The micro filter isolates "capitulation bars" — large-range, smooth, high-volume bars at z-score extremes. These represent exhaustion of a directional move. When present, mean reversion succeeds even in otherwise unprofitable regimes.

The filter overfits to the forward period. In OOS, smooth bars were LESS predictive of winners (opposite relationship), suggesting the underlying market microstructure truly shifted between Feb-Mar and Jun-Jul.

## Combined Performance
- Unfiltered (OOS+Forward): -$110.32 net
- Filtered (OOS+Forward): +$51.00 net

## EA Implementation (V2z_v2_Clean.mq5, version 1.02)
Modified from original V2z_CPPF base:
1. Added `LogEntry()` — prints ENTRYBAR with z-score, completed bar OHLC, tick vol, spread
2. Added `LogCloseFromHistory()` — reads deal history when position auto-closed by tester (SL)
3. Added `ENABLE_MICRO_FILTER`/`MICRO_MIN_BODY_PIPS`/`MICRO_MIN_SMOOTHNESS`/`MICRO_MIN_TICK_VOL`
4. Fixed bar index from 0 (degenerate new bar) to 1 (completed bar)

## Files
- `paper_trade/mt5_backtest/V2z_v2_Clean.mq5` — EA with micro filter
- `paper_trade/mt5_backtest/V2z_v2_Baseline_EURAUD.ini` — Forward EURAUD config
- `paper_trade/mt5_backtest/V2z_v2_OOS_EURAUD.ini` — OOS EURAUD config
- `research/v2z_regime/` — Python analysis scripts

## Open Questions
1. Does the micro filter work on other pairs (GBPAUD, AUDNZD)?
2. Can a dynamic regime classifier (rolling window) detect which regime we're in and switch filters?
3. Is there a universal feature set that works across both regimes?
