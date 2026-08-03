# FundedNext 5-Day Challenge — Impulse Fade Strategy

## Overview

Use the **EURUSD impulse fade** (micro-scalping) strategy to pass the FundedNext 5-day
$25K challenge ($2,000 profit target, $1,250 max daily loss).

## Strategy Details

| Parameter | Value |
|-----------|-------|
| Pair | **EURUSD only** |
| Detection | 5-pip move in 20-sec sliding window |
| Entry | Fade — trade against the impulse direction |
| Hold | 30 seconds max |
| Stop | **10-pip** hard stop |
| Lot size | **2.0 lots** |
| **Trading hours** | **14:00-19:59 UTC** (critical filter) |
| Session rationale | London afternoon + NY morning overlap = highest liquidity |
| Cost (FundedNext) | 0.8 pip spread + $3 round-turn commission |
| Expected trades | ~38/day |
| Expected WR | ~61% |
| Expected avg | +1.07 pips/trade at 1 lot ($10.70/trade) |

## The Hour Filter — Key Discovery

The strategy has **strong directional sensitivity by hour of day**. Trading during
low-liquidity hours (especially 9-12 UTC) produces net-negative results. Filtering
to 14-19 UTC eliminates the losing week from the sample and improves all metrics.

**Before vs after filter (FundedNext ticks):**
| Metric | All Hours | 14-19 UTC Only |
|--------|:---------:|:--------------:|
| WR | 58.0% | **61.3%** |
| Avg PnL | +0.74p | **+1.07p** (+45%) |
| Gross (20d) | +637p | **+819p** |
| Worst week (W28) | -189p | **-5p** |

**Cross-validated on Exness (independent 3-month sample):**
| Metric | All Hours | 14-19 UTC Only |
|--------|:---------:|:--------------:|
| WR | 57.8% | **60.9%** |
| Avg PnL | +0.90p | **+1.25p** |
| Monthly PnL | All positive | **All positive** |

The pattern is consistent across both datasets: ~+3pp WR improvement, ~+40-50%
avg PnL improvement by avoiding the low-liquidity 9-12 UTC window.

## Data Sources

1. **Exness EURUSD ticks** (Oct–Dec 2025, 2.88M ticks, 65 trading days)
   - Used for initial validation, walk-forward, and hour filter cross-check

2. **FundedNext Server 3 ticks** (Jun 29–Jul 27 2026, 2.32M ticks, ~20 trading days)
   - Real tick data from the actual challenge broker

3. **FundedNext Server 3 spread verification** (via MT5 symbol_info):

| Pair | Spread (pips) |
|------|:-------------:|
| EURUSD | 0.8 |
| GBPUSD | 0.8 |
| AUDUSD | 0.9 |
| NZDUSD | 0.8 |
| USDCAD | 0.8 |

## Backtest Results — FundedNext Tick Data (14-19 UTC, 10p stop)

| Config | n | WR | Avg | Gross |
|:------:|:-:|:--:|:---:|:-----:|
| 5p/20s 10p stop | 765 | **61.3%** | **+1.07p** | +819p |

**Weekly breakdown:**
| Week | Trades | WR | Avg | Gross |
|:----:|:------:|:--:|:---:|:-----:|
| W27 (Jun 29-Jul 2) | 426 | 54.7% | +0.32p | +137p |
| W28 (Jul 7-10) | 95 | 49.5% | -0.06p | **-5p** |
| W29 (Jul 13-16) | 227 | 75.8% | +2.88p | +654p |
| W30 (Jul 20-22) | 17 | 100% | +1.95p | +33p |

No weeks are significantly negative with the hour filter. W28 (the problematic
US holiday week) goes from -189p to essentially break-even (-5p).

## Overfit/Lookahead Validation

1. **Walk-forward (Exness):** Dec (OOS) = 59.2% WR (better than training Oct-Nov)
2. **Sign randomization test:** p < 0.001 — edge is real, not noise
3. **Cross-data validation:** Same hour-filter pattern on both Exness and FundedNext
4. **Cost sensitivity:** Positive edge up to ~1.9 pips total cost (we're at 1.1)

## Monte Carlo Results (100K iterations, trade-level resampling, 14-19 UTC filter)

Each simulation: 5 days × 38 trades/day from actual distribution.
Pass = cumulative PnL ≥ $2,000 AND no single day < -$1,250.

| Lots | Pass % | Blow-Day % | Median Days to $2K |
|:----:|:------:|:----------:|:------------------:|
| 1.0  | 51.2% | 0.0% | 7 |
| 1.5  | 85.2% | 0.0% | 4 |
| **2.0** | **94.2%** | **0.0%** | **3** |
| 2.5  | 96.9% | 0.2% | 2 |
| 3.0  | 97.8% | 0.5% | 2 |

## Recommended Config

**EURUSD 2.0 lots, 10-pip stop, 14-19 UTC only — 94.2% pass rate, 0.0% blow-day.**

- Expected $2K in median 3 days, 95% within 5 days
- Zero blow-day risk in Monte Carlo simulation
- Only trades during high-liquidity London+NY overlap

## Implementation

```
cd paper_trade
python strategies/m1_z_reversal/run_v2.py
```

Config: 2.0 lots, 10p stop, session 14-19 UTC, EURUSD only, FundedNext terminal.

## Risk Notes

1. **The hour filter is essential.** Without it, WR drops to ~58% and losing weeks appear.
2. **Slippage could add 0.1–0.3 pips.** This would reduce avg to ~0.77–0.97p — still profitable.
3. **Cost sensitivity:** Breakeven at ~1.9 pips total. FundedNext = 1.1 pips. Safe margin of 0.8 pips.
4. **No other viable pairs.** Tested GBPUSD, AUDUSD, NZDUSD, USDCAD — all negative at FundedNext costs.
5. **FundedNext free trial** (account 34535207) — use for live validation before paid challenge.
