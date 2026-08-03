# FundedNext 5-Day Challenge — Strategy Evaluation Summary

**Goal**: Pass FundedNext Stellar Lite $5K challenge ($500 profit, $250 max daily loss) in 5 days.
**Date evaluated**: Jul 28, 2026
**Data source**: FundedNext Server 3 M1, Jun 8 – Jul 24, 2026 (34 trading days)

## Bottom Line

| Strategy | Passes? | At risk? | Daily $ | Confidence |
|----------|---------|----------|---------|------------|
| V2+z trailing-stop (z>=3.5) | **No** | 100% cost-drained | −$12 | High (costs consume edge) |
| Challenge-Z fixed-hold (any z) | **No** | No edge exists | −$90 | High (32-42% WR gross) |
| **Dark Consensus (0.3 lot)** | **Yes** | ~0% daily loss risk | **$195** | Medium (34-day FN sample) |

## Key Discovery

The Dark Consensus strategy — validated across 18 months, 21K trades, 3 data sources at
61.4% WR/$594/day — **transfers cleanly to FundedNext's data feed**: 93.1% gross WR,
83.1% net WR, $650/day at 1 lot, 100% positive days in sample.

A critical bug (missing `direction[t]` multiplier) initially produced a false 48.5% WR
result on FundedNext. Fixing it revealed the true edge.

## V2+z Cannot Pass

The V2+z trailing-stop strategy has a real edge (76-82% WR gross on M1) but costs
consume 100%+ of it at every viable configuration. At FundedNext's $3/lot + spreads,
the best single pair (GBPAUD z=3.5) nets +$0.38/day. Multi-pair CPPF portfolios
lose −$964 over 6 weeks. The strategy structure (low win rate × low payoff ratio)
is incompatible with ECN commission structures.

## Dark Consensus Can Pass

Dark Consensus produces $26.00/trade net after FundedNext costs, at 83.1% WR with
max 3 consecutive losses. At 0.3 lot ($195/day), challenge profit is reached in
~3 days with near-zero daily loss risk.

## Caveats

- FundedNext-specific validation is 34 trading days — limited but consistent with
  the 18-month Dukascopy validation (61.4% WR, 0% Monte Carlo failure)
- EURUSD spread data on FundedNext has 71% zero-spread bars (corrected with floor)
- FundedNext M1 bars have near-zero return correlation with a separate MT5 parquet
  file — feed-specific re-validation is essential
- The strategy's own paper-trading plan recommends 30 live days before live capital
