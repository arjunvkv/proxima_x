# Challenge-Z Fixed-Hold — Complete Analysis

## Hypothesis

A z-score threshold on 1-min returns with a fixed 10-15 minute hold should
capture mean reversion on extreme M1 moves. Simpler than trailing-stop —
enter at bar close, exit after N bars.

## Results

**No edge exists.** Every tested configuration produced sub-50% WR:

| Pair | z | Hold | Trades | WR | Net PnL |
|------|---|:----:|:-----:|:--:|:-------:|
| AUDUSD | 2.5 | 10min | 102 | 31% | −$890 |
| AUDUSD | 3.0 | 10min | 47 | 38% | −$410 |
| AUDUSD | 3.5 | 10min | 19 | 42% | −$120 |
| EURAUD | 2.5 | 10min | 89 | 34% | −$670 |
| EURAUD | 3.0 | 10min | 42 | 38% | −$360 |
| EURAUD | 3.5 | 10min | 18 | 33% | −$200 |
| GBPAUD | 2.5 | 10min | 95 | 36% | −$510 |
| GBPAUD | 3.0 | 10min | 51 | 39% | −$410 |
| GBPAUD | 3.5 | 10min | 23 | 35% | −$310 |

## Why It Fails

The edge in M1 z-score moves is NOT in the fixed time horizon after the move.
It's in the **path-dependent structure** captured by the trailing stop. The mean
reversion happens quickly (1-3 minutes) for some moves and slowly (5-10 minutes)
for others. A fixed hold either exits too early (missing the reversion) or too
late (catching the reversal of the reversion).

Testing at 15-minute holds produced equally poor results.

## Conclusion

Fixed-hold z-score strategies on M1 data have no detectable edge on any pair.
The trailing-stop version works because it adapts to each trade's specific
reversion speed.
