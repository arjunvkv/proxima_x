# V2+z Trailing-Stop — Complete Analysis

## Hypothesis

A z-score threshold (>=3.5) on 1-min returns across 6 cross pairs (EURAUD, AUDNZD,
GBPAUD, EURNZD, GBPCAD, GBPNZD), entered with market orders and managed via
trailing stop (stop_a=3.0, trig_a=1.0, gap_a=0.05), captures mean reversion edge
on extreme M1 moves. Hold time is variable (determined by trailing stop).

## Data Source

FundedNext Server 3 M1 data: AUDUSD, EURAUD, GBPAUD (Jun 8 – Jul 24, 2026).
3-digit quotes for AUDUSD (AUDUSD is 5-digit actually, EU and GBPAUD are 5-digit).

## Key Discovery

The trailing-stop strategy HAS a real gross edge:

| Pair | z | Gross WR | Gross PnL | Spread + Comm | Net PnL |
|------|---|:--------:|:---------:|:-------------:|:-------:|
| AUDUSD | 3.5 | 76.4% | +$613 | −$800+ | negative |
| EURAUD | 3.5 | 79.2% | +$140 | −$300+ | negative |
| GBPAUD | 3.5 | 82.1% | +$233 | −$225 | +$0.38/day |

**Winners averaged $25-48, losers averaged $44-126. Payoff ratio ~0.5.**

The edge exists and is robust (>76% WR across all pairs), but the strategy
structure — many small wins, few large losses — is incompatible with ECN
commission models. Spread alone consumes 50-70% of gross PnL.

## Sweep Results

54 configurations tested: z=(3.5,3.0,2.5) × 3 pairs × sprd=(5,10,15):

- Only GBPAUD z=3.5 sprd≤10 survives costs: +$0.38/day
- Every other config: negative net
- Lower z-thresholds (3.0, 2.5) multiply trade count but cost grows faster —
  gross edge per trade is flat at $2-3 while spread cost is fixed per trade
- 6-pair CPPF portfolio: −$964 over 6 weeks Forward (with commission)

## Python Sim Was Invalid

`sim_backtest.py` used M1 close prices with `// 10**9` on already-seconds
timestamps, producing epoch=1 (1970-01-01) timestamps. All bars appeared to
be before the start date, so the warmup period never ended. It showed
+$16,492 Forward — the real MT5 tester result was −$964. The sim had 17x
look-ahead bias from M1 close-price testing of a trailing-stop strategy.

## V2+z: Not viable for FundedNext challenge

- Max realistic net at 0.75 lot: ~$12/day (6-pair CPPF)
- Need $100/day for challenge
- Daily loss limit ($250) cannot be avoided — strategy has tail risk (max DD
  5.9% in Forward, or $295 at $5K)
