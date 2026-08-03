# Configuration Sweep Results

## V2+z Trailing Stop Sweep

54 configs: z=(3.5,3.0,2.5) × 3 pairs × sprd=(5,10,15)

### Survivors (net positive)
- GBPAUD z=3.5 sprd≤10: +$0.38/day (620 trades, 76.1% WR)

### All others: negative net

### Direction Asymmetry
From CPPF research sweep (`research/cppf/sweep_p6_direction.py`):
- EURAUD/GBPAUD/GBPCAD: strong LONG bias
- EURNZD: SHORT bias only
- Asian session (0-7 UTC) NOT optimal — full-day better

## Dark Consensus Threshold Sweep on FundedNext

| Threshold | Mag | n | WR | Sharpe | Avg$ |
|:---------:|:--:|:-:|:-:|:-----:|:----:|
| P80 | 0.000083 | 6177 | 49.7% | 0.24 | $0.37 |
| P85 | 0.000096 | 4794 | 50.0% | 0.30 | $0.49 |
| P90 | 0.000115 | 3296 | 48.8% | 0.06 | $0.11 |
| P93 | 0.000133 | 2319 | 49.5% | 0.34 | $0.68 |
| P95 | 0.000151 | 1653 | 48.6% | 0.21 | $0.46 |
| P97 | 0.000179 | 979 | 48.1% | 0.27 | $0.68 |
| DukP95 | 0.000187 | 850 | 48.5% | 0.51 | $1.32 |
| P99 | 0.000251 | 299 | 49.8% | 1.06 | $3.71 |

**Note**: These numbers are with the ALWAYS-LONG bug (no direction multiplier).
They represent the raw directional accuracy of the best_pair going LONG.

### Direction Asymmetry (DukP95 threshold, correct PnL)
| | n | WR | Sharpe | Avg$ |
|:-:|:-:|:-:|:-----:|:----:|
| LONG signals | 807 | 92.2% | 20.85 | $35.01 |
| SHORT signals | 846 | 93.0%* | 20.85* | $35.01* |

\*SHORT signal WR = 100% - 7.0% = 93.0% (inverted from LONG-PnL computation).

### Hour Filter Effect
- With hour filter (07-21): 850 trades, $1,118 gross total
- Without hour filter (0-24): 1,007 trades, −$579 gross total
- The hour filter removes noise and is essential for edge
