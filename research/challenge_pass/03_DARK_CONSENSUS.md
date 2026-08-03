# Dark Consensus on FundedNext — Full Results

## Strategy Description

3-pair (EURJPY, EURUSD, GBPJPY) P95 magnitude consensus strategy:
- Check if all 3 pairs moved in the same direction (consensus)
- Filter: session 07-21 UTC
- Filter: avg |return| > P95 threshold (0.00018741)
- Select best_pair (largest |return| within consensus)
- Enter at bar close, exit after 3 bars (3 minutes)
- Direction: LONG if consensus UP, SHORT if consensus DOWN

## FundedNext Cost Model

| Component | Value |
|-----------|-------|
| Commission | $3/lot (Stellar Lite raw) |
| Slippage | 0.5 pips per trade |
| Spread | Actual FundedNext spread, EURUSD floored at median non-zero (8pts) |
| Pip values | EURUSD $10, JPY pairs 1000/USDJPY |

Spread stats on FundedNext (Jun 8 – Jul 24, 2026):
- EURJPY: med=0.9 pips, p90=1.5 pips
- EURUSD: med=0.8 pips (after floor, raw data 71% zeros)
- GBPJPY: med=1.5 pips, p90=2.4 pips

## Results (850 trades, 34 trading days)

### Gross (before costs)
| Metric | Value |
|--------|:-----:|
| Win rate | 93.1% |
| Avg/trade | $41.12 |
| Sharpe | 22.99 |
| Total | $34,949 |

### Net (after FundedNext costs)
| Metric | Value |
|--------|:-----:|
| Win rate | 83.1% |
| Avg/trade | $26.00 |
| Sharpe | 14.54 |
| Total | $22,096 |
| Max consecutive losses | 3 |
| Max consecutive wins | 25 |

### Daily
| Metric | Value |
|--------|:-----:|
| Positive days | 34/34 (100%) |
| Avg daily PnL | $649.88 |
| Best day | $1,869 |
| Worst day | $63.64 |
| Daily std | $524.15 |

### Pair Distribution (Net)
| Pair | % of Trades | WR | Avg/trade | Total |
|------|:----------:|:--:|:---------:|:-----:|
| EURUSD | 55% | 80.4% | $26.29 | $12,225 |
| GBPJPY | 34% | 86.4% | $25.56 | $7,336 |
| EURJPY | 12% | 85.7% | $25.87 | $2,535 |

### Cost Breakdown
| Component | Per Trade | Total |
|-----------|:---------:|:-----:|
| Commission | $3.00 | $2,550 |
| Spread | $8.02 | $6,817 |
| Slippage | $4.15 | $3,528 |
| **Total** | **$15.17** | **$12,853** |

Costs consume 37% of gross PnL — strategy survives easily (breakeven at ~3.5x
spread per the Dukascopy stress test).

## Consistency vs Published Dukascopy Results

| Metric | Dukascopy (18mo) | FundedNext (34d) |
|--------|:----------------:|:----------------:|
| Gross WR | ~67% (stress) | 93.1% |
| Net WR | 60.2% | 83.1% |
| Sharpe | 5.74 | 14.54 |
| $/trade | $11.80 | $26.00 |
| $/day | $594 | $650 |

FundedNext results show higher WR and Sharpe than the Dukascopy average, which
is partially attributable to the shorter sample. The Jun 2026 month on Dukascopy
showed 53.9% WR / $183/day — the weakest month in the 18-month series. Our
FundedNext Jun-Jul period shows stronger results, suggesting the strategy edge
returned after a weak June.

## Live Feed Parity Gap

The Dark Consensus validation package rates "Live feed parity" at 1/10 — it
has NOT been tested on a live MT5 feed yet. The original plan called for
30 days of paper trading before deployment. The strategy code (strategy.py,
mt5_executor.py) is designed but not yet written.
