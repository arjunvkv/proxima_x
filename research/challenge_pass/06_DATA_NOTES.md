# FundedNext Data Quality Notes

## Data Source

FundedNext Server 3 terminal at `C:\Program Files\FundedNext MT5 Terminal`
(terminal hash: `89FE26BBBAB28C077BBF5FA8C1B4DF1C`).
Data downloaded via `copy_rates_range()` for M1 timeframe, Jun 8 – Jul 24, 2026.

## Pair Coverage

| Pair | Bars | Period |
|------|:---:|--------|
| EURJPY | 50,173 | Jun 8 00:00 – Jul 24 23:59 |
| EURUSD | 49,872 | Jun 8 00:00 – Jul 24 23:59 |
| GBPJPY | 50,258 | Jun 8 00:00 – Jul 24 23:59 |
| AUDUSD | 50,007 | Jun 8 00:00 – Jul 24 23:59 |
| EURAUD | 46,724 | Jun 8 07:00 – Jul 24 23:59 |
| GBPAUD | 46,724 | Jun 8 07:00 – Jul 24 23:59 |

## Spread Quality

| Pair | Median | P90 | Zero % | Notes |
|------|:-----:|:---:|:------:|-------|
| EURJPY | 9pt (0.9p) | 15pt | 0% | Clean |
| EURUSD | 8pt (0.8p)* | 12pt | 70.7% | **71% zero spread** — floored at 8pt |
| GBPJPY | 15pt (1.5p) | 24pt | 0% | Clean |
| AUDUSD | 3pt (0.3p) | 12pt | 49.6% | Many zero spreads |
| EURAUD | 10pt (1.0p) | 17pt | 0% | Clean |
| GBPAUD | 12pt (1.2p) | 22pt | 0% | Clean |

*EURUSD spread floored at median of non-zero values.

## Cross-Pair Timestamp Alignment

| Pair Pair | Overlap % |
|-----------|:---------:|
| EURJPY × EURUSD | 99.3% |
| EURJPY × GBPJPY | 99.9% |
| EURUSD × GBPJPY | 99.9% |

Minimal alignment issues — ~0.1-0.7% of bars differ between pairs due to
different market open times or missing ticks.

## Price Quality

- Zero extreme returns: 0 1-min returns >1% across all pairs
- All OHLC values non-zero and non-NaN
- Price ranges reasonable for period:
  - EURJPY: 183.18 – 186.66
  - EURUSD: 1.1325 – 1.1621
  - GBPJPY: 212.46 – 219.61

## Data Limitations

1. **EURUSD spread unreliability**: 71% of bars report spread=0, suggesting
   FundedNext does not populate the spread field consistently for EURUSD.
   All EURUSD cost estimates use the floored value (8pt = 0.8 pips).

2. **Weekend gaps**: Standard 2-day gaps present (Sat-Sun). Gap-fill data not
   available — bars resume at Sunday 22:00 UTC (market open).

3. **No tick data**: M1 OHLC only. The 3-minute hold precision cannot be
   validated at tick level on this data.

4. **Single broker**: All FundedNext-specific testing is on one broker's data.
   Results may not transfer to other brokers' feeds.

## Feed Discrepancy with MT5 Parquet

A separate MT5 data file (`data/temp/mt5_m1_9day.parquet`, Jun 30 – Jul 18,
7 pairs) was compared against FundedNext data for the overlapping period.
Returns correlation was near zero (EURUSD: r=0.032), suggesting:

- The two feeds construct bars at different alignment boundaries
- OR the price sources differ significantly
- OR the parquet file was constructed from a different data collection method

This means FundedNext backtest results may not generalize to other brokers,
and live feed parity testing is essential before deployment.
