# FundedNext Spread Analysis

## All 3 Dark Consensus Pairs

### EURJPY (3-digit, spread in points)
| Stat | Value |
|------|-------|
| Count | 50,173 |
| Zero spread | 0 (0%) |
| Median | 9.0 pts (0.9 pips) |
| P90 | 15.0 pts (1.5 pips) |
| Max | 100 pts (10.0 pips) |
| Mean | 9.8 pts |

### EURUSD (5-digit, spread in points)
| Stat | Raw | Floored |
|------|:---:|:-------:|
| Count | 49,872 | 49,872 |
| Zero spread | 35,243 (70.7%) | 0 |
| Median | 0 | 8.0 pts (0.8 pips) |
| P90 | 12.0 pts (1.2 pips) | 12.0 pts (1.2 pips) |
| Max | 88 pts (8.8 pips) | 88 pts (8.8 pips) |

### GBPJPY (3-digit, spread in points)
| Stat | Value |
|------|-------|
| Count | 50,258 |
| Zero spread | 0 (0%) |
| Median | 15.0 pts (1.5 pips) |
| P90 | 24.0 pts (2.4 pips) |
| Max | 100 pts (10.0 pips) |
| Mean | 15.6 pts |

## The Zero-Spread EURUSD Problem

EURUSD on FundedNext Server 3 reports spread=0 for 71% of M1 bars.
This is likely NOT a true zero spread but a data artifact:
- Raw ECN accounts typically show 0.1-3.0 pip spread
- Only 29% of bars have non-zero spread, suggesting the terminal
  doesn't populate the field consistently
- The non-zero bars show median 8 points (0.8 pips), which is
  reasonable for a raw account

Our fix: floor all EURUSD spreads at 8 points (median of non-zero).
This is conservative — 75% of non-zero spreads are at or above 8 points.
