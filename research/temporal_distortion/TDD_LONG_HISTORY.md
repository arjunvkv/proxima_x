# TDD-VL Phase 7: Out-of-Sample Extension

**Date:** 2026-06-16
**Status:** BLOCKED — Insufficient tick data

---

## Requirement

Acquire minimum 12 months of tick history and repeat the entire TDD research process.

## Available Tick Data

All per-symbol tick files cover only **2026-03-12 17:33:08 to 2026-06-10 17:33:07** (~90 days).

| Symbol | Tick File | Rows | Coverage |
|--------|-----------|------|----------|
| EURJPY | `data/ticks/EURJPY_ticks.parquet` | 10,390,515 | Mar 12 → Jun 10, 2026 |
| USDJPY | `data/ticks/USDJPY_ticks.parquet` | 6,298,538 | Mar 12 → Jun 10, 2026 |
| GBPUSD | `data/ticks/GBPUSD_ticks.parquet` | 7,859,109 | Mar 12 → Jun 10, 2026 |
| EURUSD | `data/ticks/EURUSD_ticks.parquet` | 6,856,005 | Mar 12 → Jun 10, 2026 |

**Total available tick history: 3 months. 9 months short of the 12-month minimum.**

## Additional Data Sources Checked

| Source | Tick Data? | Coverage | Viable? |
|--------|-----------|----------|---------|
| `data/ticks/ticks.parquet` (merged) | Some real ticks mixed with synthetic | Mar-Jun 2026 only | NO (synthetic contamination) |
| `data/bars/*.parquet` | No (bar aggregates) | Mar-Jun 2026 | NO |
| `proxima_x/data/market/*.parquet` | No (daily OHLC) | 2019-2025 | NO |
| `data/features/*.duckdb` | No (bar-level features) | Mar-Jun 2026 | NO |
| `exports/python_reference_m1/*.parquet` | No (M1 OHLC bars) | Mar-Jun 2026 | NO |
| `archive_v2/reports/*.parquet` | No (bar aggregates) | Mar-Jun 2026 | NO |

**No tick-level data exists for any period prior to March 2026.**

## MT5 Data Loader — Available But Historical

The MT5Loader exists at `archive_v2/proxima/data/mt5_loader.py` and can fetch tick data via:
```python
loader.fetch_ticks(symbol="EURJPY", days=90, from_date=datetime(2026, 1, 1))
```

However:
- MT5 only retains ticks for ~90 days on most brokers
- Historical tick data before ~March 2026 would require a commercial data provider (Dukascopy, TrueFX, etc.)
- No historical tick archive was maintained prior to the live demo launch

## 100-Tick Bar Alternative Considered

Bar-based TDD (using 100-tick bar timestamps as "events") was tested in Phase 4 and produces equivalent results. However, the 100-tick bar files also only cover March-June 2026:

| File | Rows | Coverage |
|------|------|----------|
| `data/bars/eurjpy_bars_100t.parquet` | 103,906 | Mar-Jun 2026 |
| `data/bars/usdjpy_bars_100t.parquet` | 62,986 | Mar-Jun 2026 |

No longer-duration bar data exists either.

## 1-Second Bar Alternative

`data/bars/eurusd_bars_1s.parquet` has 2.7M rows but also only covers Mar-Jun 2026.

---

## Options for Future OOS

| Option | Timeline | Feasibility |
|--------|----------|-------------|
| Wait for live accumulation | ~9 months (Mar 2027) | HIGH (automatic) |
| Commercial tick data purchase | Immediate | Depends on budget |
| MT5 historical (if broker supports) | 1-2 days | UNKNOWN (broker-dependent) |
| Use intraday bar data from alternative provider | 1-2 weeks | MEDIUM |

---

## Verdict: **BLOCKED — Cannot complete. 3 months tick data available, 12 months required. Earliest completion: March 2027 via live accumulation.**
