# Frequency Filter Reality — Evidence Collection

**Phase 5** | **Date:** 2026-06-16
**Classification:** ALPHA_DESTROYER (tentative — insufficient evidence)

---

## Current Evidence

| Metric | Value |
|--------|-------|
| Blocked signals | 31 |
| Profitable blocked | 26 |
| Leakage rate | 83.9% |
| ADR | 1.000 |
| Executed trades | 3 |
| Evaluations | 415 |

## Statistical Sufficiency

The current evidence (31 blocked, 3 executed) is **NOT statistically sufficient** to conclude `ALPHA_DESTROYER`.

## Collection Targets

| Target | Current | Remaining |
|--------|---------|-----------|
| 100+ blocked signals | 31 | 69 |
| 50+ executed trades | 3 | 47 |

## Observation Plan

- Do NOT alter filter behavior
- Record every blocked signal (symbol, ES rank, AT rank, timestamp)
- Record every executed trade (symbol, volume, PnL)
- Re-evaluate at 100 blocked + 50 executed
- Generate `FREQUENCY_EVIDENCE_REPORT.md` at collection targets

## Hypothesis

If leakage rate remains > 60% at 100+ blocked / 50+ executed,
then the frequency filter is destroying alpha and the `ALPHA_DESTROYER`
classification is confirmed.
