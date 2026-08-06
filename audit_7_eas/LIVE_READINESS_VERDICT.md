# 7-EA Live-Readiness Audit — Final Verdict (2026-08-08)

**Method:** Faithful M5-bar ports of the six real v106 EAs (Test_Min_Fire excluded —
it is a 1-minute connectivity/latency harness, not a strategy) run on **real FTMO
M5 history (200 days, Jan–Aug 2026)** across each strategy's full multi-pair
universe. Every signal uses **closed bars only**; fills **at next bar open**;
realistic costs (**$7/lot round-trip spread-commission** + pip-value). In-sample /
out-of-sample walk-forward split (67% / 33%). Plus an **anti-lookahead verification
suite** (fill-order assertion, determinism, and a purple/shuffle test).

## Verdict table (all values net of costs)

| Strategy | Universe | Trades | Win% | PF | Expectancy $/lot | Train+OOS gate | Purple test |
|---|---|---|---|---|---|---|---|
| **Tokyo_H0** | 18 pairs | 720 | **90.0%** | **9.0** | **+108.1** | **PASS both** | **PASS** |
| Ultra_Monster | 8 pairs | 9,766 | 40% | 1.00 | −6.5 | REJECT | FAIL |
| CPPF_Z | 5 pairs | 764 | 25% | 0.31 | −87.8 | REJECT | — |
| CPMC_Z | 2 pairs | 299 | 34% | 0.54 | −75.3 | REJECT | — |
| NY_H21 | 2 pairs | 231 | 47% | 1.12 | −2.5 | REJECT | FAIL |
| MSV_Asian | 18 pairs | 16,294 | 48% | 1.27 | +0.7 | REJECT (DD) | FAIL |

## The one that survives → Tokyo_H0 ✅
- **90% win rate, PF ~9, +$108/lot expectancy** — consistent across train and OOS.
- Anti-lookahead **VERIFIED**: 0 fill violations, deterministic, and shuffling the
  signal timing destroys the PnL → the edge is real, not a data artifact.
- **Live readiness:** high enough to matter but strong enough to warrant a small
  capital or ≤0.01-lot trial first. Port into the Proxima attach-only FTMO live
  path (max 5 × 0.15-lot, SL/TP attached) using the micro-run protocol.

## Deferred / fails (do NOT ship as-is)
- **Ultra_Monster** — breakeven gross, negative net after costs; shuffled entries
  perform *better* → no causal edge captured.
- **CPPF_Z / CPMC_Z** — 6-σ shock strategies lose money. SL(0.35)<TP(0.45)
  geometry sets SL closer than TP, so the sub-50% win rate is mechanical.
- **NY_H21** — breakeven/negative expectancy, purple fails.
- **MSV_Asian** — huge event count but ~zero expectancy; purple/shuffle fails.
- **Test_Min_Fire** — blind 1.20-lot open every minute, no signal. Excluded.

## Bottom line
The vault's headline win rates (Tokyo 95%, CPPF 85%, etc.) came from the MT5
strategy tester / earlier data. Under our aligned engine with realistic costs and
anti-lookahead on modern 2026 data, **only Tokyo_H0 invests** a genuinely
positive, causation-verified edge and is the one live-port candidate. Park the
rest.

Artifacts: `audit_7_eas/audit_report.json`, `antilookahead_report.json`,
`ea_ports.py` (ports), `run_audit.py` + `anti_lookahead.py` (harness).