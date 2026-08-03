# FundedNext 5-Day Challenge — Feasibility Analysis

## Challenge Rules (Stellar Lite $5K)

- Target profit: **+$500** within 5 trading days
- Maximum daily loss: **−$250** (5% of $5K) — hard breach
- Maximum drawdown: 10% ($500) from starting balance
- Minimum trading days: 1 (can pass in a single day)
- Instruments: Forex, metals, indices (no crypto)
- Leverage: 1:100

## Dark Consensus at 1 Lot

| Metric | Value |
|--------|:-----:|
| Avg daily PnL | $650 |
| Worst day | $63.64 (all positive in sample) |
| Worst trade | −$115.00 |
| Max consecutive losses | 3 |
| Expected 5-day PnL | $2,500 – $3,900 |

## Risk Scaling Options

| Lot | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Worst Trade | Daily Loss Risk |
|:---:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|:---------------:|
| 0.2 | +$130 | +$130 | +$130 | +$130 | +$130 | −$23 | ~0% (but takes 4 days) |
| **0.3** | **+$195** | **+$195** | **+$195** | **+$195** | **−** | **−$35** | **~0% — passes in 3 days** |
| 0.5 | +$325 | +$325 | − | − | − | −$58 | ~0% — passes in 2 days |
| 0.75 | +$488 | +$488 | − | − | − | −$86 | Low risk |
| 1.0 | +$650 | − | − | − | − | −$115 | Low risk (passes day 1) |

## Recommended: 0.3 lot

**Why 0.3 lot:**
- $195/day → $975 in 5 days (2x the $500 target)
- Worst trade: −$35 (safe within $250 daily limit)
- Max 3 consecutive losses = −$105 (42% of daily limit)
- Even a worst-case scenario (3 losses + no wins) = −$105, well under −$250
- Passes challenge in 3 days with room for bad days

**Why not lower (0.2 lot):**
- $130/day → $650 in 5 days — barely over $500
- A single zero-trade day (weekend close, technical issue) could push to day 6

**Why not higher (0.5+ lot):**
- Still very safe given sample evidence, but no need — the challenge only requires $500
- Higher lot increases slippage impact on Fill-Or-Kill orders
- FundedNext may flag unusually high profits as suspicious

## Edge Robustness

The Dark Consensus validated properties support this recommendation:
- **0% Monte Carlo failure** (2000 trials) — strategy survives worst shuffles
- **Breakeven at 3.5x spread** — current spread is 1x, so 250% margin of safety
- **18 months all positive** — no losing months in either OOS or IS periods
- **All regimes positive** — every session, direction, volatility level, and day of week

## Remaining Risks

1. **Live feed parity not tested** — the strategy has never been deployed on a
   live MT5 terminal. Signal parity between Dukascopy archive and live MT5 feed
   is unvalidated. The validation package rates this 1/10.

2. **34-day sample on FundedNext** — the specific FundedNext data tested covers
   34 days. While consistent with 18 months of Dukascopy validation, the
   FundedNext feed itself could have idiosyncratic behavior.

3. **Execution precision** — the 3-minute hold requires precise timing.
   The validation shows ~$2/trade slippage cost from 60s latency, but
   systematic execution bias (always adverse slippage) could degrade results.

4. **Account restrictions** — FundedNext Stellar Lite has per-instrument
   position limits and may restrict EA/API trading.

## Recommendation

Proceed with paper trading on FundedNext demo for **30 days** (or 1,000 signals)
following the plan in the Dark Consensus validation package. Monitor signal
match rate and fill quality. If signal match rate > 90% and live PnL within
80% of model-predicted PnL, deploy at 0.3 lot on the Stellar Lite account.
