# Absorption → Price-Impact Transition — Final Research Verdict

**Date:** 2026-08-10 · **Status:** REJECTED (hypothesis A/B both null)
**Commits:** phase-1 bar study `345b8cb` · phase-2 tick study `d5abdc1` · prior
microstructure probe lineage `62eda68/b38d5eb/326754e` (commit `326754e` carries
the LAST-flag last=0 finding)
**Footprint:** engine (`proxima_ops/`), live worker (`scripts/run_core_book_live.py`),
book legs — all byte-identical (empty `git diff` verified before each commit).

---

## 1. Hypothesis under test

When directional flow is absorbed by liquidity, price moves less than expected
("absorption"); when absorption capacity deteriorates, small additional flow
produces disproportionately large move ("impact transition"). Institutional
question: is the POST-transition direction mechanically predictable — and does
the transition survive real FTMO/FundedNext costs?

Two competing directional hypotheses were tested:
- **A (reversal):** after an absorbed-flow episode, price reverts against the
  absorbed direction (absorption = exhausted demand).
- **B (continuation):** the transition fires WITH the absorbed direction
  (absorption = latent fuel finally moving price).

## 2. Data reality on the FTMO terminal (verified 2026-08-10)

| Item | Finding |
|---|---|
| Tick availability | 60 days × 7 FX majors (EURUSD, GBPUSD, USDJPY, EURJPY, GBPJPY, AUDUSD, USDCAD) ≈ 6.9M ticks; 70–92% carry the LAST flag |
| Trade price | **`last` = 0.0 on every tick** — flag is set, value is not populated. Tick-rule aggressor inference impossible |
| Volume | `volume`/`volume_real` = 0 everywhere |
| BUY/SELL bits | never set |
| Honest flow measure | only the **one-sided quote rule** (ask-lift alone = +1 buy pressure, bid-drop alone = −1), kept as an explicit pressure *proxy* — never claimed to be transaction flow |

## 3. Evidence chain (all causal, no lookahead)

1. **M5 bar level, ~200 days, 23 symbols** (`measures.py`/`study.py`):
   6 W/T configs × 4 horizons, hour-boundary same-day-guarded signals, honest
   random-position null (1000 iters). Gated ≈ raw ≈ 0, t ~ 0, per-symbol and
   per-session-block flat. Sanity: z_act 1.1–3.0σ, impact ratios 5–60× → the
   absorption→transition *profile exists*; the null is a genuine finding.
2. **Tick level, 60 days, 7 majors, native minute resolution** (`tick_study.py`):
   - **T1 (price-only):** 6 configs × 4 horizons — all |z| ≤ 3.4, sign flips
     across horizons within a config, per-symbol ≈ 0 except JPY-cross noise.
   - **T2 (flow-aware, one-sided quote pressure + Kyle-λ):** pooled means go
     NEGATIVE (reversal) as W grows (z to −7.0 at W=48 T=6 H=30) — but
     per-symbol it is **only GBPJPY** (−0.03…−0.14 price units) plus USDJPY
     with n = 1–4; EURUSD/USDCAD/AUDUSD ≈ 0.0000, EURJPY n = 0. The pooled
     negative z is a price-scale pooling artifact of the JPY crosses.
   - Raw (ungated) ≈ 0 everywhere in both variants: the base rate has no
     predictability, so any gated "edge" would have to survive the transition
     gate itself — it does not.
3. **Window-matched M5 (same 60 days):** gated means |…| < 0.014 on 4
   horizons, mixed signs → consistent with the 200-day null; the tick result
   is not a resolution artifact.

## 4. Cost & survivability context (current official rules, verified 2026-08-10)

Even granting the largest surviving effect (GBPJPY ≈ −0.05 price ≈ 5 JPY pips
over 1–4 h in the flow-gated subsample, n ≈ 11–17), round-trip costs on these
products — 12 pt spread (EURJPY stress map) + $6/lot-equivalent commission +
slippage — exceed the signal before FTMO's 5% daily / 10% static max loss or
FundedNext's 3–5% daily / 6–10% static limits are even considered. The
survivability battery (risk grid, buffers, Monte Carlo) is therefore moot:
there is **no strategy to survive**. Rules reference (FTMO vs FundedNext,
15-field, primary sources) archived in full in the delegation transcript
(2026-08-10) and summarized in §6.

## 5. Verdict

- The absorption→impact transition **is measurable** — extreme flow-absorption
  states exist (2–3% of windows), impacts are 5–60× the absorbed-flow
  expectation.
- It **does not predict direction** at any resolvable granularity on this feed
  (200-day bars, 60-day minutes, with or without the quote-pressure proxy).
- The only non-zero reading (GBPJPY reversal under the flow gate) is i) a
  proxy-based artifact concentrated in one cross, ii) opposite the
  continuation hypothesis, iii) below costs, iv) not robust per-symbol.
- Per the research contract: the **grid gate, full battery, and survivability
  analysis are not run** — their entry condition (a surviving edge in the tick
  study) was not met. §33-style mechanical spec is inapplicable; §34 mechanism
  explanation: absorption-driven impact is real at the level of *impact
  magnitude* (Kyle-λ dynamics) but the *sign* of post-transition drift is
  dominated by unobservable book depth and competing flow — the retail-visible
  footprint carries no exploitable directional information through this
  feed's resolution.

## 6. FTMO / FundedNext current rules — key facts (verified 2026-08-10, primary sources)

| Rule | FTMO 2-Step (Proxima's model) | FundedNext Stellar 1-Step / 2-Step |
|---|---|---|
| Daily loss | 5% — anchor = midnight balance − 5%×Initial; intraday profit does NOT expand room | 3% / 5% — limit = Initial×% + intraday profit (profit DOES expand room) |
| Max loss | 10% STATIC (initial-based) | 6% / 10% STATIC (initial-based) |
| Leverage | 1:100 (Swing 1:30) | 1:100 |
| Min days | 4 per phase | 2 / 5 per phase |
| News | allowed in eval; funded Standard ±2 min lockout (incl. SL/TP) | always allowed; funded credits only 40% of ±5-min high-impact profit |
| Weekend | funded Standard must be flat | allowed everywhere |
| Commission | not published (in-platform) | Instant $7/lot FX open-only; challenge FX unverified (image-only) |
| Fee | refundable with first reward (2-Step) | refundable with first payout −$25 platform fee |

Full 15-field report + sources: delegation summary
`subagent-summary-0-20260810_163008_579273.txt`.

## 7. Reusable operational traps (saved to memory/skill)

1. `copy_ticks_range` with **millisecond** timestamps throws `SystemError`;
   **seconds** work.
2. FTMO terminal history service degrades after repeated API attach/detach
   cycles (hangs/empty returns) — restart terminal64.exe, then ONE process,
   ONE attach, pull all symbols sequentially.
3. `last` price is 0 even when the LAST flag bit is set — check raw values
   before trusting any tick-rule construction.