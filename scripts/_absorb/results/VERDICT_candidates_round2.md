# VERDICT — candidate round 2: library audit + TDD — REJECTED (audit), VOL-ONLY (TDD)

**Date:** 2026-08-11 · **Window:** 200d M5 audit cache (audit) + 60d min bars from
genuine FTMO ticks (TDD) · commits: `31e0b4a` + this round.

## 1) Library audit (`lib_audit.py`) — all 20 cells REJECTED
Every engine rule never validated against costs, at mechanism windows:
thin_market_fade, vol_compress_fade, domestic_hours, day_of_week_usd,
carry_clock (US-hours, Tue-Thu), lead_lag, big_move_fade,
intraday_momentum_london, intraday_momentum, fix_reversal (TRUE fix hours
per the clock check: 3/14/17).
- Raw engine-embodied edge ≈ $0-15 per rule-block (~zero points/trade).
- Net at typical + measured spreads: NEGATIVE in 19/20 cells; the only
  non-negative net (carry_clock: +$346 typical / −$1,342 measured, PF 1.04,
  exp $2.0/lot) fails the gate (exp > $15, PF > 1.2) and dies at measured.
- The rule library contains NO hidden 5th leg. The Tokyo-h0 family (already
  in the book) remains the only cost-surviving mechanism; its siblings
  (range_reversion/round_number_bounce/session_reversion at h0) were prior-
  round sweeps and are book-adjacent, not new legs.

## 2) TDD — tick event-rate acceleration (`tdd_study.py`) — VOL CONFIRMED, DIRECTION NULL
RDSA (2026-06-16) gated this direction on tick data; the gate cleared with
the 60d genuine-tick minute bars (n_quotes = quote-event rate). Design:
hour-of-day de-sessioned log-rate z (calibration days 0-29, causal), accel
a = Δz(15 min), tests on HOLDOUT days 30-59 only, per-hour circular-shift
null (200 iters), pips per pair.

- **T1 vol: CONFIRMED.** a ≥ q75 vs ≤ q25 → |fwd| ratio 1.04-1.29; z +2.4-5.5
  at H=5/15 across ALL 7 pairs vs the null; decays to ~1-2σ at H60. Rate
  surges precede short-lived volatility amplification. REAL structure — but
  a volatility fact, not a directional edge; the book's fixed SL/TP exits
  cannot monetize it (informational only; would need vol-scaled sizing, a
  user-level engine decision).
- **T2 direction: NULL.** Signed-accel pips: G10 (+) ≤ +0.41, JPY crosses (−)
  −0.15..−0.96, USDCAD (−) — signs split by pair family, sub-pip magnitudes
  vs 2.2-7.1 pip entry costs, no LODO-dominant unified mechanism. And no
  engine rule expresses event-rate signals (zero-mutation contract) — even a
  real directional finding would be unshippable without a user approval.

## 3) Program-level close-out
RDSA's prescribed conclusion has now materialized across four independent
hunts (absorption bar+tick, fixing-window, library audit, TDD):
**directional prediction is not a reconstructable property of this feed.
The Tokyo-h0 open family is the ONLY cost-surviving directional edge in the
entire rule library + literature map.** Per the RDSA's own exit condition,
remaining research value shifts to execution/risk/capital allocation —
further directional hunts are expected-negative-return on research spend.

Engine/book/live byte-identical. Commit: this round adds lib_audit.py,
tdd_study.py, results/lib_audit.json, results/tdd_study.json.