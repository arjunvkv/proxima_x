# VERDICT — fixing-window hunt: REJECTED (no engine-expressible edge)

**Date:** 2026-08-11 (Aug) · **Window:** 2026-06-11 → 2026-08-10, 43 trading days
**Data:** FTMO M1 cache (7 majors × 60d, server clock = UTC+2) + audit M5 cache
**Scripts:** `scripts/_absorb/{fix_study,fix_gate,fix_gate2,probe_fix_signs,probe_fix_sides,probe_fix_lodo}.py`
**Results JSON:** `results/{fix_study,fix_gate,fix_gate2,probe_fix_signs,probe_fix_sides,probe_fix_lodo}.json`
**Commits:** `db58cfa` (study + initial finding) → `…` (this verdict)

## Pre-registered hypotheses (fix_study.py)
- Fix events: WMR London 16:00 BST = 17:00 server, ECB 14:15 CET = 14:15 server,
  Tokyo 09:55 JST = 02:55 server (server-clock verified from activity profile).
- T0: unconditional post-fix return vs random-minute null.
- T1: sign(pre-fix 30-min move) × post-fix return (continuation + / reversal −).
- T2: T1 on |pre-move| ≥ pair q75 days only. Entry F+5 open, holds 5–60 min.

## Stage 1 — A/B study: signal APPEARED
- Tokyo F=3 / WMR F=17 T1/T2 continuation, z = +3.8…+4.7 vs 500-iter null,
  LODO sign-flips 0/42 (the absorption two-freak-day trap did NOT apply).
- Fixed-hour specificity: true fix hour ≠ neighbors (F=2/F=4, F=16/F=18 null).
- **This is the trap that mattered instead: pooled-scale + side-selection.**

## Stage 2 — per-side decomposition (probe_fix_signs, probe_fix_sides)
- The pooled sign×post z is short-side-dominated: on down-move days JPY crosses
  continue DOWN −26…−41 pips (T2), on up-move days only +1…+7 pips.
- T2 per-side cells are **n = 4–7** (top-quartile days split by side). LODO at
  n=5 is statistically meaningless; per-day swings ±10–50 pips, some cells flip.
- Long side (the only side the legacy engine path can trade): +0.7…+7.4 pips
  T1 at n≈20, concentrated in JPY crosses; thin vs costs.

## Stage 3 — engine embodiment (fix_gate / fix_gate2, BOTH tapes)
- Embodiment: engine `session_momentum` (legacy byte-parity path), lookback 6
  M5, sessions=[F], fill_bar=1 → entry F+5 open, stop-first + 35/45-pip
  jpy SL/TP, commission 3.0/lot, spread maps (typical vs measured FTMO-demo).
- RAW edge (no costs, M1-resampled M5): **−0.6 … +3.8 pips/trade** across all
  cells; positive cells cluster +0.7…+3.8 pips (wmr F17 H60 best:
  USDJPY +3.79, EURJPY +2.76, GBPJPY +0.23).
- Costs: typical JPY-cross round trip 1.2/2.2/3.0 pips + commission; measured
  (worst-case) 3.5/5.7/7.1 pips. **Net USD negative in every cell on both
  tapes** (e.g. tokyo F3 H60: net −279 typical / −671 measured over n=126).
- Best cell (wmr F17 H60) ≈ coin flip at typical costs, dead at measured.

## Verdict: REJECT
1. No engine-expressible edge after costs — the live engine and book stay
   byte-identical; book stays at 4 legs. Session-scale fix-hour continuation
   is a **known, crowded** family (post-2016 fix manipulation case), and on
   this feed it does not clear FTMO JPY-cross spreads.
2. The headline study z was a mixed-scale pooling artifact: the pool mixes
   USDJPY/EURJPY/GBPJPY units with EURUSD-scale units, and the sign×post
   mean was carried by short-side small samples unshippable in legacy path.
3. Short-side continuation (−26…−41 pips T2) is (a) n=4–7 → not credible,
   and (b) NOT expressible in the legacy engine (BUY-only; adding a short
   rule = engine mutation = forbidden under the research contract).
4. Walk-forward/OOS, survivability grid, Jaccard-vs-leg-4 NOT run: the gate
   failed at zero cost on two independent tapes; nothing to protect. Note:
   the rule tested IS the book's session_momentum rule (sessions=[3]/[17]),
   so any residue would be a variant of leg #4, not a new mechanism.

## Banked assets
- Clock verification: cache hours = server = UTC+2 (empirically pinned via
  activity profile: London-open step-up 08→09h, daily break at 00h = 22:00 UTC,
  peak 15–17h = London/NY overlap).
- Engine-expressibility is now part of the funnel: grid-gate *before* trusting
  an A/B z — session_momentum legacy path = BUY-only + stop-first + fill-next-
  open; a pooled-sign A/B that ignores side asymmetry and cost tables can
  report phantom z.
- Per-side + per-pair-pips + n-count disclosure is the new minimum for any
  cross-pair pooled result (both hunts now bitten by pooling).