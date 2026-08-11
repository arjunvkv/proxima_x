# NOVA — pluggable vectorized engine (live + backtest), climate-gated

Status: P1 BUILT + VERIFIED (81dc8fa, 2026-08-11). P2-P4 pending owner go.

## 1. Requirements (user statement, 2026-08-11)

1. A SEPARATE engine, pluggable into the existing live-aligned engine (Proxima book).
2. Live AND backtest from ONE code path (apples-to-apples is the standing contract).
3. Performant enough to compute ALL climate factors + several strategies WITHIN
   SECONDS (targets in §8).

## 2. Non-negotiable constraints

- `proxima_ops/backtest/` stays BYTE-IDENTICAL. NOVA is new code only
  (`proxima_ops/nova/`), zero diffs in the legacy engine. The parity gates
  (`verify_engine_parity_tokyo.py`, `verify_live_backtest_parity.py`) must stay green.
- Trade-dict contract: NOVA emits the SAME trade shape as `run_strategy`
  (entry_ts/exit_ts/symbol/side/pnl_pts/gross_usd/commission/net/...), so the live
  worker, battery, journal and alignment score consume it with zero changes.
- Fill contract: byte-identical semantics to `simulate_exit` — next-bar open fill,
  stop-first SL/TP, hold-bars exit, `BIG=(1e9,1e9)` hold mode. Proven by a parity
  harness (§10), not by assertion.
- Costs: same broker truth — measured tick values (corrected maps), measured
  spreads, $3.0/lot commission, volume-per-spec.

## 3. Architecture

```
        bars cache (audit_7_eas/market/*.pqt)      live MT5 (copy_rates_from_pos)
                  |                                         |
                  +-----------------+-----------------------+
                                    v
                        FEED LAYER (normalize, epoch, gaps)
                                    v
        FACTOR ENGINE — one vectorized pass over (bars x symbols)
        per-symbol factors: ret(1..96h), ATR, realized vol, EWMA, z,
          range, session, day-part, spread state, rollover state
        cross-asset factors: USD basket strength, DXY-implied residual,
          cross correlations (gold-haven, PC1 dominance), risk composite,
          cross-asset stress (15m/1h/4h), vol-acceleration
                                    v
                     FACTOR TABLES (numpy float32, bars x syms)
                                    v
        STRATEGY INTERPRETER  — spec JSON (same format as engine) -> masks
        + vectorized fills (searchsorted SL/TP/hold) -> trade dicts
                                    v
        CLIMATE LAYER — envelope calibration (tape percentiles) ->
        per-bar continuous size multiplier + hard pause
                                    v
        BOOK AGGREGATOR — multi-strategy nets, FTMO checks
        (daily-loss throttle, maxDD, consistency) -> trade dicts out
                                    v
        CONSUMERS: live worker (per-leg), battery, triage, journal, alignment
```

Plugs into the existing worker as an ADDITIVE mode: legacy legs keep calling the
old engine; NOVA legs call NOVA; both emit compatible dicts.

## 4. Data model

- One `bars` matrix per symbol: time:Int64, OHLC float32 (57.6k rows for 200d M5).
- Factor tables: `F[name]` = 2D numpy float32 `(n_bars, n_symbols)` — one column
  per symbol, one row per bar. Cross-sectional ops (rank, top_n, z, percentile)
  are column ops per row — vectorized across symbols.
- Cross-asset block: per-bar scalars (USD basket, DXY residual, PC1, stress) as
  1D arrays indexed by bar.
- Memory: 57.6k × 35 × float32 ≈ 8 MB per factor; ~25 factors ≈ 200 MB peak.
  Fine on this box; shrink to float32 + drop rarely-used factors if needed.

## 5. Factor list (converged from GPT + internet + DRIVERS_RESEARCH)

Per-symbol: ret k-bar (k=1,3,6,12,24,48,96), ATR(14), realized vol (20d/1d),
vol percentile (rolling), vol acceleration (ATR velocity, sigma), trend strength
(directional efficiency / MA slope), range exhaustion (dist from rolling mean /
session VWAP), spread percentile (live; from journal CSVs in backtest), session
label, rollover state (swap applied), gap size (server-00:00 open − prior close).
Cross-asset: USD basket strength (5-major composite, NOT raw DXY), DXY-implied
residual (DXY vs EURUSD-implied), gold-haven correlation state, PC1 share of the
FX cross-section (correlation stress), risk composite (index/DXY/gold signs,
continuous −1..+1), cross-asset stress (15m/1h/4h returns across gold, USD
basket, US500, BTC), liquidity state (tick frequency, spread stability — live).

## 6. Strategy interpreter

- Spec: SAME `StrategySpec` JSON (feed/rule/universe/sessions/params) + new
  optional `climate` field (allowed envelope, size multipliers).
- Signal: any existing rule (session_exhaustion, session_reversion, big_move_fade,
  break-gap, cross_momentum...) as a vectorized score column; top_n/session pick
  via per-row argpartition (NOT per-trade loops).
- Fills: next-bar-open entry; exit = stop-first SL/TP hit via `searchsorted` on
  the bar's high/low series, else hold-bars exit. Emits trade dicts.
- Cost path: reuse corrected tick/spread maps + commission — same numbers as the
  legacy engine (that's what makes the parity test pass).

## 7. Climate layer

- Envelope: from the tape, per factor the observed percentile band (e.g. gold
  M5-ATR p20–p95, spread p0–p80, session set). Stored per strategy in its spec.
- Multiplier: continuous — inside envelope 1.0, mild outside → linear ramp to
  0.5, far outside → 0 (pause). No hard mid-thresholds (turnover/cutoff risk).
- Shocks: vol acceleration > +3σ or correlation break → immediate −50% (reactive
  within 1–2 bars; transition detection, not level).
- FTMO throttle ABOVE climate: day P&L −2% → half, −3.5% → stop (Layer 1
  account survival > Layer 2 climate > Layer 3 strategy).

## 8. Performance budget (honest targets, Windows box, numpy/polars)

- Cold full recompute, 35 syms × 57.6k bars, ~25 factors: **1–3 s**
- Single strategy eval (masks + fills + costs): **< 50 ms**
- Full book (8 strategies) + climate multipliers: **< 1 s**
- Live incremental update per poll (new bar per symbol, rolling sums + EWMA,
  no full recompute): **< 30 ms**
- Cold recompute every N bars is never needed — live mode maintains state.

## 9. Live/backtest symmetry

Same factor pipeline, two drivers:
- Backtest: batch compute over the tape.
- Live: incremental — running sums (returns, ATR, EWMA vol, correlation block
  via exponentially-weighted updates); each new bar costs O(1) per factor.
- Parity requirement (standing): the live incremental path must reproduce the
  batch path's factor values at every bar within float tolerance (harness
  asserts max |Δ| < 1e-4 on a 1,000-bar segment).

## 10. Plug-in proof (parity harness, the gate to build P1)

`scripts/verify_nova_parity.py` — same tape, same legacy specs, BOTH engines:
- assert identical trade sequences for the legacy rules (session_exhaustion
  Tokyo/cascade/London/usfade configs): same entry_ts, exit_ts, pnl_pts;
- assert identical nets on the book legs at their live lots;
- assert the two existing parity gates still green (legacy untouched).
Only when this passes does NOVA plug in. Numeric drift beyond float tolerance
is a fill-semantics bug, not "close enough".

## 11. Phases

- P1 (DONE 81dc8fa): feed + factor engine + interpreter + parity harness — 6,295 trades / 0 mismatches / 7 configs; 5 strategies in 0.85s vs ~150s legacy (~175x).
  Deliverable: NOVA reproduces the book's backtest numbers on the tape.
- P2: climate layer (envelopes, multipliers, shocks) + FTMO throttle.
- P3: live incremental mode + worker integration (additive legs, journal).
- P4: triage/battery on NOVA (the 20-candidate batch drops to seconds) + new
  rules (DXY residual, triangulation, commodity momentum from DRIVERS_RESEARCH).

## 12. Risks

- Numerical drift vs legacy fills → parity harness is the gate; fix semantics,
  never relax tolerance.
- Memory (~200 MB factor tables) → float32, per-symbol chunking if needed.
- Live cache invalidation → single writer per symbol keyed on bar time; the
  M5-rollover stale-refetch guard carries over.
- Overfitting climate gates → ablation test every factor (removing it must
  degrade; +5% real, +70% suspicious) — the essay/GPT rule, enforced in P2.
