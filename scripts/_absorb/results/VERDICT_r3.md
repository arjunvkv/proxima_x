# VERDICT_r3 — Three new strategies (round 3 hunt)

Date: 2026-08-11 · Research-only · engine/book/live byte-identical (git diff proxima_ops/ empty)

## Mission
Tokyo family declared not viable by user. Requirement: THREE new best strategies,
found by any means, validated through the honest battery. Data: fresh 200d M5 pull
of 9 new assets from the FTMO terminal (indices, oils, DXY.cash, refresh of
XAU/XAG/BTC/ETH) + live measured spreads + broker tick values.

## The three

### S1 — USD-regime fade on BTC (cross-asset, NEW rule needed)
Signal: DXY 4-bar-hour momentum (48 M5 bars) sign → fade BTC for 3h (36 bars),
entry next-bar open, hold-only (stops hurt: 2xATR SL too tight for 2h holds).
| stat | value |
|---|---|
| n | 36,846 |
| exp | +$38.13/lot (gate $15) |
| PF | 1.16 (diluted across 36k trades; z=+10.0) |
| WF | +$627k / +$777k (both halves positive) |
| LODO | 0/151 days flip |
| months | 7/8 positive, no consecutive negatives |
| stress 1.5x | survives (spread $1/lot, comm $6/lot dominate) |
| sides | long +$21.2, short +$43.5 both positive |
| control | BTC own-fade PF 1.04 z 3.5 — DXY conditioning more than doubles the edge (additive, not proxy) |
Mechanism: USD-regime moves overshoot in crypto; 2-3h mean reversion.
DXY->ETH dead, DXY->XAU same family (below). Weekend exposure: none (DXY has no bars).

### S2 — Index break-gap follow (4 indices pooled, NEW rule needed)
Signal: break gap = open[01:05 server] - close[23:45 prev] (CME settlement break,
server hour 0); gap in TOP tercile -> LONG at reopen, exit session close.
| stat | value |
|---|---|
| n | 154 (39/39/38/38 across US30 US500 GER40 UK100) |
| exp | +$51.66/lot |
| PF | 1.82 |
| WF | +$4,540 / +$3,416 (both positive) |
| LODO | 0/59 days flip |
| symbols | 4/4 positive (all >= +$9.49/lot) |
| z | +2.25 pooled (per-index 1.5-1.6; consistency across 4 independent indices) |
| sides | long-only by construction (top-tercile gaps) |
Fade direction uniformly negative across all 4 indices (z -0.5..-1.8) — the effect is
follow, not fade. NOTE: overnight-gap study v1 used wrong session anchors (23h
instrument) — corrected in v2; v1's z=-14 was an anchor artifact, data verified clean.

### S3 — session_exhaustion GOLD (existing engine rule — ZERO mutation shipping)
Spec: rule=session_exhaustion, universe=[XAUUSD, XAGUSD], sessions=all, top_n=3,
per_day, fill_bar=1, hold 12 bars, no stops (BIG), volume 0.15.
| stat | value |
|---|---|
| n | 300 (150d x 2 symbols) |
| exp | +$1,296/lot |
| PF | 6.23 |
| WF | 2/2 halves positive |
| LODO | 150/150 removals keep total positive |
| months | no 2 consecutive negative; profitable through gold's -4% sample downtrend |
| stress 1.5x | PF 5.6, exp +$1,210 |
| sides | ALL BUY (long-only) — flag: bear-regime risk; but survived the sample downtrend |
| entry hours | 2-3 (Asia) and 16-18 (NY afternoon) server |
| control | plain buy@fixed-hour LOSES (PF 0.73-0.95) -> the exhaustion signal does the selection, not calendar drift |
Mechanism: exhaustion-reversion intraday on metals; degenerate top_n on 2-symbol
universe = long both metals on exhaustion bars. session_reversion GOLD (two-sided)
also passes: PF 3.85, exp $1,011/lot, both sides positive — fallback if long-only is
unacceptable.

## Killed this round (honest graveyard)
- DXY->XAU/US500->XAU fades: same-bar-entry artifact (z 8-10 A/B) -> DEAD under
  exact next-bar execution (exp -$14.78/lot with stops). Hold-only survives weakly
  (PF 1.06-1.11) but May-Jun 2026 negative -> not "best".
- LBMA gold/silver fix windows: n=149, z<=1.94 — not significant (FX-fix audit pattern holds).
- Gold/silver ratio z-score reversion: direction right, PF 0.67-0.95 — economically dead.
- Index first-hour momentum (NY open): z -0.5 — dead.
- cross_momentum n_best (any universe/window/per_hour): extreme-move momentum loses
  (PF 0.18-0.79) — engine's top-2/day selection can't express every-bar momentum (ETH z+66 standalone).
- ETH DXY-fade: dead. ETH own-momentum: real standalone (z 66) but not engine-expressible.

## Engine implications
- S3 ships as spec-only + cost-map entries (XAUUSD/XAGUSD tick/spread) — the maps live
  in engine-side files: shipping = a deliberate, user-approved engine file edit.
- S1/S2 need NEW signal rules (cross-asset momentum->target; break-gap) — engine
  additions. Research kept byte-parity; shipping is a separate approval.
- Research tooling found: engine point_size() = 1e-5 for non-JPY symbols — USD
  inflated 100-1000x for new assets; corrected maps in costmaps_r3.py (compensation
  via tick-value scaling, zero engine changes). Engine simulate_exit ignores exit.mode
  and always applies FX-calibrated SL/TP — 1e9-stop trick used for honest hold runs.
