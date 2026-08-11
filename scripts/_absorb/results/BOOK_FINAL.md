# BOOK_FINAL — Core Book v2 (tokyo removed, re-evaluated)

**One-line meaning:** Tokyo is out — the live-spread probe proved its edge lives
exactly where the rollover spread tax does, so it loses money in every hour
slot. The book is now 3 proven FX legs + S3 gold (pending your OK), all inside
the FTMO risk budget.

Account profile: **$25k FTMO-funded → $1,250 daily / $2,500 maxDD guards.**
All evidence: NOVA engine, parity-verified vs legacy (0 mismatches, incl. gold).

---

## ⚠️ Why Tokyo was removed (the verification you reminded me of)

Hour-shift probe on **live daemon-measured spreads** (`core_book_spreads.csv`,
2026-08-10 session — @session:default/20260810_151323_f1979781):

| fire hour | edge (no spread) | net with live spreads | spread tax |
|---|---|---|---|
| **0** (old tokyo) | +$22,883 | **−$16,630** | **$39,514** |
| 1 | −$242 | −$15,346 | $15,105 |
| 2 | −$117 | −$1,617 | $1,500 |
| 3 | −$6,118 | −$8,409 | $2,290 |

The "buy the overnight losers at rollover" edge is real but exists **only at
server hour 0 — the exact minute the market reopens**, where spreads run
30–60 pips. Fire at 00:05 → tax > edge. Fire later → nothing to capture.
**No hour slot is viable.** A spread guard would skip every hour-0 trade and
the other hours have no edge to guard for.

**Why v1 of this book kept it:** the backtest cost model (busy-typical spread)
does NOT capture the hour-0 rollover blowout — tokyo looked like +$23.4k.
This is exactly the "measure the fire hour's live spreads before shipping any
window" gate (now a required onboarding step).

---

## Tier NOW — ships on the current worker (3 FX legs + gold pending OK)

| leg | rule | sessions | lb | top_n | hold | universe | lot | net (151d) | worstDay | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| cascade | session_exhaustion | [2,3,4] | 1440 | 8 | 24 | FX18 | 0.14 | $3,044 | $381 | already live |
| london | session_exhaustion | [7,8,9] | 1440 | 5 | 12 | FX18 | 0.23 | $2,958 | $255 | already live |
| usfade | session_exhaustion | [14-19] | 50 | 5 | 24 | FX18 | 0.45 | $14,495 | $581 | already live |
| **gold_s3** | session_exhaustion | all | 50 | 3 | 12 | XAUUSD+XAGUSD | **0.15** | **$120,739** | $318 | **NEW — needs your OK** |

**Combined (4 legs):** net **$141,236** · worstDay **$727 (58% of $1,250)** ·
maxDD **$751 (30% of $2,500)** · green days 88.1% · max loss streak 2 days ·
cost-stable (1.5× spread → net $135,227, worstDay $768, maxDD $833).
Worst combined day = all four legs red (gold −$303, usfade −$294, cascade
−$101, london −$29). NOVA runtime: **0.24s**.

**Sanity vs the alternative:** keeping tokyo at its *verified live* economics
(−$16,630 @0.52 lots) would drag the book to ≈ **$124.6k net with a bigger
tail** — and it would be paying the worst spread tax in the book for a
negative edge. Removing it is strictly better.

### gold_s3 — the new leg (pending your approval)
- **Spec:** session_exhaustion, XAUUSD+XAGUSD, sessions all, top_n 3 (degenerate
  on 2 metals = buy both on exhaustion bars), hold 12, no stops, 0.15 lots.
- **Mechanism:** exhaustion-reversion intraday on metals; entry hours 2-3 (Asia)
  and 16-18 (NY afternoon) server — NOT the rollover hour, so no tokyo-style
  spread trap (still gated by the fire-hour spread check before live).
- **Re-certified this session:** n=300, exp **$2,683/lot** (R3 certificate said
  $1,296 — older-map era; current pipeline verified NOVA==legacy 0 mismatches,
  gold costs $45/lot RT ≈ real 47-pt spread). Flag: long-only (bear-regime
  risk) — survived gold's −4% sample downtrend.
- **What shipping requires:** (1) your OK (spec-only heritage), (2) worker gold
  branch — fetch XAUUSD/XAGUSD bars + gold SL/TP resolution (BIG, hold-only),
  (3) fire-hour live-spread gate (onboarding requirement), (4) cost maps live
  research-side already.

---

## Tier NEXT — validated but cannot run on the current worker

The worker fires **legacy session_rank only**. Signed-rule and daily-cadence
strategies need a worker extension (per-hour |score| rank with causal
semantics) or NOVA live mode (P3). Nothing ships until that path exists AND
each candidate passes the causal-replay gate (R3-3b).

| leg | rule | evidence | lot @25k (30% budget) | blocker |
|---|---|---|---|---|
| big_move_fade_fx | big_move_fade (signed) | exp $49/lot PF 5.3, plateau 9/9, FTMO sim 4.65 lots @100k | 0.35 | worker signed path + causal gate |
| big_move_btc | big_move_fade (signed) | PF 3.2, sim 2.1 lots @100k (BTC=73% of net caveat) | 0.16 | same |
| s4_dxy_divergence | dxy-implied divergence (new) | exp $370/lot PF 11.8, LODO 33/33, 7/7 months, plateau 9/9, MC 0% breach | 0.14 | DAILY cadence → NOVA P3 |
| s1_btc_usd_regime | usd-regime fade (new) | exp $38/lot PF 1.16 | — | new signal rule = engine addition |
| s2_index_break_gap | break-gap follow (new) | exp $52/lot PF 1.82, 4/4 indices | — | new signal rule = engine addition |

---

## Honest notes
1. **S3 certificate discrepancy disclosed:** R3 said $1,296/lot; current
   verified pipeline says $2,683/lot. NOVA==legacy byte-parity (incl. gold)
   and the cost math (point_size×tick compensation = oz/lot exactly, $45/lot
   RT spread ≈ real 47pt) back the current number.
2. **Backtest vs live lesson re-verified:** the busy-typical spread model is
   fine for hours 2–19 (cascade/london/usfade/gold windows) but misses the
   hour-0 rollover blowout. Fire-hour spread envelope is now an onboarding
   gate for every window.
3. **Unseen-regime risk stands for everything:** 7-month sample only.
4. **VPS deploy delta:** remove the tokyo entry from `STRATS` in
   `scripts/run_core_book_live.py` + daemon restart (next tokyo window is
   21:05 UTC — do not let it fire). VPS was unreachable at book-build time;
   deploy when reachable + on your OK.

## Files
- `scripts/book_final.json` — machine-readable book (this spec)
- `scripts/book_final_battery.py` — NOVA combined-book check (0.24s runtime)
- `scripts/book_final_combined.json` — per-leg + combined evidence
