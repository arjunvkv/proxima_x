# BOOK_FINAL — Core Book v1 (updated with validated strategies)

**One-line meaning:** The live book stays the proven 4 FX legs, gains ONE new
config-only leg (S3 gold, needs your OK), and parks everything else behind the
worker's signed-rule path — because the current live worker can only fire
legacy rank rules.

Account profile: **$25k FTMO-funded → $1,250 daily / $2,500 maxDD guards.**
All evidence: NOVA engine, parity-verified vs legacy (0 mismatches, incl. gold).

---

## Tier NOW — ships on the current worker

| leg | rule | sessions | lb | top_n | hold | universe | lot | net (151d) | worstDay | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| tokyo | session_exhaustion | [0] | 6 | 3 | 12 | FX18 | 0.52 | $23,380 | $283 | already live |
| cascade | session_exhaustion | [2,3,4] | 1440 | 8 | 24 | FX18 | 0.14 | $3,044 | $381 | already live |
| london | session_exhaustion | [7,8,9] | 1440 | 5 | 12 | FX18 | 0.23 | $2,958 | $255 | already live |
| usfade | session_exhaustion | [14-19] | 50 | 5 | 24 | FX18 | 0.45 | $14,495 | $581 | already live |
| **gold_s3** | session_exhaustion | all | 50 | 3 | 12 | XAUUSD+XAGUSD | **0.15** | **$120,739** | $318 | **NEW — needs your OK** |

**Combined (5 legs):** net **$164,615** · worstDay **$465 (37% of $1,250)** ·
maxDD **$465 (19% of $2,500)** · green days 95.4% · max loss streak 1 day ·
cost-stable (1.5× spread → net $161k, worstDay $418, maxDD unchanged).
NOVA runtime for the whole book check: **0.37s**.

### gold_s3 — the new leg (pending your approval)
- **Spec:** session_exhaustion, XAUUSD+XAGUSD, sessions all, top_n 3 (degenerate
  on 2 metals = buy both on exhaustion bars), hold 12, no stops, 0.15 lots.
- **Mechanism:** exhaustion-reversion intraday on metals; entry hours 2-3 (Asia)
  and 16-18 (NY afternoon) server. Control (plain buy@fixed-hour) LOSES —
  the exhaustion signal does the selection.
- **Re-certified this session:** n=300, exp **$2,683/lot** (R3 certificate said
  $1,296 — older-map era; current pipeline verified NOVA==legacy 0 mismatches,
  gold costs $45/lot RT ≈ real 47-pt spread). Flag: long-only (bear-regime
  risk) — survived gold's −4% sample downtrend.
- **What shipping requires:** (1) your OK (spec-only heritage), (2) worker gold
  branch — fetch XAUUSD/XAGUSD bars + gold SL/TP resolution (BIG, hold-only),
  (3) cost maps already live research-side (costmaps_r3.py).
- **Why 0.15 and not more:** worstDay $318 = 25% of daily limit alone; at 0.15
  combined budget use is 37% daily / 19% maxDD — inside the "ship ~30% of
  limit" sizing rule with the other legs already consuming budget.

---

## Tier NEXT — validated but cannot run on the current worker

The worker fires **legacy session_rank only**. Signed-rule and daily-cadence
strategies need a worker extension (per-hour |score| rank with causal
semantics) or NOVA live mode (P3). Nothing here ships until that path exists
AND each candidate passes the causal-replay gate (R3-3b) — the gate that
already killed the session_reversion family.

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
2. **Early book-battery bug caught:** a missing cumsum produced a phantom
   $11,371 maxDD "breach" — corrected; the real combined maxDD is $465.
3. **Unseen-regime risk stands for everything:** 7-month sample only.
4. **Book's daily budget math:** worst combined day was gold −$318 + usfade
   −$179 + others −$58 + london −$47 + cascade +$20 = −$465 → 37% of limit.
   Two such days in a row stay under the guard; the max loss streak is 1 day.

## Files
- `scripts/book_final.json` — machine-readable book (this spec)
- `scripts/book_final_battery.py` — NOVA combined-book check (0.4s runtime)
- `scripts/book_final_combined.json` — per-leg + combined evidence
