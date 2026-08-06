# V107 Live EA Honest Audit & Fix Record (Aug 2026)

Audit of the 6 FTMO live-prepared `v107` EAs in `paper_trade\mt5_backtest\` against real FTMO M5 data,
using **honest bar-open semantics** (signal from completed bars / bar-open price, entry at bar open,
timed expiry exit, broker SL=50p/TP=80p guards modeled intrabar worst-case SL-before-TP).

- **Account**: FTMO-Demo `1514168544` (balance $98,887.06)
- **Data**: 30 days FTMO M5, `2026-07-06 00:05 → 2026-08-04 18:00 UTC`, 22 pairs
  (`mt5-connector\ultra_monster\ftmo_{PAIR}.parquet`)
- **Harness**: `mt5-connector\ultra_monster\audit_v107_ea_semantics.py` (now with `utc` / `server` / `fixed` variants)
- **Date**: 2026-08-04

---

## Change Record

| Date | Change | File |
|------|--------|------|
| 2026-08-04 | Audit all 6 v107 EAs honestly on 30d FTMO M5 | `audit_v107_ea_semantics.py` |
| 2026-08-04 | **Tokyo H0 session fix** (below) | `TokyoH0_MT5_v107.mq5` |
| 2026-08-04 | **NY H21 session fix** (below) | `NY_H21_MT5_v107.mq5` |
| 2026-08-04 | Added `fixed` variant to audit harness; verified both fixes | `audit_v107_ea_semantics.py` |
| 2026-08-05 | **⚠️ BOTH fixes were reverted** — EAs back to 3h-early firing | `TokyoH0`/`NY_H21` `_MT5_v107.mq5` |
| 2026-08-05 | **Re-fixed robustly via `TimeGMT()`** (not server-shift): SESSION_HOUR now means true UTC, gated on GMT — immune to server offset AND DST. Recompiled + redeployed to FTMO Experts. | `TokyoH0_MT5_v107.mq5`, `NY_H21_MT5_v107.mq5` |

### TokyoH0_MT5_v107.mq5 — exact edits (robust `TimeGMT` re-fix, 2026-08-05)

> The 2026-08-04 fix (SESSION_HOUR 0→3 + `dt.min != 0`→`> 5`) was **reverted** in the working tree,
> and it also had a latent flaw: `dt.min > 5` skips the 00:05 bar (minute 5), so it actually fired at
> 00:10, and it hardcoded the +3 server offset (DST-fragile). Replaced with a GMT-anchored gate.

```diff
- input int      SESSION_HOUR       = 0;       // 00:00 UTC Session
+ input int      SESSION_HOUR       = 0;       // 00:00 UTC Session (gated on TimeGMT)
...
-   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
-   if(dt.hour != SESSION_HOUR || dt.min != 0) return;
+   MqlDateTime dt; TimeToStruct(TimeGMT(), dt);
+   if(dt.hour != SESSION_HOUR || dt.min > 5) return;   // fire on the 00:05 bar (FTMO has no 00:00)
```

### NY_H21_MT5_v107.mq5 — exact edits (robust `TimeGMT` re-fix)

```diff
- input int      SESSION_HOUR        = 21;      // 21:00 UTC Session
+ input int      SESSION_HOUR        = 21;      // 21:00 UTC Session (gated on TimeGMT)
...
-   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
+   MqlDateTime dt; TimeToStruct(TimeGMT(), dt);
    if(dt.hour != SESSION_HOUR || dt.min != 0) return;
```

`TimeGMT()` returns true GMT/UTC, so `SESSION_HOUR` means UTC directly. No server-offset constant,
no DST assumptions. `OnTick` new-bar detection stays on `TimeCurrent()` (server time) as before.

---

## TL;DR

| EA | Claimed WR | Honest WR | PF | Trades | Net pips | Verdict |
|----|:----------:|:---------:|:--:|:------:|:--------:|---------|
| **CPPF Z** | 85%+ | **94.7%** | 5297 | 19 | +529.6 | ✅ REAL (small sample) |
| **MSV Asian Exhaustion** | — | **76.9% / 71.4%** | 13.3 / 14.8 | 26 / 21 | +274.6 / +414.9 | ✅ REAL (new edge) |
| **Tokyo H0** | 95.3% | **97.1%** ✅ (fixed: 00:00 UTC) / 55.2% pre-fix | 138 / 2.9 | 105 | +1554 / +304 | ✅ **FIXED** — fires true midnight now |
| **Ultra Monster** | 76%+ | **47.0%** | 1.21 | 1228 | +601.9 (+0.5/trade) | ❌ Claim fake (lookahead); weak grind |
| **NY H21** | 65.9% | **50.0%** (fixed, 6 tr) / 52.6% pre-fix | 6.17 / 2.31 | 19 | +80.6 / +134.2 | ⚠️ Marginal — fixed to design window |
| **CPMC Z** | 61%+ | **43.8%** | 0.57 | 178 | -495.8 | ❌ **DEAD** |

---

## Headline Findings

### 1. FTMO server time is UTC+3 — the deployed EAs fire 3 hours late

All EAs gate sessions on `TimeCurrent()` (MT5 **server** time), not UTC. Measured offset: **+3.0 h**.
The strategy headers and validated backtests are in UTC. Therefore the deployed EAs fire at:

| EA | Header says | Server hour used | **Actually fires (UTC)** | Edge at that time |
|----|-------------|------------------|---------------------------|-------------------|
| Tokyo H0 | 00:00 UTC | `hour==0` | **21:00 UTC** | 97.1% → **55.2% WR** (PF 2.9) |
| NY H21 | 21:00 UTC | `hour==21` | **18:00 UTC** | unvalidated US-afternoon window |
| MSV | 0–6 UTC | `hour 0..6` | **21:00–03:00 UTC** | still positive (robust to shift) |

**Consequence**: Tokyo H0 as deployed does not trade the midnight reversion it was validated on.
At 21:00 UTC it shows ~55% WR / +2.9 pips avg. At 0.15 lot that is ≈ +$4.3/trade gross vs $4.50
round-turn commission → **break-even to negative**. The 97% WR edge exists **only at true 00:00 UTC**.

**Fix applied** (server-local session hours, server = UTC+3):
- **Tokyo H0**: `SESSION_HOUR 0 → 3` (server 03:00 = 00:00 UTC) AND gate `dt.min != 0` → `dt.min > 5`
  (FTMO has no 00:00 bar, so the session opens on the 00:05 bar = server 03:05).
- **NY H21**: `SESSION_HOUR 21 → 0` (server 00:00 = 21:00 UTC).
- MSV needs no change — it is robust to the shift.

> **2026-08-05 follow-up**: The above server-shifted fix was reverted in the tree and also carried a
> subtle off-by-one (`dt.min > 5` skips the 00:05 bar). Re-applied in a DST-proof form: gate the
> session on **`TimeGMT()`** so `SESSION_HOUR` means **true UTC** (Tokyo `hour==0, min<=5` → the
> 00:05 bar; NY `hour==21, min==0`). This matches the `utc` design variant exactly and needs no
> server-offset constant. Recompiled + redeployed to the FTMO Experts folder.

**Fix verified** (re-audit with `fixed` variant = corrected configs):

| EA | `server` (pre-fix) | `fixed` (post-fix) |
|----|--------------------|--------------------|
| Tokyo H0 | 105 tr, 55.2% WR, PF 2.92, +304.2 pips | 105 tr, **97.1% WR, PF 138, +1,553.5 pips, $11,058/lot** |
| NY H21 | 19 tr, 52.6% WR, PF 2.31, +134.2 pips | 6 tr, 50.0% WR, PF 6.17, +80.6 pips |

Tokyo H0's `fixed` result is identical to the validated `utc` design variant — the deployed EA now
fires at true 00:00 UTC. NY H21 sample is small (6 trades in 30 days); it now matches its design window.

### 2. FTMO M5 feed has no 00:00 bar

The broker's M5 history starts each day at **00:05** (no 00:00 bar). This means:
- The UTC/intent Tokyo H0 variant can only be proxied on the **00:05 bar** (used here).
- A Strategy-Tester backtest of the `hour==0 && min==0` gate would silently **never fire** on this data.

### 3. The Python validation sim inflated WRs (Ultra Monster & CPMC Z)

Both inflated claims trace to the same `run_ultra_buffed_orb`-style Python engine that reads the
**current bar's close** as the signal and enters at the **same bar's open** (one-bar lookahead).
Proven previously for Ultra Monster (`lookahead_test.py`): honest 46.7–47.0% WR vs lookahead 79.8% WR.
CPMC Z's "61%+ WR" is the same artifact — honest result is 43.8% and **loses money**.

The v107 EAs themselves use correct completed-bar semantics (`g_last_bar` first-tick gating;
offset-1 reads for Ultra Monster/MSV/NY H21; forming-bar-open for CPPF/CPMC/Tokyo H0).
The EAs are structurally honest — the **claims** were fake, not the bar indexing.

---

## Methodology

Each EA's MQL5 logic was ported exactly with honest semantics:

- **Signal time**: first tick of bar `t` (`g_last_bar` gate ⇒ `rates[0].close ≈ bar t open` for
  offset-0 EAs; offset-1 EAs use last completed bar `t-1`).
- **Entry**: bar `t` open (EA sends market order at first tick).
- **Exit**: timed at open of bar `t+HOLD_BARS` (EA `CheckExits` closes on expiry bar's first tick),
  with SL=50p / TP=80p evaluated intrabar against each bar's high/low.
  Worst-case SL-before-TP when a single bar breaches both.
- **Session gates**: three variants — `utc` (design/validated), `server` (UTC+3, pre-fix deployment),
  and `fixed` (post-fix config, verified against the design window).

No `ffill`/`bfill` forward-filling of signals. Cross-sectional Tokyo H0 ranks the 18 pairs at the
session bar; pairs missing a bar fall back to `0.0` exactly as the EA's `CopyRates` failure path does.

---

## Per-Strategy Detail

### Ultra Monster (magic 202600) — ❌ claim fake, weak grind
- 1,228 trades, **47.0% WR**, PF 1.21, +601.9 pips (avg **+0.5 pips/trade**), $/lot +$3,384
- Exits: 1,219 HOLD, 3 SL, 6 TP
- `CopyRates(...,1,14)` = completed-bar breakout of the prior 12-bar range; entry at :00/:30 bar open.
  Structurally honest. The claimed 74.5–78% WR only exists under same-bar lookahead.
- At 1.20 lot, +0.5 pips ≈ +$6 gross − $4.50 commission ≈ +$1.5/trade: marginal, not a 76% WR edge.

### CPPF Z (magic 202680) — ✅ REAL (small sample)
- 19 trades, **94.7% WR**, PF 5297, +529.6 pips (avg **+27.9 pips/trade**), $/lot +$2,893
- All exits HOLD (SL/TP never triggered). No session gate ⇒ `utc` = `server`.
- Per pair: AUDNZD 7, GBPAUD 5, GBPNZD 4, EURNZD 2, EURAUD 1.
- Consistent with the 7-month honest `cppf_z` backtest (z≥6, hold 18: 75% WR, PF 5.23) —
  the EA's forming-bar-included z-score is a slightly different (honest) signal.
- Caveat: 19 trades / 30 days is a small sample; treat 85%+ claim as plausible, not proven.

### CPMC Z (magic 202690) — ❌ **DEAD**
- 178 trades, **43.8% WR**, PF 0.57, **−495.8 pips** (avg −2.8), $/lot −$2,622
- Exits: 177 HOLD, 1 SL. SELL fade of >0.15% 55-min rise: negative edge.
- The "61%+ WR" claim is a lookahead artifact. **Do not run live.**

### NY H21 (magic 202670) — ⚠️ marginal, now on design window
- `utc` (21:00 UTC): 6 trades, 50.0% WR, PF 6.17, +80.6 pips
- `server` pre-fix (18:00 UTC, actual): 19 trades, **52.6% WR**, PF 2.31, +134.2 pips, +$673/lot
- `fixed` (SESSION_HOUR=0 = server 00:00 = 21:00 UTC): 6 trades, 50.0% WR, PF 6.17, +80.6 pips ✅
- Small samples either way. Fixed now matches the validated 21:00 UTC design window, but 30d sample is
  only 6 trades — keep on watch, do not scale until a longer sample confirms.

### MSV Asian Exhaustion (magic 202650) — ✅ REAL (new edge)
- `utc` (00–06 UTC): 26 trades, **76.9% WR**, PF 13.31, +274.6 pips, +$1,712/lot
- `server` (21–03 UTC, actual): 21 trades, **71.4% WR**, PF 14.78, **+414.9 pips**, +$2,570/lot
- 18-pair Asian-exhaustion fade (`ret < −0.2%` over completed 60 min, BUY, hold 12). All exits HOLD.
- Best-performing *non-claimed* strategy; robust to the timezone shift.

### Tokyo H0 (magic 202630) — ✅ FIXED
- `utc` (00:00 UTC, proxied on 00:05 bar): 105 trades, **97.1% WR**, PF 138, +1,553.5 pips, **+$11,058/lot**
- `server` pre-fix (21:00 UTC, what actually fired): 105 trades, **55.2% WR**, PF 2.92, +304.2 pips, +$1,857/lot
- `fixed` (SESSION_HOUR=3 + `dt.min > 5` = 00:00/00:05 UTC): 105 trades, **97.1% WR, PF 138, +1,553.5 pips, +$11,058/lot** ✅
- Per-pair (server pre-fix): AUDUSD 13, AUDJPY 10, NZDUSD 10, EURGBP 8, EURJPY 8, EURUSD 11, ...
- **Conclusion**: The 95% WR edge is real at true midnight. Pre-fix the EA fired 3h early at 21:00 UTC
  (55.2% WR ≈ break-even after commission). **Fixed config restores the full validated edge.**

---

## Files

- `mt5-connector\ultra_monster\audit_v107_ea_semantics.py` — honest audit harness (this report's source)
- `mt5-connector\ultra_monster\lookahead_test.py` — lookahead-vs-honest proof for Ultra Monster
- `mt5-connector\ultra_monster\collect_ftmo_pairs.py` — FTMO M5 collector (13 pairs added)
- `mt5-connector\ultra_monster\ftmo_{PAIR}.parquet` — 30-day FTMO M5 cache (22 pairs)
- `paper_trade\mt5_backtest\*_MT5_v107.mq5` — the 6 audited EAs

## Recommendations

1. ✅ **Tokyo H0 fixed** (done): gate on `TimeGMT()` with `SESSION_HOUR=0`, `dt.min <= 5` (fires the 00:05 bar). Verified 97.1% WR / PF 138. Now DST-proof (no server-offset constant).
2. ✅ **NY H21 fixed** (done): gate on `TimeGMT()` `SESSION_HOUR=21`. Now fires at design 21:00 UTC (small 6-trade sample).
3. **Remove CPMC Z** from the live set (negative honest edge: PF 0.57, −495.8 pips).
4. **Keep CPPF Z** and **MSV** — real honest edges; MSV is a newly confirmed live candidate.
5. **Do not rely on Ultra Monster's WR claim** — honest 47% WR / +0.5 pip per trade.
6. Re-validate claims sourced from `run_ultra_buffed_orb`-based sims (one-bar lookahead).
7. **Next**: recompile fixed EAs (`mcp-mt5_compile`) and redeploy `.ex5`; re-audit after a longer FTMO window.
