# Intraday Forex Market-Structure Strategies — Concrete, Testable Definitions

Scope note: All strategies below are defined so they can be encoded as **closed-bar (M5) rules filling at next-bar open**, no lookahead. "Entry bar open" = open of the bar immediately after the trigger bar's close. Times are UTC unless stated. Every "claimed stat" is from a named source with the methodology caveat attached; none should be treated as verified — they are reproduction benchmarks. Feet-feet format: (entry) → (SL) → (TP) → (time exit / session).

---

## 1. Asian-Range Breakout at London Open (ORB / London break)
**Archetype:** Open range breakout, momentum continuation.

**Level construction (M5 on M15):**
- Asia high (AH) = MAX(high) of all bars in **00:00–06:55 UTC**.
- Asia low (AL) = MIN(low) of all bars of the same window.
- London entry window: **07:00–11:00 UTC**.

**Bar-level rules:**
- At each M5 bar OHLC, compare to AH/AL observed up to **close of prior bar** (no current-bar extreme).
- LONG if `close > AH`: enter **next bar open**. SL = one 5-pip buffer below AL (or `AL - max(1×ATR14, 2×spread)`), TP = `entry + 1.5× (AH-AL)`. Time exit 14:00 UTC.
- SHORT mirrored below AL. One signal per pair per day.

**Filters worth testing separately:** Asian range width 30–80 pips boosts reported 0.22R→0.35R (see stats); EMA-200 trend filter; exclude high-impact-news days.

**Claimed stats (conflicting — this is the headline finding):**
- BacktestEverything EA (EUR/USD, 2015–2025, 847 trades, exact rules above, ~08:00 GMT entry): **52.3% WR, 1.42 avg R:R, +0.22R expectancy**; range-filter → **+0.35R**.
- Quant-Signals "London Breakout v1" (EUR/USD 07:00–08:00 range, 5-pip buffer, 2:1): **14.7% WR, PF 0.26, −0.633R**; EMA200 filter → 27.5% / PF 0.76 / −0.174R.
- Quantified Strategies EUR/USD London breakout (03:00–08:00 London range, 08:00–11:00 window, time exits): **long breakouts LOST; short-side ~breakeven** → they recommend fading.
- FxFactory "A Simple London Breakout" (03:00–06:00 GMT box, TP≈box size, skip box>40–50 pips): **claimed 65–75% WR** (unverified).

**Falsifiable:** On my M5 tape (18 majors, 200d), pair-level expectancy: London Asian-breakout with first-close-above-entry and TP=1.5R/SL=1R/+M close-out must be > +0.15R after spread/buffer, else edge is inconsistent with +0.22R claim.

---

## 2. Kill-zone Momentum/Reversion Conditioning Tool London 07–10 / New York 12–15)
**Archetype:** The fundamentals claim is NOT a standalone entry — kill zones are **time filters** that concentrate a separate pattern. Test the *window marginal value* rather than "trade the kill zone."
**Definition:** Label London zone = bars with UTC hour in {07,08,09}; New York zone = [12,13,14]. Compare the SAME entry pattern inside vs outside the zone.

**Concrete pair of backtests to run:**
- **(a) Zone momentum:** long if first NSW-M5 bar of zone closes up vs zone open; short if down; TP 1R, SL 1R, time-exit end of zone.
- **(b) Zone ORB:** New York zone = measure 12:00–13:00 range, break at 13:00 toward prior London-direction move.

**Cited stats:**
- NY-continuation variant (EUR/USD, 2015–2025): **48% WR, 1.8:1 R:R, +0.19R expectancy**.
- Ito & Hashimoto (EBS 1999–2001): activity/persistence of spreads peak at London morning & London–NY overlap; bid–ask narrowest at overlap. This supports **better depth, not a directional edge**.
- Retail (FXCM 24M trades, published study): GBP/USD positions opened 4–5 a.m. ET net-profitable **47%** of time vs **55%** for 8–9 p.m. ET — timing > nothing, but still <50% net-profit rate at the volatile London open.

**Falsifiable hypothesis:** Mean expectancy of pattern (a) or (b) is not significantly different (Mann-Whitney p<0.05) between zone [07–10] and off-zone [10–12] UTC samples on EUR/USD, GBP/USD, EUJPY.

---

## 3. Liquidity Sweep / Stop-Hunt Rejection (prior-high/low wick reclaim)
**Archetype:** price wicks beyond a prior swing high/low, then closes back inside (rejection/reversal).

**Level construction:**
- Reference level L = prior 20-BAR high (M5) for shorts, or prior 20-bar low for longs (or prior session/day H/L).
- **Bearish sweep:** bar high > L + `0.05×ATR14` (penetration buffer) AND that bar closes back **below L**.
- **Bullish sweep:** bar low < L − `0.05×ATR14` AND close back above L.

**Bar rules (rejection):**
- Enter **next bar open** in counter-sweep direction.
- SL = sweep extreme + `0.10×ATR14` buffer; TP tested at **1R, 1.5R, 2R** separately.
- Time exit: end of London (or NY) session if no TP/SL; one trade per level.

**Cited stats:**
- **Osler (2000, NY Fed)** — the strongest citation: published FX support/resistance daily levels (DM, JPY, GBP 1996–98, 1-min quotes) produced a bounce/reversal **60.8%** of the time vs **56.2%** at random control levels (+4.6pp), effect persists ±5 business days. Caveat: old, indicative quotes. This is the *base rate* a sweep must beat.
- A sweep/reversal study (EUR/USD, wick beyond swing level + close inside, ~4,986 obs): only **30.2% reversed; 68.2% continued** → do NOT assume wick-past-level = reversal.
- Vendor EA (EUR/USD H1, 174 trades, 20-bar H/L): **39.66% WR, PF 1.39** (won on payoff, not hit rate).

**Falsifiable hypothesis:** sweep-reclaim WR (post-cost) > 40% at 1.5R on paired majors AND PF>1.1; and the difference vs random-shifted levels (edge test) is significant.

---

## 4. VWAP Mean-Reversion (stretched away from session VWAP)
**Archetype:** price stretched ≥kσ below/above session VWAP reverts.

**Level construction (M5):**
- VWAP = Σ(TypicalP × volTick) / ΣvolTick over the **session** (reset daily; decide 00:00 vs London 07:00 UTC). Prefer **tick volume**, with a stuntion if tick volume is flat (use bar-typical-price rolling VWAP fallback).
- Deviation `D = (P − VWAP) / σ_t` where σ_t = stdev of (TypicalPrice − VWAP) over the same session. Use running stdev updated through close of prior M5 bar (no lookahead).

**Bar rule (buy reversion):**
- LONG: prior M5 closes with D ≤ −2.0σ (or price touching −2σ, tested both), AND (optionally) VWAP slope ~flat in the last 6 bars (range-day filter).
- Enter next-bar **close/next-bar open at or above −2σ**.
- TP = VWAP (or VWAP−0.5σ), SL = −3σ or 1.5× typical excursion beyond entry extreme. Time exit end of session.
- SHORT mirrored at +2σ.

**Cited stats:** Computable strategies' VWAP data is equities-only (SPY 5-period MR: 8.18% CAGR, −27.93% MaxDD — not a WR). Futures/comm vendor claims cluster **54–63% WR** for 1.5–2σ ES filtered; **55–65%** generic; **75–80%** with confirmed-reversal filters (unverified). **No verified forex intraday VWAP-reversion WR exists.** Forex VWAP is broker/tick-volume definition-dependent → must fix data source.

**Falsifiable hypothesis:** reversion long D≤−2σ ⇒ TP at VWAP exits before 2σ SL on PAIR-session, net of spread, at a rate > 62% (= breakeven for 0.6R/1R payoff) and PF>1.1; else reject.

---

## 5. Session Mean-Reversion Exhaustion — Fade the Open-Range Extreme (London & NY open analog of the Tokyo hour-0 fade already validated)
**Archetype:_reversal of overextension.

**Level construction (M15→M5):**
- Session open range OR = first **30-minute** (M15×2) high-low window at session open, or **first hour** (M5×~12) for the fade — test both 0.5h/1h.
- Fade signal: price closes **beyond** OR-high (or OR-low) then an M5 closes back **inside** toward OR-midpoint, OR (stronger) closes back through the OR-open.

**Entry/exit:**
- Entry next bar in the counter-breakout direction.
- TP = session open / OR-midpoint (Test both) — fade-to-midpoint.
- SL = OR opposite extreme ± ATR buffer; time exit end of session.
- Only fade the **first** break of an OR per session.

**Cited stats:**
- Best-given fade test (futures, 2019–2026): **fading EVERY open-range break to midpoint was unprofitable for 5/15/30/60-min ranges** — 15m was worst (−$77,361, t=−4.00); no WR published. This DIRECTLY contradicts internet "80%+ ORB fade" claims.
- Financial benchmark same sample: unfiltered ORB (first close out, exit session-close) = **~50–51% WR, PF 0.91–1.07** on NQ/ES/YM 2023–24.
- Academic: **Holmberg, Lönnbark & Lundström (2013, Finance Research Letters 10(1):27–33, DOI 10.1016/j.frl.2012.09.001)** on NYSE crude **report positive ORB returns + higher success rate than fair-game benchmark** — the paper is about UNIT ranks, and its success rates (~60.6% long / ~54.2% short in sampled config) are crude-oil, open-range, daily-data based — **confirmation of EXTIR (fade) is consistent: FTJ providers about fading.**

So the falsifiable version is about the 2 categories: fade-to-middle versus your Tokyo hour-0 analog (which the parent says is already validated at Tokyo). **Hypothesis:** this exact fade rule produces positive expectancy (PF>1.1, WR>40%) at *London* (07 UTC) and *New York* (12 or 13:30 UTC) opens on ≥8/18 majors, matching the (assumed-validated) Tokyo variant. If not not, whichever des — contrasting fills are driven by data.

---

## 6. Session-Return Momentum (ride first-N-bar session direction
**Archetype:** direction of first N bars persists.
**Level (M5):** session open (London 07:00 / NY 12:00 / NY 13:30?) → compute `R1 = (Close[NthBar] − SessionOpen)` using N ∈ {1h: use 50–60 min; 1h to 2h: use ∫ ~w — start with the first **60 minute session return** (first-12 M5 bars for London, first-6 for NY if input window) or first 30-min).
- Then direction = sign(R1).

**Momentum rule (sample for London first-hour):**
- LONG if sign(R1)>0; SHORT if <0; entry at the first-hour **close** (next bar open).
- Test exits separately: next hour, session close, fixed 1R/1R, fixed 1R/2R. Volume filter: skip/hold if R1≈0 or first-hour range forces -full width.
- Compare with a mirrored **reversal** variant (opposite sign) — must run both to fairly attribute any effect.

**Cited stats:**
- Related London-continuation / NY-continuation study: NY continuation of London-morning direction **48% WR, 1.8:1, +0.19R** (EUR/USD). Supportive but modest persistence.
- Persistence academic (Breedon & Ranaldo, 2013 published-version CNY? — YES: Breedon & Ranaldo) analyzed intraday firm-quotes; **reversal more pervasive than persistence** across sessions. Larger FX-session study: direction-positive shares only **EURO/USD-44%, U.S.+5** in-sample; posting costs remove most P/L.
- Swiss National Bank (FXH 1993-2005): significant time-of-day patterns, but **reversal generally more common than persistence** → evidence AGAINST a simple first-bar momentum continuation edge.

**Falsifiable hypothesis:** `sign(first-hour return)` predicts the sign of the *next* session's open-to-close return at > 50% (p<0.05 binomial) across 18 pairs; and the momentum version beats its mirrored-reversal twin in total R after costs.

---

## 7. Prior-Day (/Oses) High-Low as Support/Resistance Rejection
**Archetype:** prior-day (or prior-session) H/L magnet/rejection.
- L_intrusions: PDH = max(high) of previono completed session (day 0 → prior 24h from UTC-N14); PDL = min(low).
- Test three variants: (a) **sweep-and-reclaim** (below PDL by `0.05×ATR14`, close back above → 1) rejection / B of previous extreme. (b) **turn-back** first touch (close above PDL = long). (c) clean breakout. Do them separate — they are different edges.

**Long PDL (rejection) bar rule:**
- M15 bar close > PDL (or closes above after a low that WICKED below PDL−0.05ATR).
- Enter next bar open, SL below the sweep-low − 0.10ATR, TP separate {1R, 1.5R, 2R, next PDH.

**Cited stats:**
- **Osler (2000 NY Fed)**: published daily S/R levels (from six FX firms, 1996–98) produced bounce in ~60.8% of the approach vs 56.2% at random levels (+4.4pp). Pro report; old data.
- A 2026 thesis (stock NVDA; not FX) using PDL/PDH as liquidity proxy with sweep: 275 trades, **43.64% WR, PF 1.18, E6.69/trade** — sub-50% WR but positive. Shows care: level-rejection edges are payoff-driven, not hit-rate-driven.

**Falsifiable hypothesis:** prior-day H/L rejection (given a sweep→reclaim) has a **post-cost WR that beats** 40% at 2R AND a PF>1.15; but (Osler) also show rejection frequency when approaching = random-shifter control: if `PDH/PDL freq − random freq` is < 2pp, edge is selected artifact.

---

## 8. Tokyo-Fix Reversal / Tokyo-Hour-0 (bonus; mostly exists around the fix)
**Archetype:** reversal tied to the 09:55 Tokyo fix (00:00/00:55? UTC 00:55) — mark: fixed revert before/after.
- **Level:** Tokyo hour-0 = 00:55–01:55?? — 09:00–10:00 JST = 00:00–01:00 UTC. FX fix ~ 09:55 JST / ~00:55 UTC. Tokyo has NO DST.
- Long/short: sell the base before fix, buy after (per BoC): sell USD pre-fix, buy USD post-fix (the aggregate dollar **depreciates ~5.0% annualized before** the Tokyo fix and **appreciates ~5.3% annualized after**).

**Bar rule (NY turn example for USD/JPY, the structurally-relevant pair):**
- SHORT 00:00–00:55 UTC window; cover 00:55; long side after 00:55 to end-of-first-hour; TP ≈ 2.5bp daily move; SL binding cost‑dominant 20. Fade hour-0 *into* fix, then reverse.

**Cited stats:** Bank of Canada (1999–2019 high-freq FT on Tokyo fix) — dollar **appreciated before fix / depreciated after ≈ ~2.5 bp/day**; **spreads at Tokyo hour-0 are wider** (EBS), costs kill most edge; a Tokyo **opening-range breakout** test gave near-zero (~40–50% WR): USD/JPY 51.2%/+0.08R, AUD/JPY 47.8%/−0.15R, EUR/JPY 49.5%/−0.03R; and proprietary "TokyoFixReversal" EA claims 61.2% WR / PF 1.75 (unverified).

**Falsifiable hypothesis:** For USD/JPY and AUD/JPY, the after-fix drift (00:55–02:00 UTC) minus costs > pre-fix drift by 2.5 bps and net >0; otherwise the "reversal" is cost-noise.