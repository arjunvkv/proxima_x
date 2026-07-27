# Research Plan V2 — High-Frequency High-Win-Rate Structural Signals

## Status
**V1 complete** — all conventional signals failed at ≥30 trades/day with positive net.

**V2+z signal found — bar-level validated BUT tick-level execution shows strategy degrades from 76-79% WR to 35-41% WR.** The bar-level backtest was an OHLC artifact. Tick-level validation is the critical path before any EA implementation.

**Current: RESEARCH PHASE — tick-level parameter tuning and cross-pair validation required.**

## Hard Constraints
- **Trades/day**: ≥30 (not negotiable)
- **Win rate**: ≥65% (not negotiable)
- **Net profit**: positive after all costs (spread + commission)
- **Session**: any/all (no session-limited)
- **Signal class**: structural (Bucket 3 — Market Ecology), not price-pattern or statistical

## What We Tried (V1 — All Failed at ≥30/day)

| Signal | Best t/day | Best WR | Why Failed |
|--------|-----------|---------|------------|
| DC EURUSD (z>1.5 h=10m) | 3.4 | 60.1% | Too few trades |
| DC EURJPY | 5.0 | 53.5% | Too few trades |
| DC GBPJPY | 6.1 | 57.9% | Too few trades |
| 10s MR EURUSD (z>3.5 h=3m) | 48.0 | 53.5% | WR too low, spread cost kills |
| Stop hunt EURJPY | 3.9 | 62.2% | Too few trades |
| OFI / VW-OFI | — | 50% | No forward signal |
| Cross-pair spillover | — | 48-50% | Random |
| Spread asymmetry | — | <55% | No edge |
| 10s breakout | — | 40-47% | Anti-signal |
| Triangular arbitrage | — | — | No USDJPY data |

## Discovery Path

### Phase 1: Asymmetric Exits (Failed)
Tested trailing stop on DC and 10s MR signals. DC too few trades (4-14/day), 10s MR negative net (avg PnL < spread cost on 10s bars).

### Phase 2: HF-DF (High-Frequency Dealer Flow) — Found Signal
**Concept**: Fade every bar's micro-direction with a tight trailing stop. Entry on spread-widen events (sr>1.05), direction = -sign(z) where z is 50-bar return z-score.

**Initial result (10s bars, sr>1.05 filter)**:
| Pair | /day | WR | Net/trade | Status |
|------|------|----|-----------|--------|
| EURUSD | 74 | 65.5% | +0.05 | ✅ |
| GBPJPY | 350 | 68.9% | +4.42 | ✅ |
| EURJPY | 262 | 61.8% | -19.49 | ❌ |

EURJPY fails because spread cost (50 MP) > avg PnL (~31 MP).

### Phase 3: M1 Breakthrough — No Filter Needed
Switching from 10s to M1 (60s) bars eliminates the need for any sr filter. Larger ATR means the trailing stop distance scales up, making spread cost proportionally smaller.

**Exness tick data → resampled to 60s bars (Oct–Dec 2025)**:
| Pair | /day | WR | Avg win | Avg loss | Net/trade |
|------|------|----|---------|----------|-----------|
| EURUSD | 1,188 | 69.5% | +0.80 | -0.19 | **+0.35** |
| EURJPY | 1,190 | 67.6% | +140.95 | -33.15 | **+34.62** |
| GBPJPY | 1,189 | 70.9% | +195.07 | -48.01 | **+64.45** |

ALL THREE PAIRS PROFITABLE. EURJPY goes from -19.49 → +34.62.

**Dukascopy M1 bid data (Oct 2024 – Jun 2026, independent source)**:
| Pair | /day | WR | Net/trade |
|------|------|----|-----------|
| EURUSD | 846 | 56.6% | +0.29 |
| EURJPY | 853 | 59.8% | +47.76 |
| GBPJPY | 853 | 59.7% | +66.26 |

Lower WR but still positive. Difference may be Dukascopy bid vs Exness mid, or different time periods.

### Phase 4: V2+z Threshold — Trade Count Reducer + WR Booster
Adding a z-score threshold filters out weak-signal bars, reducing trade count while **increasing** WR.

**CSV data (3 pairs, Oct 2024 – Jun 2026, 462 days):**
| z>= | EURUSD t/d | EURUSD WR | EURJPY t/d | EURJPY WR | GBPJPY t/d | GBPJPY WR |
|-----|-----------|----------|-----------|----------|-----------|----------|
| 0.0 | 846 | 56.6% | 853 | 59.8% | 853 | 59.7% |
| 0.5 | 560 | 72.6% | 556 | 73.2% | 553 | 73.4% |
| 1.0 | 266 | 74.9% | 272 | 74.5% | 265 | 75.4% |
| 1.5 | 125 | 76.7% | 130 | 75.8% | 126 | 76.2% |
| 2.0 | 54 | 78.1% | 57 | 76.7% | 57 | 77.3% |
| 2.5 | 23 | 79.0% | 25 | 76.8% | 24 | 78.8% |
| 3.0 | 10 | 80.3% | 11 | 79.3% | 11 | 79.5% |

**Key insight:** Adding even a minimal z>=0.5 threshold jumps WR from ~57% to ~73% while cutting trade count by ~35%. This single parameter transforms the strategy from "barely profitable after spread" to "strongly profitable."

**Fine-grained z-sweep (0.0–1.5 in 0.1 steps):**
WR jumps from ~57% to ~68% at just z>=0.1. The weakest signals (z near 0) have no edge; filtering them out cleans the trade set dramatically.

**Per-split robustness (CSV data, 3 time periods):**
| Split | EURUSD | EURJPY | GBPJPY |
|-------|--------|--------|--------|
| Q4 2024 (z>=2.0) | 77.2% WR | 71.0% WR | 73.2% WR |
| Q1 2026 (z>=2.0) | 79.1% WR | 81.0% WR | 82.8% WR |
| Q2 2026 (z>=2.0) | 77.7% WR | 76.1% WR | 77.1% WR |

Consistent 71-83% WR across all periods. No regime failure.

## The Signal — Simple Specification

```
Entry: every M1 bar close that satisfies |z| >= Z_THRESHOLD
Direction: -sign(z_score)
  z = (ret - rolling_50_mean) / rolling_50_std
Exit: asymmetric trailing stop
  stop = 0.15 × ATR(20)       # initial stop loss
  trail_trigger = 0.20 × ATR  # start trailing after this profit
  trail_gap = 0.10 × ATR      # keep stop this far behind best price
  max_hold = 54 bars (54 min)
```

**All rolling calcs use shift(1) — zero lookahead. Verified with strict shift(1) test: results near identical.**

**Z_THRESHOLD is the tuning knob:**
- z>=0.5: ~550-560 trades/day/pair, 72-73% WR — max profit
- z>=2.0: ~55-65 trades/day/pair, 76-78% WR — balanced
- z>=2.5: ~25-33 trades/day/pair, 77-79% WR — low trade count
- z>=3.0: ~10-33 trades/day/pair, 79-80% WR — minimum trades

Choose based on broker tolerance for trade frequency.

## Why It Works (Structural)

**1. Micro-directional persistence inverts at M1 scale.**
- -sign(z) predicts next-bar direction ~70% of the time
- This is mean reversion at the highest frequency — the market oscillates bid-ask at sub-minute scale
- At M1, every bar is a micro-oscillation, not a trend bar

**2. Trailing stop captures oscillation asymmetry.**
- Initial stop (0.15 ATR) catches losers immediately (avg loss = 0.19 MP for EURUSD)
- Trail trigger (0.2 ATR) activates when price moves favorably
- Trail gap (0.1 ATR) lets winners run further
- Result: payoff ratio ~4:1 (avg win / avg loss)

**3. Z-threshold filter removes noise.**
- Bars where z ≈ 0 have no directional bias (WR ≈ 50%)
- Bars where |z| > threshold have strong directional bias (WR > 70%)
- The threshold selects only the bars where microstructure oscillation is most pronounced

**4. Spread cost becomes negligible at M1 scale.**
- 10s bar ATR (EURUSD): ~1.0 → stop = 0.15, gap = 0.10, spread = 0.15 → eats 30%+ of gross
- M1 bar ATR (EURUSD): ~1.27 → stop = 0.19, gap = 0.13, spread = 0.15 → same ratio
- For JPY pairs: M1 ATR is 170-250× 10s ATR → spread cost drops to 48-59% of gross

**5. The z-threshold filter also improves slippage tolerance.**
- Higher z = larger departure from mean = larger expected move = more room for slippage
- JPY pairs naturally have larger z-magnitude per bar than USD pairs

## V2+z Validation Results

### Multi-Pair Validation (26 Dukascopy Parquet Pairs, Apr–Jun 2026)

**This is the critical finding: V2+z is a universal FX microstructure phenomenon.**

Every single one of the 26 available pairs shows 70-79% WR at z>=0.5 with 500-560 trades/day. No pair is flat or negative. This confirms the signal is not pair-specific but a fundamental property of FX microstructure.

**Combined 26-pair backtest at z>=0.5:**
| Metric | Value |
|--------|-------|
| Total pairs | 26 |
| Avg trades/day/pair | 546 |
| Combined trades/day | 14,196 |
| Combined WR | 73.4% |
| Total gross MP | +9,912,415 |
| Daily avg (0.01 lot) | $527/day |

**Portfolio scaling (all 26 pairs, 0.01 lot):**
| z>= | t/d | WR | $/day |
|-----|-----|-----|-------|
| 0.0 | 30,172 | 61.5% | $12,371 |
| 0.5 | 14,196 | 73.4% | $8,714 |
| 1.0 | 7,160 | 74.9% | $5,503 |
| 2.0 | 1,621 | 77.7% | $896 |
| 2.5 | 812 | 78.5% | $537 |
| 3.0 | 436 | 79.0% | $350 |

### Trailing Stop Order Bug (Critical Fix)

The parquet backtest script (`backtest_m1.py`) had a bug in the trailing stop logic:
- **Bug:** The code checked the trailing stop BEFORE updating it (exit-first, trail-second)
- **Effect:** On the bar where price crossed the trigger level, the old (wider) stop was checked before the new (tighter) trailing stop was applied. This caused premature exits, reducing WR from ~59% to ~24%.
- **Fix:** Trail first, then check exit. Fixed in `run_v2z_parquet_multi.py` and `run_v2z_combined.py`.

### Slippage Sensitivity

Modeled as round-trip slippage (entry + exit) added to each trade's cost:

| Pair | z>= | BE slippage | Net at 0.2p | Net at 0.5p | Net at 1.0p |
|------|-----|------------|-------------|-------------|------------|
| EURUSD | 2.0 | 0.61p | +0.16 | -0.12 | -0.21 |
| EURJPY | 2.0 | 1.17p | +55.0 | +39.6 | +23.5 |
| EURJPY | 2.5 | 1.46p | +86.6 | +68.4 | +49.4 |
| GBPJPY | 2.0 | 1.53p | +64.9 | +46.8 | +22.3 |
| GBPJPY | 2.5 | 1.96p | +106.7 | +81.4 | +44.9 |

**Key finding: JPY pairs survive slippage of 0.5-1.0p at z>=2.0+. EURUSD breaks at 0.2p.**

**FundedNext reported slippage: 0.2-0.5p round trip.** At this level:
- EURUSD at z>=2.0: barely positive (+0.16 to -0.12) — DO NOT TRADE
- EURJPY at z>=2.5: net=+68.4 MP — PROFITABLE
- GBPJPY at z>=2.5: net=+81.4 MP — PROFITABLE

**IC Markets Raw ECN (0.1-0.2p slippage):** All pairs viable at z>=0.5.

### CPPF (Cross-Pair Polarity Filter) — REJECTED
Tested two CPPF variants:
1. **Polarity voting:** Cross-pair agreement score → negative total PnL
2. **Consensus deviation:** Cross-pair residual divergence → negative total PnL

Verdict: CPPF does not add value. The V2+z signal works independently per pair.

## FundedNext Stellar 2-Step $25k Feasibility

### Cost Structure
- **Commission:** $3 per round lot (Stellar 2-Step specific rate)
- **Spread:** Variable, raw from 0.0 pips on MT5 (competitive ECN)
- **On 0.01 lot:** $0.03 commission + ~$0.03-0.06 spread = $0.06-0.09/trade all-in

### Per-Pair Profitability After All Costs (z>=2.5)

| Pair | net_mp | $/trade (0.01 lot) | -comm | net/trade | tpd | $/day |
|------|--------|-------------------|-------|-----------|-----|-------|
| GBPNZD | +2.27 | $0.2270 | $0.030 | $0.1970 | 31 | $6.11 |
| EURNZD | +1.79 | $0.1790 | $0.030 | $0.1490 | 30 | $4.47 |
| GBPAUD | +1.66 | $0.1660 | $0.030 | $0.1360 | 30 | $4.08 |
| EURAUD | +1.45 | $0.1450 | $0.030 | $0.1150 | 30 | $3.45 |
| GBPCAD | +1.16 | $0.1160 | $0.030 | $0.0860 | 31 | $2.67 |
| GBPJPY | +132.98 | $0.0891 | $0.030 | $0.0591 | 31 | $1.83 |
| CHFJPY | +132.69 | $0.0889 | $0.030 | $0.0589 | 30 | $1.77 |
| USDJPY | +127.13 | $0.0852 | $0.030 | $0.0552 | 33 | $1.82 |
| EURJPY | +96.98 | $0.0650 | $0.030 | $0.0350 | 30 | $1.05 |

**Key insight: Non-JPY crosses (EURNZD, GBPNZD, GBPAUD) outperform JPY pairs** because 1 MP = $0.10/0.01 lot vs JPY's ~$0.00067. Commission impact is proportionally smaller.

### Scenario Table

| Scenario | Pairs | t/d | WR | $/d@0.01 | $/d@0.05 | $/d@0.10 | % of $1,250 limit@0.10 |
|----------|-------|-----|-----|---------|---------|---------|----------------------|
| Top 6 non-JPY | 6 | 183 | 80% | $22.7 | $113.6 | $227.3 | 18.2% |
| Top 8 (4JPY+4non) | 8 | 245 | 79% | $24.6 | $122.9 | $245.8 | 19.7% |
| Top 12 | 12 | 371 | 79% | $33.2 | $166.0 | $332.0 | 26.6% |
| All 15 profitable | 15 | 464 | 79% | $37.9 | $189.6 | $379.2 | 30.3% |

### Risk Profile
- **Max concurrent exposure (0.10 lot):** 8 pairs × $10,000 notional = $80,000 = 3.2x leverage on $25k
- **Per-loss (0.10 lot):** ATR × 0.15 × $/pip. USD pairs ~$0.20, JPY pairs ~$0.60. At 20% loss rate (79% WR) with 245 trades/day: ~49 losers/day = ~$29/day worst-case loss
- **Max consecutive losses:** At 79% WR, expect 5-6 max. At 0.10 lot: ~$3.60
- **10-loss streak (p=10^-7):** ~$6 — impossible
- **News gap risk (20 pip NFP jump on 0.10 lot):** $20 per pair × 8 pairs = $160 — manageable
- **Daily loss limit ($1,250):** At 0.10 lot, relative worst case = ~$250 = 20% of limit

### Trade Count vs Win Rate Trade-offs

For FundedNext (where trade count may be scrutinized):
- z>=2.5, top 6-8 pairs: ~200-250 trades/day → "high frequency" but not crazy
- z>=3.0, top 6 pairs: ~100-150 trades/day → moderate frequency
- z>=2.0, top 6 pairs: ~350-400 trades/day → aggressive

At z>=2.5, 200 trades/day at 79% WR: only 42 losers/day, mostly small.

### Recommended Deployment Config (FundedNext Stellar 2-Step $25k)
- **z-threshold:** 2.5
- **Pairs (8):** GBPNZD, EURNZD, GBPAUD, EURAUD, GBPCAD, GBPJPY, CHFJPY, USDJPY
- **Lot size:** 0.05 (safe) to 0.10 (aggressive)
- **Expected daily:** $120-250/day
- **Expected monthly:** $2,600-5,500
- **Max daily loss safety margin:** 4-5× buffer at 0.10 lot

### Alternative: IC Markets Raw ECN
If FundedNext trading restrictions become an issue, IC Markets offers:
- 0.0 pip spread + $3.5/round lot commission
- No trade frequency limits (true ECN)
- V2+z at z>=0.5 on all 26 pairs: $8,714/day (0.10 lot)
- Benefit: can use lower z-threshold (more trades, more profit)

## Tick-Level Verification (CRITICAL — Strategy Degrades on Tick Execution)

**The bar-level backtest OVERESTIMATES performance.** When executed on actual tick data with tick-level stop checking, WR drops from 76-79% to 35-41%.

### Verification Setup
- **Source:** Exness tick data (EURUSD, EURJPY, GBPJPY, Oct–Dec 2025)
- **Method:** M1 bars built from ticks → PairState (live z/ATR computation) + TrailingStopManager
- **Bar comparison:** Same bars rebuilt via resample and run through hfdf_m1
- **Key check:** z-score match between bar-level and tick-level: **0.0000** (identical)

### Results (Full Data, z>=2.5)

| Pair | Months | Bar WR | Tick WR | Tick Net (raw) | Per-trade MP |
|------|--------|--------|---------|---------------|-------------|
| EURJPY | Oct–Dec 2025 | 75-79% | **35-37%** | +1.6 to +2.0 | +17-21 MP |
| GBPJPY | Oct–Dec 2025 | 77-82% | **39-41%** | +2.0 to +2.4 | +23-29 MP |
| EURUSD | Oct–Dec 2025 | 78-80% | **46-49%** | +0.01 to +0.02 | +0.1-0.2 MP |

### Key Findings

1. **Bar-level backtest is an OHLC artifact.** Trailing stop checking on once-per-minute OHLC compresses intra-bar price dynamics. Bar-level sees favorable retracements before adverse moves. Tick-level catches every tick — the first adverse wiggle triggers the stop immediately.

2. **Entry price explains nothing.** Entering at close of signal bar vs first tick of next bar shows 0.977 PnL correlation. Only 8.5% of trades flip sign. The 25pp WR gap is ENTIRELY from tick-level stop mechanics.

3. **ALL trades exit within ≤3 bars on bar-level.** The strategy is purely a quick-stop mean-reversion. At tick level, these become sub-minute holds (1-3 ticks) caught by intra-bar noise.

4. **WR degrades on larger samples.** First 500K ticks (EURJPY Oct): 51.4% WR. Full 1.1M: 34.7% WR. The strategy's tick-level performance depends heavily on market conditions.

5. **At z≥3.0, WR improves slightly** (EURJPY: 39-40%, GBPJPY: 38-41%) but per-trade PnL shrinks proportionally with trade count.

6. **First 500K sample shows viable performance** at z≥2.5 (EURJPY: 51% WR, +159 MP/trade — net +109 MP after cost). Full data's lower WR/marginal PnL may reflect October 2025's specific volatility regime.

### Implications for FundedNext Deployment

At the current tick-level WR (35-41%) and per-trade PnL (~20 MP after 50 MP cost):
- **EURJPY/GBPJPY at z≥2.5:** Marginally positive on ticks. At 0.10 lot: -$0.02 to +$0.01/trade → near zero net
- **EURUSD:** Flat to slightly positive → no edge after costs

**The strategy may not be profitable at tick-level after all costs for FundedNext.** The dollar estimates in the scenario table above are based on bar-level backtest WR and are NOT achievable.

### Path Forward

1. **Run tick-level validation on ALL 26 pairs** with full Exness or MT5 tick data to confirm the WR gap generalizes
2. **Tune for tick-level:** Test tighter initial stop (0.10 ATR), wider trail trigger (0.30 ATR), or longer max hold (108 bars)
3. **Time-segment analysis:** Identify which market hours yield tick-level profitability (Asian session may perform better)
4. **Accept lower expectations:** If tick-level WR = 40-50%, strategy may still be viable on non-JPY crosses with larger ATR at higher position sizing

**V2+z is NOT ready for live deployment.** Tick-level validation must precede any EA implementation.

## Critical Issues

1. **Trade frequency scrutiny:** FundedNext allows EA trading (fee applies) but may flag >200 trades/day. The z>=2.5 threshold with 8 pairs (~200/day) should be acceptable. Consider starting at z>=3.0 (~100/day) for safety.

2. **Slippage sensitivity:** At FundedNext's 0.2-0.5p slippage:
   - EURUSD is marginal even at z>=2.0
   - JPY pairs are fine at z>=2.5
   - Non-JPY crosses (EURNZD etc.) have higher per-trade $ value and tolerate slippage better

3. **★ Tick-level execution kills WR:** Bar-level 76-79% WR drops to 35-41% on tick execution. This is the #1 risk to deployment and invalidates all dollar estimates based on bar-level backtest.

4. **News events:** Add calendar filter as minimal survival step. Skip entry 5 min before to 5 min after high-impact events.

5. **Execution dependency:** Requires 24/7 VPS, stable broker connection. V2+z at 200 trades/day needs robust EA error handling. MT5 handles this easily.

6. **Trailing stop fill quality:** In backtest, stops fill at the exact price level. In live trading, slippage on stop orders adds cost. Mitigate by using wider stop multipliers or limit exits.

## Completed Work

- [x] M1 bar aggregation from live tick feed (MT5 M1 bars sufficient)
- [x] Live z-score computation (rolling 50 bars, verified shift(1) no-lookahead)
- [x] Live ATR computation (rolling 20 bars)
- [x] Trailing stop order management (trail-first, then check stop)
- [x] Spread cost verification — Dukascopy bid data confirms positive net
- [x] Multi-pair validation — all 26 pairs positive at z>=0.5
- [x] Slippage sensitivity — modeled 5 scenarios (0.0p to 0.5-1.0p)
- [x] Break-even slippage — computed per pair and z-threshold
- [x] Per-split OOS verification — Q4 2024, Q1 2026, Q2 2026 all positive
- [x] Independent data source validation — CSV (Dukascopy + Exness) vs parquet
- [x] Z-sweep fine-grained analysis (0.0-3.0 in 0.1-0.5 steps)
- [x] Trade count / WR trade-off mapping
- [x] CPPF evaluation — rejected (negative PnL)
- [x] FundedNext Stellar 2-Step cost model ($3/round lot)
- [x] Combined multi-pair portfolio backtest
- [x] Trailing stop order bug fix (exit after trail, not before)
- [x] EA specification document written
- [x] Recommended deployment scenario defined
- [x] Paper_trade V2+z strategy (PairState + TrailingStopManager)
- [x] Tick-level verification harness (hfdf_m1 vs PairState+TSM)
- [x] Cost unit bug fix (BASE_COST was MP, subtracted from raw PnL)
- [x] Tick-level WR gap identified: 76% (bar) → 35-41% (tick) at z>=2.5

## Remaining Work

### CRITICAL — Tick-Level Validation (Prerequisite for Deployment)
- [ ] Run tick-level validation on ALL 26 pairs with full tick data
- [ ] Determine if tick-level WR varies by pair (non-JPY crosses may perform better)
- [ ] Test parameter tuning for tick-level: (stop/trigger/gap at 0.10/0.30/0.10, 0.20/0.20/0.10)
- [ ] Time-segment analysis: identify which hours yield tick-level profitability
- [ ] Compare Exness ticks vs MT5 real ticks — verify MT5 Strategy Tester results
- [ ] If tick-level WR cannot reach ≥50% after tuning, ABANDON V2+z strategy

### Production EA Requirements (Deferred — depends on tick-level validation)
- [ ] Write MQL5 EA code implementing full spec
- [ ] Test in MT5 Strategy Tester on 6-8 pairs (backtest mode)
- [ ] Add news calendar filter (skip ±5 min high-impact events)
- [ ] Add max trade frequency governor (configurable)
- [ ] Implement daily loss limit halt

### Broker-Specific (Deferred)
- [ ] FundedNext MT5 connection test
- [ ] Commission verification on live/demo account
- [ ] Slippage monitoring — record fill vs expected price
- [ ] Spread monitoring — record actual spread per entry

### Optimization Candidates (Optional)
- [ ] Tune z-score window: 50 bars optimal?
- [ ] Tune ATR period: 20 bars or tune to market?
- [ ] Tune stop/trigger/gap ratio: current 0.15/0.20/0.10
- [ ] Test Kelly-criterion dynamic position sizing
- [ ] Test session-based variance (Asian vs London vs NY)

## Decision Gate
**The V2+z signal is NOT ready for deployment.** Bar-level validation is comprehensive, but tick-level execution reveals a critical flaw:

- This is a **CARRY-OVER from the MVS/Session lab** — marker that bar-level backtesting has been over-estimating profitability. Needs to be factored into V2+z as a separate regime check.
- ⚠️ Bar-level shows 76-79% WR at z>=2.5 ✅
- ⚠️ Tick-level shows 35-41% WR at z>=2.5 (3 pairs, 3 months) ❌
- ⚠️ Cost model requires tick-level validation to be meaningful (if WR < 50%, strategy is unprofitable)

### Next Steps
1. **Tick-level validation across ALL 26 pairs** — determine if tick-level WR is pair-dependent
2. **Parameter tuning for tick-level** — stop/trigger/gap optimization may recover WR
3. **Time-segment analysis** — identify market hours where tick-level execution works
4. **If tick-level WR cannot reach ≥50%:** pivot to alternative alpha models (V1 DC, cross-pair, regime-based)
5. **Only then:** build MQL5 EA

**Decision: RESEARCH CONTINUES — tick-level validation is the critical path.**
