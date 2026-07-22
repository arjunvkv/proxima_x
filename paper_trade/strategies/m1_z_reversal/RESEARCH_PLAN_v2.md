# M1 Z-Reversal → Shock Reaction v2 — Research Plan

## Status: STRATEGY v1 IS DEAD — Structural Failure

**Verdict from ChatGPT Brain analysis:**

> The z-reversal strategy failed because the **representation layer and execution layer are incompatible**. The bar model says "fade extreme M1 move" → tick reality says "information already consumed → entry hits microstructure noise → stop gets harvested."

The signal's edge (0.34 pips/trade at bar level) is smaller than tick-level friction (0.6-1.3 pips/trade). The strategy is operating in a **sub-spread signal regime**.

**Evidence from tick-level parameter sweep (EURJPY, Oct-Dec 2025, 2.87M ticks):**
- Every combination of z_thresh (1.5-3.5), min_stop_pips (1.5-20), stop_a (0.15-0.50), hold time (5-240 min) produces negative PnL at tick level
- Best: z=3.5, 20-pip stop: n=574 WR=49.1% PnL=-5.8 pips
- Mean reversion: WR=28.0%, PnL=-53.9 pips
- Momentum (flip): WR=24.3%, PnL=-59.5 pips
- Both directions fail → the z-score is not predicting direction, it's measuring **temporary volatility expansion**

**Do NOT continue:**
- ❌ More z optimization (already disproven)
- ❌ More trailing stop optimization (already disproven)
- ❌ More ATR filtering (ATR only says movement happened, not why)

---

## The Useful Discovery

The bar-level backtest was measuring something real — **large M1 anomalies contain information**. But the information is not "fade immediately." The information is "something happened, now classify the reaction."

**The strongest clue:**
| Strategy | n | WR | PnL | Note |
|----------|---|----|-----|------|
| Fast fade 10+pip in 5s, hold 30s | 23 | 69.6% | +30.0p | EURUSD, tick-level |
| Fast fade 15+pip in 10s, hold 10s | 16 | 75.0% | +32.7p | EURUSD, tick-level |
| EURUSD z>2.5 fixed hold 54min | 1,895 | 50.6% | +271p | EURUSD, tick-level |

The market gives edge immediately after displacement, not 54 minutes later. The short-horizon exhaustion model is the most promising research path.

---

## All Problems Encountered — From Strategy Development to Demise

### Bucket 1: Execution Model (The 40% Gap)

| # | Problem | Discovery Point | Impact |
|---|---------|-----------------|--------|
| 1 | **Bar-level backtests are structurally optimistic** for tight-stop strategies. Entry at bar.close and exit at bar.H/L models execution that doesn't exist in real tick-level trading. | tick_backtest.py comparison | 40% WR overestimate |
| 2 | **Trailing stop edge disappears at tick level**. At bar level, trailing exit via bar H/L captures reversals. At tick level, the stop fires on intra-bar noise before the reversion completes. | TrailingStopManager tick testing | 76% WR → 28% WR |
| 3 | **0.6-1.3 pips of hidden friction per trade**. Entry: bar.close vs first tick = ~0.3 pip diff. Exit: bar H/L vs actual tick crossing = ~0.3-1.0 pip diff. Combined friction exceeds the strategy's edge (0.34 pips). | Trade-by-trade comparison in _tick_backtest_comparison.py | Friction > edge |
| 4 | **41.8% of trades flip win/loss between bar and tick models**. The entry/exit price difference is enough to change the outcome of nearly half the trades. | Per-trade PnL comparison | Strategy stability zero |
| 5 | **Bar-level trailing stop assumes instant fill at favorable price.** Live: stop fills at the first tick crossing the level, which can be worse than the bar's H/L by 0.5+ pips. | Logic analysis of TrailingStopManager.update() | Overestimated win sizes |

### Bucket 2: Signal Quality

| # | Problem | Discovery Point | Impact |
|---|---------|-----------------|--------|
| 6 | **M1 z-score predicts next bar direction at only 50.8%** — barely above coin flip. The signal has almost no directional predictive power at tick-level granularity. | analyze_data.py (next-bar return analysis) | No edge to exploit |
| 7 | **The z-score measures volatility expansion, not mispricing.** It cannot distinguish between a liquidity vacuum (will revert) and information arrival (will continue). Both produce high |z| values. | ChatGPT Brain analysis | Unclassifiable signals |
| 8 | **Sub-spread signal regime.** The signal's raw edge (0.34 pips/trade at bar level, measured at H/L) is AFFIRMATIVELY SMALLER than the tick-level friction (0.6-1.3 pips). The strategy tries to capture profit that is smaller than the cost of entering/exiting. | Bar vs tick comparison of avg PnL per trade | Structural failure |
| 9 | **Both mean reversion AND momentum fail at tick level.** (WR 28.0% vs 24.3%). Direction is irrelevant — the signal does not predict direction, it only detects that movement happened. | tick_backtest.py --invert test | No direction strategy works |
| 10 | **ATR gate does not filter usefully.** Requiring ATR > 25th percentile eliminates low-volatility periods but the remaining signals still have no edge at tick level. | Parameter sweep varying atr_pctl | Gate is cosmetic |

### Bucket 3: Parameter Space is Exhausted

| # | Problem | Discovery Point | Impact |
|---|---------|-----------------|--------|
| 11 | **No profitable parameter combination exists.** Every z_thresh (1.5-3.5), stop_a (0.15-0.50), trig_a (0.20-0.50), min_stop_pips (1.5-20), hold time (5-240 min) was tested — all negative PnL at tick level on EURJPY. | Full grid scan (3,000+ configs) | Tuning is futile |
| 12 | **Wider stops don't help enough.** At 20-pip stops, WR approaches 50% but wins are small and spread cost still dominates. Best: z=3.5, 20p stop, n=574, WR=49.1%, PnL=-5.8p over 3 months. | msp=20 scan | Near breakeven only |
| 13 | **Longer holds don't help either.** Without trailing stop, WR is ~51.5% regardless of hold time (5-240 min), but losses are 5× larger than wins (avg loss -5.5p, avg win +1.1p). | Fixed-hold scan (hold 1-240 bars) | Signal too weak for any hold |
| 14 | **EURUSD fixed hold shows positive but monthly inconsistent.** z>2.5 hold 54min: Oct +487p, Nov -163p, Dec -58p. Total +271p over 3 months but driven by one good month. | Monthly breakdown (test_monthly.py) | Not robust |
| 15 | **EURJPY at 50 MP spread is too expensive.** The thin signal edge is consumed entirely by the 0.5-pip round-trip cost. EURUSD at 0.03 pip spread is the only viable pair. | Cost-adjusted comparison across pairs | EURJPY unviable |

### Bucket 4: False Assumptions & Dead Ends

| # | Problem | Discovery Point | Impact |
|---|---------|-----------------|--------|
| 16 | **"Flipped PnL" momentum estimate was wrong.** Early assumption that flipping direction would produce WR=72% at tick level was incorrect because entry prices (ask vs bid differ) and trailing stop behavior changes entirely when direction flips. | verify_momentum.py vs _tick_backtest_comparison.py | Wasted optimization cycles |
| 17 | **MT5 tick history is not available for validation.** `copy_ticks_from` and `copy_ticks_range` fail (TIMEFRAME_TICK attribute missing in MT5 Python package). Cannot cross-validate tick data from the broker used for live execution. | MT5 API testing during investigation | Validation source limited |
| 18 | **Data source discrepancy between MT5 and Exness.** M1 bars from MT5 around the original trade (17:00-17:03 UTC July 21) show EURJPY at 185.88-185.91 while live entry was at 185.944 — a 3+ pip difference. The live feed and historical MT5 data do not match for the same timestamp. | Manual comparison of live entry vs MT5 CopyRates | Data integrity uncertain |
| 19 | **The original STRATEGY_SPEC.md overfit claims were misleading.** "Survives all execution degradation," "100% of 36 parameter combinations profitable," "Random baseline 72.5% WR confirms trailing stop edge" — all based on bar-level model that was structurally optimistic. The spec proclaimed robustness against problems that don't exist at bar-level (entry delay, wider stops) but was blind to the actual problem (tick-vs-bar execution gap). | Post-hoc comparison | False confidence |
| 20 | **Tick-level fast fade has high WR but too few trades.** 10+pip moves in 5s, hold 30s: WR=69.6% but only 23 trades in 3 months (~1 trade every 4 days). Cannot build a strategy on 2 trades/month. | test_microstructure.py | Interesting but not actionable |

### Bucket 5: Infrastructure Realities

| # | Problem | Details |
|---|---------|---------|
| 21 | **Exness tick data format.** ZIP files with 5 columns (E, S, Ts, B, A). Bid/ask columns contain raw prices. Spread must be computed as ask-bid. Timestamps are UTC with millisecond precision. Data available for EURUSD, EURJPY, GBPJPY only for Oct-Dec 2025. | |
| 22 | **EURUSD average spread is 0.03 pips** (from actual tick data). Original spec claimed 0.15 MP which was 5× too high for the actual market. This made EURUSD appear less attractive than it actually is. | |
| 23 | **EURJPY average spread is 60 MP (0.6 pips)** with p99 of 600 MP (6 pips). Spread spikes during news events make the strategy even more expensive during the periods when signals are most abundant. | |
| 24 | **2.87M ticks per 3 months per pair.** ~32,000 ticks/day, ~22 ticks/minute, ~2.7 ticks/second average. During active periods, 50+ ticks/second. The tick backtester must iterate all 2.87M for a single run. Parameter sweeps need optimization (vectorized, single-pass signal collection.) | |

### Bucket 6: What We Built That Still Works

| # | Asset | Details |
|---|-------|---------|
| 25 | **tick_backtest.py** — Tick-level backtester using PairState + TrailingStopManager directly. CLI: `python tick_backtest.py EURUSD [--config] [--invert]`. Programmatic: `backtest_ticks()`, `load_ticks()`, `summary()`, `scan()`. Uses real Exness bid/ask data. Same code path as live. | |
| 26 | **Trade-by-trade PnL comparison** — `_tick_backtest_comparison.py` shows which trades flip win/loss between bar and tick models. Diagnostic tool for understanding friction impact. | |
| 27 | **Fast parameter scan framework** — `scan()` function in tick_backtest.py, plus the single-pass + simulate-exits optimization pattern. Enables efficient exploration of large parameter grids. | |
| 28 | **PairState + BarBuilder + TrailingStopManager** — The live strategy code is clean, modular, and reusable as-is for v2. The execution model (entry/exit/trail) stays unchanged; only the signal logic needs modification. | |

---

## Key Quantitative Truths

These numbers survived all testing and are the bedrock facts for v2 research:

- **Bar-level WR for z>2 reversal with trailing stop: 67.6% (EURJPY)**
- **Tick-level WR for the same: 27.8% (EURJPY)** — 40pp gap
- **M1 z>2 predicts next bar direction at: 50.8%** — no edge
- **M5 z>2 predicts next M5 bar direction at: 53.5%** — weak edge
- **M1 z-score autocorrelation: -0.013** — no persistence, signal is noise at tick level
- **Trailing stop at tick level with random direction: ~28% WR** — the trailing stop is a liability, not an edge
- **EURUSD spread: 0.03 pips average** — viable for thin-edge strategies
- **EURJPY spread: 0.6 pips average** — too expensive for this signal
- **Fast fade 10p in 5s: 69.6% WR** — strongest clue, but only 23 trades
- **EURUSD z>2.5 fixed hold 54min: +271p/3mo** — only positive config, but monthly inconsistent

## v2 Research Direction: Shock Reaction Framework

### Core Hypothesis

After extreme M1 anomalies, the tick-level reaction in the first 5-60 seconds contains exploitable information about whether the shock will revert or continue. A **reaction classification layer** between signal detection and execution can separate:
- **Liquidity shocks** (no information, reverts) → good fade
- **Information shocks** (new information, continues) → bad fade

### New Architecture

```
M1 anomaly detector (z-score)
        ↓
     Wait 30 seconds
        ↓
Microstructure reaction classifier
  ├─ retracement velocity
  ├─ tick imbalance (bid/ask pressure)
  ├─ failed extreme attempts
  ├─ spread percentile
  ├─ volatility decay
  └─ recovery efficiency
        ↓
Decision: FADE or SKIP
        ↓
Short-horizon execution (no trailing stop)
  30s / 60s / 120s fixed hold
```

### Research Questions (RQs)

#### RQ1: What happens after M1 z>2.5?
Measure the tick-level reaction at 5s, 15s, 30s, 60s, 5m post-signal:
- Retracement % of the initial impulse
- Tick imbalance (bid/ask ratio)
- Failed extreme attempts (ticks testing the extreme but failing)
- Spread behavior (widening/narrowing)
- Volatility decay (is ATR of subsequent ticks decreasing?)

#### RQ2: Can we classify event types?
Build features from RQ1 and cluster into:
- Liquidity shocks (high retracement velocity, declining extreme attempts, spread normalizing)
- Information shocks (low retracement, sustained extreme attempts, spread widening)
- Continuation events (price breaking further, high tick imbalance)

#### RQ3: Do exhaustion events produce positive expectancy?
For events classified as "liquidity shock / exhaustion":
- Trade fade direction
- Fixed holds: 30s, 60s, 120s
- No trailing stop
- Test on EURUSD first (0.03 pip spread), then EURJPY

#### RQ4: Walk-forward validation
- Train classification on Oct 2025 data
- Validate on Nov 2025
- Test on Dec 2025
- Minimum 100 trades per month before conclusions

#### RQ5: Feature importance
- Which features most predict successful vs failed fades?
- Is one feature sufficient (e.g., retracement velocity alone)?
- Or is multivariate classification required?

### Methodology

1. **Data**: Exness ticks (EURUSD, EURJPY, Oct-Dec 2025) — same as tick backtest
2. **Detection**: PairState z-score + ATR gate (unchanged from v1)
3. **Reaction phase**: For each detected anomaly, extract tick data from the 30s window post-signal
4. **Feature engineering**: Compute microstructure features from raw tick data
5. **Classification**: Simple rule-based (initially), then supervised if data supports it
6. **Execution**: Fixed hold, no trailing stop, tick-level bid/ask entry/exit
7. **Validation**: Monthly walk-forward, minimum 100 trades

## Tick Backtester Engine (Aligned to Live)

The tick backtester (`tick_backtest.py`) is the ground truth for all v2 research. It runs the **exact same Python code path** as live execution:
- `PairState` from `strategy.py` — z-score/ATR signal generation (same as live)
- `TrailingStopManager` from `strategy.py` — entry/exit/trailing (same as live)
- Real Exness bid/ask tick data — not synthetic ticks, not bar H/L approximations
- Entry at real tick prices (ask for LONG, bid for SHORT)
- Exit at real tick prices (bid for LONG, ask for SHORT)
- Spread cost deducted per trade (0.03 pips EURUSD, 0.5 pips EURJPY)

**Verified match**: The tick backtester was validated against the live paper trade deployment. The 40% WR gap between bar-level and tick-level backtests was confirmed by live trading — meaning the tick backtester is a faithful representation of real execution.

Any v2 strategy MUST pass tick-level validation before being deployed live. Bar-level proof-of-concept is acceptable for initial exploration, but the final verdict comes from the tick backtester.

### Files

```
paper_trade/strategies/m1_z_reversal/
├── RESEARCH_PLAN_v2.md        # This document
├── strategy.py                # v1 strategy (unchanged)
├── tick_backtest.py           # Tick-level backtester (reusable)
├── run.py                     # v1 live runner (unchanged)
├── STRATEGY_SPEC.md           # v1 spec + section 7 tick findings
└── TICK_BACKTEST_FRAMEWORK.md # Tick backtest framework docs
```

### Constraints

- ✅ Tick-level validation only (no bar H/L assumptions)
- ✅ Realistic bid/ask execution with spread cost
- ✅ No trailing stops
- ✅ Minimum 100 trades before conclusions
- ✅ Prefer fewer robust features over large optimization grids
- ❌ No more z-threshold tuning (already disproven)
- ❌ No more trailing stop tuning (already disproven)
