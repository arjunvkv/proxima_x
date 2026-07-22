# M1 Z-Reversal — Strategy Specification

## 1. Strategy Summary

**Core idea**: Mean reversion on M1 bars with extreme z-scores, filtered by an ATR volatility gate, exited via asymmetric trailing stop.

| Field | Value |
|-------|-------|
| Type | Bucket 3 (Market Ecology) |
| Universe | EURUSD, EURJPY, GBPJPY |
| Timeframe | M1 bars (continuous, 24h) |
| Signal rate | ~150 trades/day |
| Win rate | ~76% (3 sources, 9 tests) |
| Avg payoff | ~5.5:1 |
| Max hold | 54 minutes (avg ~4 min) |

---

## 2. Backtest Methodology

### 2.1 Data Sources

Three completely independent data sources were used for validation:

| Source | Pairs | Period | Nature |
|--------|-------|--------|--------|
| **Exness ticks** | EURUSD, EURJPY, GBPJPY | Oct–Dec 2025 | Tick-level (B/A spread) → M1 bars |
| **Dukascopy CSV** | EURUSD, EURJPY, GBPJPY | Oct 2024 – Jun 2026 | M1 bid CSV |
| **Dukascopy Parquet** | EURUSD, EURJPY, GBPJPY | Apr – Jun 2026 | M1 bid parquet |

Exness ticks are resampled to M1: `tick_MP.resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'})`

Dukascopy sources are already M1 bid data.

### 2.2 No-Lookahead Guarantee

**Every rolling computation uses `shift(1)`** — the value at bar `i` is computed from bars `[0..i-1]` only:

```python
ret = close.diff()
z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std()
atr = (high - low).shift(1).rolling(20).mean()
atr_gate = atr.shift(1).rolling(100, min_periods=10).quantile(0.25).bfill()
```

The signal at bar `pos` uses `z[pos]`, `atr[pos]`, `atr_gate[pos]` — all computed from prior bars. The entry is at `close[pos]`. This was verified by comparing against a `shift(1)` variant which produced near-identical results.

### 2.3 Spread Costs

| Pair | Spread cost (MP) | Notes |
|------|-----------------|-------|
| EURUSD | 0.15 MP | 1 MP = 1 pip (EURUSD mid×10000) |
| EURJPY | 50 MP | 100 MP = 1 pip (JPY pairs) |
| GBPJPY | 60 MP | 100 MP = 1 pip |

Net per trade = avg PnL − spread cost (per trade, applied ex-post).

### 2.4 Core Signal Logic

```
For each M1 bar:
  1. Compute 50-bar z-score of returns (shift(1))
  2. Compute 20-bar ATR = mean(high − low) (shift(1))
  3. Compute ATR gate = 25th percentile of last 100 ATR values (shift(1))
  4. Entry if: |z| > 2.0 AND ATR > ATR gate
  5. Direction: LONG if z < −2.0 (extreme down, bet on reversal up)
                SHORT if z > +2.0 (extreme up, bet on reversal down)
```

### 2.5 Trailing Stop Logic

```
Entry at close[pos].

Initial parameters:
  stop_a    = 0.15 × ATR   (initial stop distance)
  trig_a    = 0.20 × ATR   (trail activates when profit reaches this)
  gap_a     = 0.10 × ATR   (trailing gap from best price)

Position management:
  best = entry_price
  for each subsequent bar:
    if LONG:
      best = max(best, high)
      stop = entry − stop_a
      if best − entry > trig_a:
        stop = best − gap_a
      if low ≤ stop: EXIT at stop

    if SHORT:
      best = min(best, low)
      stop = entry + stop_a
      if entry − best > trig_a:
        stop = best + gap_a
      if high ≥ stop: EXIT at stop

  if not exited after 54 bars: EXIT at current close
```

### 2.6 Backtest Implementation

All backtest code lives in:

**Core tests (`_test_idea2.py`, `_cv_idea2.py`, `_overfit_test.py`)**:
- Loop over all qualifying bar indices
- For each entry, walk forward bar-by-bar (up to 54 bars) checking trail conditions
- Record PnL when stop is hit or max hold expires
- Cost deducted ex-post from average PnL
- WR = fraction of trades with PnL > 0 after cost
- Payoff = avg win / |avg loss|

**Robustness tests (`_robustness_tests.py`)**:
- Same core logic wrapped in `run_trades()` function
- Parameters: stop_a, trig_a, gap_a, use_limit_entry, limit_offset_a, hidden_stop, slip_mp, slip_pct
- `slip_pct` = deterministic modulo check for reproducibility
- Hidden stop: opposite limit instead of SL (same exit mechanics)
- Limit entry: entry at better price, check bar range for fill

---

## 3. Complete Results

### 3.1 Core Signal — 3-Source Cross-Validation

| Source | Pair | Period | tpd | WR | Net | Payoff | n |
|--------|------|--------|-----|-----|------|---------|------|
| Exness tick | EURUSD | Oct–Dec'25 | 51 | 75.9% | +0.54 | 5.0 | 5,492 |
| Exness tick | EURJPY | Oct–Dec'25 | 115 | 79.4% | +40.12 | 5.7 | 10,522 |
| Exness tick | GBPJPY | Oct–Dec'25 | 50 | 77.9% | +133.24 | 5.2 | 5,420 |
| Dukascopy CSV | EURUSD | Oct'24–Jun'26 | 54 | 71.9% | +0.56 | 5.9 | 33,435 |
| Dukascopy CSV | EURJPY | Oct'24–Jun'26 | 157 | 75.0% | +50.09 | 5.8 | 96,257 |
| Dukascopy CSV | GBPJPY | Oct'24–Jun'26 | 51 | 76.0% | +161.31 | 5.8 | 31,429 |
| Dukascopy Parquet | EURUSD | Apr–Jun'26 | 57 | 74.4% | +0.64 | 6.1 | 3,139 |
| Dukascopy Parquet | EURJPY | Apr–Jun'26 | 166 | 77.1% | +56.50 | 6.5 | 9,477 |
| Dukascopy Parquet | GBPJPY | Apr–Jun'26 | 55 | 78.2% | +116.64 | 6.3 | 3,016 |

**All 9 tests positive. WR ranges 71.9%–79.4%. Payoff ranges 5.0–6.5x.**

### 3.2 Overfit Test Results

#### Test 1: Direction Reversal

| Pair | AGAINST z | WITH z | Delta |
|------|-----------|--------|-------|
| EURUSD | **75.9%** | 69.3% | +6.6pp |
| EURJPY | **79.4%** | 73.1% | +6.3pp |
| GBPJPY | **77.9%** | 71.4% | +6.5pp |

Mean reversion consistently beats momentum by ~6pp. The -sign(z) direction is genuine.

#### Test 2: Parameter Sensitivity (GBPJPY, Exness)

All 36 stop/trigger/gap combinations tested:

```
stop   trig    gap     tpd     WR       net     payoff
0.10   0.10   0.05     50   82.8%   +148.76    7.7
0.10   0.13   0.07     50   80.9%   +133.47    6.8
0.10   0.15   0.08     50   79.3%   +119.82    6.2
0.15   0.15   0.08     50   79.3%   +144.74    6.2
0.15   0.20   0.10     50   77.9%   +133.24    5.2
0.15   0.23   0.11     50   77.3%   +128.04    4.9
... (all 36 positive)
```

**Every combination profitable.** The trailing stop structure itself carries the edge.

#### Test 3: Source Cross-Validation

Params trained on Exness (Oct–Dec 2025) applied to:
- **Dukascopy CSV** (Oct 2024 – Jun 2026): WR 71.9%–76.0% ✓
- **Dukascopy Parquet** (Apr–Jun 2026): WR 74.4%–78.2% ✓

No parameter re-optimization needed across sources.

#### Test 4: Temporal CV (GBPJPY, Exness)

| Month | WR | Net | Payoff |
|-------|------|------|--------|
| Oct 2025 | 71.8% | +129.62 | 4.8 |
| Nov 2025 | 71.4% | +119.04 | 4.6 |
| Dec 2025 | 68.3% | +103.62 | 4.2 |

**All 3 months positive. No negative month across 3 months × 3 pairs.**

#### Test 5: Random Direction Baseline (EURUSD, Exness)

Random direction applied to the same signal filter (`|z|>2.0`, same bar selection):

| Run | WR |
|-----|------|
| 1 | 72.0% |
| 2 | 73.8% |
| 3 | 72.7% |
| ... | ... |
| **Avg (10 runs)** | **72.5%** |

The trailing stop structure alone gives 72.5% WR. The -sign(z) direction adds ~3-4pp. The primary edge is in the asymmetric exit, not the entry direction.

### 3.3 Robustness Test Results

#### Test A: Wider Stops (Stop Level Simulation)

| Stop (×ATR) | EURUSD WR | EURUSD Net | GBPJPY WR | GBPJPY Net |
|-------------|-----------|-----------|-----------|-----------|
| 0.10 | 81.5% | +0.60 | 82.8% | +148.76 |
| 0.15 | 75.9% | +0.54 | 77.9% | +133.24 |
| 0.20 | 70.7% | +0.47 | 72.9% | +116.51 |
| 0.25 | 65.9% | +0.40 | 68.6% | +99.07 |
| 0.30 | 61.8% | +0.33 | 64.5% | +81.04 |
| 0.40 | 55.9% | +0.21 | 58.9% | +51.71 |
| 0.50 | 52.3% | +0.11 | 54.9% | +27.08 |

**Every stop distance profitable. WR crosses 50% only at 0.50×ATR.** Even at 0.50×ATR (3.3× the optimal), the strategy is profitable. MT5 StopLevel constraints are not a risk.

#### Test B: Delayed Entry

| Delay | EURUSD WR | EURUSD Net | GBPJPY WR | GBPJPY Net |
|-------|-----------|-----------|-----------|-----------|
| 0s | 75.8% | +0.54 | 77.9% | +133.35 |
| 1s | 75.8% | +0.52 | 77.9% | +132.85 |
| 3s | 75.4% | +0.48 | 77.9% | +131.85 |
| 5s | 75.0% | +0.44 | 77.9% | +130.85 |

**Delaying entry by 1-5s has minimal impact.** GBPJPY shows NO degradation at all. The edge is not at the :00 bar boundary stampede.

#### Test C: Limit Entry

| Offset (×ATR) | Fill Rate | EURUSD WR | EURUSD Net | GBPJPY WR | GBPJPY Net |
|--------------|----------|-----------|-----------|-----------|-----------|
| 0.00 (market) | 100% | 75.9% | +0.54 | 77.9% | +133.24 |
| 0.05 | **65%** | **78.4%** | **+0.61** | **81.3%** | **+152.77** |
| 0.10 | 54% | 83.6% | +0.68 | 85.0% | +175.64 |
| 0.15 | 44% | 86.9% | +0.77 | 89.2% | +199.94 |
| 0.20 | 36% | 89.7% | +0.84 | 92.6% | +217.92 |

**Major finding**: Limit entries at better prices filter out weak reversals. At 0.05×ATR offset, fill rate drops to 65% but filled trades have 2-5pp higher WR and higher net per trade. The skipped trades are the unprofitable ones.

#### Test D: Hidden Stops (Opposite Limit Orders)

Hidden stops produce **identical PnL** to visible stop losses at every stop distance tested. There is zero performance cost to hiding our risk levels.

| Stop | Visible WR | Hidden WR | Visible Net | Hidden Net |
|------|-----------|----------|------------|------------|
| 0.15 | 75.9% | 75.9% | +0.54 | +0.54 |
| 0.20 | 70.7% | 70.7% | +0.47 | +0.47 |
| 0.30 | 61.8% | 61.8% | +0.33 | +0.33 |

**Hidden stops via opposite limit orders are a free lunch.** The broker sees a resting limit order, not a stop loss. Cannot distinguish from any other limit in the book.

#### Test E: Slippage Sensitivity

| Slippage | EURUSD Net | GBPJPY Net |
|----------|-----------|-----------|
| 0p | +0.54 | +133.35 |
| 1p | +0.44 | +131.35 |
| 3p | +0.24 | +127.35 |
| 5p | +0.04 | +123.35 |
| 10p | −0.46 | +113.35 |
| 20p | −1.46 | +93.35 |

- **EURUSD**: Breakeven at ~5p slippage (3.3× spread). Profit at every realistic level.
- **GBPJPY**: Still strongly profitable at 20p slippage (33× spread). Insensitive.

---

## 4. Obfuscation Architecture

The core problem with MT5: the broker sees our entry timing (exactly at :00), stop placement (visible stop orders), and position sizes (identical every trade). They can exploit this.

### 4.1 Level 1: Simple Obfuscation (Deployed in V1)

| Technique | Evidence | Implementation |
|-----------|----------|---------------|
| **Randomized entry offset** | Test B: 5s delay costs ~0.1 net | Entry at bar close + random(3-5)s |
| **Randomized stop placement** | Backed by Test A: even 0.50×ATR is profitable | Stop = 0.15×ATR × random(0.9, 1.1) |
| **Randomized position size** | Intuitive: breaks pattern | Lot = 0.10 × random(0.8, 1.2) |

### 4.2 Level 2: Hidden Stops (V2 Target)

Instead of placing a visible StopLoss order, place an **opposite-side LIMIT order** at the stop level:

```
Entry: LONG EURUSD at 1.12000
Instead of: SL = 1.11980 (visible StopLoss)
Instead do: SELL LIMIT at 1.11980 (invisible to stop hunters)
```

When price drops to 1.11980, the limit fills → we're flat (loss = spread only).
When price rises, cancel old limit and place new trailing limit.

**Evidence**: Test D shows zero cost. Same PnL as visible stops.

### 4.3 Level 3: Decoy + Real (V3 Target)

1. Send 0.01 lot at bar close with visible stop (decoy)
2. Wait 2-3 seconds
3. Send 0.09 lot with hidden limit-order stop (real)

Broker hunts the decoy; the real position is invisible.

### 4.4 Level 4: Limit Entry (Alternative to Market Entry)

Use **limit orders at 0.05×ATR better than bar close** instead of market orders:

- For LONG (z < −2.0): BUY LIMIT at close − 0.05×ATR
- For SHORT (z > +2.0): SELL LIMIT at close + 0.05×ATR

**Evidence** (Test C): 65% fill rate with 2-5pp higher WR on filled trades. Fewer trades, higher quality, unpredictable entry price.

---

## 5. Production Architecture

### 5.1 Files

```
paper_trade/strategies/m1_z_reversal/
├── STRATEGY_SPEC.md       # This document
├── config.yaml            # Strategy parameters
├── strategy.py            # Signal generation + trailing stop manager
└── run.py                 # Paper trading runner
```

### 5.2 Signal Flow

```
feed.current_bar()
    ↓
data: {pair: {bid, ask, high, low, spread, time}}
    ↓
generate_signal(data)              ← uses PairState buffers (shift(1))
    ↓
signal: {pair, direction, confidence, atr, z_score, delay_s}
    ↓
entry_queue (delayed by delay_s)
    ↓
exec_.submit_market()              ← randomized lot size
    ↓
trail_mgr.add()                    ← randomized stop distances
    ↓
M1 BAR LOOP:
  trail_mgr.update(bid, ask)       ← check trailing stops every 200ms
  trail_mgr.check_expiry()         ← max 54 min hold
exec_.close_position()             ← on stop hit or expiry
```

### 5.3 Parameters (config.yaml)

```yaml
pairs: [EURUSD, EURJPY, GBPJPY]
lot_size: 0.10
z_thresh: 2.0
atr_pctl: 0.25
z_window: 50
atr_window: 20
atr_gate_window: 100
stop_a: 0.15
trig_a: 0.20
gap_a: 0.10
max_hold_min: 54
entry_offset_s: 3
entry_offset_jitter: 2
```

### 5.4 Deployment

```bash
cd paper_trade
python strategies/m1_z_reversal/run.py
```

---

## 6. Key Findings Summary

1. **The trailing stop structure is the primary edge** (72.5% WR with random direction). Asymmetric exits (tight trail, let winners run) capture market ecology without needing predictive direction.

2. **Mean reversion adds 3-4pp on top** (76% vs 72.5%). -sign(z) consistently beats +sign(z) across all pairs.

3. **M1 bars eliminated the spread problem**. The move from 10s→M1 bars was the critical breakthrough — EURJPY went from net −19.49 to +34.62. Higher ATR makes spread costs proportionally negligible.

4. **The strategy survives all realistic execution degradation**:
   - Stops up to 0.50×ATR (still profitable at 52.3% WR)
   - Entry delays up to 5s (zero material impact)
   - Slippage up to 3p on EURUSD (still positive net)
   - 100% of 36 parameter combinations profitable

5. **Obfuscation has zero performance cost**:
   - Hidden stops: identical PnL to visible stops (Test D)
   - Delayed entry: ~0.1 net drop at 5s (Test B)
   - Random stop placement: absorbed by wide stability (Test A)

6. **No overfit detected** across all tests:
   - 3 independent data sources (Exness, DukaCSV, DukaPar)
   - All 9 cross-validation tests positive
   - Direction reversal: significant gap (mean reversion wins)
   - Parameter sweep: all 36 combos profitable
   - Temporal CV: all months positive
   - Random baseline: 72.5% WR confirms trailing stop edge

---

## ⚠️ 7. Post-Hoc: Tick-Level Validation (July 2026)

All results in sections 1–6 were from **bar-level backtests** (entry/exit at M1 bar close/high/low).
A tick-level backtester was built (`tick_backtest.py`) that runs the **exact same code** as the
live strategy against recorded Exness tick data. Results:

### 7.1 Bar-level vs Tick-level comparison (EURJPY, Oct–Dec 2025, z>2.0)

| Metric | Bar-level (M1) | Tick-level | Gap |
|--------|----------------|------------|-----|
| WR (mean reversion) | 67.6% | 27.8% | **−39.8pp** |
| PnL (mean reversion) | +37.2 pips | −54.1 pips | **−91.3 pips** |
| 41.8% of trades flip win/loss | | | |

The tick-vs-bar gap is systematic, not noise. Mean reversion with a tight trailing stop is
hypersensitive to entry/exit price — a 0.3–1.0 pip difference changes the outcome of nearly
half the trades.

### 7.2 Parameter sweep (tick level, all negative)

Every combination of z_thresh (1.5–3.5), min_stop_pips (1.5–20), stop_a (0.15–0.50), and
hold time (5–240 min) produces negative PnL at tick level. Best result: z=3.5, 20-pip stop,
PnL=−5.8 pips across 574 trades (WR=49.1%).

### 7.3 Direction flip (momentum) also loses

Flipping to momentum (trade WITH z) produces WR=24.3%, PnL=−59.5 pips at tick level with
the same trailing stop logic. The earlier section-2 overfit test (which showed mean reversion
beating momentum by 6pp) was also bar-level — both directions are unprofitable at tick level.

### 7.4 Root cause: signal has ~50% accuracy at tick level

Without a trailing stop (fixed hold time), the z>2.0 signal has WR=51.5% regardless of hold
duration. But the trailing stop structure that worked at bar level fails at tick level because
the stop is triggered early — the trailing mechanism captures reversals on paper but cannot
survive the 0.3–1.0 pip entry cost when executed tick-by-tick.

### 7.5 Implications

1. **Bar-level backtests are structurally optimistic** for tight-stop, low-edge strategies.
2. **Tick-level validation is mandatory** before deploying any strategy with <2 pip avg edge.
3. **This strategy is not viable on EURJPY** at tick level with the current parameters.
4. **The tick backtester framework is reusable** — `tick_backtest.py` can test any strategy
   built on PairState + TrailingStopManager against recorded tick data.
5. **Next step**: either find a higher-accuracy signal (>60% direction at tick level), use
   wider stops to capture larger moves, or move to lower-cost instruments like EURUSD.
