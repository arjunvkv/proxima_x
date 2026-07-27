# V2+z MT5 EA Specification — Proximax V2+z Algo

## Overview
A mean-reversion / fade strategy on M1 data. Enters on every M1 bar that exceeds a z-score threshold, trades in the fade direction (short when z > 0, long when z < 0), with an ATR-based trailing stop.

## Architecture

### Pair Selection
- Deploy on **6-8 JPY-cross pairs** (EURJPY, GBPJPY, CHFJPY, NZDJPY, AUDJPY, CADJPY) + optionally EURUSD, GBPUSD
- JPY pairs preferred for higher slippage tolerance (break-even at 1-1.5p)
- All pairs run **independently** (no cross-pair correlation logic)

### Signal Generation (every M1 tick)

```
On each M1 bar (price=bid):
1. Fetch last 51 closes (current bar + 50 history)
2. Compute: ret = close - close[-1]
3. Compute: z = (ret - mean(ret[1:51])) / std(ret[1:51])
   where mean/std are of the 50 prior returns (shifted by 1)
4. If abs(z) >= Z_THRESHOLD AND trade_count_today < MAX_TRADES_PER_PAIR:
   - z > 0 → SHORT (fade the up move)
   - z < 0 → LONG (fade the down move)
```

**Critical: z-score is computed from prior 50 returns, NOT including current return.** This prevents lookahead.

### Entry
- **Order type:** Instant execution market order
- **Entry price:** Current bid (short) or ask (long)
- **Stop loss (initial):** ATR(20) × 0.15 from entry
  - `atr = iATR(_Symbol, PERIOD_M1, 20)` on prior bar close
- **Take profit:** None (trailing stop only)
- **Volume:** Fixed lot = `BASE_LOT` (default 0.01)
- **Max concurrent trades per pair:** 1 (no pyramiding)
- **Max trades per day per pair:** `MAX_TRADES_PER_PAIR` (default 100)
- **Max total daily trades across all pairs:** `MAX_TOTAL_DAILY` (default 500)

### Trailing Stop (Order Matters — Fixed)
```
On every tick for each open position:
1. Let best = position's highest high (long) or lowest low (short) since entry
2. TRAIL FIRST: 
   - If direction == LONG  AND (best - entry_price) > ATR × 0.20:
       Move SL to max(SL, best - ATR × 0.10)
   - If direction == SHORT AND (entry_price - best) > ATR × 0.20:
       Move SL to min(SL, best + ATR × 0.10)
3. THEN CHECK STOP: If current price crossed SL, close position
```

**This order is critical.** The bug found in the Python prototype checked the stop BEFORE trailing, causing premature exits and 23.8% WR instead of 59.5%.

### Exit Conditions (any triggers close)
1. **Trailing stop hit** (normal exit)
2. **Max hold bars = 54** (force close if not stopped out within ~45 min)
3. **EOD force close** at 23:50 UTC (all positions closed)

### Risk Filters

| Filter | Implementation |
|--------|---------------|
| **Daily loss limit** | Track running net PnL; if < `MAX_DAILY_LOSS`, halt all entries |
| **Max daily trades** | Per-pair counter + global counter, reset at 00:00 UTC |
| **Spread filter** | Skip entry if spread > `MAX_SPREAD` (default 3 pips for JPY pairs) |
| **News filter** | Request via `CalendarHistory` OR maintain hardcoded high-impact events; skip entry 5 min before to 5 min after |
| **Broker/server time** | Detect UTC offset via `TradingServerTimeDifference` |

### Position Sizing
```
BASE_LOT = 0.01 (default on $25k account)
Leverage: 1:100
Max active notional: ~$26,000 (0.01 lot × 6 pairs ≈ $15,600 notional JPY pairs @ ~$1,000/pair)
Risk per trade: ~$0.30-0.60 (ATR×0.15 × pip_value)
Daily max risk: ~$60-120 (at 200 trades/day) ← well within $1,250 limit
```

Optionally add a Kelly-criterion dynamic sizing:
```
Kelly fraction = (avg_win_pips / avg_loss_pips) × WR - (1 - WR)
If k > 0.10: lot = BASE_LOT × min(2.0, k / 0.10)
```

But for FundedNext compliance, fixed 0.01 lot is safest.

### Startup Sequence
```
On EA init(ExpertInitialize):
1. Verify symbol exists and is tradeable (SYMBOL_TRADE_MODE == FULL)
2. Verify free margin > MIN_MARGIN
3. Load historical rates for z-score initialization
4. Load ATR indicator handle
5. Read max daily loss / max trades from input params
6. Initialize daily counters (read from GV or file for persistence across restarts)
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Z_THRESHOLD | double | 2.0 | Minimum |z| to trigger entry |
| BASE_LOT | double | 0.01 | Fixed lot per trade |
| MAX_DAILY_LOSS | double | 500.0 | Halt if daily PnL < -500 |
| MAX_TRADES_PER_PAIR | int | 100 | Max entries per pair per day |
| MAX_TOTAL_DAILY | int | 500 | Max entries across all pairs |
| MAX_HOLD_BARS | int | 54 | Force close after N bars |
| MAX_SPREAD | double | 30.0 | Max spread in points (3 pips) |
| SL_ATR_MULT | double | 0.15 | Initial stop = ATR × this |
| TRAIL_TRIGGER_ATR | double | 0.20 | Trail activates at ATR × this |
| TRAIL_GAP_ATR | double | 0.10 | Trail gap = ATR × this |
| NEWS_BUFFER_MIN | int | 5 | Minutes before/after news to skip |

## Order of Operations (Tick Handler)

```
OnTick():  // Runs on every tick
  // Phase 1: Manage existing positions
  FOR EACH open position for this symbol:
    a. Compute current ATR(20) on M1
    b. Trail stop FIRST
    c. Check if stop hit → close
    d. Check max hold bars → close
    e. Update best price tracker
  
  // Phase 2: Consider new entry
  IF new M1 bar just formed:
    a. Compute z-score (must use prior bar close for ret calc)
    b. Check all filters:
       - Spread < MAX_SPREAD?
       - Daily loss not exceeded?
       - Daily trade count not exceeded?
       - Not in news blackout window?
    c. IF z >= Z_THRESHOLD (direction=SHORT) OR z <= -Z_THRESHOLD (direction=LONG):
       - Open market order
       - Record entry price, time
       - Increment daily counters
```

## Performance Targets (from backtest)
- **z>=2.0, 6-8 JPY pairs:** ~100-200 trades/day, ~$100-200/day on 0.01 lot
- **Expected WR:** 72-78% across JPY crosses
- **Max drawdown:** Conservatively ~$200-300/day worst case (5-8 consecutive losers at $35-40/trade)
- **Daily loss $1,250 limit:** not expected to be hit with 0.01 lot; buffer of 4-6x

## Data Persistence
- Store daily counters in `GlobalVariable` (GV) keyed by date: `"V2Z_TRADES_{YYYYMMDD}_{symbol}"`, `"V2Z_PNL_{YYYYMMDD}"`
- On EURUSD chart (or separate terminal instance) run the EA, OR run on chart 1 with `EventChartCustom` to multiplex symbols
- Alternative: run one EA instance per chart, all configured identically, on 6-8 separate charts

## Recommended Deployment
1. **IC Markets Raw ECN MT5** (0.0 pip spread, $3.5/round-turn commission on 0.01 lot ≈ $0.35 = 0.35p pip cost)
2. **VPS** in London/New York (lowest latency)
3. **6-8 charts** each running one EA instance or multiplex via `SymbolsTotal`
4. **Start at z=2.5** for safety, drop to z=2.0 if trade count is too low

## Implementation Notes
- Use `PositionSelect(Symbol)` before `PositionGetDouble/Integer` (MT5 API nuance)
- Use `NewBar` detection via `CopyTime` comparing last closed bar time to stored time
- ATR handle: `iATR(_Symbol, PERIOD_M1, 20)`
- Order send: `PositionOpen(Symbol, ORDER_TYPE_BUY/SELL, lot, price, sl, 0, comment)`
- News: maintain `MqlDateTime` array of known high-impact events, or use `TerminalInfoInteger(TERMINAL_COMMUNITY_ACCOUNT)` check (not available in all builds — fallback to hardcoded calendar)
