# Tick-Level Backtest Framework

## Purpose

The tick-level backtester (`tick_backtest.py`) runs the **exact same code path** as the live strategy (`strategy.py`) against recorded tick data. It bridges the gap between bar-level backtests (which model execution at bar boundaries) and real tick-level execution (which happens at random timestamps within the bar).

**Critical finding**: bar-level backtests can be structurally optimistic by 40% WR or more for tight-stop mean reversion strategies. The tick-level gap is not noise — it's a systematic overestimate.

## Architecture

```
tick_backtest.py  →  PairState  →  signal (same as live)
                  →  TrailingStopManager → closed trades (same as live)
                  →  backtest_ticks() returns trade list with PnL
```

The backtester reuses:
- `PairState` from `strategy.py` — z-score, ATR, signal generation, bar accumulation
- `TrailingStopManager` from `strategy.py` — entry, trailing stop, expiry
- `CONFIG` from `strategy.py` — default parameters

The only additions in the backtester:
- Tick loading from Exness ZIP files
- M1 bar seeding from ticks (to warm up rolling windows)
- Cost deduction at trade level (`COST_RAW = 0.005` = 50 MP = 0.5 pips for JPY pairs)
- `_direction_mult` config flag for direction flips

## Output

Each trade returned by `backtest_ticks()`:
```python
{
    'bar_time': int,           # bar timestamp
    'dir': 1 or -1,            # trade direction
    'entry': float,            # entry price (ask for LONG, bid for SHORT)
    'exit': float,             # exit price (bid for LONG, ask for SHORT)
    'pnl': float,              # realized PnL (raw price units, minus cost)
    'z': float,                # z-score at entry
    'atr': float,              # ATR at entry
    'entry_time': int,         # entry unix timestamp
    'exit_time': int,          # exit unix timestamp
    'dur_bars': float,         # trade duration in minutes
    'exit_reason': str,        # 'stop' or 'expiry'
}
```

## Usage

### Basic
```bash
cd paper_trade/strategies/m1_z_reversal
python tick_backtest.py              # EURJPY, Oct-Dec 2025
python tick_backtest.py EURUSD       # specific pair
python tick_backtest.py GBPJPY       # specific pair
```

### Direction flip (momentum test)
```bash
python tick_backtest.py EURJPY --invert
```

### Custom parameters
```bash
python tick_backtest.py EURJPY --config z_thresh=2.5,min_stop_pips=3.0
python tick_backtest.py EURJPY --invert --config z_thresh=2.5,min_stop_pips=3.0
```

### Programmatic
```python
from tick_backtest import backtest_ticks, load_ticks, summary

ticks = load_ticks("EURJPY")
trades = backtest_ticks(ticks, config={"z_thresh": 2.5, "min_stop_pips": 3.0})
summary(trades)

# Parameter scan
from tick_backtest import scan
grid = [
    {"z_thresh": 2.0, "min_stop_pips": 1.5},
    {"z_thresh": 2.5, "min_stop_pips": 3.0},
]
scan(grid, ticks=ticks, pair="EURJPY")
```

## Data

Exness tick data from `data/exness_ticks/`:
- Format: ZIP files named `{PAIR}_Raw_Spread_{YEAR}_{MONTH:02d}.zip`
- Columns: `E` (ignored), `S` (ignored), `Ts` (timestamp), `B` (bid), `A` (ask)
- Available pairs: EURUSD, EURJPY, GBPJPY
- Coverage: October–December 2025
- `load_ticks()` loads all months and concatenates sorted by timestamp

## Key Findings

### 1. Bar-level backtests overestimate tick-level WR by ~40%

| Metric | Bar-level (M1) | Tick-level | Gap |
|--------|----------------|------------|-----|
| WR | 67.6% | 27.8% | -39.8pp |
| PnL | +37.2 pips | -54.1 pips | -91.3 pips |

### 2. 41.8% of trades flip win/loss between bar and tick execution

The 0.3–1.0 pip difference between bar.close and the actual tick entry/exit is enough to change the outcome of nearly half the trades. This is systematic — the tight trailing stop makes the strategy hypersensitive to entry price.

### 3. The z>2.0 signal has only ~50% direction accuracy at tick level

Without a trailing stop (fixed hold time), WR is 51.5% regardless of hold time (5–240 min). But losses are 5× larger than wins (avg loss -5.5 pips, avg win +1.1 pips), producing -86 pips over 3 months regardless of hold duration. The signal is weakly predictive at tick-level granularity.

### 4. No parameter combination is profitable

Every combination of z_thresh (1.5–3.5), min_stop_pips (1.5–20), stop_a (0.15–0.50), and hold time (5–240 min) produces negative PnL at tick level. The best result is z=3.5, 20-pip stop: -5.8 pips across 574 trades (WR=49.1%).

### 5. The trailing stop is the primary edge, not direction

The asymmetric exit (tight stop, let winners run) explains 72.5% of the bar-level WR. Direction (mean reversion vs momentum) adds only ~3-4pp. At tick level, the trailing stop edge is destroyed by execution friction because the stop is triggered before the reversion completes.

### 6. Momentum (direction flip) does NOT fix the strategy

Flipping from mean reversion to momentum also loses at tick level (WR=24.3%, same PnL magnitude). The earlier "flipped PnL" estimate was incorrect because it ignored different entry prices (ask vs bid) and trailing stop behavior changes.

## Implications

1. **Any strategy with <1 pip avg edge and tight stops needs tick-level validation.** Bar-level backtests are not sufficient.
2. **To survive tick-level friction**, a strategy needs either:
   - Higher signal accuracy (>60% direction at tick level)
   - Larger per-trade wins (wider stops or longer holds)
   - Lower-cost instruments (EURUSD at 0.15 MP is cheaper than EURJPY at 50 MP)
3. **The tick backtester framework is reusable** for any strategy using PairState + TrailingStopManager. Adding a new strategy means swapping the signal generation; the execution model stays the same.
