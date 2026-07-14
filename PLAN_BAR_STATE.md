# Bar State Engine — Implementation Plan

## Architecture Overview

Replace SWPS with a **Bar State Engine** that computes 5-min trend state from cached M1 bars in TickStore. No timers, no M5 subscriptions — just partition existing M1 bars into 5-bar blocks and run WLS.

```
TickStore (M1 bars, maxlen=100 per symbol)
    │
    ├── bar[0..4]   → 5-min block 1  → WLS solve → strength vector 1
    ├── bar[5..9]   → 5-min block 2  → WLS solve → strength vector 2
    ├── bar[10..14] → 5-min block 3  → WLS solve → strength vector 3
    ├── bar[15..19] → 5-min block 4  → WLS solve → strength vector 4
    └── bar[20..24] → 5-min block 5  → WLS solve → strength vector 5
                             │
                             ▼
                Per-currency trajectory (last 5)
                        │
                        ▼
                 State metrics:
                 - Direction (slope sign)
                 - Consistency (fraction aligned)
                 - Momentum (delta / |prev|)
                 - Stability (variance of 5)
                 - Percentile (position in distribution)
```

## New File

**`currency_decomposition/features/bar_state.py`** — BarStateEngine class.

### Data Flow

```
Manager.__init__():
  self.bar_state = BarStateEngine(self.store)

Manager._process_batches()  (every 30s decision cycle):
  # After DER filter, before DRS rank
  if self.bar_state.update():
      for h in hypotheses:
          alignment = self.bar_state.alignment(h)
          h.confidence *= alignment
          if alignment < 0.20:
              print(f"[BAR-STATE] reject={h.symbol} align={alignment:.3f}", file=sys.stderr)

  hypotheses = [h for h in hypotheses if h.confidence >= MIN_CONFIDENCE]

Manager._reset_after_profit_target():
  # store.clear() already called — bar_state auto-rebuilds from empty cache
  self.bar_state.reset()
```

## BarStateEngine API

### `__init__(store)`
- Store reference to TickStore
- Separate WLSSolver instance with `lam=0.05`
- `_strength_history: dict[str, deque]` — per-currency last 5 bar-level strengths
- `_ready: bool = False`

### `update() -> bool`
Called every decision cycle. Returns True if bar state is ready.

```
1. For each symbol in SYMBOLS:
   - Get M1 bars from store._bars[symbol]
   - If len < 5: skip (not enough for even 1 block)
   - Divide into floor(len(bars) / 5) complete 5-bar blocks
   - For each NEW block (tracked by index):
       return = ln(close_of_block_5th_bar / close_of_block_1st_bar)
       Add to block_returns[block_index][symbol]

2. For each completed block:
   - Run WLSSolver.solve(block_returns[block_index])
   - Append strength vector to _strength_history per currency

3. Trim each currency to last 5 entries

4. If all 8 currencies have >= 3 entries → _ready = True
```

### `_compute_state() -> dict`

From `_strength_history[ccy]` (list of 5 floats):

```
state[ccy] = {
    "direction": sign of linear slope (+1, -1, 0),
    "consistency": fraction of last 5 bars aligned with direction,
    "slope": slope value (strength change per bar),
    "momentum": (current - prev_but_one) / abs(prev_but_one + eps),
    "stability": 1.0 - (variance of 5 / max_variance),
    "position": current percentile in [min_of_5, max_of_5],
    "current": latest strength value,
}
```

### `alignment(hypothesis) -> float`

For a hypothesis with `symbol=ABC/DEF`, `direction=+/1`:

```
base_ccy, quote_ccy = decompose(symbol)
base_state = self._state.get(base_ccy, {})
quote_state = self._state.get(quote_ccy, {})

# Direction match: does tick direction align with bar trend?
expected = +1 if hypothesis.direction > 0 else -1
# Long ABC/DEF = long base, short quote
base_aligned = sign(base_state.get("direction", 0)) == expected
quote_aligned = sign(quote_state.get("direction", 0)) == -expected

direction_score = 0.5 if base_aligned else 0.0
direction_score += 0.5 if quote_aligned else 0.0

# Weight by consistency (bar trend strength)
base_weight = base_state.get("consistency", 0.5)
quote_weight = quote_state.get("consistency", 0.5)

# Extreme penalty: avoid entering when currency is stretched
base_extreme = abs(base_state.get("position", 0.5) - 0.5) * 2  # 0 = neutral, 1 = extreme
quote_extreme = abs(quote_state.get("position", 0.5) - 0.5) * 2
extreme_penalty = 1.0 - max(base_extreme, quote_extreme) * 0.3  # max -30%

# Final
alignment = direction_score * ((base_weight + quote_weight) / 2) * extreme_penalty

# Sanity: if bar trend is very consistent (>0.8) and aligned → boost up to 1.3
if alignment > 0.6:
    boost = 1.0 + min(base_weight, quote_weight) * 0.3
    alignment *= boost

return max(0.05, min(alignment, 1.3))
```

### `reset()`
- Clear `_strength_history`
- `_ready = False`
- (store already cleared by reset_after_profit_target)

## Changes to Existing Files

### `currency_decomposition/runtime/manager.py`

1. **Import**: add `from features.bar_state import BarStateEngine`
2. **`__init__`**: add `self.bar_state = BarStateEngine(self.store)`
3. **Pipeline section (~line 328)**: Replace the SWPS block with:

```python
            # ── BAR STATE ALIGNMENT (replaces SWPS) ──────────────
            if self.bar_state.update():
                for h in hypotheses:
                    align = self.bar_state.alignment(h)
                    h.confidence = min(1.0, h.confidence * align)
                    if align < 0.20:
                        print(f"[BAR STATE] reject={h.symbol} align={align:.3f}", file=sys.stderr)
                hypotheses = [h for h in hypotheses if h.confidence >= MIN_CONFIDENCE]
                print(f"[BAR STATE] aligned={len(hypotheses)}/{self._pipeline_metrics['burst_hyp']}", file=sys.stderr)
            else:
                print("[BAR STATE] not ready — no bar alignment", file=sys.stderr)
            self._pipeline_metrics["bar_aligned"] = len(hypotheses)
```

4. **`_reset_after_profit_target`**: add `self.bar_state.reset()`

5. **Imports to remove**: 
   - `from direction.short_window_persistence import ShortWindowPersistenceScanner`
   - `from config.settings import SWPS_MIN_SCORE, SWPS_WINDOW_SIZE`

6. **Init vars to remove**:
   - `self.swps`
   - `self._strength_capture`
   - `self._swps_pick`

7. **Pipeline cleanup**: Remove the entire SWPS override block (lines ~358-400)

8. **Dashboard render**: Remove `swps_signal`, `swps_capture_count`

### `currency_decomposition/config/settings.py`
- Remove `SWPS_WINDOW_SIZE` and `SWPS_MIN_SCORE`

### `currency_decomposition/monitoring/dashboard.py`
- Remove swps-related display
- Add bar_state display block

## Alignment vs SWPS Comparison

| Aspect | SWPS (old) | Bar State (new) |
|--------|-----------|-----------------|
| Data source | Tick-level WLS strengths (5s) | M1 bar cache → 5-min WLS |
| Persistence | Flip count + direction ratio | Consistency fraction over 5 bars |
| Horizon | ~25-30s (5-6 snapshots) | ~25 min (5 × 5-min blocks) |
| Market state | No | Yes (trend, momentum, percentile) |
| Entry override | Binary (pick symbol) | Continuous (confidence multiplier) |
| Exhaustion protection | No | Yes (extreme-zone penalty) |
| Reset behavior | Manual clear | Auto from cache (store.clear) |

## Validation Plan

After implementation, shadow-compare by running live and checking:

1. **Bar state readiness time**: how many M1 bars needed before `_ready`
2. **Alignment distribution**: histogram of alignment scores at decision cycles
3. **Rejection rate**: % of hypotheses filtered by alignment < 0.20
4. **Correlation check**: max same-currency exposure of selected positions vs before
5. **Trade frequency**: trades/day vs before
