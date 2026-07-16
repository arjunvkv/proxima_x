# Minimum Trade Runtime (30s) — Implementation Plan

## Objective
Enforce a 30-second minimum runtime for every trade. Any close request (STOP_LOSS, PROFIT_TARGET, CHOP_DETECTED, NARRATIVE_DECAY, TAKE_PROFIT, etc.) that fires before the position is 30s old gets **deferred** and executed once the position reaches 30s.

## Design

**Deferral queue** on `RuntimeManager`:
- `_deferred_closes: dict[str, dict]` — keyed by position_id, stores reason + request_time
- `_deferred_all_reason: Optional[str]` — set when close-all is deferred

**Processing**: check queue at start of `_process_batches()` and in `_decision_worker()` idle path.

## Files to Modify

### 1. `currency_decomposition/config/settings.py`
Add one line:
```python
MIN_TRADE_RUNTIME_SECONDS = 30
```

### 2. `currency_decomposition/runtime/manager.py`

#### Change A — Import setting
Line 10: add `MIN_TRADE_RUNTIME_SECONDS` to the import.

#### Change B — Deferral fields in `__init__` (after line ~133)
```python
self._deferred_closes: dict[str, dict] = {}
self._deferred_all_reason: Optional[str] = None
```

#### Change C — Process deferred closes at top of `_process_batches()`
After `now = time.time()` (line 284), add:
```python
self._process_deferred_closes(now)
```

#### Change D — Call in `_decision_worker()` idle path
After `self.executor.sync()` (line 261), add:
```python
self._process_deferred_closes()
```

#### Change E — Gate `_close_all_positions()` (line 1156)
Check if ANY position < 30s. If so, store `_deferred_all_reason` and return.
```python
def _close_all_positions(self, reason: str) -> None:
    now = time.time()
    young = [p for p in self.executor.positions if now - p.entry_time < MIN_TRADE_RUNTIME_SECONDS]
    if young:
        self._deferred_all_reason = reason
        ages = [f"{p.symbol}={now-p.entry_time:.0f}s" for p in young]
        print(f"[DEFER CLOSE ALL] reason={reason} positions < {MIN_TRADE_RUNTIME_SECONDS}s: {', '.join(ages)}", file=sys.stderr)
        return
    # ... original method body unchanged ...
```

#### Change F — Gate `_close_individual_positions()` (line 1207)
Filter: positions < 30s go into `_deferred_closes`, eligible ones close immediately.
```python
def _close_individual_positions(self, positions: list, reason: str) -> None:
    now = time.time()
    eligible = []
    for pos in positions:
        age = now - pos.entry_time
        if age < MIN_TRADE_RUNTIME_SECONDS:
            self._deferred_closes[pos.id] = {"reason": reason, "request_time": now}
            print(f"[DEFER CLOSE] {pos.symbol} {pos.direction} age={age:.0f}s — deferred ({reason})", file=sys.stderr)
        else:
            eligible.append(pos)
    if not eligible:
        return
    # ... original method body using `eligible` instead of `positions` ...
```

#### Change G — Gate direct `check_stops()` path (line ~686)
Skip positions < 30s, defer them instead.
```python
for pos in to_close:
    age = time.time() - pos.entry_time
    if age < MIN_TRADE_RUNTIME_SECONDS:
        close_reason = "STOP_LOSS"
        if pos.direction == "BUY" and price_now >= pos.take_profit:
            close_reason = "TAKE_PROFIT"
        elif pos.direction == "SELL" and price_now <= pos.take_profit:
            close_reason = "TAKE_PROFIT"
        self._deferred_closes[pos.id] = {"reason": close_reason, "request_time": time.time()}
        print(f"[DEFER STOP] {pos.symbol} {pos.direction} age={age:.0f}s — deferred", file=sys.stderr)
        continue
    # ... original per-position close logic ...
```

#### Change H — Add `_process_deferred_closes()` method
```python
def _process_deferred_closes(self, now: float | None = None) -> None:
    if now is None:
        now = time.time()

    # 1. Process deferred close-all
    if self._deferred_all_reason and self.executor.positions:
        if all(now - p.entry_time >= MIN_TRADE_RUNTIME_SECONDS for p in self.executor.positions):
            reason = self._deferred_all_reason
            self._deferred_all_reason = None
            print(f"[DEFER EXECUTE] close-all now eligible ({reason})", file=sys.stderr)
            self._close_all_positions(reason)
            return

    # 2. Process deferred individual closes
    if self._deferred_closes and self.executor.positions:
        pos_map = {p.id: p for p in self.executor.positions}
        eligible_ids = [pid for pid in self._deferred_closes if pid in pos_map
                        and now - pos_map[pid].entry_time >= MIN_TRADE_RUNTIME_SECONDS]
        if eligible_ids:
            to_close = [pos_map[pid] for pid in eligible_ids]
            reason = self._deferred_closes[eligible_ids[0]]["reason"]
            for pid in eligible_ids:
                del self._deferred_closes[pid]
            self._close_individual_positions(to_close, reason)
```

## Summary

| Change | File | Δ Lines |
|--------|------|---------|
| `MIN_TRADE_RUNTIME_SECONDS = 30` | `config/settings.py` | +1 |
| Import setting | `manager.py:10` | +0 (modify line) |
| Deferral fields in `__init__` | `manager.py:~133` | +2 |
| `_process_deferred_closes()` call in `_process_batches()` | `manager.py:284` | +1 |
| `_process_deferred_closes()` call in `_decision_worker()` | `manager.py:261` | +1 |
| Gate `_close_all_positions()` | `manager.py:1156` | +5 |
| Gate `_close_individual_positions()` | `manager.py:1207` | +8 |
| Gate `check_stops()` direct path | `manager.py:~686` | +6 |
| `_process_deferred_closes()` method | `manager.py` (new) | +28 |
| **Total** | | **+52** |

---

# Chop Gate Hysteresis — Implementation Plan

## Objective
Add hysteresis to the chop gate: **block entries when SSP polarization ≥ 70%**, **unblock when ≤ 65%**. This prevents rapid flip-flopping around a single threshold.

## Behavior

| State | pol_pct | Action |
|-------|---------|--------|
| TREND | ≤ 70% | Stay in TREND |
| TREND | > 70% | Enter CHOP, block entries |
| CHOP | > 65% | Stay in CHOP, block entries |
| CHOP | ≤ 65% | Exit CHOP, reset system to cycle 1 |

## Files to Modify

### 1. `currency_decomposition/runtime/manager.py`

#### Change A — Hysteresis threshold (line ~480)
```python
# Before:
is_chop = pol_pct > 0.70
# After:
is_chop = pol_pct > 0.70 if self._chop_since == 0.0 else pol_pct > 0.65
```

#### Change B — Update `_regime_data` (lines ~498-499)
```python
# Before:
"threshold_pct": 70.0,
"gap_to_clear": max(0.0, round(pol_pct * 100 - 70.0, 1)) if is_chop else 0.0,
# After:
"threshold_pct": 70.0,
"unblock_threshold": 65.0,
"gap_to_clear": max(0.0, round(pol_pct * 100 - 65.0, 1)) if is_chop else 0.0,
```

### 2. `currency_decomposition/web_dashboard.py`

Update UI references to reflect the 65% unblock threshold.

| Line | Before | After |
|------|--------|-------|
| ~509 | `left:75%` | `left:70%` (already done) |
| ~517 | `▲ 75% clear` | `▲ 65% clear` |
| ~845 | `› 75% threshold` | `≥ 70%` |
| ~861 | `≤ 75%` | `≤ 65%` |

## Summary

| Change | File | Δ Lines |
|--------|------|---------|
| Hysteresis threshold | `manager.py:480` | +0 (modify 1 line) |
| `_regime_data` fields | `manager.py:498-499` | +1 |
| Dashboard text updates | `web_dashboard.py` | +3 |
| **Total** | | **+4** |
