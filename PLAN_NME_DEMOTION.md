# NME Demotion — Architectural Plan

## What the NME does well (keep)
- **Leader identification** — detects which currency has narrative momentum
- **NMI trajectory** — useful as contextual signal during a trade
- **Conflict detection** — flags competing narratives

## What to remove trust in
- **Direction prediction** — aligned trades can lose because the NME price forecast is wrong
- **Entry timing** — NMI level at entry does not distinguish winners from losers

## Proposed Change

Demote NME from **commander** (writes orders) to **context provider** (informs decisions).

### Current flow:
```
NME generates hypothesis (pair, direction, confidence)
  → NME direction determines entry side
  → Trade opens
```

### Proposed flow:
```
NME generates hypothesis (pair, direction suggestion, confidence)
  → Price-action confirmation (recent tick flow must match direction)
  → If confirmed: enter
  → If not confirmed: skip
  → After entry: monitor PnR recovery for early exit if needed
```

### What stays
- Confidence threshold filter (NME opinion)
- Conflict detection (skip if conflicting narratives)
- Leader-aware pair screening (only evaluate pairs involving the leader)

### What changes
- NME direction is a **suggestion**, not an order
- Actual entry requires price-action confirmation
- Post-entry PnL monitoring for early exit

## Changes to Existing Files

### `currency_decomposition/runtime/manager.py`
- NME direction validated against tick flow before committing
- Add post-entry PnL monitor
- Remove any NMI-trend-based entry gates

### `currency_decomposition/intelligence/narrative_engine.py`
- Add `suggest_direction()` — returns soft suggestion, not firm order
- Add `is_pair_relevant()` — quick leader check

### New file: `currency_decomposition/features/price_action.py`
- Confirms tick flow direction before entry
- Returns confirmation score (not binary)

## Validation
- Backtest proposed logic against same paper run
- Compare entry decisions: which trades would differ?
- Tune confirmation threshold per pair if needed
