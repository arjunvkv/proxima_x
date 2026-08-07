# Proxima Backtest→Live Engine

A **strategy-agnostic, spec-driven** engine that backtests any session/base declared
as a declarative `StrategySpec` and ships the SAME spec to the live FTMO MT5
daemon. **The engine is a frozen contract: you plug strategies INTO it, you never
modify it.** Every no-lookahead, anti-overfit, live-safety lesson this project
survived is baked into the engine so new strategies inherit them automatically.

This file is the contract. Read it before writing any new strategy. If a strategy
won't fit without editing `proxima_ops/backtest/`, you're building it wrong — the
spec is meant to express any session, bar-or-tick, long/short, mean-reversion or
momentum idea.

---

## 1. The one rule that protects the engine

> **A strategy IS a `StrategySpec` dict. Nothing else.**
> Adding a strategy = authoring ONE spec. It never touches `backtest/` code.

The engine owns, and therefore enforces, all of the following — you do **not**
reimplement them per strategy:

| Concern | Owned by engine | File |
|---|---|---|
| Data feed (bar or tick → canonical bars) | `feed.py` | `build_bars_map` / `build_tick_feed_from_archive` |
| Anti-lookahead (closed-bar signal → next-bar-open fill) | `engine.py` | `run_strategy`, `session_signal_indices` |
| Exit logic (SL/TP stop-first, hold) | `engine.py` | `simulate_exit` |
| Tick-value-correct USD PnL + commission (JPY 8.7× bug) | `pnl.py` | `trade_to_usd` |
| Validation battery (gate / train-val / walk-forward / purple / determinism) | `validation.py` | `metrics, gate, split_by_ts, walk_forward, purple_edge, determinism` |
| Determinism (polars order lesson) | engine (pure fn) | `run_strategy` |
| Live runtime manifest (attach-only, server-clock, no-cron) | `liveport.py` | `emit_live_manifest` |
| Offline live-firer replica (parity proof) | `live_sim.py` | `fire_live` |

**Rule of thumb:** if your new idea needs a change to any file above instead of a
new spec + (rarely) a new `rule`/`pick` enum value, STOP and reconsider — you are
about to break the contract. Extend the `StrategySpec`/`SignalSpec`/`ExitSpec`
dataclasses (declarative data) if you must, never alter the engine's core behavior.

---

## 2. The `StrategySpec` — what a strategy is

Plain JSON-serializable dict. Required keys: `name`, `universe`, `feed`, `signal`.

| Field | Type | Meaning |
|---|---|---|
| `name` | str | strategy id (used for state file — `proxima_ops/state/{name}_state.json`) |
| `universe` | [str] | the symbols to trade |
| `feed` | dict | `{mode: "bar"\|"tick", timeframe: "M5"}` |
| `signal.rule` | str | closed-bar rule (symbolic) |
| `signal.lookback` | int | bars of return used at the (closed) signal bar |
| `signal.pick` | str | `n_worst` \| `n_best` \| `all` — cross-section selection |
| `signal.top_n` | int | max positions per session-day |
| `signal.side` | str | `long` \| `short` \| `both` |
| `signal.fill_bar` | int | enter at open of `signal_bar + fill_bar` (default `1` = next bar; anti-lookahead) |
| `exit.mode` | str | `sl_tp_hold` \| `hold_only` \| `sl_tp` |
| `exit.hold_bars` | int | close if held ≥ N bars |
| `exit.jpy_sl_tp` | (sl, tp) | price distance for JPY pairs (e.g. `(0.35, 0.45)`) |
| `exit.non_jpy_sl_tp` | (sl, tp) | price distance for non-JPY (e.g. `(0.0035, 0.0045)`) |
| `exit.stop_first` | bool | MT5 conservative stop-first convention |
| `sessions` | [int]\|None | UTC hours that fire; `None` = any hour (tick/any-time strategies) |
| `base_lot` | float | notional position (default `0.15`) |
| `comment` | str | MT5 order comment |

### Example — the validated Tokyo_H0 spec
```python
from proxima_ops.backtest.spec import StrategySpec

TOKYO_H0 = StrategySpec(
    name="tokyo_h0",
    universe=["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD",
              "EURNZD","GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD",
              "EURGBP","EURCHF","USDCHF","AUDJPY"],
    feed={"mode": "bar", "timeframe": "M5"},
    signal=SignalSpec(rule="session_exhaustion", lookback=6, pick="n_worst",
                      top_n=5, side="long", fill_bar=1),
    exit=ExitSpec(mode="sl_tp_hold", hold_bars=12,
                  jpy_sl_tp=(0.35, 0.45), non_jpy_sl_tp=(0.0035, 0.0045)),
    sessions=[0],   # Tokyo hour 0 (UTC) only
    base_lot=0.15,
    comment="TOKYO_H0",
})
# spec.to_dict() / StrategySpec.from_dict(dict) round-trip a JSON-serialisable spec.
```

> **JPY vs non-JPY SL/TP is handled for you** (`exit.jpy_sl_tp` vs
> `exit.non_jpy_sl_tp`) — the engine picks per symbol. Never hardcode JP scale in
> your strategy file.

---

## 3. Backtest + validate a NEW strategy (offline)

1. Author your `StrategySpec` in a file under `scripts/` or `strategies/`.
2. Run it through the engine and every validation gate:

```python
from proxima_ops.backtest.feed import build_bars_map
from proxima_ops.backtest.engine import run_strategy
from proxima_ops.backtest.validation import (metrics, gate, split_by_ts,
                                             walk_forward, purple_edge, determinism)

bars = build_bars_map(spec.universe)          # offline bar oracle (audit cache)
usd  = run_strategy(bars, spec, volume=spec.base_lot)

m  = metrics(usd)                              # WR / PF / net / exp / maxDD
g  = gate(m, lot=spec.base_lot)                # hard gate: PF>1.2,net>0,exp>$15/lot,DD<20,trades[20,20k]
tr, va = split_by_ts(usd)                      # honest train/val 70/30
wf = walk_forward(usd, train_size=300, test_size=100, lot=spec.base_lot)
purple = purple_edge(bars, lambda bm: run_strategy(bm, spec, volume=spec.base_lot),
                     m["expectancy"]/spec.base_lot, iters=10)
det = determinism(lambda: run_strategy(bars, spec, volume=spec.base_lot))
```

### Interpretation — do NOT ship on the headline number alone
A new spec ships **only if the battery is coherent**, mirroring how Tokyo_H0 survived:

| Gate | Meaning | Ship if |
|---|---|---|
| `gate.passed` | hard minimums | **True** |
| `split_by_ts` val | out-of-sample net still > 0 | **True** |
| `walk_forward.stable` | net>0 in ≥60% of forward windows | **True** |
| `purple_edge == "REAL-EDGE"` | real exp > shuffled mean + 2sd (not a time artifact) | **REAL-EDGE** |
| `determinism` | identical trade count across runs | **True** |

A strategy that "looks great" but fails `purple_edge` is a curve-fit artifact —
reject it and move on; do **not** tune params to force it through (that's overfitting).

> **Events/day dilution:** a common mistake is adding session hours or raising
top_n to get more trades. Measured reality (Tokyo sweep, persisted in
`audit_7_eas/session_reversion_sweep.json`): every expansion dilutes WR / PF /
expectancy. More events is rarely a better edge. Validate the quality per lot,
not the event count.

### Sanity / speed
- Reuse the cached bar tape (`audit_7_eas/market/`). First calls read parquet; a
  full engine run over the 18-symbol universe is deterministic and cheap, so
  interactive iteration is fine.

---

## 4. From backtest to LIVE (same spec, zero reimplementation)

The spec is plain JSON, so it is **unchanged** from backtest → live. Two layers ship it:

### a) Live manifest (the contract you hand to the daemon)
```python
from proxima_ops.backtest.liveport import emit_live_manifest
emit_live_manifest(spec, out_path=r"proxima_ops/state/tokyo_h0_manifest.json",
                   terminal_path=r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe",
                   account="1514168544")
```
The manifest embeds the spec plus the runtime knobs already baked in `liveport.py`:
**attach-only, server-clock gating, fill-bar 300s tolerance, no cron, max 5
positions.** That IS the live contract — it's impossible to ship a
spec with different, unvalidated behavior.

### 4b. The live daemon (`scripts/run_tokyo_h0_live.py` is the reference impl)
The documented single-strategy live runner is `run_tokyo_h0_live.py`. It:
- attaches (never re-login, never re-init) to the running FTMO demo,
- gates on the FTMO **server clock** hour (never host wall clock — host is 11h skew),
- signals on the closed hour-0 M5 bar, fills at **next bar open** (the engine's
  anti-lookahead contract carried into live),
- ranks top-5 BUY, SL/TP stop-first, holds 12 bars, ≤5 positions,
- persists day-dedup in `proxima_ops/state/tokyo_h0_state.json`.

To generalise this to ANY spec, drive the loop off `spec` (universe/signal/exit) and
the generated manifest — do not fork the file. If your strategy's live behaviour
differs from a `spec` is capable of, fix the spec, not the runner.

### Launch / stop (manual — remember NO cron)
```bash
cd /c/Trading/Proxima_X && unset PYTHONPATH
export MT5_PATH="C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
./.venv/Scripts/python.exe scripts/run_tokyo_h0_live.py --execute --manage --daemon
```
- Stop: kill the process while it is idling (no trade is open at idle, state file
  untouched). Relaunching later with the same command is safe and correct.
- **NO cron / no auto-relaunch**: the daemon is manually kept alive. Convenience
  this file does not auto-heal.

---

## 5. The verification battery you run when you / the engine changes

These are the **regression gates** (OFFLINE, run after ANY change to
`pnl.py`/`engine.py`/`spec.py`):

```bash
cd /c/Trading/Proxima_X && unset PYTHONPATH
./.venv/Scripts/python.exe scripts/verify_engine_parity_tokyo.py    # engine reproduces audit curve
./.venv/Scripts/python.exe scripts/verify_live_backtest_parity.py   # batch engine == dummy live firer
```
Both must print **PARITY: PASS**. If either breaks, you changed the engine contract
incidentally — revert or fix before shipping.

The **live** (real-broker, on-demand — NOT an automated gate by user choice)
checks are `scripts/verify_live_micro_batch.py` (place N real 0.01-lot trades on
the FTMO demo, reconcile vs the broker ledger) and the 3×5-min replay
(`scripts/run_three_5min_trials.py`). Pause the daemon first (it holds the MT5 IPC;
two inits to the same terminal wedge IPC), then relaunch it.

---

## 6. CAUTION / invariants you must not violate on this host

1. **`unset PYTHONPATH`** before any run — the Hermes desktop session injects a
   global venv that shadows the project's `.venv`.
2. **Exactly one `terminal64.exe`** (the FTMO binary at
   `C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe`). Two MT5
   terminals — or a second in-process `initialize` to a DIFFERENT path — trip the
   single IPC channel (`-6` / `-10005`). Set `MT5_PATH` and blank creds.
3. **Server clock, never wall clock** for session/day gating (host wall clock is
   ~11h unreliable).
4. **Pause the live daemon before a live micro-batch**; a second init into the
   same terminal wedges IPC.
5. The state file (day-dedup) key is epoch-**days** consistently throughout.

---

## 7. Where to look next / extend

| Goal | Resource |
|---|---|
| Deep engine internals + all phases | `references/` (see `AGENTS.md`, `proxima_ops/backtest/` docstrings) |
| The full no-lookahead/anti-overfit battery | `proxima_ops/backtest/validation.py` |
| Tick feed (real replay tape) | `feed.build_tick_feed_from_archive` (Phase 4 archive) |
| Report per-specl | `scripts/` verify/demo scripts |

---

**Contract summary:** Author a `StrategySpec` → backtest+validate against the full
battery → ship the SAME spec to live via the (attach-only, server-clock) manifest/
runner. The engine is not edited per strategy; if you are about to edit it, you're
holding the problem upside-down. Improve the engine once (declarative data + tests),
and every strategy — current and future — inherits the fix.