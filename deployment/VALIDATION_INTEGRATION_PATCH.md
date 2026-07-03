# ValidationIntegration — Patch Guide for `run_proxima_demo.py`

6 precise insertion points to wire Phase E into the live trading pipeline.

---

## Before: Verify files exist

```bash
python -c "from validation.validation_integration import ValidationIntegration, NullValidationIntegration, OrganismState, DecisionContext, TradeOutcome; print('OK')"
python tests/test_validation_integration.py
```

---

## Patch 1: Imports (line 72–82)

**Insert after line 74** (`sys.path.insert(0, ...)`), before the existing imports:

```python
# Phase E — Validation & Promotion Framework
os.environ.setdefault("VALIDATION_ENABLED", "1")
_VALIDATION_ENABLED = os.environ.get("VALIDATION_ENABLED", "1") == "1"
if _VALIDATION_ENABLED:
    from validation.validation_integration import (
        ValidationIntegration as _ValidationIntegration,
        OrganismState,
        DecisionContext,
        TradeOutcome,
    )
else:
    from validation.validation_integration import (
        NullValidationIntegration as _ValidationIntegration,
        OrganismState,
        DecisionContext,
        TradeOutcome,
    )
```

---

## Patch 2: Constructor initialisation (end of `__init__`, after line 438)

**Insert after line 438** (`self._spread_normalizer = SpreadNormalizer()`), before `self._cycle_tpi_snapshot = ...`:

```python
        # Phase E — Observer Validation & Promotion Framework
        self._validation = _ValidationIntegration(
            evidence_path=os.path.join(os.getcwd(), "state", "evidence.parquet"),
        )
        logger.info(f"[VALIDATION] {'enabled' if _VALIDATION_ENABLED else 'disabled'} — "
                     f"{len(self._validation._module_names)} observer modules")
```

---

## Patch 3: Tick dispatch (`run_demo`, inside 10-second loop)

### Replay path (after line 1659, inside `if self._replay_mode and self._replay_feed:` block):

After `if sym:` / `self._warmup_ticks += 1`:

```python
                        # Phase E: feed tick to observer harness
                        if _VALIDATION_ENABLED:
                            self._validation.on_tick(sym, price, _ts)
```

### Live path (after line 1675, inside `else:` block, inside the for loop):

After `self._thesis_graph.tick(sym, price)`:

```python
                                # Phase E: feed tick to observer harness
                                self._validation.on_tick(sym, price, tick.get("time", 0))
```

---

## Patch 4: Feature computation (60-second cycle, after line 2816)

**After line 2816** (`time_regime = int(time_regime_arr[-1]) if len(time_regime_arr) > 0 else None`),
before `# Combined regime:`:

```python
                        # Phase E: feed organism state to observer harness
                        if _VALIDATION_ENABLED:
                            _os = OrganismState(
                                symbol=sym,
                                timestamp=int(now),
                                cycle_id=self._cycle_id,
                                topology_state=next(
                                    (t for t in ("STRUCTURED_DOMINANCE", "DIRECTED_TURBULENCE",
                                                  "DIFFUSE_CHAOS", "TRANSITIONAL_COMPRESSION",
                                                  "DISTRIBUTED")
                                     if t in str(_topo_name)), "DISTRIBUTED"),
                                trust=es_percentile,
                                pressure=float(energy_regime) / 3.0 if energy_regime else 0.5,
                                fracture=float(es_history[-1]) if len(es_history) > 0 else 0.0,
                                trust_band="HIGH" if es_percentile > 0.7 else "MEDIUM" if es_percentile > 0.4 else "LOW",
                                pressure_band="HIGH" if float(energy_regime or 2) > 2 else "MEDIUM" if float(energy_regime or 2) > 1 else "LOW",
                                rupture_flag=int(rupture_flag) if 'rupture_flag' in dir() else 0,
                                rupture_probability=rupture_state.get("rupture_probability", 0.0) if hasattr(self, '_rupture_state') else 0.0,
                                attractor_strength=0.5,
                                transition_entropy=entropy_state_s.get("normalized_entropy", 0.5) if isinstance(entropy_state_s, dict) else 0.5,
                                causal_confidence=0.5,
                                path_probability=0.5,
                                counterfactual_score=0.0,
                                cohort_instability=0.0,
                                thesis_rf_probability=self._rf_gate.prob(sym) if hasattr(self, '_rf_gate') else 0.5,
                                memory_weight=0.5,
                                production_action="BUY" if prod_signals.get(sym, 0) > 0 else "SELL" if prod_signals.get(sym, 0) < 0 else "FLAT",
                                es_percentile=es_percentile,
                                at_percentile=at_percentile,
                            )
                            self._validation.on_feature_compute(_os)
```

---

## Patch 5: Trade entry — thesis recording (60-second cycle, after line 3482)

**After line 3482** (`logger.info(f"[B4_RECORD] ...")`),
before `_entry_px = (tick["bid"] + tick["ask"]) / 2.0`:

```python
                                        # Phase E: record trade entry in validation framework
                                        if _VALIDATION_ENABLED:
                                            _dc = DecisionContext(
                                                decision_id="",
                                                timestamp=int(now),
                                                cycle_id=self._cycle_id,
                                                symbol=sym,
                                                direction=_tdir,
                                                entry_price=_entry_px,
                                                ticket=ticket,
                                                thesis_id=str(_tid),
                                                organism_at_entry=_os,
                                                observer_recommendation="EXECUTE",
                                                observer_quality=0.5,
                                            )
                                            self._validation.on_trade_entry(_dc)
```

---

## Patch 6: Trade close — thesis resolution (sync_ledger_with_mt5, after line 1343)

**After line 1343** (`_label = self._thesis_buffer.resolve(ticket, profit_money, exit_reason)`),
before `if _label is not None:`:

```python
                # Phase E: resolve observer evidence at trade close
                if _VALIDATION_ENABLED:
                    _outcome = TradeOutcome(
                        thesis_id=str(ticket),
                        symbol=symbol,
                        exit_price=exit_price,
                        pnl=profit_money,
                        success=_label == 1 if _label is not None else False,
                        exit_reason=exit_reason,
                        duration_bars=max(1, duration // 3600),
                    )
                    self._validation.on_trade_close(str(ticket), _outcome)
```

---

## Patch 7: Cycle end (60-second cycle block end)

**Before the end of the 60-second signal evaluation block** (after the cycle's main evaluation work, before the next tick loop iteration), add:

```python
                    # Phase E: cycle-end promotion checks
                    if _VALIDATION_ENABLED:
                        self._validation.on_cycle_end()
                        _vs = self._validation.stats()
                        logger.info(f"[VALIDATION_CYCLE] cycle={self._cycle_id} "
                                    f"evidence={_vs['evidence']} resolved={_vs['resolved']} "
                                    f"disagreements={_vs['disagreements']}")
```

A good insertion point is at the end of the 60-second block after `logger.info(f"[OCCUPANCY] ...")` at line 2676, before `self._global_rank_engine.compute()` at line 2679.

---

## Verification

1. Set `VALIDATION_ENABLED=0` for zero overhead on live trading
2. Run `python tests/test_validation_integration.py` — should pass 39/39
3. Check evidence parquet at `state/evidence.parquet` after running replays
4. Check logs for `[VALIDATION]` and `[VALIDATION_CYCLE]` messages

---

## Rollback

To disable Phase E at any time:
```bash
set VALIDATION_ENABLED=0
python run_proxima_demo.py demo
```
Or simply revert the 7 patches above.
