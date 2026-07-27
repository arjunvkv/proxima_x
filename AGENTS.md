# Proxima Agentic Conventions

## Pipelines

### Market Vision System (MVS)
```
from mvs.orchestrator import MVSEngine
from mvs.scheduler import MVSScheduler

# Single symbol
engine = MVSEngine("EURJPY", db_path="mvs.duckdb")
tick = engine.run_tick()
engine.detect_conflicts()
ranking = engine.update_honesty()
engine.close()

# Multi-symbol batch
scheduler = MVSScheduler(["EURJPY", "USDJPY"], db_path="mvs.duckdb")
scheduler.run_cycle(1000, report_interval=100)
scheduler.close_all()
```

### Causal Reality Attack (CRA)
```
python run_causal_reality_attack.py EURJPY
```

### Memory Physics Resolution (MPR)
```
python run_memory_physics.py EURJPY
```

### Compression Physics Investigation (CPI)
```
python run_compression_physics.py EURJPY
```

### Alpha Extraction Lab (AEL)
```
python run_alpha_extraction.py EURJPY
```

### Alpha Reality Lab (ARL)
```
python run_alpha_reality.py EURJPY
```

### Adaptive Alpha Engine (AAE)
```
python run_adaptive_alpha.py
```

### Interaction Asymmetry Lab (IAL)
```
python run_interaction_asymmetry.py EURJPY
```

### Energy Reality Lab (ERL)
```
python run_energy_reality.py EURJPY
```

### Residual Energy Project (REP)
```
python run_residual_energy.py EURJPY
```

### Proxima V1 (Production Engine)
```
python run_proxima_v1.py backtest [start] [end]
python run_proxima_v1.py walkforward
python run_proxima_v1.py paper [start] [end]
```

### Proxima V2 (Deployment Engine)
```
python run_proxima_v2.py validate    # Run full V2 validation suite
python run_proxima_v2.py paper       # Run paper trading simulation
python run_proxima_v2.py metrics     # Compare V1 vs V2 metrics
```

### Market State Vector (MSV)
```
from state.market_state import MarketStateVector

msv = MarketStateVector(history_size=50)
snapshot = msv.update(returns, weights, prior, timestamp)

# MSV outputs
snapshot.network.dispersion       # currency network dispersion
snapshot.network.agreement        # directional agreement across pairs
snapshot.residual.residual_shock  # unexplained pair movement
snapshot.residual.residual_energy # mean absolute residual

regime = msv.regime(snapshot)     # COMPRESSION / TREND / SHOCK / NEUTRAL
risk = msv.risk_score(snapshot)   # 0.0-1.0 risk score
entry_ok, reason = msv.entry_allowed(snapshot)  # binary gate
```

### MSV Asian Exhaustion Validation
```
$env:PYTHONPATH = "C:\Trading\Agentic_Trading\proxima_x\currency_decomposition"
python research/msv_validation/run_msv_final.py   # Final comprehensive validation
python research/msv_validation/run_msv_round4.py   # Direction/decomposition analysis
```

### MSV Event Layer Integration (in Proxima V2)
```
from state.market_state import MarketStateVector

class MSVEventLayer:
    """Produces PortfolioIntent when Asian FX exhaustion detected."""
    ASIA_HOURS = range(0, 7)
    DISP_PCT_THRESH = 0.95
    PREV_DECLINE_THRESH = -0.0002  # -0.02%

    def evaluate(self, snapshot, pre60_return, hour_utc) -> Optional[dict]:
        if hour_utc not in self.ASIA_HOURS: return None
        if snapshot.network.dispersion_pct < self.DISP_PCT_THRESH: return None
        if pre60_return > self.PREV_DECLINE_THRESH: return None
        return {
            "state": "ASIAN_FX_EXHAUSTION",
            "direction": "LONG",
            "confidence": 0.94,
            "expected_duration": 60,
            "universe": "FX_BASKET",
        }
```

### State Persistence Lab (SPL)
```
python run_spl.py [mode]
python run_spl.py rq1       # Persistence Driver Identification
python run_spl.py full      # Run all 10 RQs
```

### Reality Gap Analysis
```
python run_reality_gap.py [mode]
```

### Residual Live Validation Lab (RLVL)
```
python research/live_validation/run_live_validation.py
```

### Tick Collector for Engine 2 Validation
```
python research/msv_validation/backtest_engine2/tick_collector.py --duration 3600  # Collect 1hr
python research/msv_validation/backtest_engine2/tick_collector.py                  # Until Ctrl+C
python research/msv_validation/backtest_engine2/tick_collector.py --analyze        # Show collected stats
```
Collects live ticks from all 7 pairs via symbol_info_tick polling. Run during active market hours (Sun 22:00 UTC — Fri 22:00 UTC). Ticks saved to `backtest_engine2/tick_data/*.npy`. Use --duration to auto-stop after N seconds.

### Tokyo H0 (Paper Trade Strategy)
```
cd paper_trade
python strategies/tokyo_h0/run.py
```
18-pair universe, fires at 00:00 UTC, picks 3 most-declined, holds 15 min.
Validated: 77.8% OOS WR. Account 5053225887.

### Dark Consensus (Paper Trade Strategy)
```
cd paper_trade
python strategies/dark_consensus/run.py
```
7-pair magnitude gap strategy. Account 109849586.

### Combined DC + 10sMR (Paper Trade Strategy)
```
cd paper_trade
python strategies/combined_dc_mr/run.py
```
~41 trades/day at ~57% WR. DC fires on all 3 pairs (EURUSD z>1.5, EURJPY/GBPJPY z>1.75) on spread-widen recovery, hold 10min. 10sMR fires on EURUSD only (z>3.5), hold 3min. Account TBD.

### V2+z Paper (Paper Trade Strategy)
```
cd paper_trade
python strategies/v2z_paper/run.py
```
6 cross pairs (AUDNZD, EURAUD, EURNZD, GBPAUD, GBPCAD, GBPNZD). z>=2.5 threshold, trailing stop (stop_a=3, trig_a=1, gap_a=0.05), 0.5s polling, TradeTracer lifecycle logging.

**Known bugs fixed (Jul 2026):**
1. **Spread baseline** (`risk.py`): Each pair updates only its own baseline, not all 6.
2. **Phantom close** (`executor.py`): `_live_close` uses `positions_get(ticket=mt5_ticket)` from trail_mgr, not `positions_get(symbol=pair)` + magic matching.
3. **MT5 timezone** (`run.py:_get_mt5_close_info`): Server UTC+3, local UTC. History query uses `0` as from_dt to avoid timezone mismatch.
4. **Pip value cross pairs** (`components/__init__.py:pip_value_usd`): Cross pairs (EURAUD, AUDNZD, GBPCAD) now use actual quote/USD rate, not $10/pip flat. `_calc_pnl` queries live rate when mt5 available.

**Exit slippage**: `result.price` from close `order_send` differs from actual fill by 0.8-1.5 pips.

**Run 1784924962 — MT5 PnL vs old calc:**
| Trade | Old PnL | MT5 PnL | Cause |
|-------|---------|---------|-------|
| EURAUD | +$14.00 | +$9.77 | $10/pip fixed → $6.70 + 0.8pip slip |
| GBPAUD | +$14.00 | +$9.77 | $10/pip fixed → $6.70 + 1.3pip slip |
| AUDNZD | +$9.00 | +$5.21 | $10/pip fixed → $5.80 |
| GBPCAD | -$13.32 | -$17.74 | $10/pip fixed → $7.80 + 1.5pip fav exit |

### Proxima Ops — MT5 Demo Deployment
```
python run_proxima_demo.py demo       # Full demo deployment
python run_proxima_demo.py monitor    # Monitoring only
python run_proxima_demo.py report     # Daily report generation
```

## V2+z CPPF — Real MT5 Strategy Tester Results (Jul 2026)

**Critical finding**: The Python sim (`sim_backtest.py`) using M1 close prices is **invalid** — it creates massive look-ahead bias. Real MT5 tick-level backtests (Every Tick model, with spread & commission) show the strategy has an edge that is almost entirely consumed by transaction costs.

**Optimal config (6 cross pairs)**: Z_THRESHOLD=3.5, STOP_A=3.0, TRIG_A=1.0, GAP_A=0.05, BASE_LOT=0.75

**EA gotcha**: EA must NOT set `type_filling` in the tester — comment it out. `ORDER_FILLING_IOC` / `ORDER_FILLING_RETURN` causes error 10030 `[Unsupported filling mode]`.

### MT5 Tester Results (z=3.5, lot=0.75, 6 cross pairs)

**OOS (Feb-Mar 2026, zero-cost):**

| Pair | Trades | WR | Net PnL | PF | Max DD% |
|------|:-----:|:--:|:------:|:--:|:-------:|
| AUDNZD | 79 | 69.6% | +$317 | 1.32 | 1.1% |
| EURAUD | 93 | 76.3% | +$1,505 | 1.76 | 1.1% |
| EURNZD | 69 | 85.5% | +$1,523 | 3.04 | 1.0% |
| GBPAUD | 92 | 80.4% | +$2,070 | 2.02 | 0.8% |
| GBPCAD | 56 | 75.0% | +$771 | 2.17 | 0.7% |
| GBPNZD | 57 | 77.2% | +$887 | 1.73 | 1.3% |
| **Total** | **446** | **77.3%** | **+$7,073** | **1.90** | **1.3%** |

**IS (Apr-May 2026, zero-cost):**

| Pair | Trades | WR | Net PnL | PF | Max DD% |
|------|:-----:|:--:|:------:|:--:|:-------:|
| AUDNZD | 65 | 61.5% | +$253 | 1.30 | 0.8% |
| EURAUD | 58 | 69.0% | +$396 | 1.29 | 1.1% |
| EURNZD | 31 | 67.7% | +$223 | 1.28 | 1.4% |
| GBPAUD | 27 | 70.4% | +$269 | 1.50 | 0.7% |
| GBPCAD | 6 | 83.3% | +$163 | 2.65 | 0.4% |
| GBPNZD | 19 | 57.9% | -$236 | 0.69 | 1.6% |
| **Total** | **206** | **66.0%** | **+$1,068** | **1.26** | **1.6%** |

**Forward (Jun 8-Jul 25, 2026 — with commission):**

| Pair | Trades | WR | Net After Costs | PF | Max DD% |
|------|:-----:|:--:|:--------------:|:--:|:-------:|
| EURAUD | 215 | 66.5% | +$313 | 1.10 | 3.4% |
| AUDNZD | 214 | 58.4% | -$581 | 0.75 | 4.0% |
| EURNZD | 119 | 66.4% | -$52 | 0.98 | 2.9% |
| GBPAUD | 179 | 70.9% | +$499 | 1.19 | 2.3% |
| GBPCAD | 180 | 61.7% | -$351 | 0.84 | 3.3% |
| GBPNZD | 115 | 60.0% | -$792 | 0.76 | 5.9% |
| **Total** | **1,022** | **63.6%** | **-$964** | **0.93** | **5.9%** |

### Conclusions

1. **Clear edge exists**: 63-77% WR across all periods, consistent win rate structure
2. **Edge is ~$2-4/trade**: Average win $25-48, average loss $44-126, payoff ratio ~0.5
3. **Commission kills it**: Zero-cost OOS = +$7,073 (PF 1.90) → with commission Forward = -$964 (PF 0.93)
4. **Only EURAUD & GBPAUD survive commission**: +$313 and +$499 respectively
5. **GBPNZD is toxic**: -$792 on 115 trades, 5.9% max DD
6. **Python sim results were inflated 17x**: sim showed +$16,492 Forward vs real -$964 — M1 close prices created massive look-ahead bias

### Files

- `paper_trade/mt5_backtest/V2z_CPPF.mq5` — EA source (type_filling commented out)
- `paper_trade/mt5_backtest/sim_backtest.py` — **Python sim is INVALID** (M1 close-price look-ahead bias)
- MCP stored reports IDs 22-51 — actual MT5 tester results from `get_backtest_report`

## Final Reports
- `SUMMARY.md` — condensed journey summary
- `FINAL_REPORT.md` — comprehensive research report
- `ERL_REPORT.md` — Energy Storage falsification report
- `RESIDUAL_ENERGY_REPORT.md` — Residual Energy final report
- `PROXIMA_V1_ARCHITECTURE.md` — V1 system architecture
- `PROXIMA_V1_RESULTS.md` — V1 backtest and walk-forward results
- `reality/REALITY_GAP_REPORT.md` — Reality Gap Analysis report
- `research/persistence/SPL_REPORT.md` — State Persistence Lab report
- `proxima_v2/V2_REPORT.md` — Proxima V2 architecture and results
- `proxima_v2/PROXIMA_V2_CALIBRATION_REPORT.md` — V2 calibration results
- `research/live_validation/LIVE_VALIDATION_REPORT.md` — Live validation report
- `research/msv_validation/MSV_FINAL_REPORT.md` — Market State Vector final research report
