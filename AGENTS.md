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

### Proxima Ops — MT5 Demo Deployment
```
python run_proxima_demo.py demo       # Full demo deployment
python run_proxima_demo.py monitor    # Monitoring only
python run_proxima_demo.py report     # Daily report generation
```

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
