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
