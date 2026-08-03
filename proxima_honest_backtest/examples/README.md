# Proxima Honest Backtest — Examples & Reference

## Overview

A Python-only, zero-MQL5 backtesting framework that prevents lookahead bias and
overfitting by architectural design. MT5 is only used as a data source — all
backtesting runs in Python.

```
┌─────────────────────────────────────────────────────┐
│                proxima_honest_backtest               │
├───────────┬───────────┬──────────┬──────────────────┤
│  engine/  │ execution │  data/   │   validation/    │
│ (frozen   │ (broker   │ (MT5 →   │ (linter,         │
│  types,   │  profiles,│  Parquet │  gauntlet,       │
│  rolling  │  sim)     │  store)  │  walk-forward)   │
│  buffer)  │           │          │                  │
├───────────┴───────────┴──────────┴──────────────────┤
│              strategies/ + examples/                 │
└─────────────────────────────────────────────────────┘
```

---

## Architecture

### Anti-Lookahead (enforced by design)

| Mechanism | What it prevents |
|-----------|-----------------|
| `RollingBuffer` | No `shift(-n)`, `bfill`, `center=True`, `iloc[-1]` — only historical data up to `end_idx` |
| `ReadOnlyView` | Wraps dict/list — raises `TypeError` on `__setitem__` |
| Frozen dataclasses | `PointInTime`, `SignalResult`, `Trade`, `ExecutionReport` — truly immutable |
| `LookAheadLinter` | AST scanning (via `qtype`) + regex fallback — detect forbidden patterns in CI |

### Anti-Overfit (validated by gauntlet)

| Test | Library | What it checks |
|------|---------|---------------|
| Deflated Sharpe Ratio | `purgedcv` | Adjusts Sharpe for multiple testing (N strategies) |
| PBO | `purgedcv` | Probability of Backtest Overfitting |
| CPCV | `purgedcv` | Combinatorial Purged Cross-Validation |
| Sign-permutation | built-in | Randomly flip trade signs, check if edge survives |
| Regime consistency | built-in | Sharpe per market regime — stable or fluke? |
| Cost stress test | built-in | Re-run at 1x, 2x, 3x costs |
| Walk-forward | built-in | Multiple windows with embargo gap |

### Broker Profiles (5 configs)

| Profile | Spread base | Commission | Leverage |
|---------|-------------|-----------|----------|
| Exness | 0.8 pip | $0/lot | 500:1 |
| Dukascopy | 0.5 pip | $3/lot | 200:1 |
| FundedNext | 0.6 pip | $4/lot | 100:1 |
| Fusion Markets | 0.3 pip | $2.50/lot | 500:1 |
| FTMO | 0.7 pip | $3.50/lot | 200:1 |

Each profile includes spread model, slippage, latency, and fill rate — calibrated
against real trading conditions.

---

## Available Data

All stored as monthly Parquet in `data/`:

| Timeframe | Pairs | Bars/pair | Date range | Size |
|-----------|-------|-----------|------------|------|
| M5 | 18 | ~42,000 | Jan–Jul 2026 | 20.7 MB |
| H1 | 18 | ~3,500 | Jan–Jul 2026 | 3.4 MB |

Pairs: EURUSD, USDJPY, GBPUSD, AUDUSD, EURJPY, GBPJPY, EURAUD, EURNZD,
GBPAUD, GBPNZD, GBPCAD, AUDNZD, USDCAD, NZDUSD, EURGBP, EURCHF, USDCHF, AUDJPY

```python
from data.providers.mt5_provider import MT5Provider

p = MT5Provider()
df = p.load_rates("EURAUD", 2026, 7, "m5")  # 5592 bars for July 2026
```

To pull fresh data from MT5:

```python
python tools/mt5_tick_puller.py --symbols EURUSD,USDJPY --from 2026-07-01
```

---

## Creating a Strategy

Subclass `BaseStrategy` and implement `on_bar`:

```python
from typing import Any, Dict, Optional
from proxima_honest_backtest.engine.types import PointInTime, SignalResult
from proxima_honest_backtest.engine.rolling_buffer import RollingBuffer
from proxima_honest_backtest.strategies.base import BaseStrategy


class MyStrategy(BaseStrategy):
    DEFAULT_PARAMS: Dict[str, Any] = {
        "lookback": 20,
        "threshold": 2.0,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._position = 0

    def on_tick(self, tick: PointInTime, history: RollingBuffer) -> Optional[SignalResult]:
        return None  # optional: implement for tick-level

    def on_bar(self, bar: Dict[str, Any], history: RollingBuffer) -> Optional[SignalResult]:
        """Called on each bar. history contains NO future data by design."""
        if len(history) < self.parameters["lookback"]:
            return None

        closes = history.get_column("close")  # returns tuple[float, ...]
        past = closes[-(self.parameters["lookback"] + 1):-1]
        current = closes[-1]

        mean = sum(past) / len(past)
        std = (sum((p - mean)**2 for p in past) / len(past))**0.5
        if std < 1e-12:
            return None

        z = (current - mean) / std

        if self._position == 0 and abs(z) > self.parameters["threshold"]:
            direction = -1.0 if z > 0 else 1.0
            self._position = direction
            return SignalResult(bar["time"], direction,
                                min(abs(z) / 5.0, 1.0),
                                {"action": "ENTER", "z": z})

        elif self._position != 0 and abs(z) < 0.5:
            self._position = 0
            return SignalResult(bar["time"], 0.0, 0.95,
                                {"action": "EXIT"})

        return None

    def reset(self) -> None:
        self._position = 0
```

Key rules:
- `get_column("close")` returns a `tuple[float, ...]` — immutable, no lookahead
- `get_window(end_idx, length)` returns a `ReadOnlyView` — also immutable
- Never use `shift(-n)`, `bfill()`, `center=True`, or negative `iloc`
- Always reset state in `reset()`

---

## Running a Backtest

```python
from data.providers.mt5_provider import MT5Provider
from execution.execution_simulator import ExecutionSimulator
from examples.backtest_engine import BacktestEngine

# 1. Load data
p = MT5Provider()
data = p.load_rates("EURAUD", 2026, 7, "m5")
data.sort_values("time", inplace=True)
data.reset_index(drop=True, inplace=True)

# 2. Create strategy + engine
strat = MyStrategy({"lookback": 20, "threshold": 2.0})
sim = ExecutionSimulator("ftmo")  # or "exness", "dukascopy", etc.
engine = BacktestEngine(strat, sim)

# 3. Run
result = engine.run("EURAUD", data)
print(f"Trades: {result.n_trades}  PnL: ${result.net_pnl:+.2f}  WR: {result.win_rate*100:.1f}%")
print(f"Sharpe: {result.sharpe:.2f}  DD: {result.max_drawdown_pct:.2f}%  Reconciled: {result.reconciliation_pass}")
```

---

## Full Pipeline

Run the complete 7-step pipeline:

```bash
cd proxima_honest_backtest
python examples/run_pipeline.py
```

This executes:

| Step | Component | Output |
|------|-----------|--------|
| 1. Load | `MT5Provider` | 42K M5 bars |
| 2. Lint | `LookAheadLinter` | Detection of `shift(-n)`, `bfill()`, `center=True` |
| 3. Backtest | `BacktestEngine` + `ExecutionSimulator` | Trades, PnL, Sharpe, DD, reconciliation |
| 4. Gauntlet | `OverfitGauntlet` | DSR, PBO, sign-permutation, cost stress |
| 5. Walk-fwd | `WalkForwardValidator` | OOS Sharpe, consistency, decay |
| 6. Monte Carlo | `MonteCarloSimulator` | Profit probability, equity range, avg DD |
| 7. Compare | `BrokerComparer` | Strategy PnL across all 5 brokers |

Example output (EURAUD, V2+z, Jan–Jul 2026):

```
Step 3: 42,181 bars → 286 trades in 2.4s
        Net PnL: $+300.62  |  Sharpe: 15.71
        Win Rate: 60.5%    |  PF: 1.17
        Max DD: 4.43%      |  Reconciled: True

Step 4: Gauntlet PASS
        Sign-test p: 0.17

Step 5: OOS Sharpe: 1.46  |  Consistency: 100%

Step 6: Profit Prob: 84.6%  |  90% range: $9,738–$11,343
        Avg DD: 4.2%

Step 7: Exness      PnL=$+1021  WR=62.3%
        Dukascopy   PnL=$ +532  WR=64.7%
        FusionMkts  PnL=$ +564  WR=64.7%
        FTMO        PnL=$ +301  WR=60.5%
        FundedNext  PnL=$  +96  WR=58.7%
```

---

## Parameter Sweeps

```python
from research.sweep import ParameterSweep

def objective(params):
    """Return metric to maximize (e.g., Sharpe)."""
    strat = MyStrategy(params)
    engine = BacktestEngine(strat, ExecutionSimulator("ftmo"))
    result = engine.run("EURAUD", data)
    return result.sharpe if result.n_trades > 10 else -999.0

sweep = ParameterSweep(
    param_space={"lookback": [10, 20, 50, 100], "threshold": [1.5, 2.0, 3.0]},
    metric_fn=objective,
    method="grid",
)
results = sweep.run(["EURAUD"])
sweep.summarize(results)
```

---

## Broker Comparison

```python
from research.broker_comparison import BrokerComparer

comparer = BrokerComparer()
report = comparer.compare(strategy_func, "EURAUD", ticks)
print(comparer.generate_report_markdown(report))
```

---

## Walk-Forward Validation

```python
from validation.walk_forward import WalkForwardValidator

wf = WalkForwardValidator(n_windows=4, test_size=0.2, embargo_days=5)
wf_result = wf.run_simple(returns, timestamps)
print(f"OOS Sharpe: {wf_result.avg_oos_sharpe:.3f}")
print(f"Consistency: {wf_result.oos_consistency*100:.0f}%")
```

---

## Files Reference

### Engine (read-only by convention)

| File | Exports | Purpose |
|------|---------|---------|
| `engine/types.py` | `PointInTime`, `SignalResult`, `Trade`, `ExecutionReport`, `ReadOnlyView` | Frozen dataclasses, immutable dict wrapper |
| `engine/rolling_buffer.py` | `RollingBuffer` | Deque-based ring buffer, no lookahead ops |
| `engine/reconciliation.py` | `reconcile`, `reconcile_streaming` | PnL gate — trade PnL must match equity delta |

### Execution (read-only by convention)

| File | Exports | Purpose |
|------|---------|---------|
| `execution/models.py` | `SpreadModel`, `SlippageModel`, `LatencyModel`, `FillModel`, `BrokerProfile` | Pricing simulators, config-driven |
| `execution/execution_simulator.py` | `ExecutionSimulator`, `load_broker_profile`, `list_broker_profiles` | Order execution with spread/slippage/fill/latency |
| `execution/broker_profiles/` | `exness.json`, `dukascopy.json`, `fundednext.json`, `fusionmarkets.json`, `ftmo.json` | Broker configs |

### Data

| File | Exports | Purpose |
|------|---------|---------|
| `data/providers/mt5_provider.py` | `MT5Provider` | Connect to MT5, pull ticks/rates, save/load Parquet |
| `data/providers/utils.py` | `symbol_to_file_safe`, `ensure_month_dir`, `get_date_range_for_month` | Path utilities |
| `tools/mt5_tick_puller.py` | CLI | Script to pull ticks from MT5 terminal |
| `tools/ftmo_data_pull.py` | CLI | Pulled 18-pair M5+H1 data from FTMO |

### Strategies

| File | Exports | Purpose |
|------|---------|---------|
| `strategies/base.py` | `BaseStrategy` | ABC with `on_tick`, `on_bar`, `reset` |
| `strategies/examples/mean_reversion.py` | `MeanReversionStrategy` | Z-score mean reversion on bar data |
| `examples/v2z_strategy.py` | `V2zStrategy` | V2+z: z-score + trailing stop, configurable direction |

### Validation

| File | Exports | Purpose |
|------|---------|---------|
| `validation/linter.py` | `LookAheadLinter`, `LintResult` | AST scan for `shift(-n)`, `bfill()`, `center=True` |
| `validation/gauntlet.py` | `OverfitGauntlet`, `GauntletResult` | DSR, PBO, CPCV, sign-permutation, cost stress |
| `validation/walk_forward.py` | `WalkForwardValidator`, `WFResult` | Multi-window walk-forward with embargo |

### Research & Examples

| File | Exports | Purpose |
|------|---------|---------|
| `research/sweep.py` | `ParameterSweep`, `SweepResult` | Optuna/grid/random parameter search |
| `research/monte_carlo.py` | `MonteCarloSimulator`, `MCResult` | Bootstrap resampling, equity curve distributions |
| `research/broker_comparison.py` | `BrokerComparer`, `ComparisonReport` | Cross-broker performance comparison |
| `examples/backtest_engine.py` | `BacktestEngine`, `BacktestResult` | Simple event-driven backtester |
| `examples/run_pipeline.py` | CLI | Complete 7-step pipeline runner |

---

## Development

```bash
# Install
pip install -e proxima_honest_backtest/

# Run all tests (137 tests, <1s)
cd proxima_honest_backtest/
python -m pytest tests/ -q

# With optional deps
pip install "proxima_honest_backtest[all]"

# Run the full example pipeline
python examples/run_pipeline.py

# Pull new MT5 data
python tools/ftmo_data_pull.py
```
