# V2z_CPPF — MQL5 EA Specification

## Overview

V2+z Mean Reversion strategy, ported from Python `hfdf_m1` logic to MQL5.
Operates on M1 bars, enters on bar close, manages with ATR trailing stop.

## Matching Python Logic

### Z-Score Computation

```python
# Python (hfdf_m1):
closes = [...]
returns = [closes[i+1] - closes[i] for i in range(51)]  # 51 returns
cur_ret = returns[50]                     # current return
prior = returns[:50]                      # 50 prior returns
z = (cur_ret - mean(prior)) / std(prior)
```

```mql5
// MQL5 (V2z_CPPF):
// g_close_buf[0] = oldest close, [51] = newest close
// rets[i] = g_close_buf[i+1] - g_close_buf[i] for i=0..50
// rets[50] = cur_ret, rets[0..49] = prior 50
double cur_ret = rets[Z_WINDOW];
// mean + std of rets[0..Z_WINDOW-1]
```

### Entry Rules

| Condition | Action |
|-----------|--------|
| `z >= +Z_THRESHOLD` | SHORT (fade the extreme) |
| `z <= -Z_THRESHOLD` | LONG |
| Open position exists | Skip |
| Spread > MAX_SPREAD_PIPS | Skip |
| Daily trade count >= MAX_TRADES_DAY | Skip |

### Exit Rules

| Condition | Action |
|-----------|--------|
| Price hits trailing stop | Close (via `ClosePosition("stop")`) |
| Bars held >= MAX_HOLD_BARS | Close (via `ClosePosition("expiry")`) |

### Trailing Stop Logic

```
Initial stop: STOP_A * ATR(20) from entry
Trail activation: profit >= TRIG_A * ATR
Trail offset: GAP_A * ATR from best price
Stop moves: in profit direction only (ratchet)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Z_THRESHOLD | 2.5 | z-score entry threshold |
| STOP_A | 3.0 | Initial stop = STOP_A × ATR |
| TRIG_A | 1.0 | Trail activates after TRIG_A × ATR profit |
| GAP_A | 0.05 | Trail offset = GAP_A × ATR from best |
| MAX_HOLD_BARS | 54 | Maximum bars to hold |
| ATR_PERIOD | 20 | ATR lookback |
| Z_WINDOW | 50 | Number of prior returns for z-score |

## MT5 Backtest Config

**Model**: `0` = Every Tick (required for realistic fill simulation)

**Filling**: `ORDER_FILLING_IOC` (Immediate-or-Cancel — preferred for backtest)

**Supported pairs**: EURUSD, GBPUSD, EURJPY, USDJPY, EURAUD, EURNZD,
GBPAUD, GBPCAD, GBPNZD, AUDNZD, AUDCAD, NZDCAD

## File Locations

| File | Path |
|------|------|
| EA source | `MQL5\Experts\V2z_CPPF.mq5` |
| Compiled EA | `MQL5\Experts\V2z_CPPF.ex5` |
| Config (.ini, per run) | `paper_trade\mt5_backtest\bt_configs\*.ini` |
| Parameters (.set) | `MQL5\Profiles\Tester\V2z_CPPF.set` |
| Reports (.htm) | `paper_trade\mt5_backtest\bt_reports\*.htm` |
| Automation (PS) | `paper_trade\mt5_backtest\run_v2z_backtest.ps1` |
| Automation (Py) | `paper_trade\mt5_backtest\run_mt5_bt.py` |

### ⚠️ .set File Notes

- File **MUST be placed** in `MQL5\Profiles\Tester\` — not in `bt_configs\`
- Name **MUST match** EA name exactly: `V2z_CPPF.set`
- **MUST be ASCII** (no UTF-8 BOM) — use `-Encoding ASCII` in PowerShell
- MT5 auto-discovers it — no `ExpertParameters` INI key needed

### ⚠️ .ini File Notes

- **MUST start** with a `[Common]` section (empty Login/Password/Server)
- **MUST strip** `[Tester]` from `terminal.ini` before each run
- Terminal data directory must be wiped between runs (stale cache) |
