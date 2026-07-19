# Paper Trade — Plug & Play Strategy Framework

A modular paper trading system for **Proxima X**. Every research strategy gets its own folder, shares common infrastructure, and can be run independently.

```
paper_trade/
├── core/               # Shared infrastructure — never edit per strategy
│   ├── config.py       # Global strategy registry
│   ├── feed.py         # MT5 live feed + archive replay
│   ├── executor.py     # Market order execution (live + paper)
│   ├── logger.py       # CSV trade log
│   ├── risk.py         # Stateless risk checks
│   ├── stats.py        # SessionStats — live validation metrics
│   ├── registry.py     # AccountRegistry — claim/release per MT5 account
│   └── dashboard.py    # Terminal live dashboard
├── strategies/         # One folder per strategy — plug in here
│   ├── template/       # Copy this to start a new strategy
│   │   ├── strategy.py # MUST implement generate_signal(data) -> dict | None
│   │   ├── config.yaml # Strategy parameters
│   │   └── run.py      # Entry point (copy as-is)
│   └── dark_consensus/ # Validated Dark Consensus strategy
│       ├── strategy.py
│       ├── config.yaml
│       └── run.py
├── components/         # Reusable computation
│   ├── __init__.py     # Sharpe, win rate, profit factor, DD, VaR, pip value
├── live/               # Runtime output (gitignored)
│   ├── logs/           # Per-run CSV trade logs
│   ├── reports/        # Generated comparison reports
│   └── account_registry.json  # Live account claims (auto-managed)
├── check_registry.py   # CLI tool to inspect account registry
└── research/           # Historical research scripts (one subfolder per project)
```

## Plug-and-Play Contract

To add a new strategy, **copy the template folder** and implement one function:

```python
# strategies/your_strategy/strategy.py

STRATEGY_NAME = "your_strategy"

CONFIG = {
    "name": "your_strategy",
    "pairs": ["EURUSD", "GBPUSD"],
    "hold_bars": 3,
    "session_start": 7,
    "session_end": 21,
    "max_concurrent": 2,
    "max_spread_mult": 1.5,
    "max_daily_loss": 500,
    "lot_size": 1.0,
}

def generate_signal(data):
    """data: {pair: {bid, ask, time, spread}} from current feed bar.

    Returns: dict | None
        { "pair": "EURUSD", "direction": 1, "confidence": 0.8, "metadata": {} }
    """
    # --- Your signal logic here ---
    # No lookahead: only use data['bid'], data['ask'], data['time']
    return None
```

Then **set `mt5_account` to your unique MT5 login number** in `strategy.py`, then run:

```
cd paper_trade
python strategies/your_strategy/run.py
```

The framework handles:
- MT5 connection & reconnection (supports per-strategy terminal path)
- Per-strategy MT5 account isolation (duplicate account detection)
- Order execution (market orders)
- Position management (time-based exit after `hold_bars` minutes)
- Spread widening protection
- Concurrent position limits
- Session hour filtering (including weekend gate)
- Daily loss limit
- Live validation dashboard — Sharpe, win rate, profit factor, max DD, per-pair PnL
- CSV logging

## Anti-Ghost Rules

Nothing in this folder should produce phantom signals, lookahead bias, or hidden state leakage.

**Signal generation must NOT:**
- Use future prices or layout (e.g., `shift(-1)`)
- Leak state between independent runs (always start fresh)
- Use data from pairs not in `CONFIG["pairs"]`
- Reference anything outside `data` dict passed to `generate_signal`
- Use lookahead window functions (rolling mean that includes current bar is fine; rolling mean that includes next bar is not)

**Execution must NOT:**
- Modify signal after checking risk
- Use bid price for buys or ask price for sells (use ask for buys, bid for sells)
- Ignore MT5 return codes
- Submit orders outside session hours
- Exceed `max_concurrent` (handled by `risk.py`)

**Logging must NOT:**
- Modify logged data after-the-fact
- Drop rejection or error events
- Overwrite or truncate existing logs

## Running in Different Modes

### Live MT5 (default)
```
python strategies/dark_consensus/run.py
```

### Archive Replay
Change `mode` in `run.py`:
```python
feed = Feed(mode="archive", pairs=cfg["pairs"]).connect()
feed.load_archive(preloaded_data)
```

### Paper Simulation (no MT5 needed)
Same as archive replay — feed returns bid/ask prices from a simulated source or pre-loaded data dict.

## Account Isolation

Every strategy must declare a unique `mt5_account` in its `strategy.py` CONFIG:

```python
CONFIG = {
    "name": "dark_consensus",
    "mt5_account": 12345678,   # <-- your MT5 login
    "mt5_path": None,           # <-- optional: separate MT5 terminal path
    ...
}
```

**Rules enforced at startup:**
1. When a strategy starts, it **claims** its `mt5_account` in `live/account_registry.json`
2. If another strategy already holds that account → **startup rejected** with a clear message
3. Same strategy restarting is allowed (claim updates heartbeat)
4. A background thread sends a **heartbeat** every 30s — lets the registry detect crashes
5. On graceful shutdown (Ctrl+C or error), the account is **released** automatically
6. Entries with no heartbeat for >120s are treated as **stale** (crashed process) and cleaned up

### Checking Active Claims

```
python check_registry.py
```

Output:
```
    Account  Strategy             Age      PID
----------------------------------------------------
  12345678  dark_consensus       342s    20348
  23456789  template               0s    21012
```

### Running Two Strategies Simultaneously

Each needs a **different MT5 account** and **different terminal path**:

| Strategy | MT5 Account | MT5 Terminal Path |
|---|---|---|
| `dark_consensus` | 12345678 | `C:/MT5/dark/terminal64.exe` |
| `template` | 23456789 | `C:/MT5/template/terminal64.exe` |

Set `mt5_path` in CONFIG — it's passed to `mt5.initialize(path=...)`. Use `None` for the default terminal.

## Research-to-Production Pipeline

1. **Research:** scripts in `research/<project>/` — historical analysis only
2. **Validate:** backtest using Dukascopy data (27 CSVs, 9 months)
3. **Plugin:** copy template → write `strategy.py` → configure `config.yaml`
4. **Paper trade:** `python strategies/<name>/run.py` — live signals, no real capital
5. **Monitor:** dashboard shows PnL, win rate, max DD in real time
6. **Deploy:** swap `feed.mode` to `"live"`, connect to funded MT5 account

## Strategy Validation Checklist (Dark Consensus)

- [x] 9 months Dukascopy (Oct 2024–Jun 2026): all positive
- [x] Exness ticks OOS: positive
- [x] MT5 OOS: positive
- [x] Portfolio overlap: positive
- [x] 60s latency stress: positive
- [x] ATR-conditional slippage: positive
- [x] All regimes (COMPRESSION/TREND/SHOCK/NEUTRAL): positive
- [x] Max DD: $633 (0.35%)
- [x] Breakeven at 3.5× spread
- [x] Monotonic parameter plateau (P80→P99)
- [x] Fixed > rolling in low-vol regimes

See: `research/dark_research/DARK_CONSENSUS_VALIDATION_PACKAGE.md`
