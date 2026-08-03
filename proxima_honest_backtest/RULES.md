# Proxima Honest Backtest Rules

## Project Rules

1. **ZERO lookahead** — No `shift(-n)`, no `bfill`, no `center=True` rolling, no `iloc[-1]`. The RollingBuffer and static linter enforce this.

2. **Data is frozen** — All data lives in `data/` as Parquet. Never mutate in place. Never modify fixtures in tests.

3. **Engine is read-only** — `engine/` and `execution/` are read-only by convention. Strategies and user code go in `strategies/`.

4. **Reproducibility** — Every random operation must be seeded (`np.random.seed()` or `RandomState`). All seeds default to 42.

5. **Broker simulation is config-driven** — Broker profiles are JSON configs, never hardcoded. Add new profiles without changing code.

6. **Reconciliation gate** — Every backtest must pass PnL reconciliation (trade PnL = equity delta). If it doesn't match, the run is invalid.

7. **Anti-overfit gauntlet** — Every strategy must pass the gauntlet (DSR, PBO, CPCV, sign-permutation, cost stress) before deployment consideration.

8. **Walk-forward with embargo** — Never backtest on overlapping train/test sets. Minimum 5-day embargo between train and test windows.

9. **Multi-broker validation** — Every promising strategy must be tested against all 5 broker profiles (Exness, Dukascopy, FundedNext, FusionMarkets, FTMO) before any conclusion.

10. **Parameter sweeps are validated** — Optuna/random sweeps must be re-run with multiple seeds. The best parameters must survive across seeds.

## Trading Strategy Rules

1. DO use z-score based entry/exit — it normalizes across symbols and timeframes.

2. DO test at tick level before drawing conclusions from bar-level results.

3. DO compute confidence as a function of signal strength (e.g., `min(|z| / entry_z, 1.0)`).

4. DO NOT optimize on full-sample data — always walk-forward or use purged CV.

5. DO NOT use Asian-session hours as a hard filter without testing all hours.

6. DO NOT ignore spread costs — always backtest with at least one broker profile.

7. DO NOT trust a single backtest run — always run Monte Carlo simulations.

8. DO NOT use commission-free results for deployment decisions.

9. DO NOT skip the reconciliation gate — if PnL doesn't match equity, find the bug.

10. DO NOT report metrics without cost stress testing (1x, 2x, 3x costs).
