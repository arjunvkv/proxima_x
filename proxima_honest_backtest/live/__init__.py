"""Live engine — ships validated honest-backtest strategies to live MT5 unchanged.

Architecture: ONE strategy object -> ONE causal decision path (DecisionKernel)
-> multiple execution adapters (sim for backtest, LiveExecutor for live).
The live layer never re-implements strategy logic; it only supplies market data
(market_state) and execution (executor) around the shared kernel.
"""
