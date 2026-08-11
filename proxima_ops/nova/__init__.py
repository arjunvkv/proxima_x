"""NOVA — pluggable vectorized engine (live + backtest), climate-gated.

Design spec: scripts/_absorb/results/NOVA_ENGINE_DESIGN.md (2026-08-11).

The legacy engine (proxima_ops/backtest) stays BYTE-IDENTICAL. NOVA is additive:
same spec dicts, same trade-dict contract, same cost path (delegates to
pnl.trade_to_usd), vectorized factors + fills. Parity is proven by
scripts/verify_nova_parity.py.
"""
