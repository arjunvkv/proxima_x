"""Diagnostic: why only GBP pairs get traded."""
import json, sys
from collections import Counter

# 1. Dashboard log analysis
lines = open("logs/dashboard_log.jsonl").read().strip().split("\n")
entries = [json.loads(l) for l in lines]
print(f"=== Dashboard entries: {len(entries)} ===")
for e in entries[-5:]:
    cs = e["currency_strengths"]
    sorted_cs = sorted(cs.items(), key=lambda x: x[1], reverse=True)
    spread_range = sorted_cs[0][1] - sorted_cs[-1][1]
    print(f"  UPTIME={e['uptime']:.0f}s QUAL={e['graph_quality']:.3f} POS={e['positions']} TOP={e['top_symbol']} CONF={e['top_conf']}")
    print(f"    Strengths: {[f'{c}={v:+.5f}' for c,v in sorted_cs]} Range={spread_range:.5f}")

# 2. Hypothesis coverage - how many hypotheses above threshold?
import MetaTrader5 as mt5
from config.settings import SYMBOLS, BASE_CURRENCY_MAP, MIN_CONFIDENCE
from currency.graph import CurrencyGraph
from data.tick_store import TickStore
from data.mt5_adapter import MT5Adapter
from direction.hypothesis import HypothesisGenerator

# Load latest state from store
store = TickStore()
mt5.initialize()
adapter = MT5Adapter()
adapter.connect()
batch = adapter.poll_ticks()
if batch:
    store.add_ticks(batch.ticks)

returns = store.calculate_returns()
non_zero = sum(1 for v in returns.values() if v != 0.0)
print(f"\n=== Symbol coverage ===")
print(f"Symbols with non-zero returns: {non_zero}/{len(returns)}")

# Check which symbols have returns
nonzero_syms = [k for k, v in returns.items() if v != 0.0]
print(f"  Non-zero symbols: {nonzero_syms[:10]}...")
zero_syms = [k for k, v in returns.items() if v == 0.0]
if zero_syms:
    print(f"  Zero-return symbols (no data): {zero_syms}")

# 3. Currency representation in non-zero pairs
currencies_in_returns = Counter()
for sym in nonzero_syms:
    if len(sym) == 6:
        currencies_in_returns[sym[:3]] += 1
        currencies_in_returns[sym[3:]] += 1
print(f"\nCurrency representation in active pairs: {dict(currencies_in_returns.most_common())}")

mt5.shutdown()
