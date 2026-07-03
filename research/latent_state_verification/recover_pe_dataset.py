"""POSITION_EXISTS opportunity dataset recovery and analysis."""
import json
from collections import Counter

with open("C:/Trading/Agentic_Trading/proxima_x/proxima_ops/data/funnel_stats.json") as f:
    data = json.load(f)

signals = data.get("signals", {})

# Classify all signals by state
states = Counter(s.get("final_state", "UNKNOWN") for s in signals.values())
total = len(signals)

# Extract POSITION_EXISTS blocked
pe = [s for s in signals.values() if s.get("final_state") == "BLOCKED_POSITION_EXISTS"]

# Extract all non-blocked, non-generated signals (those that progressed further)
executed = [s for s in signals.values() if s.get("final_state") in ("POSITION_OPENED", "POSITION_CLOSED")]
triggered = [s for s in signals.values() if s.get("final_state") == "TRIGGERED"]
spread_blocked = [s for s in signals.values() if s.get("final_state") == "BLOCKED_SPREAD"]

print("=" * 60)
print("POSITION_EXISTS OPPORTUNITY DATASET RECOVERY")
print("=" * 60)

print(f"\nTotal signals in funnel: {total}")
print(f"\nSignal lifecycle breakdown:")
for state, count in sorted(states.items(), key=lambda x: -x[1]):
    pct = count / total * 100
    print(f"  {state:30s}: {count:5d} ({pct:5.1f}%)")

# ES comparison
print(f"\n{'='*60}")
print("ES Profile Comparison: Blocked vs Executed")
print(f"{'='*60}")

for group_name, group in [("POSITION_EXISTS blocked", pe),
                            ("Executed (closed)", executed),
                            ("Spread blocked", spread_blocked)]:
    es_vals = [s.get("es") for s in group if s.get("es") is not None]
    if es_vals:
        print(f"\n  {group_name} ({len(group)} signals):")
        print(f"    ES: mean={sum(es_vals)/len(es_vals):.4f}, "
              f"min={min(es_vals):.4f}, max={max(es_vals):.4f}")

# Symbol analysis
print(f"\n{'='*60}")
print("POSITION_EXISTS by Symbol")
print(f"{'='*60}")
sym_pe = Counter(s["symbol"] for s in pe if "symbol" in s)
total_pe = len(pe)
for sym, cnt in sorted(sym_pe.items()):
    print(f"  {sym}: {cnt:4d} ({cnt/total_pe*100:5.1f}%)")

# Compare to overall funnel symbol distribution
print(f"\n{'='*60}")
print("Comparison: Blocked vs Total Trigger Distribution")
print(f"{'='*60}")
total_by_sym = Counter(s["symbol"] for s in signals.values() if "symbol" in s)
for sym in sorted(set(list(sym_pe.keys()) + list(total_by_sym.keys()))):
    pe_cnt = sym_pe.get(sym, 0)
    tot_cnt = total_by_sym.get(sym, 0)
    pe_pct = pe_cnt / total_pe * 100 if total_pe else 0
    tot_pct = tot_cnt / total * 100 if total else 0
    print(f"  {sym}: blocked={pe_cnt:4d} ({pe_pct:5.1f}%), total={tot_cnt:4d} ({tot_pct:5.1f}%)")

# Revenue impact estimate
print(f"\n{'='*60}")
print("Opportunity Cost Estimate")
print(f"{'='*60}")

# From executed trades
ex_pnls = [s.get("pnl_money", 0) for s in executed if s.get("pnl_money") is not None]
avg_pnl = sum(ex_pnls) / len(ex_pnls) if ex_pnls else 0
print(f"\n  Executed trades: {len(ex_pnls)}")
print(f"  Average PnL per executed trade: ${avg_pnl:.2f}")
print(f"  Total PnL from executed: ${sum(ex_pnls):.2f}")
print(f"\n  Blocked opportunities: {total_pe}")
print(f"  Estimated opportunity cost at avg PnL: ${total_pe * avg_pnl:.2f}")
print(f"\n  NOTE: This assumes blocked trades would have performed like executed trades.")
print(f"  In reality, blocked trades are a DIFFERENT population (different ES levels,")
print(f"  different timing). True opportunity cost is UNKNOWN without forward tracking.")

# Recommendation
print(f"\n{'='*60}")
print("RECOMMENDATION")
print(f"{'='*60}")
print(f"""
The POSITION_EXISTS dataset ({total_pe} signals, all from {list(pe)[0].get('timestamp_generated','')[:10] if pe else 'N/A'})
cannot be backfilled with future returns because:

1. Market data (parquet) covers 2019-01 to 2025-12
2. Live demo signals are from June 2026
3. Only 3 executed trades for comparison (insufficient)

To resolve this going forward:

1. ADD forward-return tracking to the SignalFunnel
   - When a signal is blocked/executed, record the future price path
   - Use MT5 price feed to track N bars into the future
   
2. STORE in signal_ledger database:
   - future_return_H5, H20, H50 for every signal (blocked + executed)
   
3. AFTER 50+ executed trades:
   - Compare blocked vs executed outcome distributions
   - Compute: was blocking beneficial or costly?
   - Adjust position_exists logic if blocking is destroying alpha

Current takeaway: POSITION_EXISTS blocking is a LIQUIDITY CONSTRAINT
(2 positions saturate a $25k account). The blocker is correct for
risk management but OPPORTUNITY COST is unmeasured.
""")
