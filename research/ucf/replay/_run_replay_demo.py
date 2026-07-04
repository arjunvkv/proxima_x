import sys, json, time, math, random
sys.path.insert(0, '.')

print('=== PHASE 4: REPLAY SIMULATION DEMO (WITH MACRO INJECTION) ===')

from research.fsv.core.fsv_engine import FSVEngine
from research.fsv.simulation.synthetic_event_generator import SyntheticMacroGenerator
from research.ucf.replay.replay_orchestrator import ReplayOrchestrator
from research.ucf.replay.replay_macro_adapter import ReplayMacroAdapter
from research.ucf.replay.analysis.replay_metrics import ReplayMetrics
from research.ucf.replay.analysis.pnl_tracker import PnLTracker
from research.ucf.replay.execution_model.execution_simulator import ExecutionSimulator

# Build pre-loaded FSV engine
fsv_engine = FSVEngine()
gen = SyntheticMacroGenerator()
for scenario in ['crisis', 'trend', 'conflict']:
    events = gen.stress_scenario(scenario)
    for e in events:
        fsv_engine.update_with_event(e)
stream = gen.generate_event_stream(duration_seconds=3600, events_per_minute=3)
for e in stream[:200]:
    fsv_engine.update_with_event(e)
print('FSV engine ready: {} symbols with data'.format(len(fsv_engine.get_all_states())))

# Setup orchestrator with pre-loaded FSV + macro adapter
orchestrator = ReplayOrchestrator()
orchestrator.fsv_engine = fsv_engine
orchestrator.macro_adapter = ReplayMacroAdapter(fsv_engine)
orchestrator.macro_adapter.initialize()

# Generate ticks
print('Generating 5000 ticks...')
symbols = list(fsv_engine.get_all_states().keys())[:3]
ticks = orchestrator.generate_synthetic_ticks(5000, symbols)
orchestrator.load_ticks(ticks)
print('{} ticks loaded for {}'.format(len(ticks), orchestrator.symbols))

# Run replay
print('Running replay...')
replay_result = orchestrator.run_replay(batch_size=100)
total_ticks = replay_result.get("total_ticks", 0)
cycles = replay_result.get("cycles", 0)
logs = replay_result.get("logs", [])
summary = replay_result.get("summary", {})

print('Total ticks: {}'.format(total_ticks))
print('Cycles: {}'.format(cycles))
print('Avg confidence: {:.4f}'.format(summary.get("avg_confidence", 0.0)))
print('Regime distribution: {}'.format(summary.get("regime_distribution", {})))
print('Selection frequency: {}'.format(summary.get("symbol_selection_frequency", {})))

# Build replay logs + simulate trades
replay_logs = []
simulator = ExecutionSimulator()
fill_logs = []
pnl_tracker = PnLTracker()
base_prices = {"EURUSD": 1.10, "AUDUSD": 0.72, "GBPUSD": 1.25, "USDJPY": 110.0, "USDCHF": 0.92, "USDCAD": 1.35, "NZDUSD": 0.68}

for cycle_idx, log_entry in enumerate(logs):
    ranked = log_entry.get("ranked_symbols", [])
    if not ranked:
        continue
    top = ranked[0]
    symbol = top.get("symbol", "EURUSD")
    direction = top.get("direction", 0)
    ucf_score = top.get("ucf_score", 0.0)

    replay_logs.append({
        "cycle": cycle_idx,
        "symbol": symbol,
        "direction": direction,
        "confidence": ucf_score,
        "regime": log_entry.get("regime", "neutral"),
    })

    if direction == 0 or ucf_score < 0.01:
        continue

    entry_price = base_prices.get(symbol, 1.10) + random.uniform(-0.001, 0.001)
    spread_val = random.uniform(1, 5) * 0.0001
    volatility = random.uniform(0.0001, 0.001)
    trade_signal = direction * abs(ucf_score) * volatility * 10.0
    exit_price = entry_price + trade_signal

    entry_result = simulator.simulate_entry(entry_price, direction, spread_val, volatility)
    exit_result = simulator.simulate_exit(entry_price, exit_price, direction, spread_val, volatility)

    fill = simulator.simulate_trade(entry_result, exit_result)
    fill["symbol"] = symbol
    fill["cycle"] = cycle_idx
    fill["direction"] = direction
    fill["pnl"] = fill.get("net_pnl", 0.0)
    fill_logs.append(fill)
    pnl_tracker.record_trade(fill)

print('Simulated {} trades from {} cycles'.format(len(fill_logs), len(replay_logs)))

# Compute all metrics
metrics_calc = ReplayMetrics()
full_metrics = metrics_calc.compute_metrics(replay_logs, fill_logs)

report = metrics_calc.generate_report(full_metrics)
print('')
print(report)

align = full_metrics.get("alignment", {})
ucf_score = align.get("ucf_alignment_score", 0.0)
print('=== UCF ALIGNMENT SCORE: {:.3f} ({}) ==='.format(ucf_score, align.get("interpretation", "")))

pnl_summary = pnl_tracker.get_summary()
results = {
    "total_ticks": total_ticks,
    "cycles": cycles,
    "avg_confidence": summary.get("avg_confidence", 0),
    "regime_distribution": summary.get("regime_distribution", {}),
    "ucf_alignment_score": ucf_score,
    "total_pnl": round(pnl_summary.get("total_pnl", 0), 4),
    "win_rate": round(pnl_summary.get("win_rate", 0), 4),
    "sharpe": round(pnl_summary.get("sharpe_ratio", 0), 4),
    "trade_count": len(fill_logs),
}
with open('research/ucf/replay/_replay_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to _replay_results.json')
print(json.dumps(results, indent=2))
