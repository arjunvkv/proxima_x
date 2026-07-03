import sys, json, time
sys.path.insert(0, '.')
from research.fsv.phase2.testing.fundamental_backtest import FundamentalBacktest
from research.fsv.phase2.dashboard.phase2_dashboard import Phase2Dashboard, Phase2ConsoleRenderer
from research.fsv.dashboard.fsv_dashboard_spec import FSVDashboardSpec, FSVDashboardRenderer
from research.fsv.core.fsv_engine import FSVEngine
from research.fsv.simulation.synthetic_event_generator import SyntheticMacroGenerator
from research.fsv.testing.fsv_test_harness import FSVTestHarness
from research.fsv.integration.fsv_modulator import FSVModulator
from research.fsv.core.fsv_schema import FundamentalStateVector, neutral_fsv

print('=== FSV CORE TESTS ===')
harness = FSVTestHarness()
test_results = harness.run_all()
print(f'Passed: {test_results["passed"]}/{test_results["total"]} | Stability: {test_results["stability_score"]:.2f}')

engine = FSVEngine()
generator = SyntheticMacroGenerator()

for scenario in ['crisis', 'trend', 'conflict', 'api_failure']:
    events = generator.stress_scenario(scenario)
    for e in events:
        engine.update_with_event(e)

stream = generator.generate_event_stream(duration_seconds=3600, events_per_minute=3)
for e in stream[:200]:
    engine.update_with_event(e)

states = engine.get_all_states()
print(f'\nSymbols tracked: {len(states)}')
for sym, state in states.items():
    print(f'  {sym}: bias={state.bias_alignment:+.3f} macro={state.macro_pressure:+.3f} risk={state.event_risk:.3f}')

print('\n=== PHASE 2 BACKTEST ===')
backtest = FundamentalBacktest()
bt_results = backtest.run_backtest(num_cycles=50)
accuracy = backtest.evaluate_accuracy(bt_results)
print(f'Selection accuracy: {accuracy["selection_accuracy"]:.1%}')
print(f'Lift over random: {accuracy["lift_over_random"]:+.1%}')
print(f'Consistency: {accuracy["consistency_score"]:.3f}')
if 'regime_accuracy_breakdown' in accuracy:
    for regime, acc in accuracy['regime_accuracy_breakdown'].items():
        print(f'  {regime}: {acc:.1%}')

print('\n=== PHASE 2 SELECTOR TEST ===')
symbols = list(states.keys())[:3]
directions = {s: 1 for s in symbols}
convictions = {s: 0.65 for s in symbols}
fsves = {s: states[s] for s in symbols}

selector = backtest.selector
selection = selector.select_best(symbols, fsves, directions, convictions)
print(f'Selected: {selection["selected_symbol"]}')
print(f'Confidence: {selection["recommendation"]["confidence"]:.3f}')
print(f'Reason: {selection["recommendation"]["reason"]}')
print(f'Is blocking: {selection["is_blocking"]}')

print('\n=== GENERATING COMBINED REPORT ===')
full_state = FSVDashboardSpec(engine).export_full_state(engine)
full_state['test_results'] = test_results
full_state['backtest'] = accuracy
full_state['selection'] = {
    'selected': selection['selected_symbol'],
    'confidence': selection['recommendation']['confidence'],
    'reason': selection['recommendation']['reason'],
    'is_blocking': selection['is_blocking']
}

fsv_html = FSVDashboardSpec().generate_html_report(full_state)

p2_dash = Phase2Dashboard()
p2_html = p2_dash.export_html_report(symbols, fsves, directions, convictions, full_state)

combined_html = fsv_html.replace('</body>', '')
combined_html += '\n<hr><h1>PHASE 2 — TOP-3 FUNDAMENTAL SELECTION</h1>\n'
combined_html += p2_html.split('<body>')[-1]

with open('research/fsv/dashboard/fsv_full_verification_report.html', 'w') as f:
    f.write(combined_html)

print(f'Combined report saved ({len(combined_html)} bytes)')

renderer = Phase2ConsoleRenderer()
print()
print(renderer.render_selection(selection))

print('\n=== DONE ===')
