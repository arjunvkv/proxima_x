import sys, json, time
sys.path.insert(0, '.')

results = {}

# Phase 1: FSV Core
print('=== PHASE 1: FSV CORE TESTS ===')
from research.fsv.testing.fsv_test_harness import FSVTestHarness
harness = FSVTestHarness()
fsv_results = harness.run_all()
results['fsv_core'] = fsv_results
print(f'Passed: {fsv_results["passed"]}/{fsv_results["total"]} | Stability: {fsv_results["stability_score"]:.2f}')
for name, passed in fsv_results['results'].items():
    print(f'  [{"PASS" if passed else "FAIL"}] {name}')

# Phase 3: UCF
print('\n=== PHASE 3: UCF INTEGRATION TESTS ===')
from research.ucf.testing.test_ucf_system import UCFIntegrationTests
ucf_tests = UCFIntegrationTests()
ucf_results = ucf_tests.run_all()
results['ucf'] = ucf_results
print(f'Passed: {ucf_results["passed"]}/{ucf_results["total"]} | Stability: {ucf_results["stability_score"]:.2f}')
for name, passed in ucf_results['results'].items():
    print(f'  [{"PASS" if passed else "FAIL"}] {name}')
if ucf_results['failures']:
    print(f'  Failures: {ucf_results["failures"]}')

# UCF Stress Suite
print('\n=== UCF STRESS SUITE ===')
ucf_stress = ucf_tests.stress_suite()
results['ucf_stress'] = ucf_stress
print(f'Passed: {ucf_stress["passed"]}/{ucf_stress["total"]}')

# Build full system for dashboard
print('\n=== BUILDING FULL SYSTEM STATE ===')
from research.fsv.core.fsv_engine import FSVEngine
from research.fsv.simulation.synthetic_event_generator import SyntheticMacroGenerator
from research.ucf.core.unified_conviction_field import UnifiedConvictionField
from research.ucf.integration.ucf_pipeline_bridge import UCFPipelineBridge
from research.ucf.dashboard.ucf_dashboard import UCFDashboard, UCFConsoleRenderer
from research.fsv.dashboard.fsv_dashboard_spec import FSVDashboardSpec

engine = FSVEngine()
generator = SyntheticMacroGenerator()

# Feed all stress scenarios
for scenario in ['crisis', 'trend', 'conflict', 'api_failure']:
    events = generator.stress_scenario(scenario)
    for e in events:
        engine.update_with_event(e)

# Add stream data
stream = generator.generate_event_stream(duration_seconds=3600, events_per_minute=3)
for e in stream[:200]:
    engine.update_with_event(e)

states = engine.get_all_states()
symbols = list(states.keys())
print(f'Symbols tracked: {len(symbols)}')

# Run UCF Pipeline
technical_states = {}
fsv_states = {}
for sym in symbols:
    s = engine.get_state(sym, time.time())
    technical_states[sym] = {"conviction": 0.65, "direction": 1, "stability": 0.7}
    fsv_states[sym] = {
        "conviction": abs(s.bias_alignment) * 0.8 + 0.2,
        "direction": 1 if s.bias_alignment > 0.1 else (-1 if s.bias_alignment < -0.1 else 0),
        "stability": s.regime_stability
    }

regime_state = {
    "regime": "neutral",
    "regime_stability": 0.7,
    "fsv_entropy": 0.3,
    "technical_volatility": 0.2,
    "recent_prediction_error": 0.1,
    "exposure_concentration": 0.3
}

bridge = UCFPipelineBridge()
pipeline_result = bridge.process(symbols, technical_states, fsv_states, None, regime_state)

print(f'Selected: {pipeline_result["selected_symbol"]}')
print(f'Weights: T={pipeline_result["weights_used"].get("technical_weight",0):.2f} F={pipeline_result["weights_used"].get("fundamental_weight",0):.2f} M={pipeline_result["weights_used"].get("macro_weight",0):.2f} E={pipeline_result["weights_used"].get("exposure_weight",0):.2f}')
print(f'Is blocking: {pipeline_result["is_blocking"]}')
print(f'Fallback: {pipeline_result["fallback_used"]}')

# UCF Console render
renderer = UCFConsoleRenderer()
print('\n' + renderer.render_field_table(pipeline_result['field']))
print(renderer.render_summary(pipeline_result['field']))

# Generate combined HTML report
print('\n=== GENERATING COMBINED VERIFICATION REPORT ===')
full_state = FSVDashboardSpec(engine).export_full_state(engine)
full_state['test_results'] = fsv_results
full_state['ucf_results'] = ucf_results
full_state['pipeline'] = pipeline_result

fsv_html = FSVDashboardSpec().generate_html_report(full_state)

ucf_dash = UCFDashboard()
ucf_html = ucf_dash.export_html_report(pipeline_result['field'])

combined = fsv_html.replace('</body>', '')
combined += '\n<hr><h1>PHASE 3 — UNIFIED CONVICTION FIELD</h1>\n'
combined += ucf_html.split('<body>')[-1]

with open('research/fsv/dashboard/fsv_full_verification_report.html', 'w') as f:
    f.write(combined)

# Save JSON results
with open('research/fsv/dashboard/verification_results.json', 'w') as f:
    json.dump({
        'fsv_core': {k: str(v) if isinstance(v, float) else v for k, v in fsv_results.items()},
        'ucf': {k: str(v) if isinstance(v, float) else v for k, v in ucf_results.items()},
        'ucf_stress': {k: str(v) if isinstance(v, float) else v for k, v in ucf_stress.items()},
    }, f, indent=2)

print(f'Report saved: research/fsv/dashboard/fsv_full_verification_report.html')
print(f'JSON results saved: research/fsv/dashboard/verification_results.json')

# Final summary
total_tests = fsv_results['total'] + ucf_results['total']
total_passed = fsv_results['passed'] + ucf_results['passed']
overall_stability = total_passed / total_tests if total_tests > 0 else 0.0
print(f'\n=== OVERALL: {total_passed}/{total_tests} passed ({overall_stability:.1%}) ===')
print(f'FSV: {fsv_results["passed"]}/{fsv_results["total"]} | UCF: {ucf_results["passed"]}/{ucf_results["total"]}')
