"""
Proxima V2 — Live Validation Lab (RLVL)

python run_live_validation.py

Builds all 10 monitoring modules, runs on 5 assets,
generates LIVE_VALIDATION_REPORT.md

No optimization.
No recalibration.
No threshold tuning.

Observe only.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research.live_validation.live_pipeline import LivePipeline, ASSETS
from research.live_validation.live_dashboard import LiveDashboard


def main():
    print("=" * 60)
    print("PROXIMA V2 — LIVE VALIDATION LAB (RLVL)")
    print("Assets:", ASSETS)
    print("=" * 60)

    pipe = LivePipeline()
    report_data = pipe.load_and_run()

    dashboard = LiveDashboard()
    report_md = dashboard.generate_report(report_data)

    path = dashboard.save_report(report_md, "research/live_validation/LIVE_VALIDATION_REPORT.md")
    json_path = "research/live_validation/live_validation_results.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    print(f"\nReport: {path}")
    print(f"Data:   {json_path}")
    print("\n" + report_md)

    ds = report_data.get("deployment_score", {})
    print(f"\nDeployment Score: {ds.get('current_score', 0):.3f} ({ds.get('classification', 'N/A')})")
    print(f"Trend: {ds.get('trend', 'N/A')}")

    return report_data


if __name__ == "__main__":
    main()
