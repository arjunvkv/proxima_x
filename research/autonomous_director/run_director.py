import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from research.autonomous_director.director_pipeline import DirectorPipeline

SEP = "=" * 52
HEADER = f"""
{SEP}
 PROXIMA DIRECTOR REPORT
{SEP}
"""


def fmt(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}"


def run_report(args):
    pipe = DirectorPipeline()
    report = pipe.daily_report()
    cls = report.get("classification", "RESEARCH_PENDING")
    rec = report.get("recommendation", "NO_ACTION")
    print(HEADER)
    print(f"  Evidence Strength:      {fmt(report.get('evidence_strength', 0))}")
    print(f"  Research Confidence:     {fmt(report.get('research_confidence', 0))}")
    print(f"  Deployment Confidence:   {fmt(report.get('deployment_confidence', 0))}")
    print(f"  Alpha Transfer:          {fmt(report.get('alpha_transfer', 0))}")
    print(f"  Health Index:            {fmt(report.get('health_index', 0))}")
    print(f"  Contradictions:          {report.get('contradictions', 0)}")
    print(f"")
    print(f"  Biggest Risk:            {report.get('biggest_risk', 'N/A')}")
    print(f"  Biggest Strength:        {report.get('biggest_strength', 'N/A')}")
    print(f"")
    print(f"  Recommendation:          {rec}")
    reas = report.get('recommendation_reasons', [])
    if reas:
        print(f"  Reason:                  {reas[0]}")
    print(f"")
    print(f"  Classification:          {cls}")
    print(f"")
    print(f"  Hypotheses:")
    for k, v in report.get('hypotheses', {}).items():
        print(f"    {k:30s} {fmt(v)}")
    print(SEP)


def run_weekly(args):
    pipe = DirectorPipeline()
    report = pipe.weekly_report()
    print(f"\n{SEP}")
    print(f" PROXIMA WEEKLY DIRECTOR REPORT")
    print(SEP)
    print(f"  Week:                    {report.get('week_start', 'N/A')} -> {report.get('date', 'N/A')}")
    print(f"")
    print(f"  Gained Confidence:")
    for k, v in report.get('gained_confidence', {}).items():
        print(f"    + {k:30s} {fmt(v)}")
    print(f"")
    print(f"  Lost Confidence:")
    for k, v in report.get('lost_confidence', {}).items():
        print(f"    - {k:30s} {fmt(abs(v))}")
    print(f"")
    print(f"  Converging:              {report.get('converging', False)}")
    print(f"  Latest Classification:   {report.get('latest_classification', 'N/A')}")
    print(f"  Latest Recommendation:   {report.get('latest_recommendation', 'N/A')}")
    print(f"  Total Daily Reports:     {report.get('total_dailies', 0)}")
    print(f"{SEP}\n")


def run_status(args):
    pipe = DirectorPipeline()
    ev = pipe.collector.summary()
    hyp = pipe.hypotheses.summary()
    print(f"\n{SEP}")
    print(f" PROXIMA DIRECTOR STATUS")
    print(SEP)
    print(f"  Evidence Records:        {ev.get('total_records', 0)}")
    print(f"  Daily Reports:           {pipe.memory.count_dailies()}")
    print(f"  Weekly Reports:          {pipe.memory.count_weeklies()}")
    print(f"  Contradictions Found:    {pipe.contradictions.count()}")
    print(f"")
    print(f"  Research Confidence:")
    for k, v in pipe.hypotheses.all_confidences().items():
        print(f"    {k:30s} {fmt(v)}")
    print(f"{SEP}\n")


def main():
    parser = argparse.ArgumentParser(description="Proxima Autonomous Research Director")
    parser.add_argument("command", nargs="?", default="report",
                        choices=["report", "weekly", "status"],
                        help="report=daily, weekly=weekly, status=overview")
    args = parser.parse_args()
    if args.command == "report":
        run_report(args)
    elif args.command == "weekly":
        run_weekly(args)
    else:
        run_status(args)


if __name__ == "__main__":
    main()
