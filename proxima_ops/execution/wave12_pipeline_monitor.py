import json
import time
from pathlib import Path

from wave12_executor import Wave12Executor

REPORT_PATH = Path("state/pipeline_flow_report.json")
CYCLE_LOG_PATH = Path("state/wave12_cycle_log.jsonl")


def run_monitoring_session(max_cycles: int = 100, interval: int = 2):
    print(f"[MONITOR] Starting pipeline monitoring session: {max_cycles} cycles")

    executor = Wave12Executor()
    executor.session_start = time.time()

    summary = {
        "session_duration_cycles": 0,
        "total_signals_generated": 0,
        "non_zero_direction": 0,
        "threshold_pass_ge_040": 0,
        "confirm_cycle_pass": 0,
        "execution_attempts": 0,
        "executions_successful": 0,
        "executions_denied": 0,
        "no_signal_no_pos": 0,
        "position_hold_cycles": 0,
        "close_events": 0,
        "denial_reasons": {},
        "edge_presence": {},
        "per_cycle_log": [],
    }

    last_log_cycle = 0

    for cycle_i in range(1, max_cycles + 1):
        result = executor.cycle()
        summary["session_duration_cycles"] = cycle_i
        pt = result.get("pipeline_trace", {})

        gen_signals = len(pt.get("generated", []))
        summary["total_signals_generated"] += gen_signals

        non_zero_dir = sum(1 for s in pt.get("generated", []) if "dir=" in s and "dir=0" not in s.split("dir=")[1].split(" ")[0])
        summary["non_zero_direction"] += non_zero_dir

        for g in pt.get("generated", []):
            parts = g.split(" ")
            eid = parts[0] if parts else "?"
            if eid not in summary["edge_presence"]:
                summary["edge_presence"][eid] = {"appearances": 0, "threshold_pass": 0, "confirm_pass": 0}
            summary["edge_presence"][eid]["appearances"] += 1
            if "PASS" in g:
                summary["edge_presence"][eid]["threshold_pass"] += 1

        threshold_pass = len(pt.get("threshold_gate", []))
        pass_count = sum(1 for g in pt.get("threshold_gate", []) if "PASS" in g)
        summary["threshold_pass_ge_040"] += pass_count

        for g in pt.get("confirm_gate", []):
            for eid in summary["edge_presence"]:
                if g.startswith(eid) and "PASS" in g:
                    summary["edge_presence"][eid]["confirm_pass"] += 1

        confirm_pass = sum(1 for g in pt.get("confirm_gate", []) if "PASS" in g)
        summary["confirm_cycle_pass"] += confirm_pass

        denial = result.get("denial_reason", "")
        if denial:
            summary["denial_reasons"][denial] = summary["denial_reasons"].get(denial, 0) + 1

        exec_trace = pt.get("execution", "")
        if exec_trace and exec_trace != "NONE":
            summary["execution_attempts"] += 1
            if exec_trace.startswith("EXECUTED"):
                summary["executions_successful"] += 1
            elif exec_trace.startswith("DENIED") or exec_trace.startswith("FAILED"):
                summary["executions_denied"] += 1

        close_result = result.get("close_result", {})
        if close_result and close_result.get("results"):
            summary["close_events"] += len(close_result["results"])

        if result.get("decision") == "HOLD" and result.get("open_positions", 0) == 0 and not denial:
            summary["no_signal_no_pos"] += 1

        if result.get("open_positions", 0) > 0:
            summary["position_hold_cycles"] += 1

        nolog = result.get("open_positions", 0) == 0 and exec_trace in ("NONE", "NO_SIGNAL") and cycle_i - last_log_cycle < 20
        if not nolog:
            last_log_cycle = cycle_i
            per_cycle_entry = {
                "cycle": cycle_i,
                "decision": result.get("decision"),
                "gen_signals": gen_signals,
                "threshold_pass": pass_count,
                "confirm_pass": confirm_pass,
                "active_signals": result.get("active_signals", 0),
                "pipeline_exec": exec_trace if exec_trace and exec_trace != "NONE" else None,
                "denial": denial if denial else None,
                "open_positions": result.get("open_positions", 0),
                "close": close_result.get("results", [{}])[0].get("pnl") if close_result and close_result.get("results") else None,
                "segl": result.get("segl_state"),
                "regime": result.get("regime"),
                "duration_ms": round(result.get("cycle_duration", 0) * 1000, 1),
            }
            summary["per_cycle_log"].append(per_cycle_entry)

        if cycle_i % 10 == 0:
            print(f"[MONITOR] cycle {cycle_i}/{max_cycles} | "
                  f"threshold_passes={pass_count} confirm_passes={confirm_pass} "
                  f"exec_attempts={summary['execution_attempts']} "
                  f"pos_holds={summary['position_hold_cycles']}")

        time.sleep(interval / max(1, interval))

    summary["edges_seen"] = list(summary["edge_presence"].keys())
    summary["edges_with_threshold_pass"] = [
        eid for eid, data in summary["edge_presence"].items()
        if data["threshold_pass"] > 0
    ]
    summary["edges_with_confirm_pass"] = [
        eid for eid, data in summary["edge_presence"].items()
        if data["confirm_pass"] > 0
    ]
    summary["overall_flow"] = {
        "total_signals_sum": summary["total_signals_generated"],
        "non_zero_direction": summary["non_zero_direction"],
        "threshold_pass": summary["threshold_pass_ge_040"],
        "confirm_cycle_pass": summary["confirm_cycle_pass"],
        "execution_attempts": summary["execution_attempts"],
        "executions_successful": summary["executions_successful"],
        "executions_denied": summary["executions_denied"],
        "no_signal_no_pos_cycles": summary["no_signal_no_pos"],
        "position_hold_cycles": summary["position_hold_cycles"],
        "close_events": summary["close_events"],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[MONITOR] Report saved to {REPORT_PATH}")

    return summary


if __name__ == "__main__":
    import sys
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    summary = run_monitoring_session(max_cycles=cycles, interval=interval)

    print("\n=== PIPELINE FLOW SUMMARY ===")
    overall = summary["overall_flow"]
    print(f"  Total cycle executions: {summary['session_duration_cycles']}")
    print(f"  Signals generated (sum): {overall['total_signals_sum']}")
    print(f"  Non-zero direction:      {overall['non_zero_direction']}")
    print(f"  Threshold pass (>=0.40): {overall['threshold_pass']}")
    print(f"  Confirm cycle pass:      {overall['confirm_cycle_pass']}")
    print(f"  Execution attempts:      {overall['execution_attempts']}")
    print(f"  Executions successful:   {overall['executions_successful']}")
    print(f"  Executions denied:       {overall['executions_denied']}")
    print(f"  No-signal no-pos cycles: {overall['no_signal_no_pos_cycles']}")
    print(f"  Position hold cycles:    {overall['position_hold_cycles']}")
    print(f"  Close events:            {overall['close_events']}")
    print(f"\n  Denial reasons breakdown:")
    for reason, count in sorted(summary.get("denial_reasons", {}).items(), key=lambda x: -x[1]):
        print(f"    {count:4d}x {reason}")
    print(f"\n  Edge presence:")
    for eid, data in sorted(summary.get("edge_presence", {}).items()):
        print(f"    {eid}: appearances={data['appearances']}, "
              f"threshold_pass={data['threshold_pass']}, "
              f"confirm_pass={data['confirm_pass']}")
