"""Tests for stress_activation_runner.py."""

import json
import os
import tempfile
import unittest

from stress_activation_runner import StressActivationRunner


class TestStressActivationRunner(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_log(self, lines: list[dict]):
        path = os.path.join(self.tmpdir, "wave12_cycle_log.jsonl")
        with open(path, "w") as f:
            for entry in lines:
                f.write(json.dumps(entry) + "\n")
        return path

    # ------------------------------------------------------------------
    # Empty / missing log
    # ------------------------------------------------------------------

    def test_log_not_found(self):
        runner = StressActivationRunner(log_path="does_not_exist.jsonl")
        report = runner.analyze()
        self.assertEqual(report["hypothetical_trades"], 0)
        self.assertEqual(report["actual_trades"], 0)
        self.assertEqual(report["missed_trades_due_to_confirm"], 0)
        self.assertIn("ANALYSIS ONLY", report["warning"])

    def test_empty_log(self):
        path = self._write_log([])
        runner = StressActivationRunner(log_path=path)
        report = runner.analyze()
        self.assertEqual(report["hypothetical_trades"], 0)

    # ------------------------------------------------------------------
    # No-signal cycles
    # ------------------------------------------------------------------

    def test_no_signals(self):
        path = self._write_log([
            {"cycle": 1, "total_signals": 0, "decision": "HOLD",
             "pipeline_trace": {"generated": [], "threshold_gate": [],
                                "confirm_gate": [], "governor_gate": [],
                                "execution": "NO_SIGNAL"}},
        ])
        runner = StressActivationRunner(log_path=path)
        report = runner.analyze(n_recent_cycles=1000)
        self.assertEqual(report["hypothetical_trades"], 0)
        self.assertEqual(report["actual_trades"], 0)

    # ------------------------------------------------------------------
    # Signal passed confirm (CROSS_PASS)
    # ------------------------------------------------------------------

    def test_signal_passed_confirm(self):
        path = self._write_log([
            {"cycle": 10, "total_signals": 1, "active_symbol": "EURUSD",
             "confirm_cycles": 3, "denial_reason": "",
             "pipeline_trace": {
                 "generated": ["edge_01 EURUSD mean_reversion dir=1 conf=0.55 -> PASS"],
                 "threshold_gate": ["edge_01: PASS dir=1 conf=0.55"],
                 "confirm_gate": ["edge_01: CROSS_PASS (cycles=3/2)"],
                 "governor_gate": [],
                 "execution": "PASSED ALL GATES",
             }},
        ])
        runner = StressActivationRunner(log_path=path)
        report = runner.analyze(n_recent_cycles=1000)
        self.assertEqual(report["hypothetical_trades"], 1)
        self.assertEqual(report["actual_trades"], 1)
        self.assertEqual(report["missed_trades_due_to_confirm"], 0)

    # ------------------------------------------------------------------
    # Signal denied by insufficient confirm
    # ------------------------------------------------------------------

    def test_signal_denied_by_confirm(self):
        path = self._write_log([
            {"cycle": 20, "total_signals": 1, "active_symbol": "USDJPY",
             "confirm_cycles": 1, "denial_reason": "Insufficient cross-projection confirm: 1/2",
             "pipeline_trace": {
                 "generated": ["edge_02 USDJPY mean_reversion dir=-1 conf=0.51 -> PASS"],
                 "threshold_gate": ["edge_02: PASS dir=-1 conf=0.51"],
                 "confirm_gate": ["edge_02: cross_cyc=0/2 (waiting)"],
                 "governor_gate": ["segl_state=ARMED ready_to_exec=YES intent=True"],
                 "execution": "DENIED cross_confirm=1/2",
             }},
        ])
        runner = StressActivationRunner(log_path=path)
        report = runner.analyze(n_recent_cycles=1000)
        self.assertEqual(report["hypothetical_trades"], 1)
        self.assertEqual(report["actual_trades"], 0)
        self.assertEqual(report["missed_trades_due_to_confirm"], 1)

    # ------------------------------------------------------------------
    # Mix of passed and denied
    # ------------------------------------------------------------------

    def test_mixed_scenario(self):
        path = self._write_log([
            # Denied by confirm
            {"cycle": 1, "total_signals": 1, "active_symbol": "USDJPY",
             "confirm_cycles": 1, "denial_reason": "Insufficient cross-projection confirm: 1/2",
             "pipeline_trace": {
                 "generated": ["edge_01 USDJPY mean_reversion dir=-1 conf=0.51 -> PASS"],
                 "threshold_gate": ["edge_01: PASS dir=-1 conf=0.51"],
                 "confirm_gate": ["edge_01: cross_cyc=0/2 (waiting)"],
                 "governor_gate": [],
                 "execution": "DENIED cross_confirm=1/2",
             }},
            # Passed confirm
            {"cycle": 2, "total_signals": 1, "active_symbol": "GBPUSD",
             "confirm_cycles": 2, "denial_reason": "",
             "pipeline_trace": {
                 "generated": ["edge_02 GBPUSD vol_expansion dir=1 conf=0.62 -> PASS"],
                 "threshold_gate": ["edge_02: PASS dir=1 conf=0.62"],
                 "confirm_gate": ["edge_02: CROSS_PASS (cycles=2/2)"],
                 "governor_gate": [],
                 "execution": "PASSED ALL GATES",
             }},
            # No signal
            {"cycle": 3, "total_signals": 0, "decision": "HOLD",
             "pipeline_trace": {"generated": [], "threshold_gate": [],
                                "confirm_gate": [], "governor_gate": [],
                                "execution": "NO_SIGNAL"}},
        ])
        runner = StressActivationRunner(log_path=path)
        report = runner.analyze(n_recent_cycles=1000)
        self.assertEqual(report["hypothetical_trades"], 2)
        self.assertEqual(report["actual_trades"], 1)
        self.assertEqual(report["missed_trades_due_to_confirm"], 1)
        self.assertEqual(report["opportunity_cost"]["missed_signals"], 1)
        self.assertEqual(report["opportunity_cost"]["missed_confirm1_signals"], 1)
        self.assertIn("USDJPY", report["opportunity_cost"]["by_symbol"])
        self.assertIn("GBPUSD", report["opportunity_cost"]["by_symbol"])
        self.assertEqual(report["opportunity_cost"]["by_symbol"]["USDJPY"]["hypothetical"], 1)
        self.assertEqual(report["opportunity_cost"]["by_symbol"]["USDJPY"]["actual"], 0)
        self.assertEqual(report["opportunity_cost"]["by_symbol"]["GBPUSD"]["hypothetical"], 1)
        self.assertEqual(report["opportunity_cost"]["by_symbol"]["GBPUSD"]["actual"], 1)

    # ------------------------------------------------------------------
    # Signal entered confirm but denied for other reason (not confirm)
    #   — should count in hypothetical but not in missed_due_to_confirm
    # ------------------------------------------------------------------

    def test_denied_other_reason_not_confirm(self):
        path = self._write_log([
            {"cycle": 5, "total_signals": 1, "active_symbol": "EURJPY",
             "confirm_cycles": 4, "denial_reason": "VEL blocked: exposure_smoothing",
             "pipeline_trace": {
                 "generated": ["edge_03 EURJPY mean_reversion dir=1 conf=0.55 -> PASS"],
                 "threshold_gate": ["edge_03: PASS dir=1 conf=0.55"],
                 "confirm_gate": ["edge_03: CROSS_PASS (cycles=3/2)"],
                 "governor_gate": [],
                 "execution": "DENIED VEL",
             }},
        ])
        runner = StressActivationRunner(log_path=path)
        report = runner.analyze(n_recent_cycles=1000)
        self.assertEqual(report["hypothetical_trades"], 1)
        self.assertEqual(report["actual_trades"], 1)  # confirm passed
        self.assertEqual(report["missed_trades_due_to_confirm"], 0)

    # ------------------------------------------------------------------
    # n_recent_cycles slicing
    # ------------------------------------------------------------------

    def test_n_recent_cycles_slicing(self):
        entries = []
        for i in range(1, 21):
            entries.append({
                "cycle": i,
                "total_signals": 1 if i >= 10 else 0,
                "active_symbol": "USDJPY",
                "confirm_cycles": 2 if i >= 10 else 0,
                "denial_reason": "",
                "pipeline_trace": {
                    "generated": [f"edge_{i:02d} USDJPY PASS"],
                    "threshold_gate": [f"edge_{i:02d}: PASS"],
                    "confirm_gate": [f"edge_{i:02d}: CROSS_PASS"] if i >= 10 else [],
                    "governor_gate": [],
                    "execution": "PASSED ALL GATES" if i >= 10 else "NO_SIGNAL",
                },
            })
        path = self._write_log(entries)

        # Only last 5 cycles (16-20) — all have signals
        runner = StressActivationRunner(log_path=path)
        report = runner.analyze(n_recent_cycles=5)
        self.assertEqual(report["hypothetical_trades"], 5)
        self.assertEqual(report["actual_trades"], 5)

        # Only 2 cycles
        report2 = runner.analyze(n_recent_cycles=2)
        self.assertEqual(report2["hypothetical_trades"], 2)
        self.assertEqual(report2["actual_trades"], 2)

    # ------------------------------------------------------------------
    # Warning message
    # ------------------------------------------------------------------

    def test_warning_present(self):
        path = self._write_log([
            {"cycle": 1, "total_signals": 1, "active_symbol": "EURUSD",
             "confirm_cycles": 1, "denial_reason": "Insufficient cross-projection confirm: 1/2",
             "pipeline_trace": {
                 "generated": ["edge_01 EURUSD mean_reversion dir=1 conf=0.55 -> PASS"],
                 "threshold_gate": ["edge_01: PASS dir=1 conf=0.55"],
                 "confirm_gate": ["edge_01: cross_cyc=0/2 (waiting)"],
                 "governor_gate": [],
                 "execution": "DENIED cross_confirm=1/2",
             }},
        ])
        runner = StressActivationRunner(log_path=path)
        report = runner.analyze()
        self.assertIn("ANALYSIS ONLY", report["warning"])


if __name__ == "__main__":
    unittest.main()
