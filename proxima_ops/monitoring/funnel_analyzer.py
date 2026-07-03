import json
import os
import logging
from collections import defaultdict

logger = logging.getLogger("proxima_ops.funnel_analyzer")

DEFAULT_TRACE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "state", "live_pipeline_trace.jsonl"
)

VEL_REASON_CATEGORIES = ["temporal_spacing", "exposure_smoothing", "burst_prevention"]

TRACE_KEYS = [
    "cycle", "timestamp", "signals_generated", "signals_detail",
    "threshold_pass_count", "confirm_gate", "governor", "vel",
    "circuit_breaker", "execution", "open_positions", "pipeline_funnel"
]


class FunnelAnalyzer:
    def __init__(self, trace_path: str = None):
        self._trace_path = trace_path or DEFAULT_TRACE_PATH
        logger.info("FunnelAnalyzer initialized with trace_path=%s", self._trace_path)

    def load_trace(self, path: str = None) -> list[dict]:
        resolved = path or self._trace_path
        if not os.path.exists(resolved):
            logger.warning("Trace file not found: %s", resolved)
            return []
        entries = []
        with open(resolved, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    logger.error("JSON decode error at line %d: %s", line_no, e)
        logger.info("Loaded %d trace entries from %s", len(entries), resolved)
        return entries

    def _get_confirm_pass(self, entry: dict) -> bool:
        cg = entry.get("confirm_gate", {})
        if isinstance(cg, dict):
            return cg.get("confirm_pass", False)
        return False

    def _get_governor_authorized(self, entry: dict) -> bool:
        gov = entry.get("governor", {})
        if isinstance(gov, dict):
            return gov.get("authorized", False)
        return False

    def _get_vel_allowed(self, entry: dict) -> bool:
        v = entry.get("vel", {})
        if isinstance(v, dict):
            return v.get("allowed", False)
        return False

    def _get_vel_reason(self, entry: dict) -> str:
        v = entry.get("vel", {})
        if isinstance(v, dict):
            return v.get("reason", "")
        return ""

    def _get_executed(self, entry: dict) -> bool:
        ex = entry.get("execution", {})
        if isinstance(ex, dict):
            return ex.get("decision", "HOLD") != "HOLD"
        return False

    def _get_pipeline_funnel(self, entry: dict) -> dict:
        pf = entry.get("pipeline_funnel", {})
        if isinstance(pf, dict):
            return pf
        return {}

    def _extract_vel_category(self, reason: str) -> str:
        if not reason:
            return "unknown"
        reason_lower = reason.lower()
        for cat in VEL_REASON_CATEGORIES:
            if reason_lower.startswith(cat):
                return cat
        return "other"

    def _extract_symbols(self, entry: dict) -> set[str]:
        symbols: set[str] = set()
        signals_detail = entry.get("signals_detail", [])
        if isinstance(signals_detail, list):
            for sd in signals_detail:
                if isinstance(sd, dict):
                    sym = sd.get("symbol")
                    if sym:
                        symbols.add(sym)
        cg = entry.get("confirm_gate", {})
        if isinstance(cg, dict):
            confirm_map = cg.get("confirm_map", {})
            if isinstance(confirm_map, dict):
                for key in confirm_map:
                    parts = key.split("_")
                    if len(parts) >= 2:
                        sym = "_".join(parts[:-1])
                        symbols.add(sym)
        return symbols

    def funnel_report(self, entries: list[dict]) -> dict:
        if not entries:
            return {"total_cycles": 0, "stages": {}, "kill_chain": [], "dominant_blocker": "none"}
        total = len(entries)
        cycles_with_signals = 0
        total_signals = 0
        cycles_with_threshold = 0
        total_threshold = 0
        cycles_with_confirm = 0
        cycles_governor_armed = 0
        cycles_vel_allowed = 0
        vel_blocked_reasons: dict[str, int] = defaultdict(int)
        total_executed = 0

        for entry in entries:
            sig_count = entry.get("signals_generated", 0)
            total_signals += sig_count
            if sig_count > 0:
                cycles_with_signals += 1

            thresh_count = entry.get("threshold_pass_count", 0)
            total_threshold += thresh_count
            if thresh_count > 0:
                cycles_with_threshold += 1

            if self._get_confirm_pass(entry):
                cycles_with_confirm += 1

            if self._get_governor_authorized(entry):
                cycles_governor_armed += 1

            vel_allowed = self._get_vel_allowed(entry)
            if vel_allowed:
                cycles_vel_allowed += 1
            else:
                reason = self._get_vel_reason(entry)
                cat = self._extract_vel_category(reason)
                vel_blocked_reasons[cat] += 1

            if self._get_executed(entry):
                total_executed += 1

        stages = {
            "signal_generation": {
                "total_cycles_with_signals": cycles_with_signals,
                "avg_signals_per_cycle": round(total_signals / total, 4) if total else 0.0,
                "pct_cycles_with_any_signal": f"{(cycles_with_signals / total * 100):.1f}%",
            },
            "threshold_pass": {
                "total_cycles_with_threshold_pass": cycles_with_threshold,
                "avg_threshold_per_cycle": round(total_threshold / total, 4) if total else 0.0,
                "pct_cycles_with_any_pass": f"{(cycles_with_threshold / total * 100):.1f}%",
            },
            "confirm_pass": {
                "total_cycles_with_confirm_pass": cycles_with_confirm,
                "pct_cycles_with_confirm": f"{(cycles_with_confirm / total * 100):.1f}%",
            },
            "governor_ready": {
                "total_cycles_governor_armed": cycles_governor_armed,
                "pct_cycles_armed": f"{(cycles_governor_armed / total * 100):.1f}%",
            },
            "vel_allowed": {
                "total_cycles_vel_allowed": cycles_vel_allowed,
                "pct_cycles_vel_allowed": f"{(cycles_vel_allowed / total * 100):.1f}%",
                "blocked_reasons": {
                    "temporal_spacing": vel_blocked_reasons.get("temporal_spacing", 0),
                    "exposure_smoothing": vel_blocked_reasons.get("exposure_smoothing", 0),
                    "burst_prevention": vel_blocked_reasons.get("burst_prevention", 0),
                },
            },
            "executed": {
                "total_trades": total_executed,
                "execution_rate": f"{(total_executed / total * 100):.1f}%",
            },
        }

        kill_chain_raw = [
            ("start", total),
            ("threshold", cycles_with_threshold),
            ("confirm", cycles_with_confirm),
            ("governor", cycles_governor_armed),
            ("vel", cycles_vel_allowed),
            ("executed", total_executed),
        ]
        kill_chain = []
        prev = None
        for stage, count in kill_chain_raw:
            if prev is None:
                drop = 0.0
            else:
                if prev > 0:
                    drop = (prev - count) / prev * 100
                else:
                    drop = 0.0
            kill_chain.append({
                "stage": stage,
                "cycles_reaching": count,
                "drop_from_prev" if prev is not None else "drop_pct": f"{drop:.1f}%",
            })
            prev = count

        return {
            "total_cycles": total,
            "stages": stages,
            "kill_chain": kill_chain,
            "dominant_blocker": self.dominant_blocker(entries),
        }

    def per_symbol_report(self, entries: list[dict]) -> dict:
        if not entries:
            return {}
        symbol_data: dict[str, dict] = {}
        for entry in entries:
            symbols = self._extract_symbols(entry)
            sig_count = entry.get("signals_generated", 0)
            thresh_count = entry.get("threshold_pass_count", 0)
            confirm_pass = self._get_confirm_pass(entry)
            gov_auth = self._get_governor_authorized(entry)
            vel_allowed = self._get_vel_allowed(entry)
            executed = self._get_executed(entry)

            for sym in symbols:
                if sym not in symbol_data:
                    symbol_data[sym] = {
                        "total_cycles": 0,
                        "cycles_with_signals": 0,
                        "cycles_with_threshold": 0,
                        "cycles_with_confirm": 0,
                        "cycles_governor_armed": 0,
                        "cycles_vel_allowed": 0,
                        "cycles_executed": 0,
                    }
                sd = symbol_data[sym]
                sd["total_cycles"] += 1
                if sig_count > 0:
                    sd["cycles_with_signals"] += 1
                if thresh_count > 0:
                    sd["cycles_with_threshold"] += 1
                if confirm_pass:
                    sd["cycles_with_confirm"] += 1
                if gov_auth:
                    sd["cycles_governor_armed"] += 1
                if vel_allowed:
                    sd["cycles_vel_allowed"] += 1
                if executed:
                    sd["cycles_executed"] += 1

        result = {}
        for sym, sd in symbol_data.items():
            tc = sd["total_cycles"]
            result[sym] = {
                "total_cycles": tc,
                "signals": {
                    "count": sd["cycles_with_signals"],
                    "pct": f"{(sd['cycles_with_signals'] / tc * 100):.1f}%" if tc else "0.0%",
                },
                "threshold": {
                    "count": sd["cycles_with_threshold"],
                    "pct": f"{(sd['cycles_with_threshold'] / tc * 100):.1f}%" if tc else "0.0%",
                },
                "confirm": {
                    "count": sd["cycles_with_confirm"],
                    "pct": f"{(sd['cycles_with_confirm'] / tc * 100):.1f}%" if tc else "0.0%",
                },
                "governor": {
                    "count": sd["cycles_governor_armed"],
                    "pct": f"{(sd['cycles_governor_armed'] / tc * 100):.1f}%" if tc else "0.0%",
                },
                "vel": {
                    "count": sd["cycles_vel_allowed"],
                    "pct": f"{(sd['cycles_vel_allowed'] / tc * 100):.1f}%" if tc else "0.0%",
                },
                "executed": {
                    "count": sd["cycles_executed"],
                    "pct": f"{(sd['cycles_executed'] / tc * 100):.1f}%" if tc else "0.0%",
                },
            }
        return result

    def windowed_report(self, entries: list[dict], window: int = 100) -> list[dict]:
        if not entries or window <= 0:
            return []
        reports = []
        for start in range(0, len(entries), window):
            chunk = entries[start:start + window]
            if not chunk:
                continue
            report = self.funnel_report(chunk)
            report["window_start"] = start
            report["window_end"] = start + len(chunk) - 1
            first_cycle = chunk[0].get("cycle", 0)
            last_cycle = chunk[-1].get("cycle", 0)
            report["cycle_range"] = f"{first_cycle}-{last_cycle}"
            reports.append(report)
        return reports

    def kill_distribution(self, entries: list[dict]) -> dict:
        if not entries:
            return {}
        total = len(entries)
        cycles_with_threshold = sum(1 for e in entries if e.get("threshold_pass_count", 0) > 0)
        cycles_with_confirm = sum(1 for e in entries if self._get_confirm_pass(e))
        cycles_governor = sum(1 for e in entries if self._get_governor_authorized(e))
        cycles_vel = sum(1 for e in entries if self._get_vel_allowed(e))
        cycles_exec = sum(1 for e in entries if self._get_executed(e))

        stages = [
            ("threshold", total, cycles_with_threshold),
            ("confirm", cycles_with_threshold, cycles_with_confirm),
            ("governor", cycles_with_confirm, cycles_governor),
            ("vel", cycles_governor, cycles_vel),
            ("executed", cycles_vel, cycles_exec),
        ]
        result = {}
        for name, came_in, passed in stages:
            if came_in > 0:
                stopped_pct = (came_in - passed) / came_in * 100
            else:
                stopped_pct = 0.0
            result[name] = {
                "reaching": came_in,
                "passed": passed,
                "stopped_pct": f"{stopped_pct:.1f}%",
            }
        return result

    def dominant_blocker(self, entries: list[dict]) -> str:
        if not entries:
            return "none"
        total = len(entries)
        stages = [
            ("threshold", total, sum(1 for e in entries if e.get("threshold_pass_count", 0) > 0)),
            ("confirm", 0, 0),
            ("governor", 0, 0),
            ("vel", 0, 0),
            ("executed", 0, 0),
        ]
        thresh_c = sum(1 for e in entries if e.get("threshold_pass_count", 0) > 0)
        confirm_c = sum(1 for e in entries if self._get_confirm_pass(e))
        gov_c = sum(1 for e in entries if self._get_governor_authorized(e))
        vel_c = sum(1 for e in entries if self._get_vel_allowed(e))
        exec_c = sum(1 for e in entries if self._get_executed(e))

        drops = [
            ("threshold", total, thresh_c),
            ("confirm", thresh_c, confirm_c),
            ("governor", confirm_c, gov_c),
            ("vel", gov_c, vel_c),
            ("executed", vel_c, exec_c),
        ]
        best_stage = "none"
        best_drop = -1.0
        for name, prev_c, curr_c in drops:
            if prev_c > 0:
                drop = (prev_c - curr_c) / prev_c * 100
                if drop > best_drop:
                    best_drop = drop
                    best_stage = name
        return best_stage


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    analyzer = FunnelAnalyzer()
    entries = analyzer.load_trace()
    if not entries:
        logger.warning("No trace entries found. Run a trading cycle first to generate trace data.")
        print(json.dumps({"error": "no trace data"}, indent=2))
        return
    report = analyzer.funnel_report(entries)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
