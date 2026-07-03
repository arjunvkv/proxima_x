import json
import logging
import os
import re
import time

logger = logging.getLogger("proxima_ops.dashboard.unified_state_aggregator")

STATE_DIR = "state"

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


class UnifiedStateAggregator:
    def __init__(self, state_dir: str = STATE_DIR):
        self._state_dir = state_dir

    def _read_json(self, filename: str, default=None):
        path = os.path.join(self._state_dir, filename)
        if not os.path.exists(path):
            logger.debug("File not found: %s", path)
            return default
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return default

    def _read_jsonl_last(self, filename: str, default=None):
        path = os.path.join(self._state_dir, filename)
        if not os.path.exists(path):
            logger.debug("File not found: %s", path)
            return default
        try:
            with open(path, "r") as f:
                last = None
                for line in f:
                    line = line.strip()
                    if line:
                        last = json.loads(line)
                return last
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return default

    def _grep_log(self, filename: str, patterns: list[str], max_lines: int = 10):
        path = os.path.join(self._state_dir, filename)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return {}

        results = {}
        for line in reversed(lines):
            for pat in patterns:
                m = re.search(pat, line)
                if m:
                    results[pat] = m
            if len(results) >= len(patterns):
                break
        return results

    def aggregate(self) -> dict:
        trace = self._read_jsonl_last("live_pipeline_trace.jsonl", default={})
        monitor = self._read_json("live_monitor_state.json", default={})
        regime = self._read_json("regime_exposure_state.json", default={})
        watchdog = self._read_json("mt5_watchdog_state.json", default={})
        cb = self._read_json("circuit_breaker_state.json", default={})
        lifecycle = self._read_json("trade_lifecycle_state.json", default={})
        activation = self._read_json("activation_watch_state.json", default={})
        amplifier = self._read_json("exposure_amplifier_state.json", default={})
        distributor = self._read_json("distribution_observer_state.json", default={})
        staircase = self._read_json("volume_staircase_state.json", default={})
        mt5_snapshot = self._read_json("live_mt5_state.json", default={})
        log_matches = self._grep_log(
            "live_bg_stderr.log",
            patterns=[
                r"\[PIPELINE\] cycle=(\d+)",
            ],
        )

        md_summary = trace.get("md_summary", {})
        rsi_by_symbol = {}
        atr_by_symbol = {}
        for sym, data in md_summary.items():
            if isinstance(data, dict):
                rsi_by_symbol[sym] = data.get("rsi")
                atr_by_symbol[sym] = data.get("atr")

        regime_labels = list(regime.get("per_symbol", {}).keys()) if regime else []

        signals_detail = trace.get("signals_detail", [])
        last_signal = signals_detail[-1] if signals_detail else {}
        last_signal_edge = last_signal.get("edge_id", None)
        last_signal_confidence = last_signal.get("confidence", None)
        last_signal_direction_raw = last_signal.get("direction", None)
        dir_map = {1: "BUY", -1: "SELL", 0: "NEUTRAL"}
        last_signal_direction = dir_map.get(last_signal_direction_raw, str(last_signal_direction_raw)) if last_signal_direction_raw is not None else None

        confirm_gate = trace.get("confirm_gate", {})
        confirm_map = confirm_gate.get("confirm_map", {})
        confirm_count = max(confirm_map.values()) if confirm_map else 0

        governor = trace.get("governor", {})
        segl_state = governor.get("segl_state", None)
        governor_decision = governor.get("reason", None)

        vel = trace.get("vel", {})
        vel_block_reason = vel.get("reason", None) if not vel.get("allowed", True) else None

        cb_triggered = cb.get("triggered", False) if cb else False

        cycle_count = trace.get("cycle", 0)

        pipeline_match = log_matches.get(r"\[PIPELINE\] cycle=(\d+)")
        if pipeline_match and cycle_count == 0:
            try:
                cycle_count = int(pipeline_match.group(1))
            except (IndexError, ValueError):
                pass

        mt5_acct = mt5_snapshot.get("account_summary", {}) if mt5_snapshot else {}
        open_positions = len(mt5_snapshot.get("open_positions", [])) if mt5_snapshot else 0
        if not open_positions and trace:
            open_positions = trace.get("open_positions", 0)

        balance = mt5_acct.get("balance", None)
        equity = mt5_acct.get("equity", None)
        open_pnl = 0.0
        if mt5_snapshot:
            positions = mt5_snapshot.get("open_positions", [])
            open_pnl = sum(p.get("profit", 0) + p.get("swap", 0) for p in positions)

        if MT5_AVAILABLE and mt5.initialize():
            try:
                acct = mt5.account_info()
                if acct:
                    balance = acct.balance
                    equity = acct.equity
                pos = mt5.positions_get()
                if pos is not None:
                    open_positions = len(pos)
            except Exception as exc:
                logger.warning("MT5 live query error: %s", exc)

        execution_decision = trace.get("execution", {}).get("decision", None)

        mof_score_val = None
        mof_state_val = None
        if governor:
            reason_str = governor.get("reason", "")
            m = re.search(r"mof_score=([\d.]+)", str(reason_str))
            if m:
                try:
                    mof_score_val = float(m.group(1))
                except ValueError:
                    pass

        monitor_summary = monitor.get("summarize", {}) if monitor else {}
        expectancy_data = monitor.get("expectancy", {}) if monitor else {}
        total_trades = expectancy_data.get("total_closed_trades", 0)
        win_rate = expectancy_data.get("win_rate", 0.0)
        expectancy = expectancy_data.get("expectancy", 0.0)

        if not total_trades:
            lifecycle_trades = lifecycle.get("trade_history", []) if lifecycle else []
            total_trades = len(lifecycle_trades)
            wins = sum(1 for t in lifecycle_trades if t.get("pnl", 0) is not None and t.get("pnl", 0) > 0)
            total_closed = sum(1 for t in lifecycle_trades if t.get("pnl") is not None)
            win_rate = round(wins / total_closed, 4) if total_closed > 0 else 0.0
            pnls = [t["pnl"] for t in lifecycle_trades if t.get("pnl") is not None]
            expectancy = round(sum(pnls) / len(pnls), 4) if pnls else 0.0

        amplifier_mult = 1.0
        if amplifier:
            amplifier_mult = amplifier.get("multiplier", 1.0)
        loss_streak = amplifier.get("loss_streak", 0) if amplifier else 0

        staircase_phase = staircase.get("current_phase", 1) if staircase else 1

        cb_rules = {}
        if cb:
            cb_rules = {
                k: cb[k] for k in [
                    "triggered", "trigger_reasons", "consecutive_mt5_failures",
                    "slippage_rolling_avg", "session_pnl", "drawdown_limit",
                ] if k in cb
            }

        drawdown_from_peak = 0.0
        if balance and mt5_acct:
            mt5_positions = mt5_snapshot.get("open_positions", []) if mt5_snapshot else []
            mt5_equity_val = equity if equity else balance
            drawdown_from_peak = round((balance - mt5_equity_val) / balance * 100, 2)

        cb_session_pnl = cb.get("session_pnl", 0.0) if cb else 0.0
        daily_pnl = cb_session_pnl

        activation_events = []
        if activation:
            counters = activation.get("event_counters", {})
            last_cycles = activation.get("last_event_cycle", {})
            recent = activation.get("recent_events", {})
            for event_type in [
                "rsi_extreme", "atr_spike", "edge_confidence_hit",
                "threshold_pass", "confirm_progress", "governor_arm",
                "vel_block", "execution_attempt",
            ]:
                activation_events.append({
                    "type": event_type,
                    "count": counters.get(event_type, 0),
                    "last_cycle": last_cycles.get(event_type, 0),
                    "recent_timestamps": recent.get(event_type, []),
                })

        dist_regime = "unknown"
        if distributor:
            occ = distributor.get("regime_occupancy", {})
            regime_labels_local = list(occ.keys()) if occ else []
            if regime_labels_local:
                all_regimes = set()
                for sym_regs in occ.values():
                    if isinstance(sym_regs, dict):
                        all_regimes.update(sym_regs.keys())
                dist_regime = ", ".join(sorted(all_regimes)) if all_regimes else "unknown"

        process_alive = os.path.exists(os.path.join(self._state_dir, "live_pipeline_trace.jsonl"))
        mt5_connected = MT5_AVAILABLE and mt5.terminal_info() is not None if hasattr(mt5, "terminal_info") else bool(mt5_snapshot)

        now = time.time()

        result = {
            "timestamp": now,
            "market_state": {
                "rsi_by_symbol": rsi_by_symbol,
                "atr_by_symbol": atr_by_symbol,
                "regime_labels": regime_labels,
            },
            "execution_state": {
                "segl_state": segl_state,
                "open_positions": open_positions,
                "open_pnl": round(open_pnl, 2),
                "balance": balance,
                "equity": equity,
                "last_signal_edge": last_signal_edge,
                "last_signal_confidence": last_signal_confidence,
                "last_signal_direction": last_signal_direction,
                "confirm_count": confirm_count,
                "governor_decision": governor_decision,
                "vel_block_reason": vel_block_reason,
                "circuit_breaker_triggered": cb_triggered,
                "cycle_count": cycle_count,
            },
            "risk_state": {
                "drawdown_from_peak": drawdown_from_peak,
                "circuit_breaker_rules": cb_rules,
                "loss_streak": loss_streak,
                "daily_pnl": round(daily_pnl, 2),
            },
            "performance_state": {
                "staircase_phase": staircase_phase,
                "amplifier_multiplier": amplifier_mult,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "expectancy": expectancy,
            },
            "monitoring_state": {
                "activation_events": activation_events,
                "distribution_regime": dist_regime,
            },
            "system_health": {
                "process_alive": process_alive,
                "mt5_connected": mt5_connected,
                "last_cycle_time": trace.get("timestamp", now) if trace else now,
            },
        }
        return result

    def aggregate_to_file(self, output_path: str = "state/unified_system_state.json"):
        data = self.aggregate()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        try:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Unified system state written to %s", output_path)
        except Exception as exc:
            logger.error("Failed to write unified state to %s: %s", output_path, exc)
