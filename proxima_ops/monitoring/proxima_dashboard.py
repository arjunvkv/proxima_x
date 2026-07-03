import json
import os
import time
from collections import deque
from typing import Optional

from proxima_x.proxima_ops.execution.execution_ledger import ExecutionLedger


class ProximaDashboard:
    def __init__(self, ledger: Optional[ExecutionLedger] = None):
        self.snapshots = deque(maxlen=100)
        self.current = {}
        self._ledger = ledger or ExecutionLedger()

    def update(self, cycle_data: dict):
        self.current = cycle_data
        self.snapshots.append(cycle_data)

    def _v(self, key: str, default="N/A"):
        val = self.current.get(key, default)
        return val if val is not None else default

    def render(self) -> str:
        stats = self._ledger.get_stats()
        c = self.current

        mt5_status = self._v("mt5_status", "DISCONNECTED")
        balance = self._v("balance", 0.0)
        equity = self._v("equity", 0.0)
        floating_pnl = self._v("floating_pnl", 0.0)
        realized_pnl = self._v("realized_pnl", 0.0)
        open_positions = self._v("open_positions", [])
        open_count = len(open_positions) if isinstance(open_positions, list) else 0
        margin = self._v("margin", 0.0)
        margin_level = self._v("margin_level", "N/A")
        mof_state = self._v("mof_state", "N/A")
        mof_score = self._v("mof_score", 0.0)
        rf_drift = self._v("rf_drift", 0.0)
        rf_readiness = self._v("rf_readiness", "N/A")
        segl_state = self._v("segl_state", "N/A")
        intent_status = self._v("intent_status", "N/A")
        drift_status = self._v("drift_status", "OK")
        active_edge = self._v("active_edge", "edge_04")
        edge_signal = self._v("edge_signal", "")
        edge_confidence = self._v("edge_confidence", 0.0)
        last_exec_time = self._v("last_exec_time", "N/A")
        last_exec_result = self._v("last_exec_result", "N/A")
        last_rej_time = self._v("last_rej_time", "N/A")
        last_rej_reason = self._v("last_rej_reason", "N/A")
        lifecycle_health = self._v("lifecycle_health", "OK")
        cycle_label = self._v("cycle", "N/A")
        decision = self._v("decision", "N/A")
        state = self._v("state", "N/A")
        pnl = self._v("pnl", "N/A")
        last_signal_time = self._v("last_signal_time", "N/A")
        arming_eligible = self._v("arming_eligible", "N/A")

        total_trades = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0)
        total_pnl = stats.get("total_pnl", 0)
        avg_win = stats.get("avg_win", 0)
        avg_loss = stats.get("avg_loss", 0)
        profit_factor = stats.get("profit_factor", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)

        max_dd = self._v("max_drawdown", 0.0)

        win_rate_str = f"{win_rate:.1f}%" if total_trades > 0 else "N/A"
        pf_str = f"{profit_factor:.2f}" if total_trades > 0 else "N/A"
        avg_win_str = f"${avg_win:.2f}" if wins > 0 else "N/A"
        avg_loss_str = f"${avg_loss:.2f}" if losses > 0 else "N/A"

        mt5_line = "CONNECTED" if str(mt5_status).upper() in ("CONNECTED", "TRUE", "1", "YES") else "DISCONNECTED"

        lines = []
        lines.append("\u2554" + "\u2550" * 54 + "\u2557")
        lines.append("\u2551" + "              PROXIMA GOVERNANCE KERNEL              " + "\u2551")
        lines.append("\u2551" + "              Live Operations Dashboard               " + "\u2551")
        lines.append("\u255a" + "\u2550" * 54 + "\u255d")
        lines.append("")
        lines.append(
            f"MT5: {mt5_line:<14s} Balance: ${balance:>8,.2f}   Equity: ${equity:>8,.2f}"
        )
        lines.append(
            f"Open Positions: {open_count:<3d}        Margin: ${margin:>8,.2f}   Margin Level: {margin_level}"
        )
        if open_count > 0 and isinstance(open_positions, list):
            for pos in open_positions:
                sym = pos.get("symbol", "?")
                direc = pos.get("direction", pos.get("type", "?"))
                vol = pos.get("volume", 0)
                price = pos.get("price", pos.get("open_price", 0))
                pos_pnl = pos.get("pnl", pos.get("profit", 0))
                lines.append(f"    {sym:<8s} {direc:<6s} {vol:<5.2f} @ {price:<10.5f}  PnL: ${pos_pnl:<8.2f}")
        lines.append(f"Floating PnL: ${floating_pnl:>8,.2f}     Realized PnL: ${realized_pnl:>8,.2f}")
        lines.append("")
        lines.append("\u2500\u2500 Governance State \u2500\u2500")
        lines.append(
            f"SEGL: {str(segl_state):<12s}  MOF: {mof_state}({mof_score})  RF Drift: {rf_drift}"
        )
        lines.append(
            f"RF Readiness: {rf_readiness:<12s}  Intent: {intent_status:<12s}  Drift: {drift_status}"
        )
        lines.append(f"Lifecycle: {lifecycle_health:<12s}")
        lines.append("")
        lines.append("\u2500\u2500 Edge State \u2500\u2500")
        lines.append(f"Active: {active_edge} {edge_signal}   Confidence: {edge_confidence}")
        lines.append(f"Last Signal: {last_signal_time:<20s}   Arming Eligible: {arming_eligible}")
        lines.append("")
        lines.append("\u2500\u2500 Execution History \u2500\u2500")
        lines.append(
            f"Total Trades: {total_trades:<3d}   Win Rate: {win_rate_str:<8s}   Profit Factor: {pf_str:<8s}"
        )
        lines.append(
            f"Avg Win: {avg_win_str:<12s}   Avg Loss: {avg_loss_str:<12s}   Max DD: ${max_dd:>7,.2f}"
        )
        lines.append("")
        lines.append("\u2500\u2500 Last Cycle \u2500\u2500")
        lines.append(f"Cycle: {str(cycle_label):<10s}   Decision: {str(decision):<16s}   State: {str(state):<12s}   PnL: {str(pnl)}")
        lines.append("")
        lines.append("\u2500\u2500 Last Execution \u2500\u2500")
        lines.append(f"Time: {last_exec_time:<20s}   Result: {last_exec_result}")
        lines.append("")
        if last_rej_time != "N/A":
            lines.append("\u2500\u2500 Last Rejection \u2500\u2500")
            lines.append(f"Time: {last_rej_time:<20s}   Reason: {last_rej_reason}")
            lines.append("")
        return "\n".join(lines)

    def get_latest(self) -> dict:
        return self.current

    def get_history(self, n: int = 5) -> list:
        return list(self.snapshots)[-n:]

    def get_summary(self) -> dict:
        n = len(self.snapshots)
        if n == 0:
            return {"snapshots": 0}
        keys = {"balance", "equity", "floating_pnl", "rf_drift", "mof_score", "edge_confidence"}
        agg = {}
        for k in keys:
            vals = [s.get(k) for s in self.snapshots if isinstance(s.get(k), (int, float))]
            if vals:
                agg[f"{k}_min"] = min(vals)
                agg[f"{k}_max"] = max(vals)
                agg[f"{k}_avg"] = sum(vals) / len(vals)
                agg[f"{k}_last"] = vals[-1]
        agg["snapshots"] = n
        stats = self._ledger.get_stats()
        agg.update({f"ledger_{k}": v for k, v in stats.items()})
        return agg

    def log_event(self, cycle_data: dict):
        log_path = os.path.join("state", "dashboard_log.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        entry = {
            "timestamp": time.time(),
            "data": cycle_data,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
