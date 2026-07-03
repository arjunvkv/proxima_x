import json
import logging
import os
import time
from collections import defaultdict, deque


logger = logging.getLogger("proxima_ops.monitoring.live_monitor")

STATE_PATH = "state/live_monitor_state.json"


class LiveMonitor:
    def __init__(self, state_path: str = STATE_PATH, max_history: int = 1000):
        self._state_path = state_path
        self._max_history = max_history
        self.cycle_history: deque[dict] = deque(maxlen=max_history)
        self.confirm_levels: dict[str, int] = {"0/2": 0, "1/2": 0, "2/2": 0}
        self.symbol_agreement: dict[str, dict] = defaultdict(
            lambda: {"qualifying_cycles": 0, "cross_pass_cycles": 0, "total_cycles": 0}
        )
        self.edge_persistence: dict[str, dict] = defaultdict(
            lambda: {"appearances": 0, "threshold_passes": 0, "confirm_passes": 0, "first_seen": 0, "last_seen": 0}
        )
        self.trades: list[dict] = []
        self._open_by_ticket: dict[int, dict] = {}
        self._cycle_count: int = 0
        self._session_start: float = time.time()
        self._load_state()

    # ── Public API ─────────────────────────────────────────────────────────

    def record_cycle(self, cycle_data: dict) -> None:
        self._cycle_count += 1
        snapshot = self._build_snapshot(cycle_data)
        self.cycle_history.append(snapshot)
        self._track_confirm_levels(cycle_data)
        self._track_symbol_agreement(cycle_data)
        self._track_edge_persistence(cycle_data)
        self._track_trade_lifecycle(cycle_data)

    def summarize(self) -> dict:
        return {
            "monitor_version": "1.0.0",
            "session_start": self._session_start,
            "session_duration": time.time() - self._session_start,
            "total_cycles": self._cycle_count,
            "cycles_in_history": len(self.cycle_history),
            "confirm_levels": dict(self.confirm_levels),
            "symbol_agreement": dict(self.symbol_agreement),
            "edge_persistence": dict(self.edge_persistence),
            "expectancy": self.expectancy_report(),
            "confirm_stability": self.confirm_stability_report(),
            "total_trades_recorded": len(self.trades),
        }

    def expectancy_report(self) -> dict:
        closed = [t for t in self.trades if t.get("close_time") is not None]
        total_pnl = sum(t.get("pnl", 0.0) for t in closed)
        total_trades = len(closed)
        expectancy = total_pnl / total_trades if total_trades > 0 else 0.0
        by_symbol: dict[str, list[float]] = defaultdict(list)
        for t in closed:
            by_symbol[t.get("symbol", "UNKNOWN")].append(t.get("pnl", 0.0))
        per_symbol = {}
        for sym, pnls in by_symbol.items():
            n = len(pnls)
            per_symbol[sym] = {
                "trades": n,
                "total_pnl": round(sum(pnls), 2),
                "avg_pnl": round(sum(pnls) / n, 4) if n > 0 else 0.0,
            }
        wins = sum(1 for t in closed if t.get("pnl", 0.0) > 0)
        losses = sum(1 for t in closed if t.get("pnl", 0.0) <= 0)
        return {
            "total_closed_trades": total_trades,
            "total_pnl": round(total_pnl, 2),
            "expectancy": round(expectancy, 4),
            "win_count": wins,
            "loss_count": losses,
            "win_rate": round(wins / total_trades, 4) if total_trades > 0 else 0.0,
            "per_symbol": per_symbol,
            "open_trades": len(self._open_by_ticket),
        }

    def confirm_stability_report(self) -> dict:
        levels = dict(self.confirm_levels)
        total_confirmed = sum(levels.values())
        level_pcts = {}
        for k, v in levels.items():
            level_pcts[k] = round(v / total_confirmed, 4) if total_confirmed > 0 else 0.0
        sym_agree = {}
        for sym, data in self.symbol_agreement.items():
            tc = data["total_cycles"]
            sym_agree[sym] = {
                "qualifying_cycles": data["qualifying_cycles"],
                "cross_pass_cycles": data["cross_pass_cycles"],
                "total_cycles_observed": tc,
                "qualifying_rate": round(data["qualifying_cycles"] / tc, 4) if tc > 0 else 0.0,
                "cross_pass_rate": round(data["cross_pass_cycles"] / tc, 4) if tc > 0 else 0.0,
            }
        edges_alive = {
            eid: data for eid, data in self.edge_persistence.items()
            if data["last_seen"] >= self._cycle_count - 5
        }
        return {
            "confirm_level_distribution": levels,
            "confirm_level_pcts": level_pcts,
            "symbol_agreement": sym_agree,
            "edge_persistence": dict(self.edge_persistence),
            "edges_currently_alive": len(edges_alive),
            "total_edges_seen": len(self.edge_persistence),
        }

    def save_state(self) -> None:
        state = {
            "confirm_levels": dict(self.confirm_levels),
            "symbol_agreement": {k: dict(v) for k, v in self.symbol_agreement.items()},
            "edge_persistence": {k: dict(v) for k, v in self.edge_persistence.items()},
            "trades": self.trades,
            "_open_by_ticket": {str(k): v for k, v in self._open_by_ticket.items()},
            "_cycle_count": self._cycle_count,
            "_session_start": self._session_start,
        }
        if os.path.dirname(self._state_path):
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        with open(self._state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info("LiveMonitor state saved to %s", self._state_path)

    def load_state(self) -> bool:
        return self._load_state()

    # ── Internal: snapshot builder ─────────────────────────────────────────

    @staticmethod
    def _build_snapshot(cycle_data: dict) -> dict:
        pt = cycle_data.get("pipeline_trace", {})
        gen = pt.get("generated", [])
        thresh = pt.get("threshold_gate", [])
        confirm = pt.get("confirm_gate", [])
        return {
            "cycle": cycle_data.get("cycle", 0),
            "timestamp": time.time(),
            "decision": cycle_data.get("decision", "UNKNOWN"),
            "signals_generated": len(gen),
            "threshold_passed": sum(1 for g in thresh if "PASS" in g),
            "confirm_passed": sum(1 for g in confirm if "PASS" in g),
            "confirm_cycles": cycle_data.get("confirm_cycles", 0),
            "segl_state": cycle_data.get("segl_state", ""),
            "vel_decision": cycle_data.get("vel_decision", ""),
            "active_edge": cycle_data.get("active_edge", ""),
            "active_symbol": cycle_data.get("active_symbol", ""),
            "active_direction": cycle_data.get("active_direction", ""),
            "active_confidence": cycle_data.get("active_confidence", 0.0),
            "denial_reason": cycle_data.get("denial_reason", ""),
            "open_positions": cycle_data.get("open_positions", 0),
            "balance": cycle_data.get("balance", 0.0),
            "mof_state": cycle_data.get("mof_state", ""),
            "regime": cycle_data.get("regime", ""),
            "execution_result": cycle_data.get("execution_result"),
            "close_result": cycle_data.get("close_result"),
            "close_reason": cycle_data.get("close_reason", ""),
        }

    # ── Internal: confirm-level tracking ──────────────────────────────────

    def _track_confirm_levels(self, cycle_data: dict) -> None:
        cc = cycle_data.get("confirm_cycles", 0)
        if cc >= 2:
            self.confirm_levels["2/2"] += 1
        elif cc >= 1:
            self.confirm_levels["1/2"] += 1
        else:
            self.confirm_levels["0/2"] += 1

    # ── Internal: symbol agreement tracking ───────────────────────────────

    def _track_symbol_agreement(self, cycle_data: dict) -> None:
        pt = cycle_data.get("pipeline_trace", {})
        generated = pt.get("generated", [])
        confirm_entries = pt.get("confirm_gate", [])

        qualifying_symbols: set[str] = set()
        for g in generated:
            sym = self._extract_symbol_from_generated(g)
            if sym and "PASS" in g:
                qualifying_symbols.add(sym)

        cross_pass_eids: set[str] = set()
        for entry in confirm_entries:
            if "CROSS_PASS" in entry:
                eid = self._extract_eid(entry)
                if eid:
                    cross_pass_eids.add(eid)

        eid_to_symbol: dict[str, str] = {}
        for g in generated:
            eid = self._extract_eid(g)
            sym = self._extract_symbol_from_generated(g)
            if eid and sym:
                eid_to_symbol[eid] = sym

        cross_pass_symbols: set[str] = set()
        for eid in cross_pass_eids:
            if eid in eid_to_symbol:
                cross_pass_symbols.add(eid_to_symbol[eid])

        for sym in qualifying_symbols:
            self.symbol_agreement[sym]["qualifying_cycles"] += 1
        for sym in cross_pass_symbols:
            self.symbol_agreement[sym]["cross_pass_cycles"] += 1
        for sym in self.symbol_agreement:
            self.symbol_agreement[sym]["total_cycles"] = self._cycle_count

    @staticmethod
    def _extract_symbol_from_generated(entry: str) -> str | None:
        parts = entry.split(" ")
        if len(parts) >= 3:
            candidate = parts[1]
            if candidate in ("EURUSD", "GBPUSD", "EURJPY", "USDJPY"):
                return candidate
        return None

    # ── Internal: edge persistence tracking ───────────────────────────────

    def _track_edge_persistence(self, cycle_data: dict) -> None:
        pt = cycle_data.get("pipeline_trace", {})
        generated = pt.get("generated", [])
        threshold_gate = pt.get("threshold_gate", [])
        confirm_gate = pt.get("confirm_gate", [])
        seen_eids: set[str] = set()
        for g in generated:
            eid = self._extract_eid(g)
            if eid:
                seen_eids.add(eid)
                self.edge_persistence[eid]["appearances"] += 1
                self.edge_persistence[eid]["last_seen"] = self._cycle_count
                if self.edge_persistence[eid]["first_seen"] == 0:
                    self.edge_persistence[eid]["first_seen"] = self._cycle_count
                if "PASS" in g:
                    self.edge_persistence[eid]["threshold_passes"] += 1
        for tg in threshold_gate:
            eid = self._extract_eid(tg)
            if eid and "PASS" in tg:
                self.edge_persistence[eid]["threshold_passes"] += 1
                seen_eids.add(eid)
        for cg in confirm_gate:
            eid = self._extract_eid(cg)
            if eid:
                seen_eids.add(eid)
                if "CROSS_PASS" in cg:
                    self.edge_persistence[eid]["confirm_passes"] += 1

    @staticmethod
    def _extract_eid(entry: str) -> str | None:
        parts = entry.split(" ")
        if parts:
            candidate = parts[0]
            if candidate.startswith("edge_"):
                return candidate
        return None

    # ── Internal: trade lifecycle tracking ────────────────────────────────

    def _track_trade_lifecycle(self, cycle_data: dict) -> None:
        exec_result = cycle_data.get("execution_result")
        if exec_result and exec_result.get("success"):
            ticket = exec_result.get("ticket", 0)
            if ticket and ticket not in self._open_by_ticket:
                data_cycle = cycle_data.get("cycle", self._cycle_count)
                trade = {
                    "signal_id": exec_result.get("signal_id", ""),
                    "symbol": exec_result.get("symbol", ""),
                    "direction": exec_result.get("direction", ""),
                    "volume": exec_result.get("volume", 0.0),
                    "entry_price": exec_result.get("price", 0.0),
                    "entry_time": time.time(),
                    "cycle_entry": data_cycle,
                    "fusion_sources": exec_result.get("fusion_sources", []),
                    "fusion_is_erl": exec_result.get("fusion_is_erl", False),
                    "vel_decision": cycle_data.get("vel_decision", ""),
                    "ticket": ticket,
                    "close_time": None,
                    "exit_price": None,
                    "pnl": None,
                    "close_reason": None,
                    "segl_entry": cycle_data.get("segl_state", ""),
                    "mof_entry": cycle_data.get("mof_state", ""),
                }
                self._open_by_ticket[ticket] = trade

        close_result = cycle_data.get("close_result")
        if close_result and close_result.get("success"):
            data_cycle = cycle_data.get("cycle", self._cycle_count)
            results_list = close_result.get("results", [])
            for r in results_list:
                ticket = r.get("ticket", 0)
                if ticket in self._open_by_ticket:
                    trade = self._open_by_ticket.pop(ticket)
                    trade["close_time"] = time.time()
                    trade["cycle_close"] = data_cycle
                    trade["exit_price"] = r.get("exit_price", 0.0)
                    trade["pnl"] = r.get("pnl", 0.0)
                    trade["close_reason"] = cycle_data.get("close_reason", "UNKNOWN")
                    entry_cycle = trade.get("cycle_entry", data_cycle)
                    trade["hold_cycles"] = data_cycle - entry_cycle
                    self.trades.append(trade)
                else:
                    logger.warning("Close for unknown open ticket=%s", ticket)

    # ── Internal: state persistence ───────────────────────────────────────

    def _load_state(self) -> bool:
        if self._state_path == ":memory:":
            return False
        if not os.path.exists(self._state_path):
            return False
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self.confirm_levels = defaultdict(int, state.get("confirm_levels", {}))
            raw_sa = state.get("symbol_agreement", {})
            self.symbol_agreement = defaultdict(
                lambda: {"qualifying_cycles": 0, "cross_pass_cycles": 0, "total_cycles": 0},
                {k: defaultdict(int, v) for k, v in raw_sa.items()},
            )
            raw_ep = state.get("edge_persistence", {})
            self.edge_persistence = defaultdict(
                lambda: {"appearances": 0, "threshold_passes": 0, "confirm_passes": 0, "first_seen": 0, "last_seen": 0},
                {k: defaultdict(int, v) for k, v in raw_ep.items()},
            )
            self.trades = state.get("trades", [])
            raw_open = state.get("_open_by_ticket", {})
            self._open_by_ticket = {int(k): v for k, v in raw_open.items()}
            self._cycle_count = state.get("_cycle_count", 0)
            self._session_start = state.get("_session_start", time.time())
            logger.info("LiveMonitor state loaded from %s (%d cycles)", self._state_path, self._cycle_count)
            return True
        except Exception as e:
            logger.warning("Failed to load LiveMonitor state: %s", e)
            return False
