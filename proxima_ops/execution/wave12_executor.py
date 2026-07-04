import sys, os, json, time, traceback, math, logging
from collections import deque
from typing import Optional, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np

logger = logging.getLogger("proxima_ops.execution.wave12_executor")

from proxima_x.proxima_ops.execution.mt5_connector import MT5Connector
from proxima_x.proxima_ops.execution.order_manager import OrderManager
from proxima_x.proxima_ops.execution.symbol_direction_lock import SymbolDirectionLock
from proxima_x.proxima_ops.governance.selective_execution_governor import SelectiveExecutionGovernor
from proxima_x.proxima_ops.governance.execution_state_machine import ExecutionStateMachine, ExecutionState
from proxima_x.proxima_ops.governance.intent_constraint_layer import IntentConstraintLayer
from proxima_x.proxima_ops.execution.execution_ledger import ExecutionLedger, TradeEvent
from proxima_x.proxima_ops.monitoring.proxima_dashboard import ProximaDashboard
from proxima_x.proxima_ops.monitoring.reconciliation_engine import ReconciliationEngine
from proxima_x.proxima_ops.monitoring.broker_reconciliation import BrokerReconciliation
from proxima_x.proxima_ops.risk.edge_signal_mapper import EdgeSignalMapper, _compute_rsi
from proxima_x.proxima_ops.risk.regime_classifier import MarketRegimeClassifier, RegimeGatedFilter
from proxima_x.proxima_ops.risk.edge_redundancy_layer import EdgeRedundancyLayer
from proxima_x.proxima_ops.risk.signal_fusion_engine import SignalFusionEngine
from proxima_x.proxima_ops.execution.volume_expansion_layer import VolumeExpansionLayer
from proxima_x.proxima_ops.execution.validation_staircase import ValidationStaircase
from proxima_x.proxima_ops.execution.exposure_amplifier_layer import ExposureAmplifierLayer
from proxima_x.proxima_ops.monitoring.live_monitor import LiveMonitor
from proxima_x.proxima_ops.monitoring.mt5_watchdog import MT5Watchdog
from proxima_x.proxima_ops.monitoring.trade_lifecycle_tracker import TradeLifecycleTracker
from proxima_x.proxima_ops.monitoring.circuit_breaker import CircuitBreaker
from proxima_x.proxima_ops.monitoring.pipeline_trace_logger import PipelineTraceLogger
from proxima_x.proxima_ops.monitoring.regime_exposure_tracker import RegimeExposureTracker
from proxima_x.proxima_ops.monitoring.activation_watch import ActivationWatch
from proxima_x.proxima_ops.monitoring.distribution_observer import DistributionObserver

try:
    from proxima_x.proxima_ops.intelligence.symbol_universe_selector import SymbolUniverseSelector
    _HAS_SIL = True
except ImportError:
    SymbolUniverseSelector = None
    _HAS_SIL = False


SYMBOLS = ["EURUSD", "GBPUSD", "EURJPY", "USDJPY"]
FALLBACK_SYMBOLS = ["EURUSD", "GBPUSD", "EURJPY", "USDJPY"]


class Wave12Executor:
    def __init__(self, spec_path: str = "state/wave12_experiment_spec.json",
                 ledger_path: str = "state/execution_ledger.jsonl",
                 mt5_connector=None):
        with open(spec_path) as f:
            self.spec = json.load(f)
        self.ledger = ExecutionLedger(ledger_path)
        self.dashboard = ProximaDashboard()
        self.reconciler = ReconciliationEngine(self.ledger)
        self.mt5 = mt5_connector if mt5_connector is not None else MT5Connector()
        self.order_manager = OrderManager(self.mt5)
        self._hold_tracker: dict[int, int] = {}
        self._trade_state_persistence_path = "state/trade_state_persistence.json"
        self._state_restored = False
        self.governor = SelectiveExecutionGovernor()
        self.state_machine = self.governor.state_machine
        self.intent_layer = IntentConstraintLayer()
        self.signal_mapper = EdgeSignalMapper()
        self.regime_classifier = MarketRegimeClassifier()
        self.regime_filter = RegimeGatedFilter(self.regime_classifier)
        self.er_layer = EdgeRedundancyLayer()
        self.fusion_engine = SignalFusionEngine()
        self.vel = VolumeExpansionLayer()
        self.live_monitor = LiveMonitor()
        self.mt5_watchdog = MT5Watchdog()
        self.trade_lifecycle = TradeLifecycleTracker()
        self.circuit_breaker = CircuitBreaker()
        self.pipeline_trace_logger = PipelineTraceLogger()
        self.regime_tracker = RegimeExposureTracker()
        self.activation_watch = ActivationWatch()
        self.distribution_observer = DistributionObserver()
        self.cycle_count = 0
        self.cycle_times = deque(maxlen=20)
        self.stopping_triggered = False
        self.stop_reason = ""
        self.session_start = time.time()
        self._cycles_with_position = 0
        self._max_hold_cycles = 5
        self._last_active_edge = None
        self.confirm_cycles: dict[str, int] = {}
        self._broker_reconciled = False
        self._broker_reconciler = BrokerReconciliation(self.mt5, self.ledger)
        self._trade_entry_atr = 0.0
        self._trade_mfe = 0.0
        self._trade_mae = 0.0
        self._trade_entry_price = 0.0
        self._trade_side = ""
        self.staircase = ValidationStaircase()
        self.exposure_amplifier = ExposureAmplifierLayer()
        self._amplifier_enabled = True
        self.sdl = SymbolDirectionLock()
        self._symbol_selector = SymbolUniverseSelector() if _HAS_SIL else None
        self._last_heartbeat_cycle = 0
        self._active_symbols = list(FALLBACK_SYMBOLS)
        self._previous_universe = None
        self._prev_heartbeat_top3 = None
        logging.info("POSITIONS_DEBUG: CACHE_BYPASS_ACTIVE init OK")

    # ── Trade State Persistence ────────────────────────────────────────
    def _load_all_trade_state(self) -> dict:
        if os.path.exists(self._trade_state_persistence_path):
            with open(self._trade_state_persistence_path) as f:
                return json.load(f)
        return {}

    def _save_trade_state(self):
        os.makedirs("state", exist_ok=True)
        existing = self._load_all_trade_state()
        current_time = time.time()
        for ticket, hold_count in self._hold_tracker.items():
            ticket_str = str(ticket)
            old_entry = existing.get(ticket_str, {}).get("entry_timestamp", current_time)
            existing[ticket_str] = {
                "hold_cycle_count": hold_count,
                "entry_timestamp": old_entry,
                "last_cycle_timestamp": current_time,
                "mfe": round(self._trade_mfe, 6),
                "mae": round(self._trade_mae, 6),
                "entry_atr": round(self._trade_entry_atr, 6),
                "entry_price": round(self._trade_entry_price, 5),
                "side": self._trade_side,
            }
        active_tickets = set(str(t) for t in self._hold_tracker.keys())
        for t_str in list(existing.keys()):
            if t_str not in active_tickets:
                del existing[t_str]
        with open(self._trade_state_persistence_path, "w") as f:
            json.dump(existing, f, indent=2)

    def _remove_trade_state(self, ticket: int):
        state = self._load_all_trade_state()
        ticket_str = str(ticket)
        if ticket_str in state:
            del state[ticket_str]
            with open(self._trade_state_persistence_path, "w") as f:
                json.dump(state, f, indent=2)

    def _restore_trade_state(self, positions: list):
        self._state_restored = True
        if not positions:
            return
        persisted = self._load_all_trade_state()
        logger.info(f"[TRADE_STATE] Restoring state for {len(positions)} open positions")
        for pos in positions:
            ticket = pos.get("ticket", 0) if isinstance(pos, dict) else pos.ticket
            ticket_str = str(ticket)
            if ticket_str in persisted:
                ps = persisted[ticket_str]
                self._hold_tracker[ticket] = ps.get("hold_cycle_count", 0)
                self._trade_entry_atr = ps.get("entry_atr", 0.0)
                self._trade_mfe = ps.get("mfe", 0.0)
                self._trade_mae = ps.get("mae", 0.0)
                self._trade_entry_price = ps.get("entry_price", 0.0)
                self._trade_side = ps.get("side", "")
                logger.info(
                    f"[TRADE_STATE] Restored ticket={ticket} hold={self._hold_tracker[ticket]} "
                    f"mfe={self._trade_mfe} mae={self._trade_mae} atr={self._trade_entry_atr}"
                )
            else:
                pos_time = pos.get("time", 0) if isinstance(pos, dict) else pos.time
                symbol = pos.get("symbol", "") if isinstance(pos, dict) else pos.symbol
                # Use server time from tick (MT5 times are server-local, not UTC)
                tick = self.mt5.get_tick(symbol) if symbol else None
                if tick:
                    server_now = tick["time"]
                else:
                    server_now = int(time.time() + 10800)  # fallback UTC+3
                elapsed_sec = max(0, server_now - pos_time)
                estimated_cycles = max(1, int(elapsed_sec / 3))
                self._hold_tracker[ticket] = estimated_cycles
                if symbol:
                    self._trade_entry_atr = self.order_manager._compute_atr(symbol)
                opn = float(pos.get("price_open", 0) if isinstance(pos, dict) else pos.price_open)
                typ = pos.get("type", "") if isinstance(pos, dict) else pos.type
                self._trade_entry_price = opn
                self._trade_side = typ
                logger.info(
                    f"[TRADE_STATE] Estimated ticket={ticket} hold={estimated_cycles} "
                    f"(elapsed={elapsed_sec:.0f}s) entry_price={opn} side={typ}"
                )
        self._save_trade_state()
        logger.info(f"[TRADE_STATE] Restore complete. hold_tracker={dict(self._hold_tracker)}")

    def _get_open_positions(self) -> list:
        """Always fetch positions fresh from MT5 API — no caching."""
        positions = self.mt5.get_positions()
        logging.info(f"POSITIONS_DEBUG: SOURCE=MT5_API count={len(positions)}")
        return positions

    def _refresh_symbol_universe(self) -> list:
        """Get current symbol universe from SIL, with fallback to hardcoded list."""
        try:
            if self._symbol_selector is not None:
                symbols = self._symbol_selector.select_universe()
                if symbols and len(symbols) >= 4:
                    if symbols != self._previous_universe:
                        self._previous_universe = list(symbols)
                        logging.info(f"[SYMBOL_UNIVERSE] count={len(symbols)} symbols={symbols}")
                    else:
                        logging.info(f"[SYMBOL_UNIVERSE] Universe unchanged ({len(symbols)} symbols)")
                    self._active_symbols = symbols
                    return symbols
        except Exception as exc:
            logging.warning(f"[SYMBOL_UNIVERSE] SIL failed: {exc} — using fallback")
        self._active_symbols = list(FALLBACK_SYMBOLS)
        return self._active_symbols

    def _fetch_market_data(self) -> dict:
        """Fetch OHLC rates for all 4 symbols. Returns {symbol: closes_ndarray, ...}."""
        closes = {}
        highs = {}
        lows = {}
        prices = {}
        for sym in self._active_symbols:
            rates = self.mt5.get_rates(sym, count=100, timeframe="M1")
            if rates and len(rates) > 20:
                c = np.array([r["close"] for r in rates], dtype=np.float64)
                h = np.array([r["high"] for r in rates], dtype=np.float64)
                l = np.array([r["low"] for r in rates], dtype=np.float64)
                closes[sym] = c
                highs[sym] = h
                lows[sym] = l
                prices[sym] = float(c[-1])
        return {"closes": closes, "highs": highs, "lows": lows, "prices": prices}

    def _sweep_signals(self, md: dict) -> list:
        """Generate signals for ALL edges. Returns list sorted by quality desc."""
        if not md["closes"]:
            logger.info("[SWEEP_DEBUG] no closes data available")
            return []
        avail = list(md["closes"].keys())
        n_edges = self.signal_mapper.edge_count
        logger.info("[SWEEP_DEBUG] edges=%d symbols=%s closes_keys=%s", n_edges, avail, list(md["closes"].keys()))
        first_edge = self.signal_mapper.edges[0] if self.signal_mapper.edges else None
        if first_edge:
            sym = first_edge.get("symbol", "?")
            has_closes = sym in md["closes"]
            n_closes = len(md["closes"].get(sym, []))
            logger.info("[SWEEP_DEBUG] first_edge=%s sym=%s has_closes=%s n_closes=%d",
                        first_edge.get("id", "?"), sym, has_closes, n_closes)
        signals = self.signal_mapper.generate_all(
            closes_by_symbol=md["closes"],
            highs_by_symbol=md["highs"],
            lows_by_symbol=md["lows"],
            prices_by_symbol=md["prices"],
        )
        logger.info("[SWEEP_DEBUG] generated %d signals", len(signals))
        # Debug: check RSI for first few mean_reversion edges
        from proxima_x.proxima_ops.risk.edge_signal_mapper import _compute_rsi
        for sym in self._active_symbols:
            c = md["closes"].get(sym)
            if c is not None and len(c) >= 20:
                rsi_arr = _compute_rsi(c)
                logger.info("[SWEEP_DEBUG] %s RSI=%.1f (last=%.2f, prev=%.2f, n=%d)", sym, rsi_arr[-1], c[-1], c[-2], len(c))
        if len(signals) > 0:
            logger.info("[SWEEP_DEBUG] top: %s %.4f dir=%d",
                        signals[0].get("edge_id","?"), signals[0].get("confidence",0),
                        signals[0].get("direction",0))
        else:
            # Check why no signals — test first edge manually
            for e in self.signal_mapper.edges[:3]:
                sym = e.get("symbol", "?")
                closes = md["closes"].get(sym)
                if closes is None or len(closes) < 20:
                    logger.info("[SWEEP_DEBUG] edge %s: no closes (%s)", e.get("id","?"), "None" if closes is None else f"{len(closes)} bars")
                    continue
                from proxima_x.proxima_ops.risk.edge_signal_mapper import _STRATEGY_FUNCTIONS
                func = _STRATEGY_FUNCTIONS.get(e.get("strategy",""))
                if func:
                    try:
                        d, c, ec, dr = func(closes, e.get("params", {}))
                        logger.info("[SWEEP_DEBUG] edge %s manual: dir=%d conf=%.4f ecdf=%.4f drift=%d",
                                    e.get("id","?"), d, c, ec, dr)
                    except Exception as ex:
                        logger.info("[SWEEP_DEBUG] edge %s error: %s", e.get("id","?"), ex)
        signals.sort(
            key=lambda s: s.get("confidence", 0) * s.get("edge_pf", 1),
            reverse=True,
        )
        return signals

    def _build_regime_dashboard(self, pressure_signals: list, momentum_signals: list) -> dict:
        """Build regime observation dashboard from ERL outputs. Non-influential."""
        dashboard = {}
        for s in pressure_signals + momentum_signals:
            sym = s.get("symbol", "?")
            if sym not in dashboard:
                dashboard[sym] = {"pressure_score": 0, "momentum_score": 0, "direction": 0, "regime_label": "neutral"}
            strat = s.get("strategy", "")
            conf = s.get("confidence", 0)
            if strat == "pressure":
                dashboard[sym]["pressure_score"] = max(dashboard[sym]["pressure_score"], conf)
                dashboard[sym]["direction"] = s.get("direction", 0)
            elif strat == "momentum":
                dashboard[sym]["momentum_score"] = max(dashboard[sym]["momentum_score"], conf)
                dashboard[sym]["direction"] = s.get("direction", 0)
        for sym, data in dashboard.items():
            p, m = data["pressure_score"], data["momentum_score"]
            if p > 0.5 and m > 0.5:
                data["regime_label"] = "compression_transition"
            elif p > 0.5:
                data["regime_label"] = "compression"
            elif m > 0.5:
                data["regime_label"] = "transition"
            else:
                data["regime_label"] = "neutral"
        return dashboard

    def _compute_market_state(self, spread: int) -> dict:
        if spread <= 20:
            mof_state = "INFORMATION_RICH"
            mof_score = min(0.99, 0.85 + (20 - spread) * 0.003)
        elif spread <= 50:
            mof_state = "STRUCTURE_LIMITED"
            mof_score = 0.60 + 0.25 * (50 - spread) / 30
        else:
            mof_state = "NOISE"
            mof_score = max(0.1, 0.50 * max(0, 100 - spread) / 100)

        return {
            "mof_state": mof_state,
            "mof_score": round(mof_score, 4),
            "mof_gating_consistent": mof_state in ("STRUCTURE_LIMITED", "INFORMATION_RICH"),
        }

    def cycle(self) -> dict:
        self.cycle_count += 1
        self.vel.record_cycle()
        cycle_start = time.time()
        cycle_data = {"cycle": self.cycle_count}

        try:
            connected = self.mt5.connect()
            cycle_data["mt5_connected"] = connected
            if not connected:
                cycle_data["error"] = "MT5 connection failed"
                cycle_data["decision"] = "SKIP"
                self._record_cycle(cycle_data)
                return cycle_data

            self._refresh_symbol_universe()
            positions = self._get_open_positions()
            account_info = self.mt5.get_account()
            cycle_data["open_positions"] = len(positions)
            cycle_data["balance"] = account_info.get("balance", 0) if account_info else 0

            if not self._broker_reconciled:
                recon_report = self._broker_reconciler.reconcile(positions=positions)
                self._broker_reconciled = True
                cycle_data["broker_reconciliation"] = {
                    "closes_appended": recon_report["closes_appended"],
                    "completed_trades": recon_report["total_completed_trades"],
                    "staircase_phase": recon_report.get("staircase_after", {}).get("current_phase", 1),
                    "orphan_entries": len(recon_report.get("orphan_ledger_entries", [])),
                }
                if recon_report["closes_appended"] > 0:
                    logger.info(
                        f"[BROKER_RECON] Appended {recon_report['closes_appended']} missing close events. "
                        f"Staircase: {recon_report['total_completed_trades']} trades"
                    )
                reconciled_trades = recon_report.get("total_completed_trades", self.staircase.completed_trades)
                self.staircase.set_trades(reconciled_trades)

            md = self._fetch_market_data()
            core_signals = self._sweep_signals(md)
            # ERL: perception layer only — no influence on execution
            erp_pressure = self.er_layer.generate_pressure_signals(md["closes"], md["highs"], md["lows"])
            erp_momentum = self.er_layer.generate_momentum_signals(md["closes"], md["highs"], md["lows"])
            regime_dashboard = self._build_regime_dashboard(erp_pressure, erp_momentum)
            cycle_data["regime_dashboard"] = regime_dashboard
            all_signals = self.fusion_engine.fuse(core_signals, [], [])  # CORE-only dedup
            cycle_data["total_signals"] = len(all_signals)
            cycle_data["core_signals"] = len(core_signals)
            cycle_data["erp_pressure"] = len(erp_pressure)
            cycle_data["erp_momentum"] = len(erp_momentum)
            pipeline_trace = {"cycle": self.cycle_count, "generated": [], "threshold_gate": [], "confirm_gate": [], "governor_gate": [], "execution": None}

            for s in all_signals:
                eid = s.get("edge_id", "?")
                sym = s.get("symbol", "?")
                d = s.get("direction", 0)
                c = s.get("confidence", 0)
                strat = s.get("strategy", "?")
                dir_status = "ACTIVE_DIR" if d != 0 else "NO_DIR"
                conf_status = f"conf={c:.4f}"
                thresh_pass = d != 0 and c >= 0.40
                pipeline_trace["generated"].append(f"{eid} {sym} {strat} dir={d} {conf_status} -> {'PASS' if thresh_pass else 'FAIL'}")

            tick = self.mt5.get_tick("EURJPY")
            spread = tick.get("spread", 30) if tick else 30
            market = self._compute_market_state(spread)

            for s in all_signals:
                if s.get("direction", 0) != 0:
                    conf = s.get("confidence", 0)
                    if 0.35 <= conf < 0.40:
                        logger.info("Marginal signal %s: confidence %.3f in [0.35, 0.40) band - not executing",
                                     s.get("edge_id", "?"), conf)

            pre_gate = []
            for s in all_signals:
                eid = s.get("edge_id", "?")
                sym = s.get("symbol", "?")
                d = s.get("direction", 0)
                c = s.get("confidence", 0)
                direction_ok = d != 0
                conf_ok = c >= 0.40
                if not direction_ok:
                    pipeline_trace["threshold_gate"].append(f"{eid}: direction=0 (dead)")
                elif not conf_ok:
                    pipeline_trace["threshold_gate"].append(f"{eid}: conf={c:.4f} < 0.40")
                else:
                    pipeline_trace["threshold_gate"].append(f"{eid}: PASS dir={d} conf={c:.4f}")
                    pre_gate.append(s)

            active_signals = []
            for s in pre_gate:
                eid = s.get("edge_id", "?")
                sym = s.get("symbol", "?")
                s_dir = "BUY" if s.get("direction", 0) > 0 else "SELL"
                ccount = self.confirm_cycles.get(f"{sym}_{s_dir}", 0)
                has_active = s.get("has_active_signal", "True")
                if has_active == "False":
                    pipeline_trace["confirm_gate"].append(f"{eid}: has_active_signal=False (dead field)")
                elif ccount < 2:
                    pipeline_trace["confirm_gate"].append(f"{eid}: cross_cyc={ccount}/2 (waiting)")
                    active_signals.append(s)
                else:
                    pipeline_trace["confirm_gate"].append(f"{eid}: CROSS_PASS (cycles={ccount}/2)")
                    active_signals.append(s)
            cycle_data["active_signals"] = len(active_signals)
            cycle_data["pipeline_trace"] = pipeline_trace

            regime_label = self.regime_classifier.classify(
                closes_by_symbol=md["closes"],
                highs_by_symbol=md["highs"],
                lows_by_symbol=md["lows"],
                cycle=self.cycle_count,
            )
            gated_signals = self.regime_filter.filter_signals(all_signals)
            gated_active = len([s for s in all_signals
                            if s.get("direction", 0) != 0
                            and s.get("confidence", 0) >= 0.35
                            and s.get("strategy", "") in self.regime_classifier.get_active_strategies()])
            cycle_data["regime"] = regime_label
            cycle_data["regime_shadow_gated_active"] = gated_active
            cycle_data["regime_shadow_gated_total"] = len([s for s in all_signals
                                                      if s.get("strategy", "") in self.regime_classifier.get_active_strategies()])
            cycle_data["regime_shadow_filtered_out"] = len(all_signals) - len(gated_signals)

            best_signal = active_signals[0] if active_signals else None
            signal_id = best_signal.get("edge_id", "none") if best_signal else "none"
            symbol = best_signal.get("symbol", "EURJPY") if best_signal else "EURJPY"
            direction = "BUY" if best_signal and best_signal.get("direction", 0) > 0 else "SELL"
            direction = best_signal.get("side", direction) if best_signal else direction
            confidence = best_signal.get("confidence", 0) if best_signal else 0

            cycle_data["active_edge"] = signal_id
            cycle_data["active_symbol"] = symbol
            cycle_data["active_direction"] = direction
            cycle_data["active_confidence"] = confidence

            # Cross-projection confirmation: track by (symbol, direction)
            qualifying_pairs: set[str] = set()
            for s in all_signals:
                if s.get("direction", 0) != 0 and s.get("confidence", 0) >= 0.40:
                    s_dir = "BUY" if s.get("direction", 0) > 0 else "SELL"
                    qualifying_pairs.add(f"{s.get('symbol', '')}_{s_dir}")
            for k in qualifying_pairs:
                self.confirm_cycles[k] = self.confirm_cycles.get(k, 0) + 1
            for k in list(self.confirm_cycles.keys()):
                if k not in qualifying_pairs:
                    self.confirm_cycles[k] = 0
            confirm_count = self.confirm_cycles.get(f"{symbol}_{direction}", 0) if best_signal else 0
            cycle_data["confirm_cycles"] = confirm_count
            identity_key = f"{symbol}_{direction}"
            matched = identity_key in qualifying_pairs and confirm_count > 1
            logger.debug(
                "[CONFIRM_DEBUG] cycle=%d identity_key=%s edge_id=%s symbol=%s direction=%s "
                "confidence=%.4f matched=%s confirm_count=%d state_map=%s",
                self.cycle_count, identity_key, signal_id, symbol, direction,
                confidence, matched, confirm_count, dict(self.confirm_cycles),
            )

            if len(positions) == 0:
                self._cycles_with_position = 0
            else:
                self._cycles_with_position += 1

            has_signal = best_signal is not None
            sig_state = "ACTIVE" if has_signal else "NONE"

            logger.info(
                "[GOVPIPE_DEBUG] governance_pipeline_approved=%s source=has_signal(best_signal=%s) "
                "active_signals=%d cycle=%d confirm_count=%d",
                has_signal, signal_id if best_signal else "None",
                len(active_signals), self.cycle_count, confirm_count
            )
            if not has_signal:
                logger.info(
                    "[GOVPIPE_DEBUG] BLOCKING_CHECK: best_signal=None active_signals=%d pre_gate=%d "
                    "confirm_threshold=2 confirm_count=%d",
                    len(active_signals), len(pre_gate), confirm_count
                )

            logger.info(
                "[ENVELOPE_DEBUG] edge_envelope_pass=%s source=has_signal(best_signal=%s) "
                "cycle=%d [NOTE: envelope check stubbed to has_signal — no envelope logic wired]",
                has_signal, signal_id if best_signal else "None", self.cycle_count
            )
            if not has_signal:
                logger.info(
                    "[ENVELOPE_DEBUG] BLOCKING_CHECK: best_signal=None active_signals=%d "
                    "envelope_threshold=N/A (stub — no real envelope check implemented)",
                    len(active_signals)
                )

            self.governor.process_signal(
                signal={
                    "id": signal_id, "symbol": symbol, "direction": direction,
                    "side": direction, "state": sig_state,
                    "confidence": confidence,
                    "strategy": best_signal.get("strategy", "") if best_signal else "none",
                    "volume": self.spec["instrument"]["volume"],
                    "timestamp": time.time(),
                    "price": best_signal.get("price", 0) if best_signal else 0,
                },
                mof_state=market["mof_state"],
                mof_score=market["mof_score"],
                mof_gating_consistent=market["mof_gating_consistent"],
                portfolio_conflict=0.05,
                governance_pipeline_approved=has_signal,
                rf_drift=0.02,
                lifecycle_orphans=0,
                edge_envelope_pass=has_signal,
            )

            system_state = self.governor.evaluate_system_state()
            segl_state = system_state["state"]
            cycle_data["segl_state"] = segl_state

            intent_result = self.intent_layer.evaluate_decision_against_intent({
                "objective": "SIGNAL_IDENTITY",
                "conflict": None,
                "outcome": f"cycle_{self.cycle_count}_signal_{signal_id}",
                "traded_away_priority": None,
            })
            intent_compliant = intent_result.get("conforms", True)
            cycle_data["intent_compliant"] = intent_compliant
            pipeline_trace["governor_gate"].append(f"segl_state={segl_state} ready_to_exec={'YES' if segl_state == 'ARMED' and intent_compliant else 'NO'} intent={intent_compliant}")

            decision = "HOLD"
            exec_authorized = False

            if len(positions) == 0 and not best_signal:
                pipeline_trace["execution"] = "NO_SIGNAL no best_signal passed all gates"
            if len(positions) == 0 and best_signal:
                if confirm_count < 2:
                    cycle_data["denial_reason"] = f"Insufficient cross-projection confirm: {confirm_count}/2"
                    pipeline_trace["execution"] = f"DENIED cross_confirm={confirm_count}/2"
                elif segl_state == "ARMED" and intent_compliant:
                    authorized, reason = self.governor.authorize_execution({
                        "id": signal_id, "symbol": symbol
                    })
                    authorized, reason = self.governor.authorize_execution({
                        "id": signal_id, "symbol": symbol
                    })
                    can_x, x_reason = self.state_machine.can_transition(ExecutionState.EXECUTING)
                    if not authorized:
                        pipeline_trace["execution"] = f"DENIED governor.authorize_execution='{reason}'"
                        cycle_data["denial_reason"] = reason
                    elif not can_x:
                        pipeline_trace["execution"] = f"DENIED state_machine cannot transition to EXECUTING: {x_reason}"
                        cycle_data["denial_reason"] = x_reason
                    else:
                        vel_block_rate = self.trade_lifecycle.vel_audit_summary().get("block_rate", 0.0)
                        cb_allow, cb_reason = self.circuit_breaker.check_execution_attempt(
                            symbol, direction, len(positions),
                            self.mt5_watchdog.check_integrity(),
                            vel_block_rate,
                        )
                        cycle_data["cb_decision"] = cb_reason
                        if not cb_allow:
                            decision = "HOLD"
                            cycle_data["denial_reason"] = f"CircuitBreaker: {cb_reason}"
                            pipeline_trace["execution"] = f"DENIED CB: {cb_reason}"
                        else:
                            vel_allow, vel_reason = self.vel.should_allow_execution(
                                symbol, direction, self.staircase.current_phase
                            )
                            cycle_data["vel_decision"] = vel_reason
                            self.trade_lifecycle.record_vel_decision(symbol, direction, vel_allow, vel_reason)
                            if not vel_allow:
                                decision = "HOLD"
                                cycle_data["denial_reason"] = f"VEL blocked: {vel_reason}"
                                pipeline_trace["execution"] = f"DENIED VEL: {vel_reason}"
                            else:
                                tick_obj = self.mt5.get_tick(symbol)
                                if tick_obj is None:
                                    decision = "HOLD"
                                    cycle_data["execution_error"] = "No tick data"
                                    cycle_data["denial_reason"] = "No tick data"
                                    pipeline_trace["execution"] = "FAILED tick data unavailable"
                                else:
                                    price = tick_obj["ask"] if direction == "BUY" else tick_obj["bid"]
                                    staircase_volume = self.staircase.get_volume()
                                    if self._amplifier_enabled:
                                        amplifier_mult = self.exposure_amplifier.get_multiplier(
                                            recent_trade_pnls=[],
                                            current_drawdown=0.0,
                                            volatility_regime="normal",
                                        )
                                    else:
                                        amplifier_mult = 1.0
                                    final_volume = round(staircase_volume * amplifier_mult, 2)
                                    trade_index = self.staircase.completed_trades + 1
                                    self.mt5_watchdog.record_order_attempt({
                                        "symbol": symbol, "direction": direction,
                                        "volume": final_volume, "price": price,
                                        "type": "MARKET",
                                    })
                                    logger.info(
                                        f"[STAIRCASE_EXEC] Trade #{trade_index} | "
                                        f"Phase {self.staircase.current_phase} | "
                                        f"Lot {final_volume} (base={staircase_volume} mult={amplifier_mult})"
                                    )
                                    self.vel.record_execution(symbol)
                                    cycle_data["volume_composition"] = {
                                        "base_volume": staircase_volume,
                                        "amplifier_multiplier": amplifier_mult,
                                        "final_volume": final_volume,
                                    }
                                    pipeline_trace["amplifier_mult"] = amplifier_mult
                                    result = self.order_manager.place_order(
                                        symbol, direction,
                                        final_volume, price=price
                                    )
                                    if result is not None:
                                        self.sdl.lock(symbol, direction)
                                        self.mt5_watchdog.record_order_result(result)
                                        self.circuit_breaker.record_slippage(0.0)
                                        ticket = result.get("ticket", 0)
                                        pipeline_trace["execution"] = f"EXECUTED ticket={ticket} {symbol} {direction} vol={final_volume} price={price}"
                                        self._hold_tracker[ticket] = 0
                                        self.state_machine.transition(ExecutionState.EXECUTING,
                                                                        f"Executing {signal_id}", "cycle")
                                        self._last_active_edge = signal_id
                                        cycle_data["execution_result"] = {
                                            "success": True,
                                            "ticket": ticket,
                                            "price": price,
                                            "symbol": symbol,
                                            "direction": direction,
                                            "signal_id": signal_id,
                                            "fusion_sources": best_signal.get("_fusion_sources", []),
                                            "fusion_is_erl": best_signal.get("_fusion_is_erl", False),
                                        }
                                        decision = "EXECUTE"
                                        trade_data = {
                                            "ticket": ticket, "symbol": symbol,
                                            "direction": direction, "volume": final_volume,
                                            "entry_price": price, "signal_id": signal_id,
                                            "fusion_sources": best_signal.get("_fusion_sources", []),
                                            "confirm_path": pipeline_trace.get("confirm_gate", []),
                                            "vel_decision": vel_reason,
                                        }
                                        self.trade_lifecycle.open_trade(trade_data)
                                        self.governor.record_execution(
                                            signal_id=signal_id, symbol=symbol,
                                            action=direction,
                                            mof_state=market["mof_state"],
                                            mof_score=market["mof_score"],
                                            rf_drift=0.02,
                                        )
                                        event = TradeEvent(
                                            event_type="trade_opened",
                                            signal_id=signal_id, symbol=symbol,
                                            direction=direction,
                                            volume=final_volume,
                                            entry_price=price,
                                            mt5_ticket=result.get("ticket", 0),
                                            segl_state=segl_state,
                                            mof_state=market["mof_state"],
                                            mof_score=market["mof_score"],
                                            rf_drift=0.02,
                                            portfolio_conflict=0.05,
                                            frequency_budget_remaining=0,
                                            authorization_path=[signal_id],
                                            intent_compliant=intent_compliant,
                                            lifecycle_match=True,
                                        )
                                        cycle_data["event_id"] = self.ledger.append(event)
                                        exec_authorized = True
                                        self.distribution_observer.record_trade(cycle_data.get("execution_result", {}))
                                        self._trade_entry_atr = self.order_manager._compute_atr(symbol)
                                        self._trade_mfe = 0.0
                                        self._trade_mae = 0.0
                                        self._trade_entry_price = float(price)
                                        self._trade_side = direction
                                        self._save_trade_state()
                                    else:
                                        self.circuit_breaker.record_mt5_failure()
                                        cycle_data["execution_error"] = "MT5 place_order returned None"
                                        decision = "HOLD"
                                        pipeline_trace["execution"] = "FAILED MT5 place_order returned None"
                else:
                    pipeline_trace["execution"] = f"DENIED segl_state={segl_state} intent={intent_compliant}"
                    cycle_data["denial_reason"] = f"State={segl_state} or !intent"

            elif len(positions) > 0:
                if not self._state_restored:
                    self._restore_trade_state(positions)
                active_tickets = {p.get("ticket", 0) if isinstance(p, dict) else p.ticket for p in positions}
                for t in list(self._hold_tracker.keys()):
                    if t not in active_tickets:
                        del self._hold_tracker[t]

                for pos in positions:
                    ticket = pos.get("ticket", 0) if isinstance(pos, dict) else pos.ticket
                    if ticket not in self._hold_tracker:
                        self._hold_tracker[ticket] = 0
                    self._hold_tracker[ticket] += 1

                min_hold = min(
                    (self._hold_tracker.get(
                        pos.get("ticket", 0) if isinstance(pos, dict) else pos.ticket, 0
                    ) for pos in positions),
                    default=0
                )
                hold_time = min_hold
                cycle_data["hold_cycles"] = hold_time

                for pos in positions:
                    cur = float(pos.get("price_current", 0) if isinstance(pos, dict) else pos.price_current)
                    opn = float(pos.get("price_open", 0) if isinstance(pos, dict) else pos.price_open)
                    typ = pos.get("type", "") if isinstance(pos, dict) else pos.type
                    if self._trade_entry_price == 0:
                        self._trade_entry_price = opn
                        self._trade_side = typ
                    adv = (self._trade_entry_price - cur) if self._trade_side == "SELL" else (cur - self._trade_entry_price)
                    self._trade_mfe = max(self._trade_mfe, adv)
                    self._trade_mae = min(self._trade_mae, adv)
                cycle_data["mfe"] = round(self._trade_mfe, 5)
                cycle_data["mae"] = round(self._trade_mae, 5)
                self._save_trade_state()

                sl_tp_hit = False
                for pos in positions:
                    current_price = pos.get("price_current", 0) if isinstance(pos, dict) else pos.price_current
                    pos_sl = pos.get("sl", 0) if isinstance(pos, dict) else pos.sl
                    pos_tp = pos.get("tp", 0) if isinstance(pos, dict) else pos.tp
                    pos_type = pos.get("type", direction) if isinstance(pos, dict) else pos.type

                    if pos_sl and pos_tp:
                        if pos_type == "BUY":
                            if current_price <= pos_sl:
                                sl_tp_hit = True
                                break
                            if current_price >= pos_tp:
                                sl_tp_hit = True
                                break
                        else:
                            if current_price >= pos_sl:
                                sl_tp_hit = True
                                break
                            if current_price <= pos_tp:
                                sl_tp_hit = True
                                break

                    profit = pos.get("profit", 0.0) if isinstance(pos, dict) else pos.profit
                    if profit <= -2.0:
                        sl_tp_hit = True
                        break

                reverse_signal_strength = 0.0
                if best_signal and positions:
                    pos_type = positions[0].get("type", direction) if isinstance(positions[0], dict) else positions[0].type
                    signal_side = "BUY" if best_signal.get("direction", 0) > 0 else "SELL"
                    if signal_side != pos_type:
                        reverse_signal_strength = best_signal.get("confidence", 0)

                cycle_data["reverse_signal_strength"] = reverse_signal_strength

                should_close = False
                close_reason = ""
                if sl_tp_hit:
                    should_close = True
                    close_reason = "SL/TP_HIT"
                elif hold_time < 3:
                    should_close = False
                    close_reason = f"HOLD:{hold_time}/3"
                elif reverse_signal_strength > 0.65 and hold_time >= 3:
                    should_close = True
                    close_reason = f"REVERSE:{reverse_signal_strength:.2f}"
                elif hold_time >= 60:
                    mfe_mae_range = self._trade_mfe - self._trade_mae
                    if self._trade_entry_atr > 0 and mfe_mae_range < 0.5 * self._trade_entry_atr:
                        should_close = True
                        close_reason = "STAGNATION_60"
                    else:
                        close_reason = f"STAGNATE_WATCH:{hold_time}"
                else:
                    should_close = False
                    close_reason = "NO_TRIGGER"
                cycle_data["close_reason"] = close_reason

                if should_close:
                    results = []
                    all_ok = True
                    for pos in positions:
                        ticket = pos.get("ticket", 0) if isinstance(pos, dict) else pos.ticket
                        ok = self.mt5.close_order(ticket)
                        pnl = pos.get("profit", 0.0) if isinstance(pos, dict) else pos.profit
                        results.append({
                            "ticket": ticket, "success": ok, "pnl": pnl,
                            "symbol": pos.get("symbol", symbol) if isinstance(pos, dict) else pos.symbol,
                            "direction": pos.get("type", direction) if isinstance(pos, dict) else pos.type,
                            "volume": pos.get("volume", 0) if isinstance(pos, dict) else pos.volume,
                            "entry_price": pos.get("price_open", 0) if isinstance(pos, dict) else pos.price_open,
                            "exit_price": pos.get("price_current", 0) if isinstance(pos, dict) else pos.price_current,
                        })
                        if not ok:
                            all_ok = False
                    cycle_data["close_result"] = {"success": all_ok, "results": results}
                    if all_ok:
                        decision = "CLOSE"
                        for r in results:
                            evt = TradeEvent(
                                event_type="trade_closed",
                                signal_id=self._last_active_edge or "unknown",
                                symbol=r.get("symbol", symbol),
                                direction=r.get("direction", direction),
                                volume=r.get("volume", 0),
                                entry_price=r.get("entry_price", 0),
                                exit_price=r.get("exit_price", 0),
                                mt5_ticket=r.get("ticket", 0),
                                pnl=r.get("pnl", 0),
                                segl_state=segl_state,
                                mof_state=market["mof_state"],
                                mof_score=market["mof_score"],
                                rf_drift=0.02,
                                portfolio_conflict=0.05,
                                frequency_budget_remaining=0,
                                authorization_path=[self._last_active_edge or "unknown"],
                                intent_compliant=intent_compliant,
                                lifecycle_match=True,
                            )
                            cycle_data["event_id"] = self.ledger.append(evt)
                        self.governor.record_execution(
                            signal_id=self._last_active_edge or "unknown",
                            symbol=symbol, action="CLOSE",
                            mof_state=market["mof_state"],
                            mof_score=market["mof_score"],
                            rf_drift=0.02,
                        )
                        exec_authorized = True
                        self.staircase.increment_trades()
                        self._cycles_with_position = 0
                        self._trade_entry_atr = 0.0
                        self._trade_mfe = 0.0
                        self._trade_mae = 0.0
                        self._trade_entry_price = 0.0
                        self._trade_side = ""
                        for r in results:
                            self.sdl.release(r.get("symbol", symbol))
                            self._remove_trade_state(r["ticket"])
                            self.trade_lifecycle.close_trade(r["ticket"], {
                                "exit_price": r.get("exit_price", 0),
                                "reason": close_reason,
                                "pnl": r.get("pnl", 0),
                            })
                            self.circuit_breaker.record_trade_result({
                                "pnl": r.get("pnl", 0),
                                "ticket": r["ticket"],
                                "symbol": r.get("symbol", symbol),
                            })

            cycle_data["decision"] = decision
            cycle_data["mof_state"] = market["mof_state"]
            cycle_data["mof_score"] = market["mof_score"]
            self.governor.record_cycle()

            positions = self._get_open_positions()
            cycle_data["open_positions"] = len(positions)

            ledger_trades = self.ledger.get_open_trades()
            ledger_tickets = {t.mt5_ticket for t in ledger_trades if t.mt5_ticket}
            mt5_tickets = {p.get("ticket", 0) for p in positions}
            cycle_data["lifecycle_match"] = (mt5_tickets == ledger_tickets)
            cycle_data["orphan_mt5"] = list(mt5_tickets - ledger_tickets)
            cycle_data["orphan_ledger"] = list(ledger_tickets - mt5_tickets)

            reconcile_result = self.reconciler.reconcile(
                positions, segl_state,
                {"cycle": self.cycle_count, "decision": decision}
            )
            cycle_data["reconciliation"] = reconcile_result

            stats = self.ledger.get_stats()
            cycle_data["trade_stats"] = {
                "total_trades": stats["total_trades"],
                "total_pnl": stats["total_pnl"],
                "win_rate": stats["win_rate"],
                "profit_factor": stats["profit_factor"],
            }

            self.dashboard.update(cycle_data)
            self.dashboard.log_event(cycle_data)

            stop_check = self._check_stopping(cycle_data)
            cycle_data["stopping_triggered"] = stop_check["triggered"]
            cycle_data["stop_reason"] = stop_check["reason"]
            if stop_check["triggered"]:
                self.stopping_triggered = True
                self.stop_reason = stop_check["reason"]

            cycle_data["cycle_duration"] = time.time() - cycle_start
            if self.cycle_count % 10 == 0 or pipeline_trace.get("execution") == "EXECUTED":
                pt = pipeline_trace
                exec_str = pt.get("execution", "NONE")
                gen_ok = sum(1 for g in pt["generated"] if "PASS" in g)
                gen_total = len(pt["generated"])
                thresh_pass = sum(1 for g in pt["threshold_gate"] if "PASS" in g)
                confirm_pass = sum(1 for g in pt["confirm_gate"] if "PASS" in g)
                logger.info(
                    f"[PIPELINE] cycle={self.cycle_count} signals={gen_total}({gen_ok}pass) "
                    f"thresh={thresh_pass} confirm={confirm_pass} "
                    f"exec={exec_str} segl={segl_state}"
                )

            self.pipeline_trace_logger.record_cycle(
                cycle_data, pipeline_trace, md, all_signals
            )

            self.activation_watch.check_cycle(cycle_data, pipeline_trace, md, all_signals)

            self.regime_tracker.record_cycle(cycle_data, md)

            if self.cycle_count - self._last_heartbeat_cycle >= 50:
                self._last_heartbeat_cycle = self.cycle_count
                self._regime_heartbeat(md)

            cycle_data["staircase_describe"] = self.staircase.describe()
            cycle_data["amplifier_describe"] = self.exposure_amplifier.describe()
            self.distribution_observer.record_cycle(cycle_data)

        except Exception as e:
            cycle_data["error"] = str(e)
            cycle_data["traceback"] = traceback.format_exc()
            cycle_data["decision"] = "ERROR"

        self._record_cycle(cycle_data)
        self.live_monitor.record_cycle(cycle_data)
        return cycle_data

    def _regime_heartbeat(self, md: dict):
        closes = md.get("closes", {})
        rsil = {}
        for sym, arr in closes.items():
            if arr is not None and len(arr) >= 20:
                rsi_arr = _compute_rsi(arr)
                rsil[sym] = round(float(rsi_arr[-1]), 1)
        if rsil:
            max_sym = max(rsil, key=rsil.get)
            min_sym = min(rsil, key=rsil.get)
            top3 = sorted(rsil, key=rsil.get, reverse=True)[:3]

            # RSI dispersion index — std dev of RSI values
            rsi_values = list(rsil.values())
            dispersion = round(float(np.std(rsi_values, ddof=0)), 2)

            # Regime compression ratio — fraction of symbols with RSI in 45–55
            compressed = sum(1 for v in rsi_values if 45 <= v <= 55)
            compression_ratio = round(compressed / len(rsi_values), 2)

            # SIL top 3 shift tracking
            top3_changed = (self._prev_heartbeat_top3 is not None
                            and top3 != self._prev_heartbeat_top3)
            self._prev_heartbeat_top3 = top3

            logger.info(
                f"[HEARTBEAT] cycle={self.cycle_count} "
                f"RSI range={rsil[min_sym]}({min_sym})–{rsil[max_sym]}({max_sym}) "
                f"top3={top3} "
                f"dispersion={dispersion} "
                f"compression_ratio={compression_ratio} "
                f"top3_changed={top3_changed} "
                f"universe={self._active_symbols}"
            )

    def _check_stopping(self, cycle_data: dict) -> dict:
        for condition in self.spec.get("stopping_conditions", []):
            cond = condition["condition"]
            threshold = condition["threshold"]

            if cond == "drift_detected":
                drift = cycle_data.get("reconciliation", {}).get("drift_score", 0)
                if drift > threshold:
                    return {"triggered": True, "reason": f"Drift {drift:.4f} > {threshold}"}

            elif cond == "mof_instability":
                if cycle_data.get("mof_state") in ("BLACKOUT", "DEGRADED"):
                    cycle_data.setdefault("_mof_unstable_count", 0)
                    cycle_data["_mof_unstable_count"] += 1
                    if cycle_data["_mof_unstable_count"] >= threshold:
                        return {"triggered": True, "reason": "MOF unstable"}
                else:
                    cycle_data["_mof_unstable_count"] = 0

            elif cond == "sample_size_reached":
                trades = cycle_data.get("trade_stats", {}).get("total_trades", 0)
                if trades >= threshold:
                    return {"triggered": True, "reason": f"Sample {trades}>={threshold}"}

            elif cond == "lifecycle_incoherence":
                mismatches = 0 if cycle_data.get("lifecycle_match", True) else 1
                cycle_data.setdefault("_lifecycle_errors", 0)
                if not cycle_data.get("lifecycle_match", True):
                    cycle_data["_lifecycle_errors"] += 1
                if cycle_data["_lifecycle_errors"] >= threshold:
                    return {"triggered": True, "reason": "Too many lifecycle errors"}

            elif cond == "intent_violation":
                if not cycle_data.get("intent_compliant", True):
                    return {"triggered": True, "reason": "Intent violation"}

        return {"triggered": False, "reason": ""}

    def _record_cycle(self, cycle_data: dict):
        os.makedirs("state", exist_ok=True)
        with open("state/wave12_cycle_log.jsonl", "a") as f:
            f.write(json.dumps(cycle_data, default=str) + "\n")

    def run(self, max_cycles: int = 200, cycle_interval: int = 2) -> dict:
        session_data = {
            "session_start": self.session_start,
            "max_cycles": max_cycles,
            "completed_cycles": 0,
            "total_executions": 0,
            "stopping_reason": "",
            "cycles": []
        }
        for i in range(max_cycles):
            if self.stopping_triggered:
                session_data["stopping_reason"] = self.stop_reason
                break
            result = self.cycle()
            session_data["cycles"].append(result)
            session_data["completed_cycles"] += 1
            if result.get("decision") in ("EXECUTE", "CLOSE"):
                session_data["total_executions"] += 1
            if i < max_cycles - 1 and not self.stopping_triggered:
                time.sleep(cycle_interval)

        session_data["session_end"] = time.time()
        session_data["session_duration"] = session_data["session_end"] - session_data["session_start"]
        session_data["final_dashboard"] = self.dashboard.render()
        session_data["final_stats"] = self.ledger.get_stats()
        session_data["ledger_integrity"] = self.ledger.integrity_check()

        os.makedirs("state", exist_ok=True)
        with open("state/wave12_session_report.json", "w") as f:
            json.dump(session_data, f, indent=2, default=str)

        return session_data

    def get_status(self) -> dict:
        return {
            "cycle_count": self.cycle_count,
            "stopping_triggered": self.stopping_triggered,
            "stop_reason": self.stop_reason,
            "session_duration": time.time() - self.session_start if self.session_start else 0,
            "dashboard_snapshots": len(self.dashboard.snapshots),
            "ledger_events": len(self.ledger._events),
            "spec": self.spec.get("experiment_id", "unknown")
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PROXIMA Wave 12 Live Operations Executor")
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--spec", type=str, default="state/wave12_experiment_spec.json")
    args = parser.parse_args()
    executor = Wave12Executor(spec_path=args.spec)
    session = executor.run(max_cycles=args.cycles, cycle_interval=args.interval)
    print(f"\nSession: {session['completed_cycles']} cycles, {session['total_executions']} execs")
    if session.get("stopping_reason"):
        print(f"Stop: {session['stopping_reason']}")
    print(f"Stats: {json.dumps(session.get('final_stats', {}), indent=2)}")


if __name__ == "__main__":
    main()
