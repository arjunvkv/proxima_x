import os
import time
import signal
import threading
import sys
from queue import Queue, Empty
from pathlib import Path
from typing import Optional

from config.settings import EXECUTION_MODE, MAX_POSITIONS, LOT_SIZE, MAX_TOTAL_LOTS, PROFIT_TARGET, STOP_LOSS_AMOUNT, BURST_TOP_N, MIN_CONFIDENCE, CHOP_CLOSE_DELAY_SECONDS, MIN_TRADE_RUNTIME_SECONDS
from config.settings import CURRENCY_LIST, BASE_CURRENCY_MAP, WLS_DIRECT_MODE
from data.models import TickBatch
from data.mt5_adapter import MT5Adapter
from data.tick_store import TickStore
from currency.graph import CurrencyGraph
from direction.hypothesis import HypothesisGenerator
from features.bar_state import BarStateEngine
from portfolio.drs import DRS
from portfolio.concentration import CurrencyConcentration
from risk.safety import RiskEngine
from execution.paper import PaperExecutor
from execution.mt5_executor import MT5Executor
from persistence.snapshot import SnapshotManager
from monitoring.dashboard import Dashboard
from monitoring.trade_journal import TradeJournal
from .health import HealthMonitor
from .trade_lifecycle import TradeLifecycleLogger
from features.participation_burst import ParticipationBurstEngine
from features.directional_efficiency import DirectionalEfficiency
from narrative import NarrativeEngine, NarrativeInput, NarrativeState
from narrative.overlay import narrative_quality, narrative_health_score
from dashboard.nme_dashboard import NMEDashboard
from dashboard.bar_dashboard import BarStateDashboard


class RuntimeManager:
    def __init__(self):
        self.running = False
        self._regime_data: dict = {"polarized_ssp_pct": 0.0, "regime": "N/A", "entries_blocked": False}
        self._chop_since: float = 0.0
        self.shutdown_event = threading.Event()
        self._force_exit = False

        self.mt5 = MT5Adapter()
        self.store = TickStore()
        self.burst = ParticipationBurstEngine()
        self.efficiency = DirectionalEfficiency()
        self.graph = CurrencyGraph()
        self.generator = HypothesisGenerator()
        self.drs = DRS()
        self.bar_state = BarStateEngine(self.mt5, self.store)
        self.risk = RiskEngine()
        if EXECUTION_MODE == "live":
            self.executor = MT5Executor(self.mt5)
        else:
            self.executor = PaperExecutor()
        self.snapshot = SnapshotManager()
        self.dashboard = Dashboard()
        self.journal = TradeJournal()
        self._symbol_trade_history = {}
        # Preload past trades from journal file
        import os
        import json
        j_path = os.path.join("logs", "trade_journal.jsonl")
        if os.path.exists(j_path):
            try:
                with open(j_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        sym = record.get("symbol")
                        event = record.get("event")
                        if not sym:
                            continue
                        if event == "open":
                            self._symbol_trade_history[sym] = {
                                "status": "RUNNING",
                                "direction": record.get("direction"),
                                "entry_price": record.get("entry_price"),
                                "sl": record.get("sl") or record.get("stop_loss", 0.0),
                                "tp": record.get("tp") or record.get("take_profit", 0.0),
                                "ts": record.get("ts", 0),
                            }
                        elif event == "close":
                            self._symbol_trade_history[sym] = {
                                "status": "CLOSED",
                                "direction": record.get("direction"),
                                "entry_price": record.get("entry_price"),
                                "exit_price": record.get("exit_price"),
                                "pnl": record.get("pnl"),
                                "reason": record.get("exit_reason") or record.get("reason"),
                                "ts": record.get("ts", 0),
                            }
            except Exception:
                pass
        self.health = HealthMonitor()
        self.nme = NarrativeEngine()
        self.nme_dashboard = NMEDashboard()
        self.bar_dashboard = BarStateDashboard()
        self._trade_lifecycle = TradeLifecycleLogger()

        self._tick_queue = Queue(maxsize=1000)
        self._tick_thread: Optional[threading.Thread] = None
        self._decision_thread: Optional[threading.Thread] = None

        self._cycle_count = 0
        self._last_decision = 0.0
        self._last_snapshot = 0.0
        self._start_time = 0.0
        self._mt5_audit = None
        self._production_ready = False
        self._pipeline_metrics = {"generated": 0, "ranked": 0, "selected": 0, "risk_approved": 0, "executed": 0, "bar_aligned": 0}
        self._available_symbols: list[str] = []
        self._excluded_symbols: list[dict] = []
        self._last_discovery = 0.0
        self.last_exec_fail: Optional[str] = None
        self._top_burst_pairs: list[str] = []
        self._currency_bursts: dict[str, float] | None = None
        self._persistence: dict[str, dict] | None = None
        self._currency_der: dict[str, float] | None = None
        self._der_persistence: dict[str, dict] | None = None
        self._top_der_pairs: list[str] = []
        self._nme_narrative: Optional[NarrativeState] = None
        self._pnl_reset_done = False
        self._nme_trade_snapshots: list[dict] = []
        self._position_trajectory: dict[str, list] = {}
        self._narrative_epoch_id: Optional[str] = None
        self._narrative_peak_health: float = 0.0
        self._narrative_decay_exit: bool = False
        self._narrative_decay_cycles: int = 0
        self._last_batch_pnl: float = 0.0
        self._batch_open_cycle: int = 0
        self._deferred_closes: dict[str, dict] = {}
        self._deferred_all_reason: str | None = None
        self.dashboard_process = None

    def start(self) -> None:
        self._setup_signal_handlers()
        
        # Start standalone web dashboard in a separate background process
        try:
            import subprocess
            dashboard_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web_dashboard.py")
            self.dashboard_process = subprocess.Popen(
                [sys.executable, dashboard_script, "7700"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True
            )
            print("\033[92m[DASHBOARD ACTIVE] Game-level Web HUD running on http://localhost:7700\033[0m")
        except Exception as e:
            print(f"[DASHBOARD] Failed to auto-start web dashboard: {e}", file=sys.stderr)

        print("Connecting to MT5...")
        if not self.mt5.connect():
            print("ERROR: Cannot connect to MT5. Ensure terminal is running.")
            sys.exit(1)
        print("MT5 connected.")
        if EXECUTION_MODE == "live":
            discovery = self.executor.discover_symbols()
            self._available_symbols = discovery["available"]
            self._excluded_symbols = discovery["excluded"]
        else:
            from config.settings import SYMBOLS
            self._available_symbols = list(SYMBOLS)

        state = self.snapshot.load()
        if state:
            print(f"Snapshot loaded (market_timestamp={state.get('market_timestamp', 0):.1f})")
        else:
            print("FACTORY BOOT: No snapshot found.")

        self._start_time = time.time()
        self.running = True

        self._tick_thread = threading.Thread(target=self._tick_worker, daemon=True)
        self._decision_thread = threading.Thread(target=self._decision_worker, daemon=True)
        self._tick_thread.start()
        self._decision_thread.start()

        self._check_stop_file()

        try:
            while self.running and not self.shutdown_event.is_set():
                if Path("STOP").exists():
                    print("STOP file detected. Shutting down...")
                    self.shutdown_event.set()
                    break
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.running = False
        self.shutdown_event.set()

        if hasattr(self, 'dashboard_process') and self.dashboard_process:
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=2.0)
            except Exception:
                try:
                    self.dashboard_process.kill()
                except Exception:
                    pass

        if self._tick_thread and self._tick_thread.is_alive():
            self._tick_thread.join(timeout=3.0)
        if self._decision_thread and self._decision_thread.is_alive():
            self._decision_thread.join(timeout=5.0)

        state = {
            "market_timestamp": time.time(),
            "currency_strengths": self.graph.strengths_raw(),
            "graph_quality": self.graph.quality(),
            "positions": [(p.id, p.symbol, p.direction, p.entry_price) for p in self.executor.positions],
            "trade_count": len(self.risk.trades),
            "uptime": time.time() - self._start_time
        }
        if self.snapshot.save(state):
            print("Snapshot saved.")
        else:
            print("WARNING: Snapshot save failed.")

        self.nme.close()
        self.mt5.disconnect()
        print("Shutdown complete.")

    def _tick_worker(self) -> None:
        poll_interval = 5.0
        while not self.shutdown_event.is_set():
            try:
                batch = self.mt5.poll_ticks()
                try:
                    self._tick_queue.put(batch, timeout=0.1)
                except Exception:
                    try:
                        self._tick_queue.get_nowait()
                        self._tick_queue.put(batch, timeout=0.1)
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(poll_interval)

    def _decision_worker(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                batches = []
                batch = self._tick_queue.get(timeout=1.0)
                batches.append(batch)
                while not self._tick_queue.empty():
                    try:
                        batches.append(self._tick_queue.get_nowait())
                    except Empty:
                        break

                self._process_batches(batches)
            except Empty:
                self.executor.sync()
                self._process_deferred_closes()
                to_close_indiv_sl = self.risk.check_individual_stop_loss(self.executor.positions)
                if to_close_indiv_sl:
                    pnls = [f"{p.symbol}=${p.pnl:.2f}" for p in to_close_indiv_sl]
                    print(f"[PER-POSITION STOP LOSS] closing {len(to_close_indiv_sl)}: {', '.join(pnls)}", file=sys.stderr)
                    self._close_individual_positions(to_close_indiv_sl, "STOP_LOSS")
                total_pnl = sum(p.pnl or 0 for p in self.executor.positions)
                if total_pnl <= -100.0 and self.executor.positions:
                    print(f"[BATCH STOP LOSS] total_pnl={total_pnl:.2f} <= -$100 — closing all", file=sys.stderr)
                    self._close_all_positions("STOP_LOSS")
                    self._reset_after_profit_target()
                    self.dashboard.latest_event = {"event": "STOP_LOSS", "time": time.time()}
                elif self.risk.check_profit_target(self.executor.positions):
                    self._close_all_positions("PROFIT_TARGET")
                    self._reset_after_profit_target()
                    self.dashboard.latest_event = {"event": "PROFIT_TARGET", "time": time.time()}
                continue
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue

    def _process_batches(self, batches: list[TickBatch]) -> None:
        now = time.time()
        self._process_deferred_closes(now)

        for batch in batches:
            self.store.add_ticks(batch.ticks)
            for tick in batch.ticks:
                self.burst.update(tick.symbol, tick.volume)
                self.efficiency.update(tick.symbol, tick.mid)
            self.executor.update_prices(batch.ticks)
            for pos in self.executor.positions:
                if pos.id not in self._position_trajectory:
                    self._position_trajectory[pos.id] = []
            to_close_indiv_sl = self.risk.check_individual_stop_loss(self.executor.positions)
            if to_close_indiv_sl:
                pnls = [f"{p.symbol}=${p.pnl:.2f}" for p in to_close_indiv_sl]
                print(f"[PER-POSITION STOP LOSS] closing {len(to_close_indiv_sl)}: {', '.join(pnls)}", file=sys.stderr)
                self._close_individual_positions(to_close_indiv_sl, "STOP_LOSS")
            # to_close_indiv_pt = self.risk.check_individual_profit_target(self.executor.positions)
            # if to_close_indiv_pt:
            #     pnls = [f"{p.symbol}=${p.pnl:.2f}" for p in to_close_indiv_pt]
            #     print(f"[PER-POSITION PROFIT TARGET] closing {len(to_close_indiv_pt)}: {', '.join(pnls)}", file=sys.stderr)
            #     self._close_individual_positions(to_close_indiv_pt, "PROFIT_TARGET")
            total_pnl = sum(p.pnl or 0 for p in self.executor.positions)
            if total_pnl <= -100.0 and self.executor.positions:
                print(f"[BATCH STOP LOSS] total_pnl={total_pnl:.2f} <= -$100 — closing all", file=sys.stderr)
                self._close_all_positions("STOP_LOSS")
                self._reset_after_profit_target()
                self.dashboard.latest_event = {"event": "STOP_LOSS", "time": now}
            elif self.risk.check_profit_target(self.executor.positions):
                self._close_all_positions("PROFIT_TARGET")
                self._reset_after_profit_target()
                self.dashboard.latest_event = {"event": "PROFIT_TARGET", "time": now}
            # ── Close all on chop safety net ────────────────
            if (self._regime_data.get("regime") == "CHOP"
                    and self.executor.positions
                    and self._chop_since > 0
                    and (time.time() - self._chop_since) >= CHOP_CLOSE_DELAY_SECONDS):
                print(f"[CHOP CLOSE] chop persisted {CHOP_CLOSE_DELAY_SECONDS}s — closing all", file=sys.stderr)
                self._close_all_positions("CHOP_DETECTED")
                self._chop_since = 0.0

        returns = self.store.calculate_returns()
        if self._available_symbols:
            returns = {s: v for s, v in returns.items() if s in self._available_symbols}

        if now - self._last_decision >= 5.0:
            solve_start = time.time()
            freshness_weights = {sym: self.store.freshness(sym) for sym in returns}
            topology_weights = self.graph.topology.pair_weights([s for s, v in returns.items() if v != 0.0])
            weights = {sym: freshness_weights.get(sym, 0) * topology_weights.get(sym, 0) for sym in returns}
            self.graph.update(returns, weights, now, available_count=len(self._available_symbols))
            solve_ms = (time.time() - solve_start) * 1000
            self.health.record_solve(solve_ms)

        hypotheses = []

        if now - self._last_decision >= 30.0:
            self._last_decision = now

            self._pipeline_metrics.update({
                "generated": 0,
                "burst_hyp": 0,
                "ranked": 0,
                "selected": 0,
                "risk_approved": 0,
                "executed": 0,
            })

            if self._cycle_count % 10 == 0 and EXECUTION_MODE == "live":
                discovery = self.executor.discover_symbols()
                self._available_symbols = discovery["available"]
                self._excluded_symbols = discovery["excluded"]
            hypotheses = self.generator.generate_all(self.graph, now)
            hypotheses = [h for h in hypotheses if h.symbol in self._available_symbols]
            self._pipeline_metrics["generated"] = len(hypotheses)
            self._top_burst_pairs = self.burst.get_top_burst_pairs(BURST_TOP_N)
            self._currency_bursts = None
            self._persistence = None
            if hypotheses:
                print(f"[TRACE] gen={len(hypotheses)} top.conf={hypotheses[0].confidence:.3f} "
                      f"top.symbol={hypotheses[0].symbol} top.dir={'BUY' if hypotheses[0].direction>0 else 'SELL'} "
                      f"top.spread={hypotheses[0].base_strength-hypotheses[0].quote_strength:.6f}",
                      file=sys.stderr)
            if self._top_burst_pairs:
                self._currency_bursts = self.burst.get_currency_bursts(returns)
                self._persistence = self.burst.get_persistence()
                burst_lookup = {p: i for i, p in enumerate(self._top_burst_pairs)}
                for h in hypotheses:
                    rank = burst_lookup.get(h.symbol)
                    if rank is None:
                        h.confidence = h.confidence * 0.5
                    elif rank >= 3:
                        h.confidence = h.confidence * 0.8
            # ── DIRECTIONAL EFFICIENCY FILTER ─────────────────────
            if hypotheses:
                der_values = self.efficiency.get_all_der()
                der_filtered = []
                for h in hypotheses:
                    der = der_values.get(h.symbol, 0.0)
                    if der < 0.10:
                        print(f"[TRACE] der_reject={h.symbol} der={der:.3f}", file=sys.stderr)
                        continue
                    prev_der = self.efficiency.get_previous_der(h.symbol)
                    der_change = der - prev_der
                    effective_der = der + 0.5 * der_change
                    der_factor = 0.8 + 0.4 * effective_der
                    der_factor = min(1.2, max(0.0, der_factor))
                    h.confidence = min(1.0, h.confidence * der_factor)
                    der_filtered.append(h)
                print(f"[TRACE] der_pass={len(der_filtered)}/{len(hypotheses)}", file=sys.stderr)
                hypotheses = der_filtered
                self.efficiency.finalize_cycle()
            self._pipeline_metrics["burst_hyp"] = len(hypotheses)

            # ── BAR STATE ALIGNMENT (replaces SWPS) ─────────────
            if self.bar_state.update():
                pre = len(hypotheses)
                for h in hypotheses:
                    align = self.bar_state.alignment(h.symbol, h.direction)
                    h.confidence = min(1.0, h.confidence * align)
                    if align < 0.40:
                        print(f"[BAR STATE] reject={h.symbol} align={align:.3f}", file=sys.stderr)
                hypotheses = [h for h in hypotheses if h.confidence >= MIN_CONFIDENCE]
                print(f"[BAR STATE] aligned={len(hypotheses)}/{pre}  {self.bar_state.get_summary()}", file=sys.stderr)
            else:
                print("[BAR STATE] not ready — no bar alignment", file=sys.stderr)
            self._pipeline_metrics["bar_aligned"] = len(hypotheses)

            self.executor.sync()

            if self.executor.sync_failed:
                print("[EXECUTION BLOCKED] Position state unknown — sync failed", file=sys.stderr)
                return

            self.risk.set_positions(self.executor.positions)
            self.drs.set_positions(self.executor.positions)

            # ── NARRATIVE DECAY EXIT ─────────────────────────────────
            if os.environ.get("DISABLE_NARRATIVE_DECAY", "0") == "1":
                if self._narrative_decay_exit:
                    self._narrative_decay_exit = False
            if self._narrative_decay_exit and self.executor.positions:
                total_pnl = sum(p.pnl or 0 for p in self.executor.positions)
                print(f"[NARRATIVE DECAY] closing all — pos={len(self.executor.positions)} pnl=${total_pnl:.2f}", file=sys.stderr)
                self._close_all_positions("NARRATIVE_DECAY")
                self._reset_after_profit_target()
                return

            _a = self.graph.execution_allowed(returns)
            _c = self.graph.connectivity_score(returns)
            _ap = self.graph._active_pair_count
            _q = self.graph.state.quality
            nme_ready = self._nme_narrative is not None
            print(f"[GATE] allowed={_a} prod={self._production_ready} nme={nme_ready} hyp={len(hypotheses)} ap={_ap} q={_q:.3f} conn={_c:.3f} direct={WLS_DIRECT_MODE}", file=sys.stderr)
            gate_pass = (WLS_DIRECT_MODE or (_a and self._production_ready)) and nme_ready
            if gate_pass:
                ranked = self.drs.rank(hypotheses)
                self._pipeline_metrics["ranked"] = len(ranked)
                pos_count = self.executor.position_count()
                open_count = pos_count
                print(f"[POSITION STATE] executor={len(self.executor.positions)} count={pos_count}", file=sys.stderr)
                if open_count >= MAX_POSITIONS:
                    candidates = []
                    print(
                        f"[ENTRY SKIP] — open_count={open_count} >= MAX={MAX_POSITIONS}",
                        file=sys.stderr
                    )
                elif self.risk.cooldown_active():
                    candidates = []
                    remain = int(self.risk._profit_cooldown_until - now)
                    print(f"[ENTRY SKIP] — profit cooldown {remain}s remaining", file=sys.stderr)
                else:
                    candidates = ranked

                self._pipeline_metrics["selected"] = len(candidates)

                # ── REGIME GATE: detect chop via polarized SSP distribution ────
                polar_ssp = 0
                total_ssp = 0
                try:
                    for rsym in self._available_symbols:
                        rprice = getattr(self.executor, "_last_prices", {}).get(rsym, 0.0)
                        if rprice == 0.0:
                            rprice = getattr(self.mt5, "_last_bar_close", {}).get(rsym, 0.0)
                        if rprice == 0.0:
                            continue
                        rssp = self.bar_state.get_structural_swing_position(rsym, rprice)
                        if rssp is None:
                            continue
                        for v in (rssp.get("buy_ssp"), rssp.get("sell_ssp")):
                            if isinstance(v, (int, float)):
                                total_ssp += 1
                                if v < 0.3 or v > 0.7:
                                    polar_ssp += 1
                except Exception:
                    pass
                pol_pct = (polar_ssp / total_ssp) if total_ssp > 4 else 0.0
                is_chop = pol_pct > 0.70 if self._chop_since == 0.0 else pol_pct > 0.65
                if is_chop and self._chop_since == 0.0:
                    self._chop_since = time.time()
                    print(f"[REGIME BLOCK] chop — polar_ssp={pol_pct:.0%} blocking all entries", file=sys.stderr)
                    candidates = []
                elif is_chop:
                    candidates = []
                else:
                    if self._chop_since > 0.0:
                        print(f"[CHOP UNBLOCK] chop cleared — resetting system to cycle 1", file=sys.stderr)
                        self._reset_after_profit_target()
                    self._chop_since = 0.0
                chop_minutes = round((time.time() - self._chop_since) / 60.0, 1) if is_chop and self._chop_since > 0 else 0.0
                self._regime_data = {
                    "polarized_ssp_pct": round(pol_pct * 100, 0),
                    "regime": "CHOP" if is_chop else "TREND",
                    "entries_blocked": is_chop,
                    "chop_minutes": chop_minutes,
                    "threshold_pct": 70.0,
                    "unblock_threshold": 65.0,
                    "gap_to_clear": max(0.0, round(pol_pct * 100 - 65.0, 1)) if is_chop else 0.0,
                }

                cycle_submitted = set()
                self.last_exec_fail = None
                for h in candidates:
                    if self.executor.position_count() >= MAX_POSITIONS:
                        break
                    dir_label = "BUY" if h.direction > 0 else "SELL"
                    symdir = f"{h.symbol}:{dir_label}"
                    if symdir in cycle_submitted:
                        continue
                    # ── DUPLICATE POSITION CHECK ───────────────────
                    if any(p.symbol == h.symbol and p.direction == dir_label for p in self.executor.positions):
                        print(f"[DUPLICATE BLOCK] {h.symbol} {dir_label} already active — open positions: {[(p.symbol, p.direction) for p in self.executor.positions]}", file=sys.stderr)
                        continue
                    # ── BAR STATE ENTRY GATE (final check) ─────────
                    bar_align = self.bar_state.alignment(h.symbol, h.direction)
                    if bar_align < 0.40:
                        print(f"[BAR GATE] reject={h.symbol} align={bar_align:.3f} conf={h.confidence:.3f}", file=sys.stderr)
                        continue
                    approved = self.risk.approve(h)
                    if not approved:
                        print(f"[RISK BLOCK] {h.symbol} drs={h.drs_score:.3f}", file=sys.stderr)
                        continue
                    self._pipeline_metrics["risk_approved"] += 1
                    nme_info = f" nme_leader={self._nme_narrative.identity.leader} nme_dir={'BUY' if self._nme_narrative.identity.direction>0 else 'SELL'} nme_nmi={self._nme_narrative.nmi:.2f}" if self._nme_narrative is not None else " nme=None"
                    print(f"[ENTRY] {h.symbol} dir={'BUY' if h.direction>0 else 'SELL'} "
                          f"base_str={h.base_strength:.6f} quote_str={h.quote_strength:.6f} "
                          f"spread={h.base_strength-h.quote_strength:.6f} conf={h.confidence:.3f} "
                          f"bar_align={bar_align:.3f}{nme_info}",
                          file=sys.stderr)
                    # ── CALCULATE ENTRY SWING TP & DOLLAR SL ─────────
                    open_price = self.bar_state._forming_open.get(h.symbol)
                    current_price = getattr(self.executor, "_last_prices", {}).get(h.symbol, 0.0)
                    if current_price == 0.0:
                        current_price = self.mt5._last_bar_close.get(h.symbol, 0.0)
                    if current_price == 0.0 and open_price is not None:
                        current_price = open_price

                    # ── SWING-STATE ENTRY GATE ─────────────────────
                    if current_price > 0.0:
                        ssp_data = self.bar_state.get_structural_swing_position(h.symbol, current_price)
                        msp_data = None
                        stats = self.bar_state.get_swing_stats(h.symbol)
                        if stats is not None:
                            avg_up = stats["avg_upside"]
                            avg_dn = stats["avg_downside"]
                            msp_data = self.bar_state.get_micro_swing_positions(h.symbol, avg_up, avg_dn)
                        swing_result = self.bar_state.classify_swing_state(h.direction, ssp_data, msp_data)
                        swing_state = swing_result.get("swing_state", "")
                        position_state = swing_result.get("position_state", "")
                        decision = swing_result.get("decision", "")
                        # Block EXHAUSTED
                        if decision == "BLOCK":
                            print(f"[SWING BLOCK] {h.symbol} {dir_label} state={swing_state} pos={position_state}", file=sys.stderr)
                            continue
                        # Block LATE (override CAUTION → BLOCK)
                        if swing_state == "LATE":
                            print(f"[SWING BLOCK] {h.symbol} {dir_label} state=LATE (late entry)", file=sys.stderr)
                            continue
                        # Direction-checked BREAKOUT: block BUY if sell_ssp>1 (BREAKOUT_DOWN), SELL if buy_ssp>1 (BREAKOUT_UP)
                        if ssp_data is not None:
                            if h.direction > 0 and ssp_data.get("sell_ssp", 0) > 1.0:
                                print(f"[SWING BLOCK] {h.symbol} BUY blocked — sell_ssp={ssp_data['sell_ssp']:.2f} (BREAKOUT_DOWN)", file=sys.stderr)
                                continue
                            if h.direction < 0 and ssp_data.get("buy_ssp", 0) > 1.0:
                                print(f"[SWING BLOCK] {h.symbol} SELL blocked — buy_ssp={ssp_data['buy_ssp']:.2f} (BREAKOUT_UP)", file=sys.stderr)
                                continue
                        # Noise filter: avg swing < 1.0 pip
                        if stats is not None:
                            avg_swing_pips = (stats["avg_upside"] - stats["avg_downside"]) * 10000
                            if abs(avg_swing_pips) < 1.0:
                                print(f"[SWING BLOCK] {h.symbol} avg_swing={abs(avg_swing_pips):.1f}p < 1.0p (noise)", file=sys.stderr)
                                continue
                        # Block exhausted swing: rem_up_price == 0 (BUY) or rem_dn_price == 0 (SELL)
                        if stats is not None and open_price is not None and open_price > 0:
                            rem_dn_exhausted = max(0.0, current_price - (open_price + avg_dn))
                            rem_up_exhausted = max(0.0, (open_price + avg_up) - current_price)
                            if h.direction > 0 and rem_up_exhausted == 0:
                                print(f"[SWING BLOCK] {h.symbol} BUY — rem_up=0 (swing exhausted)", file=sys.stderr)
                                continue
                            if h.direction < 0 and rem_dn_exhausted == 0:
                                print(f"[SWING BLOCK] {h.symbol} SELL — rem_dn=0 (swing exhausted)", file=sys.stderr)
                                continue

                    sl_price = None
                    tp_price = None

                    if current_price > 0.0:
                        # 1. Calculate usd_quote_rate
                        quote_ccy = h.symbol[3:6]
                        suffix = h.symbol[6:] if len(h.symbol) > 6 else ""
                        
                        last_prices = {}
                        last_prices.update(self.mt5._last_bar_close)
                        last_prices.update(getattr(self.executor, "_last_prices", {}))
                        
                        usd_quote_rate = 1.0
                        if quote_ccy != "USD":
                            pair1 = f"USD{quote_ccy}{suffix}"
                            if pair1 in last_prices:
                                usd_quote_rate = last_prices[pair1]
                            else:
                                pair2 = f"{quote_ccy}USD{suffix}"
                                if pair2 in last_prices:
                                    rate = last_prices[pair2]
                                    usd_quote_rate = 1.0 / rate if rate > 0 else 1.0
                        
                        # 2. Dollar-based stop loss (-60.0 USD)
                        sl_usd_abs = abs(STOP_LOSS_AMOUNT) if STOP_LOSS_AMOUNT else 60.0
                        sl_dist_usd = sl_usd_abs * usd_quote_rate / (LOT_SIZE * 100000)
                        if h.direction > 0: # BUY
                            sl_price = current_price - sl_dist_usd
                        else: # SELL
                            sl_price = current_price + sl_dist_usd
                        
                        # 3. Swing-based take profit
                        stats = self.bar_state.get_swing_stats(h.symbol)
                        if stats is not None and open_price is not None and open_price > 0:
                            avg_dn = stats["avg_downside"]
                            avg_up = stats["avg_upside"]
                            rem_dn_price = max(0.0, current_price - (open_price + avg_dn))
                            rem_up_price = max(0.0, (open_price + avg_up) - current_price)
                            
                            if h.direction > 0: # BUY
                                tp_price = current_price + rem_up_price * 1.5
                            else: # SELL
                                tp_price = current_price - rem_dn_price * 1.5

                        # 4. Minimum TP guard: skip if TP < 2.0 pips from entry
                        if tp_price is not None and current_price > 0:
                            min_tp_dist = 0.0002 if "JPY" not in h.symbol else 0.02
                            tp_dist = abs(tp_price - current_price)
                            if tp_dist < min_tp_dist:
                                print(f"[TP BLOCK] {h.symbol} {dir_label} tp_dist={tp_dist:.5f} < {min_tp_dist} (<2p min)", file=sys.stderr)
                                continue

                    result = self.executor.execute(h, sl=sl_price, tp=tp_price)
                    if result.success:
                        cycle_submitted.add(symdir)
                        self._pipeline_metrics["executed"] += 1
                        if self._batch_open_cycle == 0:
                            self._batch_open_cycle = self._cycle_count
                        self.drs.record_position(
                            next(p for p in self.executor.positions if p.id == result.position_id)
                        )
                        if self._nme_narrative is not None:
                            self._nme_trade_snapshots.append({
                                "time": time.time(),
                                "symbol": h.symbol,
                                "direction": "BUY" if h.direction > 0 else "SELL",
                                "leader": self._nme_narrative.identity.leader,
                                "nmi": round(self._nme_narrative.nmi, 2),
                                "phase": self._nme_narrative.phase.value,
                            })
                        self.risk.set_positions(self.executor.positions)
                        sp = self.graph.get_strength_persistence()
                        pos = next((p for p in self.executor.positions if p.id == result.position_id), None)
                        pos_sl = pos.stop_loss if pos else 0.0
                        pos_tp = pos.take_profit if pos else 0.0
                        self.journal.record_open(
                            position_id=result.position_id,
                            symbol=h.symbol,
                            direction="BUY" if h.direction > 0 else "SELL",
                            entry_price=result.price,
                            volume=LOT_SIZE,
                            confidence=h.confidence,
                            drs_score=h.drs_score,
                            strengths=self.graph.strengths_raw(),
                            peaks={c: v["peak"] for c, v in sp.items()},
                            troughs={c: v["trough"] for c, v in sp.items()},
                            streaks={c: v["streak"] for c, v in sp.items()},
                            bursts=self.burst.get_currency_bursts(returns) if self._currency_bursts is None else (self._currency_bursts or {}),
                            sl=pos_sl,
                            tp=pos_tp,
                        )
                        self._symbol_trade_history[h.symbol] = {
                            "status": "RUNNING",
                            "direction": "BUY" if h.direction > 0 else "SELL",
                            "entry_price": result.price,
                            "sl": pos_sl,
                            "tp": pos_tp,
                            "ts": time.time(),
                        }
                    else:
                        self.last_exec_fail = f"{h.symbol} {result.reason}"
                        print(f"[EXEC FAIL] {h.symbol} {result.reason}", file=sys.stderr)

        if self.executor.positions and not self._trade_lifecycle.is_active:
            self._trade_lifecycle.open_batch(self.executor.positions, self._nme_narrative)

        prices = {p.symbol: p.current_price for p in self.executor.positions}
        to_close = self.risk.check_stops(prices)
        sp = self.graph.get_strength_persistence()
        strengths_now = self.graph.strengths_raw()
        bursts_now = self.burst.get_currency_bursts(returns) if self._currency_bursts is None else (self._currency_bursts or {})
        for pos in to_close:
            price_now = prices.get(pos.symbol, pos.entry_price)
            age = time.time() - pos.entry_time
            close_reason = "STOP_LOSS"
            if pos.direction == "BUY" and price_now >= pos.take_profit:
                close_reason = "TAKE_PROFIT"
            elif pos.direction == "SELL" and price_now <= pos.take_profit:
                close_reason = "TAKE_PROFIT"
            if age < MIN_TRADE_RUNTIME_SECONDS:
                if close_reason == "TAKE_PROFIT":
                    pass
                else:
                    self._deferred_closes[pos.id] = {"reason": close_reason, "request_time": time.time()}
                    print(f"[DEFER STOP] {pos.symbol} {pos.direction} age={age:.0f}s ← {MIN_TRADE_RUNTIME_SECONDS}s — deferred ({close_reason})", file=sys.stderr)
                    continue
            traj = self._position_trajectory.pop(pos.id, [])

            if traj:
                entry = traj[0]
                final = traj[-1]
                peak_pnl = max(t.get("pnl", 0) for t in traj)
                peak_h = max(t["health"] for t in traj)
                peak_nmi = max(t["nmi"] for t in traj)
                phases = list(dict.fromkeys(t["phase"] for t in traj))
                print(f"[NARRATIVE TRAJECTORY] {pos.symbol} {pos.direction} "
                      f"entry: h={entry['health']} n={entry['nmi']} p={entry['phase']} pnl=${entry.get('pnl',0):.2f} "
                      f"peak_pnl=${peak_pnl:.2f} peak_h={peak_h:.2f} "
                      f"exit: h={final['health']} n={final['nmi']} p={final['phase']} pnl=${final.get('pnl',0):.2f} "
                      f"phases={'→'.join(phases)} cycles={len(traj)} ({close_reason})", file=sys.stderr)
            r = self.executor.close_position(pos.id, price_now, close_reason)
            self.drs.remove_position(pos.symbol)
            self.journal.record_close(
                position_id=pos.id,
                exit_price=r.price,
                pnl=pos.pnl or 0,
                reason=r.reason or close_reason,
                strengths=strengths_now,
                peaks={c: v["peak"] for c, v in sp.items()},
                troughs={c: v["trough"] for c, v in sp.items()},
                streaks={c: v["streak"] for c, v in sp.items()},
                bursts=bursts_now,
            )
            self._symbol_trade_history[pos.symbol] = {
                "status": "CLOSED",
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "exit_price": r.price,
                "pnl": pos.pnl or 0,
                "reason": r.reason or close_reason,
                "ts": time.time(),
            }

        if now - self._last_snapshot >= 300.0:
            self._last_snapshot = now
            state = {
                "market_timestamp": now,
                "currency_strengths": self.graph.strengths_raw(),
                "graph_quality": self.graph.quality(),
                "positions": [(p.id, p.symbol, p.direction, p.entry_price) for p in self.executor.positions],
            }
            self.snapshot.save(state)

        freshness = self.store.all_freshness()
        health = self.health.check(
            mt5_ok=self.mt5.connected,
            tick_freshness=freshness,
            graph_quality=self.graph.quality(),
            snapshot_ok=True
        )
        self.risk.health = health

        top_h = None
        if hypotheses:
            h = hypotheses[0]
            top_h = {
                "symbol": h.symbol,
                "direction": h.direction,
                "confidence": h.confidence,
                "drs_score": h.drs_score,
                "base_strength": h.base_strength,
                "quote_strength": h.quote_strength,
            }

        total_pnl = sum(p.pnl or 0 for p in self.executor.positions)
        factor_exposure = dict(self.drs._net_currency_exposure())
        observability = dict(self.graph.state.observability)
        z_scores = self.graph.strength_zscore()
        stability = self.graph.strength_stability()
        edge_balance = self.graph.topology.currency_edge_balance(returns)
        missing_syms = self.graph.missing_symbols(returns)
        missing_impact = self.graph.missing_currency_impact(returns)
        health_report = self.graph.health_report(returns)

        concentration = CurrencyConcentration().calculate(self.executor.positions)
        max_conc = max(concentration.values()) if concentration else 0.0
        if max_conc > 0.50:
            health.state = "DEGRADED"

        stress_test = self.graph.currency_stress_test(returns)
        if self._currency_bursts is None:
            currency_bursts = self.burst.get_currency_bursts(returns)
            persistence = self.burst.get_persistence()
        else:
            currency_bursts = self._currency_bursts
            persistence = self._persistence
        strength_persistence = self.graph.get_strength_persistence()
        self._currency_der = self.efficiency.get_currency_efficiency(returns)
        self.efficiency.update_persistence(self._currency_der)
        self._der_persistence = self.efficiency.get_persistence()
        self._top_der_pairs = self.efficiency.get_top_pairs(3)
        recent_fails = self.executor.recent_failures()

        self._update_production_readiness(health_report, max_conc)

        self._cycle_count += 1
        cd_remain = max(0, int(self.risk._profit_cooldown_until - now)) if self.risk.cooldown_active() else 0

        nme_input = NarrativeInput(
            cycle=self._cycle_count,
            currency_strengths=self.graph.strengths_raw(),
            currency_bursts=currency_bursts or {},
            currency_der=self._currency_der or {},
            graph_quality=health.graph_quality,
            tick_quality=health.tick_quality,
            reliability=observability,
        )
        nme_state = self.nme.update(nme_input)
        self._nme_narrative = nme_state
        if self._position_trajectory and self._nme_narrative is not None:
            traj_health = narrative_health_score(self._nme_narrative)
            snap = {
                "cycle": self._cycle_count,
                "health": round(traj_health, 3),
                "nmi": round(self._nme_narrative.nmi, 3),
                "phase": self._nme_narrative.phase.value,
                "strength_delta": round(self._nme_narrative.strength_delta or 0, 6),
                "age": self._nme_narrative.age,
            }
            pos_pnls = {p.id: round(p.pnl or 0, 2) for p in self.executor.positions}
            for pid in list(self._position_trajectory):
                snap["pnl"] = pos_pnls.get(pid, 0)
                self._position_trajectory[pid].append(dict(snap))

        # ── NARRATIVE DECAY DETECTION ────────────────────────────
        DECAY_THRESHOLD = 0.70
        DECAY_REQUIRED_CYCLES = 5
        DEATH_REQUIRED_CYCLES = 7
        if self._nme_narrative is not None:
            detect_health = narrative_health_score(self._nme_narrative)
            epoch_id = f"{self._nme_narrative.identity.leader}_{self._nme_narrative.identity.direction}"
            if epoch_id != self._narrative_epoch_id:
                self._narrative_epoch_id = epoch_id
                self._narrative_peak_health = detect_health
                self._narrative_decay_exit = False
                self._narrative_decay_cycles = 0
            if detect_health > self._narrative_peak_health:
                self._narrative_peak_health = detect_health
                self._narrative_decay_cycles = 0
            elif detect_health < self._narrative_peak_health * DECAY_THRESHOLD:
                self._narrative_decay_cycles += 1
                if self._narrative_decay_cycles >= DECAY_REQUIRED_CYCLES:
                    if os.environ.get("DISABLE_NARRATIVE_DECAY", "0") != "1":
                        self._narrative_decay_exit = True
                    print(f"[NARRATIVE DECAY] DETECTED — health {self._narrative_peak_health:.2f}→{detect_health:.2f} peak_nmi={self._nme_narrative.nmi:.2f} epoch={epoch_id}", file=sys.stderr)
            else:
                self._narrative_decay_cycles = 0
        elif self._narrative_epoch_id is not None and not self._narrative_decay_exit:
            detect_health = 0.5
            if detect_health < self._narrative_peak_health * DECAY_THRESHOLD:
                self._narrative_decay_cycles += 1
                if self._narrative_decay_cycles >= DEATH_REQUIRED_CYCLES:
                    if os.environ.get("DISABLE_NARRATIVE_DECAY", "0") != "1":
                        self._narrative_decay_exit = True
                    print(f"[NARRATIVE DECAY] DETECTED — health {self._narrative_peak_health:.2f}→{detect_health:.2f} epoch={self._narrative_epoch_id} (narrative died, {self._narrative_decay_cycles} cycles)", file=sys.stderr)
            else:
                self._narrative_decay_cycles = 0
        elif self._narrative_epoch_id is None and self.executor.positions and not self._narrative_decay_exit:
            detect_health = 0.5
            self._narrative_peak_health = 0.9
            if detect_health < self._narrative_peak_health * DECAY_THRESHOLD:
                self._narrative_decay_cycles += 1
                if self._narrative_decay_cycles >= DEATH_REQUIRED_CYCLES:
                    if os.environ.get("DISABLE_NARRATIVE_DECAY", "0") != "1":
                        self._narrative_decay_exit = True
                    print(f"[NARRATIVE DECAY] STALE POSITIONS ON RESTART — {self._narrative_decay_cycles} cycles without narrative pos={len(self.executor.positions)}", file=sys.stderr)
        if self.executor.positions:
            self._trade_lifecycle.log_cycle(
                self._cycle_count,
                self.executor.positions,
                self._nme_narrative,
                self.graph.strengths_raw(),
                currency_bursts if currency_bursts is not None else (self._currency_bursts or {}),
                self._currency_der or {},
                health.graph_quality,
            )

        nme_snapshot = self.nme.get_state()
        nme_market = {
            "cycle": self._cycle_count,
            "currency_strengths": self.graph.strengths_raw(),
            "currency_bursts": currency_bursts or {},
            "currency_der": self._currency_der or {},
            "graph_quality": health.graph_quality,
            "tick_quality": health.tick_quality,
            "reliability": observability,
        }
        nme_output = self.nme_dashboard.render(nme_snapshot, nme_market)

        forming_returns = None
        pair_agreements = None
        terminal_trends = None
        swing_overlay = None
        bar_state_dict = self.bar_state.get_state()
        if self.bar_state._preloaded:
            forming_returns = {s: self.bar_state.forming_return(s) for s in self._available_symbols}
            # ── SWING OVERLAY (M5 bar swing stats per symbol for display) ──
            swing_overlay = {}
            for sym in self._available_symbols:
                if sym not in BASE_CURRENCY_MAP:
                    continue
                stats = self.bar_state.get_swing_stats(sym)
                form_ret = self.bar_state.forming_return_from_open(sym)
                if stats is None:
                    continue
                base, quote = BASE_CURRENCY_MAP[sym]
                base_s = bar_state_dict.get(base, {}).get("current", 0)
                quote_s = bar_state_dict.get(quote, {}).get("current", 0)
                
                open_price = self.bar_state._forming_open.get(sym)
                if open_price is None or open_price <= 0:
                    continue
                
                current_price = 0.0
                for p in self.executor.positions:
                    if p.symbol == sym:
                        current_price = p.current_price or p.entry_price
                        break
                if current_price == 0.0:
                    current_price = getattr(self.executor, "_last_prices", {}).get(sym, 0.0)
                if current_price == 0.0:
                    current_price = self.mt5._last_bar_close.get(sym, 0.0)
                if current_price == 0.0:
                    current_price = open_price
                
                # Downsides/upsides in price units
                avg_dn = stats["avg_downside"]
                avg_up = stats["avg_upside"]
                form_price_diff = current_price - open_price
                
                # Pip size (handles JPY and broker suffixes)
                is_jpy = "JPY" in sym
                pip_size = 0.01 if is_jpy else 0.0001
                
                avg_dn_pips = round(avg_dn / pip_size, 1)
                avg_up_pips = round(avg_up / pip_size, 1)
                form_pips = round(form_price_diff / pip_size, 1)
                
                # Remaining downside / upside to stop in price units
                rem_dn_price = max(0.0, current_price - (open_price + avg_dn))
                rem_up_price = max(0.0, (open_price + avg_up) - current_price)
                
                # Extract quote currency and broker suffix (e.g. .m)
                quote_ccy = sym[3:6]
                suffix = sym[6:] if len(sym) > 6 else ""
                
                last_prices = {}
                last_prices.update(self.mt5._last_bar_close)
                last_prices.update(getattr(self.executor, "_last_prices", {}))
                
                usd_quote_rate = 1.0
                if quote_ccy != "USD":
                    pair1 = f"USD{quote_ccy}{suffix}"
                    if pair1 in last_prices:
                        usd_quote_rate = last_prices[pair1]
                    else:
                        pair2 = f"{quote_ccy}USD{suffix}"
                        if pair2 in last_prices:
                            rate = last_prices[pair2]
                            usd_quote_rate = 1.0 / rate if rate > 0 else 1.0
                
                # Calculations in price units
                buy_sl = open_price + avg_dn
                buy_tp = current_price + rem_up_price * 0.80
                buy_sl_usd = (rem_dn_price * LOT_SIZE * 100000) / usd_quote_rate
                buy_tp_usd = ((buy_tp - current_price) * LOT_SIZE * 100000) / usd_quote_rate if buy_tp > current_price else 0.0
                
                sell_sl = open_price + avg_up
                sell_tp = current_price - rem_dn_price * 0.80
                sell_sl_usd = (rem_up_price * LOT_SIZE * 100000) / usd_quote_rate
                sell_tp_usd = ((current_price - sell_tp) * LOT_SIZE * 100000) / usd_quote_rate if sell_tp < current_price else 0.0
                
                # Get historical trade record
                hist = self._symbol_trade_history.get(sym)
                hist_data = None
                if hist:
                    is_jpy = "JPY" in sym
                    dec = 3 if is_jpy else 5
                    if hist["status"] == "RUNNING":
                        pos = next((p for p in self.executor.positions if p.symbol == sym), None)
                        current_pnl = pos.pnl if pos else 0.0
                        hist["peak_pnl"] = max(hist.get("peak_pnl", 0.0), current_pnl)
                        
                        hist_data = {
                            "status": "RUNNING",
                            "direction": hist["direction"],
                            "entry_price": f"{hist['entry_price']:.{dec}f}",
                            "peak_pnl": round(hist["peak_pnl"], 2),
                        }
                    else:
                        hist_data = {
                            "status": "CLOSED",
                            "direction": hist["direction"],
                            "exit_price": f"{hist['exit_price']:.{dec}f}",
                            "pnl": round(hist.get("pnl", 0.0), 2),
                            "reason": "SL" if "STOP_LOSS" in hist["reason"] else ("TP" if "TAKE_PROFIT" in hist["reason"] else hist["reason"]),
                        }
                
                # Per-position swing reach (how far through the expected swing has this trade moved)
                swing_reach = None
                open_pos = next((p for p in self.executor.positions if p.symbol == sym), None)
                if open_pos:
                    entry_p = open_pos.entry_price
                    curr_p = open_pos.current_price or entry_p
                    if open_pos.direction == "BUY":
                        price_moved = curr_p - entry_p
                        expected_range = avg_up  # in price units
                    else:
                        price_moved = entry_p - curr_p
                        expected_range = abs(avg_dn)  # avg_dn is negative
                    reach_pct = (price_moved / expected_range * 100) if expected_range > 0 else 0.0
                    moved_pips = round(price_moved / pip_size, 1)
                    swing_reach = {
                        "direction": open_pos.direction,
                        "pct": round(reach_pct, 1),
                        "moved_pips": moved_pips,
                        "expected_pips": round(expected_range / pip_size, 1),
                        "pnl": round(open_pos.pnl or 0.0, 2),
                    }

                ssp_data = self.bar_state.get_structural_swing_position(sym, current_price)
                msp_data = self.bar_state.get_micro_swing_positions(sym, avg_up, avg_dn)
                swing_analysis_data = None
                if ssp_data is not None and msp_data is not None:
                    buy_cls = self.bar_state.classify_swing_state(1.0, ssp_data, msp_data)
                    sell_cls = self.bar_state.classify_swing_state(-1.0, ssp_data, msp_data)
                    swing_analysis_data = {
                        "buy": {
                            "state": buy_cls.get("swing_state", "--"),
                            "ssp": ssp_data.get("buy_ssp"),
                            "msp": msp_data.get("buy_msp"),
                        },
                        "sell": {
                            "state": sell_cls.get("swing_state", "--"),
                            "ssp": ssp_data.get("sell_ssp"),
                            "msp": msp_data.get("sell_msp"),
                        },
                        "position_state": buy_cls.get("position_state", "--"),
                        "range_price": round((avg_up - avg_dn) / pip_size, 1),
                        "range_expansion": ssp_data.get("range_expansion", 1.0),
                        "vol_expansion": ssp_data.get("vol_expansion", 1.0),
                    }

                swing_overlay[sym] = {
                    "avg_down": avg_dn_pips,
                    "avg_up": avg_up_pips,
                    "forming_pips": form_pips,
                    "buy_tp": round(buy_tp, 5),
                    "buy_tp_usd": round(buy_tp_usd, 2),
                    "buy_sl": round(buy_sl, 5),
                    "buy_sl_usd": round(buy_sl_usd, 2),
                    "sell_tp": round(sell_tp, 5),
                    "sell_tp_usd": round(sell_tp_usd, 2),
                    "sell_sl": round(sell_sl, 5),
                    "sell_sl_usd": round(sell_sl_usd, 2),
                    "base_str": round(base_s, 6),
                    "quote_str": round(quote_s, 6),
                    "history": hist_data,
                    "swing_reach": swing_reach,
                    "swing_analysis": swing_analysis_data,
                }
            if bar_state_dict:
                scores = []
                for sym, (base, quote) in BASE_CURRENCY_MAP.items():
                    tick_v = self.graph.strength(base, raw=True) - self.graph.strength(quote, raw=True)
                    tick_dir = 1 if tick_v > 1e-10 else (-1 if tick_v < -1e-10 else 0)
                    bb = bar_state_dict.get(base, {}).get("direction", 0)
                    bq = bar_state_dict.get(quote, {}).get("direction", 0)
                    bar_dir = 1 if bb > 0 and bq <= 0 else (-1 if bb < 0 and bq >= 0 else 0)
                    scores.append((abs(tick_v), sym, tick_dir, bar_dir))
                scores.sort(reverse=True)
                pair_agreements = [(s, td, bd) for _, s, td, bd in scores]

                # Compute terminal trends: last 5 ticks vs last 5 bars per pair
                tick_history = list(self.graph._strength_history)
                bar_history = self.bar_state.get_strength_history()
                terminal_trends = []
                max_spreads = []
                for _, sym, td, bd in scores[:10]:
                    base, quote = BASE_CURRENCY_MAP[sym]
                    tick_seq = []
                    for snap in tick_history[-5:]:
                        spread = snap.get(base, 0) - snap.get(quote, 0)
                        tick_seq.append(1 if spread > 1e-10 else (-1 if spread < -1e-10 else 0))
                    tick_spread_val = self.graph.strength(base, raw=True) - self.graph.strength(quote, raw=True)
                    bar_seq = []
                    bar_base = bar_history.get(base, [])
                    bar_quote = bar_history.get(quote, [])
                    n = min(len(bar_base), len(bar_quote))
                    for i in range(n):
                        spread = bar_base[i] - bar_quote[i]
                        bar_seq.append(1 if spread > 1e-10 else (-1 if spread < -1e-10 else 0))
                    bar_spread_val = (bar_base[-1] if bar_base else 0) - (bar_quote[-1] if bar_quote else 0)
                    terminal_trends.append((sym, tick_seq, bar_seq, td, tick_spread_val, bar_spread_val))
                    max_spreads.extend([abs(tick_spread_val), abs(bar_spread_val)])
                max_mag = max(max_spreads) if max_spreads else 1e-12
        bar_output = self.bar_dashboard.render(
            bar_state=bar_state_dict,
            bar_summary=self.bar_state.get_summary(),
            currency_strengths=self.graph.strengths_raw(),
            strength_persistence=strength_persistence,
            cycle_count=self._cycle_count,
            forming_returns=forming_returns,
            pair_agreements=pair_agreements,
            terminal_trends=terminal_trends,
            max_mag=max_mag,
        )

        self.dashboard.render(
            mode=EXECUTION_MODE.upper(),
            health=health,
            currency_strengths=self.graph.strengths_raw(),
            positions=self.executor.positions_summary(),
            top_hypothesis=top_h,
            pnl=total_pnl,
            trade_count=self._cycle_count,
            factor_exposure=factor_exposure,
            observability=observability,
            z_scores=z_scores,
            stability=stability,
            edge_balance=edge_balance,
            missing_symbols=missing_syms,
            missing_impact=missing_impact,
            health_report=health_report,
            concentration=concentration,
            production_ready=self._production_ready,
            stress_test=stress_test,
            recent_failures=recent_fails,
            exec_fail=self.last_exec_fail,
            unavailable_symbols=[e["symbol"] for e in self._excluded_symbols],
            pipeline_metrics=self._pipeline_metrics,
            total_lots=self.executor.total_lots(),
            max_total_lots=MAX_TOTAL_LOTS,
            lot_size=LOT_SIZE,
            profit_target=PROFIT_TARGET,
            stop_loss_amount=STOP_LOSS_AMOUNT,
            cooldown_active=self.risk.cooldown_active(),
            cooldown_remaining=cd_remain,
            currency_bursts=currency_bursts,
            persistence=persistence,
            strength_persistence=strength_persistence,
            currency_der=self._currency_der,
            der_persistence=self._der_persistence,
            top_der_pairs=self._top_der_pairs,
            wls_direct=WLS_DIRECT_MODE,
            top_burst_pairs=self._top_burst_pairs,
            burst_state=self._pipeline_metrics.get("burst_state"),
            bar_state_summary=self.bar_state.get_summary(),
            available_symbols_count=len(self._available_symbols),
            configured_symbols_count=len(self._available_symbols) + len(self._excluded_symbols),
            nme_output=nme_output,
            bar_output=bar_output,
            nme_trade_snapshots=self._nme_trade_snapshots,
            swing_overlay=swing_overlay,
            regime_data=self._regime_data,
        )

    def _process_deferred_closes(self, now: float | None = None) -> None:
        if now is None:
            now = time.time()

        # 1. Process deferred close-all
        if self._deferred_all_reason and self.executor.positions:
            if all(now - p.entry_time >= MIN_TRADE_RUNTIME_SECONDS for p in self.executor.positions):
                reason = self._deferred_all_reason
                self._deferred_all_reason = None
                print(f"[DEFER EXECUTE] close-all now eligible ({reason})", file=sys.stderr)
                self._close_all_positions(reason)
                return

        # 2. Process deferred individual closes
        if self._deferred_closes and self.executor.positions:
            pos_map = {p.id: p for p in self.executor.positions}
            eligible_ids = [pid for pid in self._deferred_closes if pid in pos_map
                            and now - pos_map[pid].entry_time >= MIN_TRADE_RUNTIME_SECONDS]
            if eligible_ids:
                to_close = [pos_map[pid] for pid in eligible_ids]
                reason = self._deferred_closes[eligible_ids[0]]["reason"]
                for pid in eligible_ids:
                    del self._deferred_closes[pid]
                self._close_individual_positions(to_close, reason)

    def _close_all_positions(self, reason: str) -> None:
        now = time.time()
        young = [p for p in self.executor.positions if now - p.entry_time < MIN_TRADE_RUNTIME_SECONDS]
        if young:
            self._deferred_all_reason = reason
            ages = [f"{p.symbol}={now-p.entry_time:.0f}s" for p in young]
            print(f"[DEFER CLOSE ALL] reason={reason} positions < {MIN_TRADE_RUNTIME_SECONDS}s: {', '.join(ages)}", file=sys.stderr)
            return
        held = list(self.executor.positions)
        self._trade_lifecycle.close_batch(reason, held)
        sp = self.graph.get_strength_persistence()
        strengths_now = self.graph.strengths_raw()
        bursts_now = self.burst.get_currency_bursts({}) if self._currency_bursts is None else (self._currency_bursts or {})
        results = self.executor.close_all({}, reason)
        ok = sum(1 for r in results if r.success)
        total_pnl = sum(p.pnl or 0 for p in held)
        print(f"[CLOSE ALL] reason={reason} {ok}/{len(results)} ok, total_pnl={total_pnl:.2f}", file=sys.stderr)
        for pos, r in zip(held, results):
            traj = self._position_trajectory.pop(pos.id, [])
            if traj:
                entry = traj[0]
                final = traj[-1]
                peak_pnl = max(t.get("pnl", 0) for t in traj)
                peak_h = max(t["health"] for t in traj)
                peak_nmi = max(t["nmi"] for t in traj)
                pnl_at_peak_h = max((t.get("pnl", 0) for t in traj if t["health"] == peak_h), default=0)
                phases = list(dict.fromkeys(t["phase"] for t in traj))
                print(f"[NARRATIVE TRAJECTORY] {pos.symbol} {pos.direction} "
                      f"entry: h={entry['health']} n={entry['nmi']} p={entry['phase']} pnl=${entry.get('pnl',0):.2f} "
                      f"peak_pnl=${peak_pnl:.2f} peak_h={peak_h:.2f} "
                      f"exit: h={final['health']} n={final['nmi']} p={final['phase']} pnl=${final.get('pnl',0):.2f} "
                      f"phases={'→'.join(phases)} cycles={len(traj)} ({reason})", file=sys.stderr)
            if not r.success:
                print(f"[CLOSE ALL]   close fail: id={r.position_id} reason={r.reason}", file=sys.stderr)
            self.journal.record_close(
                position_id=pos.id,
                exit_price=r.price,
                pnl=pos.pnl or 0,
                reason=reason,
                strengths=strengths_now,
                peaks={c: v["peak"] for c, v in sp.items()},
                troughs={c: v["trough"] for c, v in sp.items()},
                streaks={c: v["streak"] for c, v in sp.items()},
                bursts=bursts_now,
            )
            self._symbol_trade_history[pos.symbol] = {
                "status": "CLOSED",
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "exit_price": r.price,
                "pnl": pos.pnl or 0,
                "reason": reason,
                "ts": time.time(),
            }
            self.drs.remove_position(pos.symbol)
        self.executor.sync()
        self.risk.set_positions(self.executor.positions)

    def _close_individual_positions(self, positions: list, reason: str) -> None:
        now = time.time()
        eligible = []
        for pos in positions:
            age = now - pos.entry_time
            if age < MIN_TRADE_RUNTIME_SECONDS:
                self._deferred_closes[pos.id] = {"reason": reason, "request_time": now}
                print(f"[DEFER CLOSE] {pos.symbol} {pos.direction} age={age:.0f}s ← {MIN_TRADE_RUNTIME_SECONDS}s — deferred ({reason})", file=sys.stderr)
            else:
                eligible.append(pos)
        if not eligible:
            return
        sp = self.graph.get_strength_persistence()
        strengths_now = self.graph.strengths_raw()
        bursts_now = self.burst.get_currency_bursts({}) if self._currency_bursts is None else (self._currency_bursts or {})
        for pos in eligible:
            traj = self._position_trajectory.pop(pos.id, [])
            if traj:
                entry = traj[0]
                final = traj[-1]
                peak_pnl = max(t.get("pnl", 0) for t in traj)
                peak_h = max(t["health"] for t in traj)
                peak_nmi = max(t["nmi"] for t in traj)
                phases = list(dict.fromkeys(t["phase"] for t in traj))
                print(f"[NARRATIVE TRAJECTORY] {pos.symbol} {pos.direction} "
                      f"entry: h={entry['health']} n={entry['nmi']} p={entry['phase']} pnl=${entry.get('pnl',0):.2f} "
                      f"peak_pnl=${peak_pnl:.2f} peak_h={peak_h:.2f} "
                      f"exit: h={final['health']} n={final['nmi']} p={final['phase']} pnl=${final.get('pnl',0):.2f} "
                      f"phases={'→'.join(phases)} cycles={len(traj)} ({reason})", file=sys.stderr)
            r = self.executor.close_position(pos.id, pos.current_price, reason)
            if r and r.success:
                self.journal.record_close(
                    position_id=pos.id,
                    exit_price=r.price,
                    pnl=pos.pnl or 0,
                    reason=reason,
                    strengths=strengths_now,
                    peaks={c: v["peak"] for c, v in sp.items()},
                    troughs={c: v["trough"] for c, v in sp.items()},
                    streaks={c: v["streak"] for c, v in sp.items()},
                    bursts=bursts_now,
                )
                self._symbol_trade_history[pos.symbol] = {
                    "status": "CLOSED",
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "exit_price": r.price,
                    "pnl": pos.pnl or 0,
                    "reason": reason,
                    "ts": time.time(),
                }
                self.drs.remove_position(pos.symbol)
        self.executor.sync()
        self.risk.set_positions(self.executor.positions)

    def _reset_after_profit_target(self) -> None:
        self.graph.reset()
        self.store.clear()
        self.burst.reset()
        self.efficiency.reset()
        self.bar_state.reset()
        self._top_burst_pairs = []
        self._currency_bursts = None
        self._persistence = None
        self._currency_der = None
        self._der_persistence = None
        self._top_der_pairs = []
        self._cycle_count = 0
        self._pipeline_metrics = {k: 0 for k in self._pipeline_metrics}
        self.drs = None
        from portfolio.drs import DRS
        self.drs = DRS()
        self.risk.reset_state()
        self.nme = NarrativeEngine()
        self._nme_narrative = None
        self._pnl_reset_done = False
        self._nme_trade_snapshots.clear()
        self._position_trajectory.clear()
        self._narrative_epoch_id = None
        self._narrative_peak_health = 0.0
        self._narrative_decay_exit = False
        self._narrative_decay_cycles = 0
        self._last_batch_pnl = 0.0
        self._batch_open_cycle = 0
        self._trade_lifecycle.discard()
        print("[HARD RESET] WLS state, burst data, tick store, DRS, risk, and NME cleared", file=sys.stderr)

    def _update_production_readiness(self, health_report: dict, max_concentration: float) -> None:
        if WLS_DIRECT_MODE:
            self._production_ready = True
            return
        if self.executor.sync_failed:
            self._production_ready = False
            return
        if health_report.get("confidence_level") == "LOW":
            self._production_ready = False
            return
        if health_report.get("connectivity", 0) < 0.45:
            self._production_ready = False
            return
        if max_concentration > 0.60:
            self._production_ready = False
            return
        if self._cycle_count >= 1 and self._mt5_audit is None:
            try:
                self._mt5_audit = self.mt5.audit_symbols()
            except Exception:
                self._mt5_audit = {"missing": []}
        if self._mt5_audit is not None and len(self._mt5_audit.get("missing", [])) > 10:
            pass
        self._production_ready = True

    def _setup_signal_handlers(self) -> None:
        def handler(sig, frame):
            if self._force_exit:
                import os
                os._exit(1)
            self._force_exit = True
            print("\nShutting down gracefully... (Ctrl+C again for force kill)")
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, handler)

    def _check_stop_file(self) -> None:
        stop_path = Path("STOP")
        if stop_path.exists():
            stop_path.unlink()
