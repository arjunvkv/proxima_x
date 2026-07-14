import os
import time
import signal
import threading
import sys
from queue import Queue, Empty
from pathlib import Path
from typing import Optional

from config.settings import EXECUTION_MODE, MAX_POSITIONS, LOT_SIZE, MAX_TOTAL_LOTS, PROFIT_TARGET, STOP_LOSS_AMOUNT, BURST_TOP_N, MIN_CONFIDENCE
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
            to_close_indiv_pt = self.risk.check_individual_profit_target(self.executor.positions)
            if to_close_indiv_pt:
                pnls = [f"{p.symbol}=${p.pnl:.2f}" for p in to_close_indiv_pt]
                print(f"[PER-POSITION PROFIT TARGET] closing {len(to_close_indiv_pt)}: {', '.join(pnls)}", file=sys.stderr)
                self._close_individual_positions(to_close_indiv_pt, "PROFIT_TARGET")
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
                cycle_submitted = set()
                self.last_exec_fail = None
                for h in candidates:
                    if self.executor.position_count() >= MAX_POSITIONS:
                        break
                    if h.symbol in cycle_submitted:
                        continue
                    # ── DUPLICATE POSITION CHECK ───────────────────
                    dir_label = "BUY" if h.direction > 0 else "SELL"
                    if any(p.symbol == h.symbol and p.direction == dir_label for p in self.executor.positions):
                        print(f"[DUPLICATE BLOCK] {h.symbol} {dir_label} already active", file=sys.stderr)
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
                    result = self.executor.execute(h)
                    if result.success:
                        cycle_submitted.add(h.symbol)
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
                        )
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
                      f"phases={'→'.join(phases)} cycles={len(traj)} (STOP_LOSS)", file=sys.stderr)
            r = self.executor.close_position(pos.id, prices.get(pos.symbol, pos.entry_price), "STOP_LOSS")
            self.drs.remove_position(pos.symbol)
            self.journal.record_close(
                position_id=pos.id,
                exit_price=r.price,
                pnl=pos.pnl or 0,
                reason=r.reason or "STOP_LOSS",
                strengths=strengths_now,
                peaks={c: v["peak"] for c, v in sp.items()},
                troughs={c: v["trough"] for c, v in sp.items()},
                streaks={c: v["streak"] for c, v in sp.items()},
                bursts=bursts_now,
            )

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
        bar_state_dict = self.bar_state.get_state()
        if self.bar_state._preloaded:
            forming_returns = {s: self.bar_state.forming_return(s) for s in self._available_symbols}
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
        )

    def _close_all_positions(self, reason: str) -> None:
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
            self.drs.remove_position(pos.symbol)
        self.executor.sync()
        self.risk.set_positions(self.executor.positions)

    def _close_individual_positions(self, positions: list, reason: str) -> None:
        sp = self.graph.get_strength_persistence()
        strengths_now = self.graph.strengths_raw()
        bursts_now = self.burst.get_currency_bursts({}) if self._currency_bursts is None else (self._currency_bursts or {})
        for pos in positions:
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
