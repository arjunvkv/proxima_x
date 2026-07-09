import time
import signal
import threading
import sys
from collections import deque
from queue import Queue, Empty
from pathlib import Path
from typing import Optional

from config.settings import EXECUTION_MODE, MAX_POSITIONS, LOT_SIZE, MAX_TOTAL_LOTS, PROFIT_TARGET
from config.settings import SWPS_MIN_SCORE, SWPS_WINDOW_SIZE
from config.settings import CURRENCY_LIST, BASE_CURRENCY_MAP, WLS_DIRECT_MODE
from data.models import TickBatch
from data.mt5_adapter import MT5Adapter
from data.tick_store import TickStore
from currency.graph import CurrencyGraph
from direction.hypothesis import HypothesisGenerator
from direction.short_window_persistence import ShortWindowPersistenceScanner
from portfolio.drs import DRS
from portfolio.concentration import CurrencyConcentration
from risk.safety import RiskEngine
from execution.paper import PaperExecutor
from execution.mt5_executor import MT5Executor
from persistence.snapshot import SnapshotManager
from monitoring.dashboard import Dashboard
from .health import HealthMonitor
from features.participation_burst import ParticipationBurstEngine

class RuntimeManager:
    def __init__(self):
        self.running = False
        self.shutdown_event = threading.Event()
        self._force_exit = False

        self.mt5 = MT5Adapter()
        self.store = TickStore()
        self.burst = ParticipationBurstEngine()
        self.graph = CurrencyGraph()
        self.generator = HypothesisGenerator()
        self.drs = DRS()
        self.swps = ShortWindowPersistenceScanner(
            currencies=CURRENCY_LIST,
            pairs=BASE_CURRENCY_MAP
        )
        self._strength_capture = deque(maxlen=SWPS_WINDOW_SIZE)
        self.risk = RiskEngine()
        if EXECUTION_MODE == "live":
            self.executor = MT5Executor(self.mt5)
        else:
            self.executor = PaperExecutor()
        self.snapshot = SnapshotManager()
        self.dashboard = Dashboard()
        self.health = HealthMonitor()

        self._tick_queue = Queue(maxsize=1000)
        self._tick_thread: Optional[threading.Thread] = None
        self._decision_thread: Optional[threading.Thread] = None

        self._cycle_count = 0
        self._last_decision = 0.0
        self._last_snapshot = 0.0
        self._start_time = 0.0
        self._mt5_audit = None
        self._production_ready = False
        self._pipeline_metrics = {"generated": 0, "ranked": 0, "selected": 0, "risk_approved": 0, "executed": 0}
        self._available_symbols: list[str] = []
        self._excluded_symbols: list[dict] = []
        self._last_discovery = 0.0
        self.last_exec_fail: Optional[str] = None
        self._swps_pick: Optional[tuple] = None
        self._top_burst_pairs: list[str] = []

    def start(self) -> None:
        self._setup_signal_handlers()
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

        if self._tick_thread and self._tick_thread.is_alive():
            self._tick_thread.join(timeout=3.0)
        if self._decision_thread and self._decision_thread.is_alive():
            self._decision_thread.join(timeout=5.0)

        state = {
            "market_timestamp": time.time(),
            "currency_strengths": self.graph.strengths(),
            "graph_quality": self.graph.quality(),
            "positions": [(p.id, p.symbol, p.direction, p.entry_price) for p in self.executor.positions],
            "trade_count": len(self.risk.trades),
            "uptime": time.time() - self._start_time
        }
        if self.snapshot.save(state):
            print("Snapshot saved.")
        else:
            print("WARNING: Snapshot save failed.")

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
                continue
            except Exception:
                continue

    def _process_batches(self, batches: list[TickBatch]) -> None:
        now = time.time()

        for batch in batches:
            self.store.add_ticks(batch.ticks)
            for tick in batch.ticks:
                self.burst.update(tick.symbol, tick.volume)
            self.executor.update_prices(batch.ticks)

        if self.risk.check_profit_target(self.executor.positions):
            self._close_all_positions("PROFIT_TARGET")
            self.dashboard.latest_event = {"event": "PROFIT_TARGET", "time": now}

        returns = self.store.calculate_returns()
        if self._available_symbols:
            returns = {s: v for s, v in returns.items() if s in self._available_symbols}

        if now - self._last_decision >= 5.0:
            solve_start = time.time()
            freshness_weights = {sym: self.store.freshness(sym) for sym in returns}
            topology_weights = self.graph.topology.pair_weights([s for s, v in returns.items() if v != 0.0])
            burst_weights = self.burst.get_paired_weights(list(returns.keys()))
            weights = {sym: freshness_weights.get(sym, 0) * topology_weights.get(sym, 0) * burst_weights.get(sym, 1.0) for sym in returns}
            self.graph.update(returns, weights, now)
            self._strength_capture.append(self.graph.strengths())
            solve_ms = (time.time() - solve_start) * 1000
            self.health.record_solve(solve_ms)

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
            self._top_burst_pairs = self.burst.get_top_burst_pairs(3)
            if self._top_burst_pairs:
                burst_filtered = [h for h in hypotheses if h.symbol in self._top_burst_pairs]
                if burst_filtered:
                    hypotheses = burst_filtered
            self._pipeline_metrics["burst_hyp"] = len(hypotheses)

            self.executor.sync()

            if self.executor.sync_failed:
                import sys
                print("[EXECUTION BLOCKED] Position state unknown — sync failed", file=sys.stderr)
                return

            self.risk.set_positions(self.executor.positions)
            self.drs.set_positions(self.executor.positions)

            import sys
            _a = self.graph.execution_allowed(returns)
            _c = self.graph.connectivity_score(returns)
            _ap = self.graph._active_pair_count
            _q = self.graph.state.quality
            print(f"[GATE] allowed={_a} prod={self._production_ready} hyp={len(hypotheses)} ap={_ap} q={_q:.3f} conn={_c:.3f} direct={WLS_DIRECT_MODE}", file=sys.stderr)
            gate_pass = WLS_DIRECT_MODE or (_a and self._production_ready)
            if gate_pass:
                self._swps_pick = None

                if not WLS_DIRECT_MODE and len(self._strength_capture) >= SWPS_WINDOW_SIZE:
                    swps_ranked = self.swps.rank(list(self._strength_capture))

                    for candidate_symbol, candidate_direction, candidate_score in swps_ranked:
                        print(
                            f"[SWPS CANDIDATE] {candidate_symbol} "
                            f"{candidate_direction} score={candidate_score:.4f}",
                            file=sys.stderr
                        )

                        if candidate_score < SWPS_MIN_SCORE:
                            continue

                        if candidate_symbol not in self._available_symbols:
                            print(
                                f"[SWPS SKIP] {candidate_symbol} unavailable in MT5",
                                file=sys.stderr
                            )
                            continue

                        self._swps_pick = (
                            candidate_symbol,
                            candidate_direction,
                            candidate_score
                        )
                        break

                if self._swps_pick:
                    symbol, direction, score = self._swps_pick
                    print(
                        f"[SWPS SELECTED] {symbol} {direction} score={score:.4f}",
                        file=sys.stderr
                    )

                    matching = [
                        h for h in hypotheses
                        if h.symbol == symbol
                    ]

                    if not matching:
                        generated = self.generator.generate_all(self.graph, now)
                        matching = [
                            h for h in generated
                            if h.symbol == symbol
                        ]

                    if matching:
                        h = matching[0]
                        h.direction = 1.0 if direction == "BUY" else -1.0
                        h.confidence = min(h.confidence * (1.0 + score), 1.0)
                        hypotheses = [h]
                        print(
                            f"[SWPS OVERRIDE] {symbol} {direction} score={score:.4f}",
                            file=sys.stderr
                        )
                    else:
                        print(
                            f"[SWPS ABORT] {symbol} no executable hypothesis",
                            file=sys.stderr
                        )
                        hypotheses = []

                ranked = self.drs.rank(hypotheses)
                self._pipeline_metrics["ranked"] = len(ranked)
                pos_count = self.executor.position_count()
                open_count = pos_count
                print(f"[POSITION STATE] executor={len(self.executor.positions)} count={pos_count}", file=sys.stderr)
                if open_count >= MAX_POSITIONS:
                    selected = []
                    print(
                        f"[DRS SELECT] skipped — open_count={open_count} >= MAX={MAX_POSITIONS}",
                        file=sys.stderr
                    )

                elif self.risk.cooldown_active():
                    selected = []
                    remain = int(self.risk._profit_cooldown_until - now)
                    print(f"[DRS SELECT] skipped — profit cooldown {remain}s remaining", file=sys.stderr)
                else:
                    if (
                        self._swps_pick
                        and len(ranked) == 1
                        and open_count < MAX_POSITIONS
                    ):
                        selected = [ranked[0]]
                        print(
                            f"[SWPS DIRECT SELECT] "
                            f"{ranked[0].symbol} "
                            f"drs={ranked[0].drs_score:.3f}",
                            file=sys.stderr
                        )
                    else:
                        selected = self.drs.select(ranked, open_count)
                        print(f"[DRS SELECT] {self.drs.last_selection_trace}", file=sys.stderr)
                        replacements = self.drs.replacement_candidates(ranked)
                        if replacements:
                            print(f"[DRS REPLACE] candidates: {replacements}", file=sys.stderr)

                self._pipeline_metrics["selected"] = len(selected)
                cycle_submitted = set()
                self.last_exec_fail = None
                for h in selected:
                    if self.executor.position_count() >= MAX_POSITIONS:
                        break
                    if h.symbol in cycle_submitted:
                        continue
                    approved = self.risk.approve(h)
                    if not approved:
                        print(f"[RISK BLOCK] {h.symbol} drs={h.drs_score:.3f}", file=sys.stderr)
                        continue
                    self._pipeline_metrics["risk_approved"] += 1
                    result = self.executor.execute(h)
                    if result.success:
                        cycle_submitted.add(h.symbol)
                        self._pipeline_metrics["executed"] += 1
                        self.drs.record_position(
                            next(p for p in self.executor.positions if p.id == result.position_id)
                        )
                        self.risk.set_positions(self.executor.positions)
                    else:
                        import sys
                        self.last_exec_fail = f"{h.symbol} {result.reason}"
                        print(f"[EXEC FAIL] {h.symbol} {result.reason}", file=sys.stderr)

        prices = {p.symbol: p.current_price for p in self.executor.positions}
        to_close = self.risk.check_stops(prices)
        for pos in to_close:
            self.executor.close_position(pos.id, prices.get(pos.symbol, pos.entry_price), "STOP_LOSS")
            self.drs.remove_position(pos.symbol)

        if now - self._last_snapshot >= 300.0:
            self._last_snapshot = now
            state = {
                "market_timestamp": now,
                "currency_strengths": self.graph.strengths(),
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
        currency_bursts = self.burst.get_currency_bursts()
        persistence = self.burst.get_persistence()
        strength_persistence = self.graph.get_strength_persistence()
        recent_fails = self.executor.recent_failures()

        self._update_production_readiness(health_report, max_conc)

        self._cycle_count += 1
        cd_remain = max(0, int(self.risk._profit_cooldown_until - now)) if self.risk.cooldown_active() else 0

        swps_signal = None
        if self._swps_pick:
            sym, d, sc = self._swps_pick
            swps_signal = {"symbol": sym, "direction": d, "score": sc}

        self.dashboard.render(
            mode=EXECUTION_MODE.upper(),
            health=health,
            currency_strengths=self.graph.strengths(),
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
            swps_signal=swps_signal,
            swps_capture_count=len(self._strength_capture),
            currency_bursts=currency_bursts,
            persistence=persistence,
            strength_persistence=strength_persistence,
            wls_direct=WLS_DIRECT_MODE,
            top_burst_pairs=self._top_burst_pairs,
        )

    def _close_all_positions(self, reason: str) -> None:
        held_symbols = [(p.symbol, p.id) for p in self.executor.positions]
        results = self.executor.close_all({}, reason)
        ok = sum(1 for r in results if r.success)
        total_pnl = sum(p.pnl or 0 for p in self.executor.positions)
        import sys
        print(f"[PROFIT TARGET] close_all: {ok}/{len(results)} ok, total_pnl={total_pnl:.2f}", file=sys.stderr)
        for r in results:
            if not r.success:
                print(f"[PROFIT TARGET]   close fail: id={r.position_id} reason={r.reason}", file=sys.stderr)
        for sym, pid in held_symbols:
            self.drs.remove_position(sym)
        self.executor.sync()
        self.risk.set_positions(self.executor.positions)

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

