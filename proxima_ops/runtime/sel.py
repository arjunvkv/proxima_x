"""
SEL — Sovereign Execution Loop

Continuous execution loop driven by market ticks.
Pipeline: SES → ECL → EFK — real-time sovereign order flow.

process_tick()
  1. EDEK.on_tick() processes tick event
  2. If EDEK says triggered_evaluation → call full decision pipeline
  3. Return result

process_cycle()
  1. ETD.evaluate() — check trigger conditions
  2. If triggered: SES.evaluate() — final sovereign decision
  3. If SES says emit: ECL.commit() — lock execution window
  4. If lock acquired: EFK.finalize() — emit MT5 order
  5. SFPF.persist() — save state
  6. Return full pipeline trace
"""

import time
from datetime import datetime, timezone
from typing import Optional


class SovereignExecutionLoop:
    """Continuous execution loop driven by market ticks.

    Parameters
    ----------
    ses : SingleExecutionSovereign
        Sovereign authority for MT5 order emission.
    ecl : ExecutionCommitmentLock
        Non-interruptible execution window lock.
    efk : ExecutionFinalityKernel
        Final pipeline stage converting decision → MT5 order.
    etd : object
        Event Trigger Detector — exposes ``evaluate(signal, tick) -> dict``
        with key ``"triggered"`` (bool).
    edek : EventDrivenExecutionKernel
        Tick event processor — exposes ``on_tick(symbol, tick) -> dict``
        with key ``"triggered_evaluation"`` (bool).
    sfpf : StateFreezingFix
        State persister — exposes ``persist(ses_state, ecl_state, efk_state) -> bool``.
    mrbl : object or None
        Market Replay Bridge Layer — optional connector with ``connect()``
        and ``subscribe(symbols)``.
    symbol_universe : list of str, optional
        Symbols to subscribe on start (default empty).
    """

    def __init__(
        self,
        ses,
        ecl,
        efk,
        etd,
        edek,
        sfpf,
        mrbl,
        symbol_universe: Optional[list] = None,
    ) -> None:
        self._ses = ses
        self._ecl = ecl
        self._efk = efk
        self._etd = etd
        self._edek = edek
        self._sfpf = sfpf
        self._mrbl = mrbl
        self._symbol_universe = symbol_universe or []

        # runtime state
        self._running: bool = False
        self._total_ticks_processed: int = 0
        self._total_cycles_processed: int = 0
        self._total_orders_emitted: int = 0
        self._last_tick_time: Optional[str] = None
        self._last_order_time: Optional[str] = None
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # process_tick
    # ------------------------------------------------------------------

    def process_tick(
        self,
        symbol: str,
        tick: dict,
        emancipation: dict = None,
    ) -> dict:
        """Process a single market tick.

        1. Delegates to EDEK.on_tick().
        2. If triggered_evaluation is True, runs the full decision pipeline
           via process_cycle() using emancipation inputs when available.

        Parameters
        ----------
        symbol : str
            Market symbol (e.g. "EURUSD").
        tick : dict
            Tick data with at least ``"bid"`` and ``"ask"``.
        emancipation : dict, optional
            Pre-computed emancipation values, with keys matching
            ``process_cycle`` keyword arguments (erf, loef_density,
            dce_action, dce_confidence, tamk_authorized, etc.).
            When None, hardcoded defaults (HOLD, erf=0, etc.) are used.

        Returns
        -------
        dict
            ``{
                "tick_processed": bool,
                "decision_triggered": bool,
                "order_emitted": bool,
                "ticket": int or None,
                "state_before": str,
                "state_after": str,
                "pipeline": dict or None,
            }``
        """
        state_before = "running" if self._running else "stopped"
        result: dict = {
            "tick_processed": False,
            "decision_triggered": False,
            "order_emitted": False,
            "ticket": None,
            "state_before": state_before,
            "state_after": state_before,
            "pipeline": None,
            "rejection_reason": None,
        }

        try:
            if not self._running:
                return result

            # 1. EDEK.on_tick()
            edek_result = self._edek.on_tick(symbol, tick)
            event_processed = bool(edek_result.get("event_processed", False))
            triggered = bool(edek_result.get("triggered_evaluation", False))

            result["tick_processed"] = event_processed
            self._total_ticks_processed += 1 if event_processed else 0

            if event_processed:
                self._last_tick_time = datetime.now(timezone.utc).isoformat()

            # 2. If triggered, run full decision pipeline
            if triggered:
                result["decision_triggered"] = True

                # Use emancipation inputs when available, else hardcoded defaults
                em = emancipation or {}
                cycle_result = self.process_cycle(
                    signal=em.get("signal", {"symbol": symbol, "direction": "HOLD"}),
                    mt5_tick=tick,
                    mt5_account=em.get("mt5_account", {}),
                    open_positions=em.get("open_positions", []),
                    sil_scores=em.get("sil_scores", {}),
                    rsi_dict=em.get("rsi_dict", {}),
                    activation=em.get("activation", {}),
                    readiness=em.get("readiness", {}),
                    governor_state=em.get("governor_state", "IDLE"),
                    cb_triggered=em.get("cb_triggered", False),
                    cb_latch_cycles=em.get("cb_latch_cycles", 0),
                    confirm_counts=em.get("confirm_counts", {}),
                    signals=em.get("signals", []),
                    erf=em.get("erf", 0.0),
                    loef_density=em.get("loef_density", 0.0),
                    gmci_score=em.get("gmci_score", 0.0),
                    aeem_escape=em.get("aeem_escape", 0.0),
                    rfg=em.get("rfg", 0.0),
                    eprg_reachability=em.get("eprg_reachability", 0.0),
                    era_result=em.get("era_result"),
                    tamk_result=em.get("tamk_result"),
                )

                result["order_emitted"] = bool(cycle_result.get("order_emitted", False))
                result["ticket"] = cycle_result.get("ticket")
                result["pipeline"] = cycle_result.get("pipeline")
                result["rejection_reason"] = cycle_result.get("rejection_reason")
                self._edek.reset_trigger_state()

            # Determine state_after from EDEK trigger_state
            trigger_state = edek_result.get("trigger_state", "UNKNOWN")
            result["state_after"] = trigger_state

        except Exception:
            result["tick_processed"] = False
            result["state_after"] = "ERROR"

        return result

    # ------------------------------------------------------------------
    # process_cycle
    # ------------------------------------------------------------------

    def process_cycle(
        self,
        signal: dict,
        mt5_tick: dict,
        mt5_account: dict,
        open_positions: list,
        sil_scores: dict,
        rsi_dict: dict,
        activation: dict,
        readiness: dict,
        governor_state: str,
        cb_triggered: bool,
        cb_latch_cycles: int,
        confirm_counts: dict,
        signals: list,
        erf: float = 0.0,
        loef_density: float = 0.0,
        gmci_score: float = 0.0,
        aeem_escape: float = 0.0,
        rfg: float = 0.0,
        eprg_reachability: float = 0.0,
        era_result: Optional[dict] = None,
        tamk_result: Optional[dict] = None,
    ) -> dict:
        """Execute one full sovereignty decision cycle.

        Pipeline steps
        -------------
        1. ETD.evaluate()  — check trigger conditions
        2. SES.evaluate()  — final sovereign decision
        3. ECL.commit()    — lock execution window (if SES says emit)
        4. EFK.finalize()  — emit MT5 order (if lock acquired)
        5. SFPF.persist()  — save state for continuity

        Parameters
        ----------
        signal : dict
            Primary signal dict (keys: symbol, direction, confidence, etc.).
        mt5_tick : dict
            Latest MT5 tick (keys: bid, ask, time).
        mt5_account : dict
            MT5 account info.
        open_positions : list
            Currently open positions.
        sil_scores : dict
            Symbol Intelligence Layer scores.
        rsi_dict : dict
            RSI indicator values.
        activation : dict
            Activation/readiness state.
        readiness : dict
            Readiness state.
        governor_state : str
            Current governor state label.
        cb_triggered : bool
            Circuit breaker triggered flag.
        cb_latch_cycles : int
            Circuit breaker latch cycle count.
        confirm_counts : dict
            Confirmation count map.
        signals : list
            Full signal list (may include best_signal).
        erf : float
            Energy Recapture Fraction.
        loef_density : float
            LOEF opportunity density.
        gmci_score : float
            GMCI diagnostic score.
        aeem_escape : float
            AEEM escape velocity.
        rfg : float
            RFG diagnostic value.
        eprg_reachability : float
            EPRG reachability score.

        Returns
        -------
        dict
            ``{
                "cycle_processed": bool,
                "decision": "BUY" | "SELL" | "HOLD",
                "order_emitted": bool,
                "ticket": int or None,
                "pipeline": {
                    "etd_triggered": bool,
                    "ses_authorized": bool,
                    "ecl_locked": bool,
                    "efk_executed": bool
                },
                "error": str or None
            }``
        """
        result: dict = {
            "cycle_processed": False,
            "decision": "HOLD",
            "order_emitted": False,
            "ticket": None,
            "pipeline": {
                "etd_triggered": False,
                "ses_authorized": False,
                "ecl_locked": False,
                "efk_executed": False,
            },
            "rejection_reason": None,
            "error": None,
        }

        try:
            # ---- pipeline trace accumulator ----
            etd_triggered: bool = False
            ses_authorized: bool = False
            ecl_locked: bool = False
            efk_executed: bool = False

            # ---- 1. ETD.evaluate() ----
            etd_result = self._etd.evaluate(
                erf=erf,
                loef_density=loef_density,
                dce_confidence=signal.get("confidence", 0.0),
                dce_action=signal.get("direction", "HOLD"),
                tamk_authorized=not cb_triggered,
            )
            etd_triggered = bool(etd_result.get("trigger", False))

            if not etd_triggered:
                result["cycle_processed"] = True
                result["pipeline"]["etd_triggered"] = False
                self._total_cycles_processed += 1
                return result

            result["pipeline"]["etd_triggered"] = True
            decision = signal.get("direction", "HOLD") or "HOLD"
            result["decision"] = decision

            # ---- 2. SES.evaluate() ----
            final_era_result = era_result if era_result is not None else {
                "valid": True, "reality_alignment_score": 0.8,
                "adjusted_price": mt5_tick.get("ask" if decision == "BUY" else "bid")
                if mt5_tick else None,
                "adjusted_volume": 0.01
            }
            final_tamk_result = tamk_result if tamk_result is not None else {
                "authorized": not cb_triggered,
                "override_active": cb_latch_cycles > 100
            }

            ses_result = self._ses.evaluate(
                decision={"action": decision, "symbol": signal.get("symbol"),
                          "confidence": signal.get("confidence", 0.0),
                          "action_value": signal.get("confidence", 0.0)},
                era_result=final_era_result,
                tamk_result=final_tamk_result,
                loef_result={"opportunity_density": loef_density,
                             "top_k_symbols": [signal.get("symbol")] if signal.get("symbol") else []},
                signal=signal,
                mt5_tick=mt5_tick,
            )
            emit_order = bool(ses_result.get("emit_order", False))
            ses_authorized = emit_order
            result["pipeline"]["ses_authorized"] = emit_order
            result["rejection_reason"] = ses_result.get("rejection_reason")

            if not emit_order:
                result["cycle_processed"] = True
                self._total_cycles_processed += 1
                return result

            # ---- 3. ECL.commit() ----
            order_params = ses_result.get("order_params")
            commit_result = self._ecl.commit(order_params, ses_result)
            ecl_locked = bool(commit_result.get("locked", False))
            result["pipeline"]["ecl_locked"] = ecl_locked

            if not ecl_locked:
                result["cycle_processed"] = True
                self._total_cycles_processed += 1
                return result

            # ---- 4. EFK.finalize() ----
            ecl_state = self._ecl.get_lock_state()
            efk_result = self._efk.finalize(
                ses_result=ses_result,
                ecl_state=ecl_state,
                awns_result={},
                mt5_tick=mt5_tick,
                signal=signal,
            )
            efk_executed = bool(efk_result.get("order_emitted", False))
            result["pipeline"]["efk_executed"] = efk_executed

            if efk_executed:
                # ECL.release() after successful execution
                self._ecl.release()
                ticket = efk_result.get("ticket")
                result["order_emitted"] = True
                result["ticket"] = ticket
                self._total_orders_emitted += 1
            else:
                # Release the lock if execution failed to avoid deadlock
                self._ecl.release()
                self._last_order_time = datetime.now(timezone.utc).isoformat()

            # ---- 5. SFPF.persist() ----
            self._sfpf.persist(
                ses_state=ses_result,
                ecl_state=ecl_state,
                efk_state=efk_result,
            )

            result["cycle_processed"] = True
            self._total_cycles_processed += 1

        except Exception as exc:
            result["cycle_processed"] = False
            result["error"] = str(exc)

        return result

    # ------------------------------------------------------------------
    # start / stop
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the execution loop.

        * Sets internal ``running`` flag to True.
        * If MRBL is available, calls ``connect()`` and ``subscribe()`` with
          the configured symbol universe.
        * Starts EDEK.

        Returns
        -------
        bool
            True if start succeeded.
        """
        try:
            if self._running:
                return True

            self._running = True
            self._start_time = time.time()

            # Start EDEK
            self._edek.start()

            # MRBL connect + subscribe
            if self._mrbl is not None:
                try:
                    self._mrbl.connect()
                    if self._symbol_universe:
                        self._mrbl.subscribe_ticks(self._symbol_universe)
                except Exception:
                    # MRBL failure is non-fatal
                    pass

            return True

        except Exception:
            self._running = False
            return False

    def is_running(self) -> bool:
        try:
            return self._running
        except Exception:
            return False

    def stop(self) -> bool:
        """Stop the execution loop.

        * Clears the ``running`` flag.
        * Stops EDEK.

        Returns
        -------
        bool
            True if stop succeeded.
        """
        try:
            self._running = False
            self._edek.stop()
            return True

        except Exception:
            return False

    # ------------------------------------------------------------------
    # get_state
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return a snapshot of the current loop state.

        Returns
        -------
        dict
            ``{
                "running": bool,
                "total_ticks_processed": int,
                "total_cycles_processed": int,
                "total_orders_emitted": int,
                "last_tick_time": str or None,
                "last_order_time": str or None,
                "uptime_seconds": float
            }``
        """
        uptime = 0.0
        if self._start_time is not None:
            uptime = time.time() - self._start_time

        return {
            "running": self._running,
            "total_ticks_processed": self._total_ticks_processed,
            "total_cycles_processed": self._total_cycles_processed,
            "total_orders_emitted": self._total_orders_emitted,
            "last_tick_time": self._last_tick_time,
            "last_order_time": self._last_order_time,
            "uptime_seconds": uptime,
        }
