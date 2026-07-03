"""
EFK — Execution Finality Kernel

Final pipeline stage converting decision → MT5 order packet.
Zero "pending decision" states guaranteed on exit.

Contract
--------
SES result (ses_result) must carry ``emit_order``.
ECL state    (ecl_state)    must carry ``locked``.
Only when BOTH are True does a real/simulated order get emitted.
In all other cases the pipeline is terminated cleanly with no pending decisions.
"""

import hashlib
import time


class ExecutionFinalityKernel:
    """
    Sole finality gate that turns a sovereign decision into an MT5 order.

    Parameters
    ----------
    mt5_connector : optional
        Any object exposing ``place_order(symbol, order_type, volume, price, ...)``
        returning a dict with a ``ticket`` key on success.
        When *None* the kernel runs in simulation mode and generates a mock ticket.
    """

    def __init__(self, mt5_connector=None):
        self.mt5_connector = mt5_connector

    # ------------------------------------------------------------------
    def finalize(
        self,
        ses_result: dict,
        ecl_state: dict,
        awns_result: dict,
        mt5_tick: dict,
        signal: dict,
    ) -> dict:
        """
        Execute finality — emit MT5 order or terminate the pipeline cleanly.

        Parameters
        ----------
        ses_result : dict
            Output of ``SingleExecutionSovereign.evaluate()``.
            Required keys: ``emit_order``, ``order_params``.
        ecl_state : dict
            Execution Control Loop state snapshot.
            Required key: ``locked``.
        awns_result : dict
            Output of AWNS (not used in current logic but passed for audit).
        mt5_tick : dict
            Latest market tick. Expected keys: ``bid``, ``ask``, ``time``.
        signal : dict
            The best-signal dict from the signal mapper.

        Returns
        -------
        dict with keys:
            order_emitted        : bool    – True if order successfully emitted
            mt5_order_result     : dict|None – raw result from place_order
            ticket               : int|None – MT5 ticket number if filled
            pipeline_terminated  : bool    – True when pipeline is fully ended
            pending_decisions    : int     – always 0 (no dangling state)
            finality_latency_ms  : float   – wall-clock time in ms for finalize()
            error                : str|None – error description, if any
        """
        start = time.perf_counter()
        try:
            # ---- determine whether to emit ----
            emit_order_flag = bool(ses_result.get("emit_order", False))
            ecl_locked = bool(ecl_state.get("locked", False))
            should_emit = emit_order_flag and ecl_locked

            if not should_emit:
                latency = (time.perf_counter() - start) * 1000.0
                return {
                    "order_emitted": False,
                    "mt5_order_result": None,
                    "ticket": None,
                    "pipeline_terminated": True,
                    "pending_decisions": 0,
                    "finality_latency_ms": latency,
                    "error": None,
                }

            # ---- build order packet ----
            order_params = ses_result.get("order_params", {})
            symbol = order_params.get("symbol") or (signal or {}).get("symbol", "UNKNOWN")
            action = order_params.get("action", "BUY")
            volume = order_params.get("volume", 0.01)
            price = order_params.get("price", 0.0)
            order_type = order_params.get("order_type", "MARKET")

            # ---- emit via connector or simulation ----
            mt5_order_result = None
            ticket = None

            if self.mt5_connector is not None:
                try:
                    mt5_order_result = self.mt5_connector.place_order(
                        symbol=symbol,
                        order_type=action,  # BUY/SELL
                        volume=volume,
                        price=price,
                    )
                    if mt5_order_result is not None:
                        ticket = mt5_order_result.get("ticket")
                except Exception as exc:
                    latency = (time.perf_counter() - start) * 1000.0
                    return {
                        "order_emitted": False,
                        "mt5_order_result": None,
                        "ticket": None,
                        "pipeline_terminated": True,
                        "pending_decisions": 0,
                        "finality_latency_ms": latency,
                        "error": f"MT5 place_order raised: {exc}",
                    }
            else:
                # ---- simulation mode: generate a plausible mock ticket ----
                raw = f"{symbol}{action}{volume}{time.time_ns()}"
                mock_ticket = int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)
                # keep ticket in a reasonable positive range
                ticket = mock_ticket % 10_000_000 + 1_000_000
                mt5_order_result = {
                    "ticket": ticket,
                    "symbol": symbol,
                    "type": action,
                    "volume": volume,
                    "price": price,
                }

            # ---- finalise ----
            latency = (time.perf_counter() - start) * 1000.0
            return {
                "order_emitted": ticket is not None,
                "mt5_order_result": mt5_order_result,
                "ticket": ticket,
                "pipeline_terminated": True,
                "pending_decisions": 0,
                "finality_latency_ms": latency,
                "error": None,
            }

        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000.0
            return {
                "order_emitted": False,
                "mt5_order_result": None,
                "ticket": None,
                "pipeline_terminated": True,
                "pending_decisions": 0,
                "finality_latency_ms": latency,
                "error": str(exc),
            }
