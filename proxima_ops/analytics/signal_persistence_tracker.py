import logging
from collections import defaultdict, deque
from typing import Any

logger = logging.getLogger("proxima_ops.analytics.signal_persistence_tracker")


class SignalPersistenceTracker:
    """Track how long signals persist per symbol, confirm survival probability,
    and which stage they drop off at (threshold vs confirm vs governor)."""

    def __init__(self, max_history: int = 500):
        self.max_history = max_history

        # edge_id -> record
        #   record = {
        #       "first_seen": int,
        #       "last_seen": int,
        #       "lifetime": int,          # cycles this signal has been alive
        #       "symbol": str,
        #       "direction": int,
        #       "confirm_key": str,       # "{symbol}_{BUY|SELL}"
        #       "max_confirm": int,       # highest confirm count observed
        #       "reached_confirm_2": bool,
        #       "failure_stage": str|None,  # "threshold" | "confirm" | "governor" | "unknown"
        #       "died_at_cycle": int|None,
        #   }
        self._signal_records: dict[str, dict[str, Any]] = {}

        # Ring buffers keyed by cycle number (most recent last)
        self._active_history: deque[set[str]] = deque(maxlen=max_history)
        self._pipeline_trace_history: deque[dict] = deque(maxlen=max_history)
        self._cycle_numbers: deque[int] = deque(maxlen=max_history)

        # Counters for which stage signals died at
        self._stage_drops: dict[str, int] = {
            "threshold": 0,
            "confirm": 0,
            "governor": 0,
            "unknown": 0,
        }

        # Per-symbol accumulation
        #   {symbol: {"signal_count": int, "lifetimes": list[int], "confirm_pass_count": int}}
        self._per_symbol: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"signal_count": 0, "lifetimes": [], "confirm_pass_count": 0}
        )

        self._total_cycles: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_signal_cycle(
        self,
        cycle: int,
        signals: list,
        confirm_counts: dict,
        governor_state: str,
        pipeline_trace: dict,
    ) -> None:
        """Record one cycle's worth of signal activity.

        Parameters
        ----------
        cycle : int
            Current executor cycle number.
        signals : list[dict]
            Active signals from ``_sweep_signals()``.  Each dict must have
            at least ``edge_id``, ``symbol``, ``direction`` keys.
        confirm_counts : dict[str, int]
            ``self.confirm_cycles`` from the executor, keyed by
            ``{symbol}_{BUY|SELL}``.
        governor_state : str
            SEGL state string from the executor cycle.
        pipeline_trace : dict
            Pipeline trace with keys ``threshold_gate``, ``confirm_gate``,
            ``governor_gate``, ``execution``.
        """
        try:
            self._total_cycles += 1
            current_ids: set[str] = set()
            signal_info: dict[str, dict[str, Any]] = {}

            # ---- 1. Index current signals by edge_id ----
            for s in signals:
                eid = s.get("edge_id")
                if eid is None:
                    continue
                current_ids.add(eid)
                sym: str = s.get("symbol", "?")
                direction: int = s.get("direction", 0)
                s_dir = "BUY" if direction > 0 else "SELL"
                confirm_key = f"{sym}_{s_dir}"
                signal_info[eid] = {
                    "symbol": sym,
                    "direction": direction,
                    "confirm_key": confirm_key,
                    "current_confirm": confirm_counts.get(confirm_key, 0),
                }

            # ---- 2. Update / create signal records ----
            for eid, info in signal_info.items():
                rec = self._signal_records.get(eid)
                if rec is None:
                    # New signal
                    self._signal_records[eid] = {
                        "first_seen": cycle,
                        "last_seen": cycle,
                        "lifetime": 1,
                        "symbol": info["symbol"],
                        "direction": info["direction"],
                        "confirm_key": info["confirm_key"],
                        "max_confirm": info["current_confirm"],
                        "reached_confirm_2": info["current_confirm"] >= 2,
                        "failure_stage": None,
                        "died_at_cycle": None,
                    }
                    sym = info["symbol"]
                    self._per_symbol[sym]["signal_count"] += 1
                else:
                    # Existing signal – update lifetime & confirm progress
                    rec["last_seen"] = cycle
                    rec["lifetime"] += 1
                    if info["current_confirm"] > rec["max_confirm"]:
                        rec["max_confirm"] = info["current_confirm"]
                    if info["current_confirm"] >= 2:
                        rec["reached_confirm_2"] = True

            # ---- 3. Detect signals that died since previous cycle ----
            if self._cycle_numbers and self._cycle_numbers[-1] == cycle - 1:
                prev_ids = self._active_history[-1]
                died_ids = prev_ids - current_ids
                prev_trace = (
                    self._pipeline_trace_history[-1]
                    if self._pipeline_trace_history
                    else {}
                )
                for eid in died_ids:
                    rec = self._signal_records.get(eid)
                    if rec is None or rec["died_at_cycle"] is not None:
                        continue
                    rec["died_at_cycle"] = cycle
                    stage = self._determine_failure_stage(
                        edge_id=eid,
                        pipeline_trace=prev_trace,
                    )
                    rec["failure_stage"] = stage
                    self._stage_drops[stage] = self._stage_drops.get(stage, 0) + 1

                    sym = rec.get("symbol", "?")
                    ps = self._per_symbol.get(sym)
                    if ps is not None:
                        ps["lifetimes"].append(rec["lifetime"])
                        if rec.get("reached_confirm_2", False):
                            ps["confirm_pass_count"] += 1

            # ---- 4. Append current cycle to ring buffers ----
            self._active_history.append(current_ids)
            self._pipeline_trace_history.append(pipeline_trace)
            self._cycle_numbers.append(cycle)

            logger.debug(
                "SignalPersistenceTracker recorded cycle %d: %d active, %d total tracked",
                cycle,
                len(current_ids),
                len(self._signal_records),
            )

        except Exception as exc:
            logger.error("Error recording signal cycle %d: %s", cycle, exc, exc_info=True)

    def report(self) -> dict:
        """Produce a summary report of signal persistence.

        Returns
        -------
        dict
            Schema defined in the task specification.
        """
        try:
            total_signals = len(self._signal_records)
            if total_signals == 0:
                return {
                    "avg_signal_lifetime": 0.0,
                    "confirm_survival_rate": 0.0,
                    "dominant_failure_stage": "unknown",
                    "stage_drop_rates": {
                        "threshold": 0.0,
                        "confirm": 0.0,
                        "governor": 0.0,
                        "unknown": 0.0,
                    },
                    "per_symbol": {},
                }

            # -- Average signal lifetime --
            lifetimes = [r["lifetime"] for r in self._signal_records.values()]
            avg_lifetime = sum(lifetimes) / len(lifetimes)

            # -- Confirm survival rate: fraction that ever reached confirm=2 --
            reached_confirm = sum(
                1 for r in self._signal_records.values() if r.get("reached_confirm_2", False)
            )
            confirm_survival_rate = reached_confirm / total_signals

            # -- Dominant failure stage (among dead signals) --
            total_drops = sum(self._stage_drops.values())
            if total_drops > 0:
                dominant = max(self._stage_drops, key=lambda k: self._stage_drops[k])
            else:
                dominant = "unknown"

            stage_drop_rates = {
                stage: round((count / total_drops) * 100.0, 2) if total_drops > 0 else 0.0
                for stage, count in self._stage_drops.items()
            }

            # -- Per-symbol breakdown (computed from _signal_records to
            #    include both alive and dead signals) --
            sym_data: dict[str, dict[str, Any]] = {}
            for rec in self._signal_records.values():
                sym = rec.get("symbol", "?")
                if sym not in sym_data:
                    sym_data[sym] = {"lifetimes": [], "confirm_passes": 0, "count": 0}
                sym_data[sym]["count"] += 1
                sym_data[sym]["lifetimes"].append(rec["lifetime"])
                if rec.get("reached_confirm_2", False):
                    sym_data[sym]["confirm_passes"] += 1

            per_symbol_report: dict[str, dict[str, Any]] = {}
            for sym, data in sym_data.items():
                sym_lifetimes = data["lifetimes"]
                sym_count = data["count"]
                avg_lt = (
                    sum(sym_lifetimes) / len(sym_lifetimes) if sym_lifetimes else 0.0
                )
                confirm_rate = data["confirm_passes"] / sym_count if sym_count > 0 else 0.0
                per_symbol_report[sym] = {
                    "signal_count": sym_count,
                    "avg_lifetime": round(avg_lt, 2),
                    "confirm_rate": round(confirm_rate, 4),
                }

            return {
                "avg_signal_lifetime": round(avg_lifetime, 2),
                "confirm_survival_rate": round(confirm_survival_rate, 4),
                "dominant_failure_stage": dominant,
                "stage_drop_rates": stage_drop_rates,
                "per_symbol": per_symbol_report,
            }

        except Exception as exc:
            logger.error("Error generating SignalPersistenceTracker report: %s", exc, exc_info=True)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _determine_failure_stage(
        self,
        edge_id: str,
        pipeline_trace: dict,
    ) -> str:
        """Determine which pipeline gate the signal failed at.

        Inspects the pipeline trace from the **last cycle where the signal
        was still present** to see which gate it did *not* pass.
        """
        try:
            # -- threshold gate --
            thresh_entries = pipeline_trace.get("threshold_gate", [])
            for entry in thresh_entries:
                if entry.startswith(f"{edge_id}:"):
                    if "PASS" not in entry:
                        return "threshold"
                    break  # passed threshold

            # -- confirm gate --
            confirm_entries = pipeline_trace.get("confirm_gate", [])
            for entry in confirm_entries:
                if entry.startswith(f"{edge_id}:"):
                    if "CROSS_PASS" not in entry:
                        return "confirm"
                    break  # passed confirm

            # -- governor gate / execution --
            gov_entries = pipeline_trace.get("governor_gate", [])
            for entry in gov_entries:
                if "ready_to_exec=YES" not in entry:
                    return "governor"

            execution = pipeline_trace.get("execution", "")
            if isinstance(execution, str):
                if "DENIED" in execution or "NO_SIGNAL" in execution:
                    return "governor"

            # If the signal passed every gate but still disappeared,
            # it likely expired naturally or was out-competed.
            return "unknown"

        except Exception as exc:
            logger.debug("Error determining failure stage for %s: %s", edge_id, exc)
            return "unknown"
