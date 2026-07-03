import os
import json
import logging
from typing import Optional, List

logger = logging.getLogger("proxima_ops.runtime.plr")


class PositionLifecycleReactor:
    """Position Lifecycle Reactor (PLR) - Event-driven position exit evaluation."""

    def __init__(self, mrbl, state_path: str = "state/plr_state.json", max_hold_cycles: int = 15):
        self.mrbl = mrbl
        self._state_path = state_path
        self._max_hold_cycles = max_hold_cycles
        self._hold_tracker: dict[int, int] = {}
        self._load_state()

    def _load_state(self):
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self._hold_tracker = {int(k): int(v) for k, v in state.items()}
            logger.info("[PLR] Restored hold tracker state: %s", self._hold_tracker)
        except Exception as e:
            logger.warning("[PLR] Load state failed: %s", e)

    def _save_state(self):
        parts = os.path.split(self._state_path)
        if parts[0]:
            os.makedirs(parts[0], exist_ok=True)
        try:
            with open(self._get_state_path(), "w") as f:
                json.dump(self._hold_tracker, f, indent=2)
        except Exception as e:
            logger.warning("[PLR] Save state failed: %s", e)

    def _get_state_path(self) -> str:
        return self._state_path

    def on_position_event(self, positions: List[dict]) -> List[dict]:
        """Process latest open positions and evaluate exit policies.

        Returns list of executed close dicts.
        """
        closed_results = []
        active_tickets = set()

        for pos in positions:
            ticket = pos.get("ticket")
            if not ticket:
                continue
            active_tickets.add(ticket)

            # Increment hold cycle for active position
            current_hold = self._hold_tracker.get(ticket, 0)
            new_hold = current_hold + 1
            self._hold_tracker[ticket] = new_hold

            # Evaluate exit policies
            should_close = False
            reason = "HOLD"

            # 1. SL/TP Check (from broker or local PnL limits)
            profit = pos.get("profit", 0.0)
            volume = pos.get("volume", 0.01)
            
            # Scaled SL and TP thresholds based on volume (tighter thresholds for demo demonstration)
            sl_threshold = -0.15 * (volume / 0.01)
            tp_threshold = 0.20 * (volume / 0.01)

            if profit <= sl_threshold:
                should_close = True
                reason = "SL_HIT"
            elif profit >= tp_threshold:
                should_close = True
                reason = "TP_HIT"

            # 2. H20 Temporal Exit (exit after max_hold_cycles)
            if not should_close and new_hold >= self._max_hold_cycles:
                should_close = True
                reason = "H20_EXIT"

            # Execute close if policy triggers
            if should_close:
                logger.info("[PLR_TRIGGER] Closing ticket=%d symbol=%s reason=%s profit=%.2f hold_cycles=%d",
                            ticket, pos.get("symbol"), reason, profit, new_hold)
                success = self.mrbl.close_position(ticket)
                closed_results.append({
                    "ticket": ticket,
                    "symbol": pos.get("symbol"),
                    "reason": reason,
                    "profit": profit,
                    "hold_cycles": new_hold,
                    "success": success,
                })
                # Remove from hold tracker immediately on close
                self._hold_tracker.pop(ticket, None)

        # Cleanup hold tracker for positions that are no longer active
        for t in list(self._hold_tracker.keys()):
            if t not in active_tickets:
                self._hold_tracker.pop(t, None)

        self._save_state()
        return closed_results
