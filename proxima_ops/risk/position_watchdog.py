"""RHL-8: Position Watchdog — MT5 vs ledger cross-verification with auto-healing."""

import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.risk.watchdog")


class PositionWatchdog:
    def __init__(self, position_manager=None):
        self._mismatches: list[dict] = []
        self._state = "HEALTHY"
        self._pm = position_manager
        self._last_mismatch_hash = None
        self._hydrated_count = 0
        self._purged_count = 0

    def set_position_manager(self, pm):
        self._pm = pm

    def verify(self, mt5_positions: list[dict], ledger_positions: list[dict]) -> dict:
        mt5_tickets = {p.get("ticket") for p in mt5_positions if p.get("ticket")}
        ledger_tickets = {p.get("ticket") for p in ledger_positions if p.get("ticket")}

        mt5_only = mt5_tickets - ledger_tickets
        ledger_only = ledger_tickets - mt5_tickets

        mismatches = []

        # Auto-hydrate MT5-only orphans
        for t in sorted(mt5_only):
            mt5_pos = next((p for p in mt5_positions if p.get("ticket") == t), None)
            if mt5_pos and self._pm:
                try:
                    self._pm.hydrate_from_mt5(mt5_pos)
                    self._hydrated_count += 1
                    logger.info(f"[WATCHDOG] Hydrated orphan MT5 position {t}")
                except Exception as e:
                    mismatches.append({"ticket": t, "issue": "MT5_only", "detail": f"hydration_failed:{e}"})
            else:
                mismatches.append({"ticket": t, "issue": "MT5_only", "detail": "no_position_manager"})

        # Auto-purge ledger-only ghosts
        for t in sorted(ledger_only):
            if self._pm:
                try:
                    self._pm.close_ghost_position(t)
                    self._purged_count += 1
                    logger.info(f"[WATCHDOG] Purged ghost ledger position {t}")
                except Exception as e:
                    mismatches.append({"ticket": t, "issue": "ledger_only", "detail": f"purge_failed:{e}"})
            else:
                mismatches.append({"ticket": t, "issue": "ledger_only", "detail": "no_position_manager"})

        # Volume/PnL mismatches on common positions
        for mp in mt5_positions:
            t = mp.get("ticket")
            lp = next((p for p in ledger_positions if p.get("ticket") == t), None)
            if lp is None:
                continue
            if abs(mp.get("volume", 0) - lp.get("volume", 0)) > 0.001:
                mismatches.append({"ticket": t, "issue": "volume_mismatch",
                                   "mt5_vol": mp.get("volume"), "ledger_vol": lp.get("volume")})
            if abs(mp.get("profit", 0) - lp.get("profit", 0)) > 0.50:
                mismatches.append({"ticket": t, "issue": "pnl_mismatch",
                                   "mt5_pnl": mp.get("profit"), "ledger_pnl": lp.get("profit")})

        # Spam suppression: only log when mismatch set changes
        current_hash = hash(tuple(sorted(
            (m["ticket"], m["issue"]) for m in mismatches
        ))) if mismatches else 0

        if mismatches and current_hash != self._last_mismatch_hash:
            self._last_mismatch_hash = current_hash
            self._state = "CRITICAL_POSITION_MISMATCH"
            logger.warning(f"Position watchdog: {len(mismatches)} mismatch(es) — {mismatches[0]['issue']}")
            for m in mismatches[:3]:
                logger.warning(f"  [{m['issue']}] ticket={m['ticket']} {m.get('detail','')}")
        elif not mismatches:
            if self._state != "HEALTHY":
                logger.info(f"[WATCHDOG] State restored to HEALTHY (hydrated={self._hydrated_count}, purged={self._purged_count})")
            self._state = "HEALTHY"
            self._last_mismatch_hash = None

        if mismatches:
            self._mismatches.extend(mismatches)

        return {"state": self._state, "mismatch_count": len(mismatches), "mismatches": mismatches[:5]}
