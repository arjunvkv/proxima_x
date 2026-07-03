"""ECL — Execution Commitment Lock.

Once SES commits ALLOW, enforce a non-interruptible execution window.
Prevent post-ALLOW re-evaluation or rollback.
"""

import hashlib
import time


class ExecutionCommitmentLock:
    """Execution Commitment Lock — prevents non-interruptible execution
    window violation after SES commits ALLOW.

    Tracks lock state, generates unique commit identifiers, and enforces
    single-commit semantics within a configured window of cycles.
    """

    def __init__(self, lock_cycles: int = 3):
        """Initialise the lock.

        Args:
            lock_cycles: Number of cycles the lock remains active after
                         commit (default 3).
        """
        self._lock_cycles = lock_cycles
        self._locked = False
        self._commit_id = None
        self._order_params = None
        self._lock_cycle = None          # cycle number when committed
        self._lock_duration = None       # total cycles lock was held
        self._lock_expiry_cycle = None   # cycle on which lock expires

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def commit(self, order_params: dict, ses_result: dict) -> dict:
        """Attempt to acquire the execution commitment lock.

        The lock is acquired only when the SES result signals
        ``emit_order == True``.  If the lock is already held the call
        is rejected.

        Args:
            order_params: Order parameters to freeze under the lock.
            ses_result:   SES evaluation result (must contain at least
                          ``emit_order`` and optionally ``cycle``).

        Returns:
            A dict with the following keys:

            - **locked** ― ``True`` if the lock was acquired.
            - **order_params** ― frozen order params (or ``None``).
            - **lock_expiry_cycle** ― cycle when the lock expires (or
              ``None``).
            - **commit_id** ― unique commit identifier (or ``None``).
            - **rejection_reason** ― human-readable reason for rejection
              (or ``None`` on success).
        """
        try:
            if self._locked:
                return {
                    "locked": False,
                    "order_params": None,
                    "lock_expiry_cycle": None,
                    "commit_id": None,
                    "rejection_reason": (
                        "Commit lock already active — cannot double-commit"
                    ),
                }

            emit_order = ses_result.get("emit_order", False)
            if not emit_order:
                return {
                    "locked": False,
                    "order_params": None,
                    "lock_expiry_cycle": None,
                    "commit_id": None,
                    "rejection_reason": (
                        "SES did not emit ALLOW — lock not acquired"
                    ),
                }

            # Unique commit identifier derived from a nanosecond timestamp
            raw = str(time.time_ns())
            commit_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

            current_cycle = ses_result.get("cycle", 0)
            lock_expiry_cycle = current_cycle + self._lock_cycles

            self._locked = True
            self._commit_id = commit_id
            self._order_params = order_params
            self._lock_cycle = current_cycle
            self._lock_expiry_cycle = lock_expiry_cycle
            self._lock_duration = 0

            return {
                "locked": True,
                "order_params": self._order_params,
                "lock_expiry_cycle": self._lock_expiry_cycle,
                "commit_id": self._commit_id,
                "rejection_reason": None,
            }

        except Exception as exc:
            return {
                "locked": False,
                "order_params": None,
                "lock_expiry_cycle": None,
                "commit_id": None,
                "rejection_reason": f"Lock commit error: {exc}",
            }

    def get_lock_state(self) -> dict:
        """Return the current lock state.

        Returns:
            A dict with the following keys:

            - **locked** ― whether the lock is currently held.
            - **commit_id** ― active commit identifier (or ``None``).
            - **lock_cycles_remaining** ― configured lock duration (in
              cycles) when locked, otherwise ``0``.
            - **order_params** ― frozen order parameters (or ``None``).
            - **committed_at_cycle** ― cycle number at commit time (or
              ``None``).
        """
        try:
            if not self._locked:
                return {
                    "locked": False,
                    "commit_id": None,
                    "lock_cycles_remaining": 0,
                    "order_params": None,
                    "committed_at_cycle": None,
                }

            return {
                "locked": self._locked,
                "commit_id": self._commit_id,
                "lock_cycles_remaining": self._lock_cycles,
                "order_params": self._order_params,
                "committed_at_cycle": self._lock_cycle,
            }

        except Exception as exc:
            return {
                "locked": False,
                "commit_id": None,
                "lock_cycles_remaining": 0,
                "order_params": None,
                "committed_at_cycle": None,
                "error": str(exc),
            }

    def release(self) -> dict:
        """Release the execution commitment lock.

        Can be called manually or triggered externally on lock expiry.

        Returns:
            A dict with the following keys:

            - **released** ― ``True`` if the lock was successfully
              released.
            - **commit_id** ― the commit identifier that was released.
            - **lock_duration** ― total number of cycles the lock was
              held (equal to the configured ``lock_cycles`` if the lock
              was active).
        """
        try:
            if not self._locked:
                return {
                    "released": False,
                    "commit_id": self._commit_id,
                    "lock_duration": 0,
                }

            commit_id = self._commit_id
            duration = self._lock_cycles

            self._locked = False
            self._commit_id = None
            self._order_params = None
            self._lock_cycle = None
            self._lock_duration = duration
            self._lock_expiry_cycle = None

            return {
                "released": True,
                "commit_id": commit_id,
                "lock_duration": duration,
            }

        except Exception as exc:
            return {
                "released": False,
                "commit_id": self._commit_id,
                "lock_duration": self._lock_duration or 0,
                "error": str(exc),
            }
