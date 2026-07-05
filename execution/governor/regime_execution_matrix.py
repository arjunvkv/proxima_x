from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RegimeType(Enum):
    SHADOW = 0
    MICRO = 1
    FULL = 2
    UNKNOWN = 3


@dataclass
class ExecutionPermissionsMatrix:
    regime: RegimeType
    allowed_actions: list[str]
    max_exposure: float     # 0.0 to 1.0
    max_position_size: float # 0.0 to 1.0
    description: str


class RegimeExecutionMatrix:
    """Defines execution permissions per regime."""

    def __init__(self):
        # Default permission matrix
        self._matrix = {
            RegimeType.SHADOW: ExecutionPermissionsMatrix(
                regime=RegimeType.SHADOW,
                allowed_actions=[
                    "HOLD",
                    "REDUCE_LIGHT",
                    "TRANSITION_PREP",
                    "EMERGENCY_STOP",
                ],
                max_exposure=0.05,
                max_position_size=0.05,
                description="SHADOW: Observe only. Limited to position reduction and transition preparation."
            ),
            RegimeType.MICRO: ExecutionPermissionsMatrix(
                regime=RegimeType.MICRO,
                allowed_actions=[
                    "HOLD",
                    "BUY_LIGHT",
                    "REDUCE_LIGHT",
                    "REDUCE_MODERATE",
                    "REDUCE_STRONG",
                    "EXIT_ALL",
                    "TRANSITION_PREP",
                    "EMERGENCY_STOP",
                ],
                max_exposure=0.25,
                max_position_size=0.20,
                description="MICRO: Limited exposure. No strong directional bets."
            ),
            RegimeType.FULL: ExecutionPermissionsMatrix(
                regime=RegimeType.FULL,
                allowed_actions=[
                    "BUY_STRONG",
                    "BUY_MODERATE",
                    "BUY_LIGHT",
                    "HOLD",
                    "REDUCE_LIGHT",
                    "REDUCE_MODERATE",
                    "REDUCE_STRONG",
                    "EXIT_ALL",
                    "TRANSITION_PREP",
                    "EMERGENCY_STOP",
                ],
                max_exposure=0.80,
                max_position_size=0.50,
                description="FULL: Full execution capabilities with risk limits."
            ),
            RegimeType.UNKNOWN: ExecutionPermissionsMatrix(
                regime=RegimeType.UNKNOWN,
                allowed_actions=[
                    "HOLD",
                    "REDUCE_LIGHT",
                    "REDUCE_MODERATE",
                    "EMERGENCY_STOP",
                ],
                max_exposure=0.02,
                max_position_size=0.02,
                description="UNKNOWN: Maximal restriction. Only defensive actions."
            ),
        }

    # ------------------------------------------------------------------
    # Internal regime resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_regime(regime: RegimeType | str | int) -> RegimeType:
        """Resolve a regime value to a RegimeType enum.

        Accepts:
          - RegimeType enum      → direct lookup
          - str ("SHADOW", ...)  → case-insensitive mapping
          - int (0, 1, 2, 3)    → RegimeType mapping

        Returns ``RegimeType.UNKNOWN`` for any unresolvable value (safe default).
        """
        if isinstance(regime, RegimeType):
            return regime

        if isinstance(regime, str):
            try:
                return RegimeType[regime.upper().strip()]
            except (KeyError, ValueError):
                return RegimeType.UNKNOWN

        if isinstance(regime, int):
            try:
                return RegimeType(regime)
            except (ValueError, TypeError):
                return RegimeType.UNKNOWN

        return RegimeType.UNKNOWN

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_permissions(self, regime: RegimeType | str | int) -> ExecutionPermissionsMatrix:
        """Get permission matrix for a regime (accepts enum, string, or int)."""
        resolved = self._resolve_regime(regime)
        return self._matrix.get(resolved, self._matrix[RegimeType.UNKNOWN])

    def is_action_allowed(self, action: str, regime: RegimeType | str | int) -> bool:
        """Check if a specific action is allowed in given regime."""
        permissions = self.get_permissions(regime)
        return action.upper().strip() in permissions.allowed_actions

    def get_max_exposure(self, regime: RegimeType | str | int) -> float:
        """Get max exposure fraction for regime."""
        return self.get_permissions(regime).max_exposure

    def get_max_position_size(self, regime: RegimeType | str | int) -> float:
        """Get max position size fraction for regime."""
        return self.get_permissions(regime).max_position_size

    def update_permissions(
        self, regime: RegimeType, permissions: ExecutionPermissionsMatrix
    ) -> None:
        """Override permissions for a regime (for dynamic adjustment)."""
        self._matrix[regime] = permissions
