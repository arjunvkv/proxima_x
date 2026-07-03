"""Intent Stabilization & Utility Constraint Definition Layer (ISUCD).

Defines the formal utility hierarchy, trade-off resolution policy, 
and system intent for the PROXIMA Governance Kernel.

This is the semantic constitution — not a runtime execution layer,
but the formal specification against which ALL runtime behavior is evaluated.
"""
from __future__ import annotations
from enum import Enum, auto
from typing import Optional


class UtilityObjective(Enum):
    """Hierarchical utility objectives ordered by priority (highest first)."""
    SAFETY_INTEGRITY = auto()
    SIGNAL_IDENTITY = auto()
    STATE_CORRECTNESS = auto()
    FREQUENCY_DISCIPLINE = auto()
    PORTFOLIO_STABILITY = auto()
    PERFORMANCE_FIDELITY = auto()


class ConflictType(Enum):
    """Known conflict types that require resolution."""
    MOF_VS_RF = auto()
    SIGNAL_VS_PORTFOLIO = auto()
    FREQUENCY_VS_OPPORTUNITY = auto()
    IDENTITY_VS_DIVERSITY = auto()
    STATE_VS_EXECUTION = auto()


class UtilityPriority:
    """Priority table mapping objectives to their rank.

    Lower number = higher priority.
    Priority 0 = absolute (never violated).
    """
    PRIORITY_TABLE = {
        UtilityObjective.SAFETY_INTEGRITY: 0,
        UtilityObjective.SIGNAL_IDENTITY: 1,
        UtilityObjective.STATE_CORRECTNESS: 2,
        UtilityObjective.FREQUENCY_DISCIPLINE: 3,
        UtilityObjective.PORTFOLIO_STABILITY: 4,
        UtilityObjective.PERFORMANCE_FIDELITY: 5,
    }

    @classmethod
    def dominates(cls, a: UtilityObjective, b: UtilityObjective) -> bool:
        return cls.PRIORITY_TABLE.get(a, 99) < cls.PRIORITY_TABLE.get(b, 99)


class ConflictResolution:
    """Deterministic conflict resolution rules."""

    RULES = {
        ConflictType.MOF_VS_RF: {
            "policy": "MOF_TRUMPS",
            "rationale": "Market Observability Factor reflects direct position-state "
                         "awareness. RF drift is a secondary signal. When MOF indicates "
                         "BLACKOUT/DEGRADED/NOISE, execution is denied regardless of RF.",
            "always_wins": UtilityObjective.SAFETY_INTEGRITY,
        },
        ConflictType.SIGNAL_VS_PORTFOLIO: {
            "policy": "PORTFOLIO_CONFLICT_TRUMPS",
            "rationale": "Portfolio conflict measures real instrument overlap. "
                         "Signal confidence is theoretical. A validated conflict "
                         "barrier blocks marginal signal advantage.",
            "always_wins": UtilityObjective.PORTFOLIO_STABILITY,
        },
        ConflictType.FREQUENCY_VS_OPPORTUNITY: {
            "policy": "FREQUENCY_DISCIPLINE_TRUMPS",
            "rationale": "Frequency budget is a hard safety constraint. "
                         "No execution may exceed MAX_ACTUATIONS_PER_WINDOW "
                         "regardless of signal quality or market conditions.",
            "always_wins": UtilityObjective.FREQUENCY_DISCIPLINE,
        },
        ConflictType.IDENTITY_VS_DIVERSITY: {
            "policy": "IDENTITY_ANCHOR_WINS",
            "rationale": "edge_04 is the system identity anchor. When it conflicts "
                         "with other edges (portfolio overlap, direction conflict), "
                         "edge_04 priority is absolute.",
            "always_wins": UtilityObjective.SIGNAL_IDENTITY,
        },
        ConflictType.STATE_VS_EXECUTION: {
            "policy": "STATE_CORRECTNESS_ABSOLUTE",
            "rationale": "No execution may occur outside valid state transitions. "
                         "State machine integrity is inviolable. If state does not "
                         "permit a transition, execution is denied unconditionally.",
            "always_wins": UtilityObjective.STATE_CORRECTNESS,
        },
    }

    @classmethod
    def resolve(cls, conflict: ConflictType) -> dict:
        return cls.RULES.get(conflict, {
            "policy": "UNKNOWN",
            "rationale": "No resolution policy defined for this conflict type",
            "always_wins": None,
        })


class IntentConstraintLayer:
    """Formal intent constraint layer for PROXIMA Governance Kernel.

    This is a specification layer, not a runtime enforcement layer.
    It defines WHAT constitutes correct behavior.
    Runtime enforcement is delegated to the SelectiveExecutionGovernor
    and its sub-components.

    Use this layer for:
    - Auditing whether runtime decisions conform to system intent
    - Validating that behavioral changes stay within intent bounds
    - Verifying that no drift toward unintended optimization occurs
    """

    def __init__(self):
        self._intent_statement = self._build_intent_statement()
        self._objective_hierarchy = list(UtilityPriority.PRIORITY_TABLE.keys())
        self._resolution_policies = ConflictResolution.RULES

    def _build_intent_statement(self) -> str:
        return (
            "PROXIMA is a governance kernel that maintains bounded decision "
            "integrity for edge signals. "
            "It does not optimize for profit, anticipate market behavior, "
            "or execute outside its safety envelope. "
            "Its purpose is to ensure every execution decision passes a "
            "deterministic gate of constraints, "
            "and that system state remains recoverable to OBSERVE "
            "under all conditions."
        )

    @property
    def intent(self) -> str:
        return self._intent_statement

    def get_priority(self, objective: UtilityObjective) -> int:
        return UtilityPriority.PRIORITY_TABLE.get(objective, 99)

    def get_hierarchy(self) -> list[dict]:
        return [
            {"rank": rank, "objective": obj.name, "priority": True}
            for rank, (obj, _) in enumerate(
                sorted(UtilityPriority.PRIORITY_TABLE.items(),
                       key=lambda x: x[1])
            )
        ]

    def get_conflict_policy(self, conflict: ConflictType) -> dict:
        return ConflictResolution.resolve(conflict)

    def get_all_conflict_policies(self) -> dict:
        return dict(ConflictResolution.RULES)


    def evaluate_decision_against_intent(self, decision: dict) -> dict:
        """Evaluate a runtime decision against the intent specification.

        Args:
            decision: dict with keys:
                - 'objective': UtilityObjective name or value
                - 'conflict': ConflictType name or value (optional)
                - 'outcome': str describing what happened
                - 'traded_away_priority': the objective that was deprioritized

        Returns:
            dict with 'conforms', 'violations', 'rationale'
        """
        violations = []
        objective = decision.get("objective")
        conflict = decision.get("conflict")
        traded_away = decision.get("traded_away_priority")

        if objective and traded_away:
            obj_priority = self.get_priority(objective)
            trade_priority = self.get_priority(traded_away)
            if trade_priority < obj_priority:
                violations.append(
                    f"Priority violation: {traded_away} (rank {trade_priority}) "
                    f"has higher priority than {objective} (rank {obj_priority}) "
                    f"but was traded away"
                )

        if conflict:
            policy = self.get_conflict_policy(conflict)
            if policy.get("always_wins"):
                winner_priority = self.get_priority(policy["always_wins"])
                if objective and self.get_priority(objective) != winner_priority:
                    violations.append(
                        f"Conflict resolution violation: {conflict} policy "
                        f"specifies {policy['always_wins']} as winner, "
                        f"but {objective} was chosen"
                    )

        return {
            "conforms": len(violations) == 0,
            "violations": violations,
            "rationale": self._intent_statement if not violations
                         else "Decision violates intent specification",
        }


    def describe(self) -> dict:
        return {
            "layer": "IntentConstraintLayer",
            "intent": self._intent_statement,
            "objective_hierarchy": self.get_hierarchy(),
            "conflict_policies": self.get_all_conflict_policies(),
            "allowed_failure_envelope": {
                "max_acceptable_drift": {
                    "mof_score": 0.05,
                    "balance_per_cycle": 50.0,
                    "timing_ms": 500,
                },
                "max_acceptable_suppression": {
                    "false_positive_rate": 0.30,
                    "max_cycles_without_execution": 100,
                },
                "max_acceptable_loss": {
                    "per_trade": 0.0,
                    "per_session": 0.0,
                },
                "max_acceptable_latency": {
                    "signal_processing_ms": 50,
                    "decision_evaluation_ms": 50,
                    "total_cycle_ms": 1000,
                },
            },
        }
