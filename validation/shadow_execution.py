import logging
from typing import Dict, List, Optional
from collections import defaultdict

from signals.reflexive_arbitration import ReflexiveArbitrationEngine

logger = logging.getLogger("proxima_demo")


RECOMMENDATIONS = ["EXECUTE", "HESITATE", "AVOID", "REVIEW"]


def _production_to_rec(action: str) -> str:
    """Map production action to observer recommendation space for comparison.

    BUY/SELL  -> EXECUTE   (production is taking a directional trade)
    FLAT      -> HESITATE  (production is equivocal / standing aside)
    Everything else falls through to HESITATE as a safe default.
    """
    if action in ("BUY", "SELL"):
        return "EXECUTE"
    return "HESITATE"


class ShadowExecutionEngine:
    """Pure observer that shadows every production trading decision against
    D7 (ReflexiveArbitrationEngine) recommendations without affecting execution.

    Records every disagreement and computes outcome differences so the system
    can later evaluate whether the observer would have performed better.
    """

    def __init__(self, d7: Optional[ReflexiveArbitrationEngine] = None):
        self.d7 = d7 if d7 is not None else ReflexiveArbitrationEngine()
        self._decisions: List[dict] = []
        self._outcomes: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Core shadow cycle
    # ------------------------------------------------------------------

    def shadow_decision(self, decision_state: dict,
                        production_action: str) -> dict:
        """Shadow a single production decision.

        Calls D7.evaluate() to obtain the observer recommendation, compares
        it against the production action via the mapping in _production_to_rec,
        and records the result internally.

        Parameters
        ----------
        decision_state : dict
            State dict consumed by D7 — must contain at minimum
            ``thesis_rf_probability``, ``memory_weight``,
            ``counterfactual_score``, ``rupture_probability``,
            ``path_probability``, ``causal_confidence``.
        production_action : str
            One of ``BUY``, ``SELL``, or ``FLAT``.

        Returns
        -------
        dict
            ``decision_id``, ``production_action``,
            ``observer_recommendation``, ``observer_quality``,
            ``disagreement`` (bool), ``would_execute`` (bool).
        """
        d7_result = self.d7.evaluate(decision_state)

        rec = d7_result["recommendation"]
        prod_rec = _production_to_rec(production_action)
        disagreement = rec != prod_rec
        would_execute = rec == "EXECUTE"

        record = {
            "decision_id": d7_result["decision_id"],
            "production_action": production_action,
            "observer_recommendation": rec,
            "observer_quality": d7_result["decision_quality"],
            "disagreement": disagreement,
            "would_execute": would_execute,
            "decision_state": dict(decision_state),
        }
        self._decisions.append(record)

        return {
            "decision_id": d7_result["decision_id"],
            "production_action": production_action,
            "observer_recommendation": rec,
            "observer_quality": d7_result["decision_quality"],
            "disagreement": disagreement,
            "would_execute": would_execute,
        }

    def resolve_outcome(self, decision_id: str, pnl: float,
                        success: bool):
        """Record the actual outcome of a previously shadowed decision.

        Parameters
        ----------
        decision_id : str
            The ``decision_id`` returned by ``shadow_decision``.
        pnl : float
            Realised PnL of the production decision.
        success : bool
            Whether the production decision is considered a success.
        """
        for d in self._decisions:
            if d["decision_id"] == decision_id:
                self._outcomes[decision_id] = {
                    "decision_id": decision_id,
                    "production_action": d["production_action"],
                    "observer_recommendation": d["observer_recommendation"],
                    "disagreement": d["disagreement"],
                    "pnl": pnl,
                    "success": success,
                }
                self.d7.observe_outcome(decision_id, success)
                logger.info(
                    "[SHADOW_OUTCOME] %s pnl=%.2f success=%s",
                    decision_id, pnl, success,
                )
                return
        logger.warning("[SHADOW_OUTCOME] decision %s not found — ignoring",
                       decision_id)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def disagreement_matrix(self) -> dict:
        """Produce a confusion-matrix-style summary of production vs observer.

        Returns
        -------
        dict
            ``total_decisions``, ``agreements``, ``disagreements``,
            ``disagreement_rate``, ``production_precision``,
            ``observer_precision``, and a ``matrix`` keyed by the four
            recommendation categories (EXECUTE, HESITATE, AVOID, REVIEW).
            Each matrix cell carries ``agreed``, ``disagreed``,
            ``production_pnl`` and ``observer_pnl`` counts.
        """
        total = len(self._decisions)
        if total == 0:
            return self._empty_matrix()

        agreements = 0
        disagreements = 0

        prod_correct = 0
        prod_total = 0
        obs_correct = 0
        obs_total = 0

        matrix = {
            rec: {"agreed": 0, "disagreed": 0,
                  "production_pnl": 0.0, "observer_pnl": 0.0}
            for rec in RECOMMENDATIONS
        }

        for d in self._decisions:
            cat = _production_to_rec(d["production_action"])
            if d["disagreement"]:
                matrix[cat]["disagreed"] += 1
                disagreements += 1
            else:
                matrix[cat]["agreed"] += 1
                agreements += 1

            outcome = self._outcomes.get(d["decision_id"])
            if outcome is not None:
                matrix[cat]["production_pnl"] += outcome["pnl"]
                obs_pnl = self._estimate_observer_pnl(
                    d["production_action"],
                    d["observer_recommendation"],
                    outcome["pnl"],
                )
                matrix[cat]["observer_pnl"] += obs_pnl

                # Production precision: when production mapped to EXECUTE,
                # was the trade successful?
                if _production_to_rec(d["production_action"]) == "EXECUTE":
                    prod_total += 1
                    if outcome["success"]:
                        prod_correct += 1

                # Observer precision: was the observer's recommendation
                # consistent with the eventual outcome?
                obs_total += 1
                if self._observer_would_succeed(
                    d["observer_recommendation"], outcome["success"],
                ):
                    obs_correct += 1

        return {
            "total_decisions": total,
            "agreements": agreements,
            "disagreements": disagreements,
            "disagreement_rate": round(disagreements / total, 4),
            "production_precision": (
                round(prod_correct / prod_total, 4) if prod_total > 0 else 0.0
            ),
            "observer_precision": (
                round(obs_correct / obs_total, 4) if obs_total > 0 else 0.0
            ),
            "matrix": matrix,
        }

    def outcome_comparison(self) -> dict:
        """Compare what actually happened vs what D7 would have caused.

        Returns
        -------
        dict
            ``production_total_pnl``, ``observer_total_pnl``,
            ``production_win_rate``, ``observer_win_rate``,
            ``pnl_delta`` (observer - production),
            ``agreement_pnl``, ``disagreement_pnl_production``,
            ``disagreement_pnl_observer``.
        """
        prod_total_pnl = 0.0
        obs_total_pnl = 0.0
        prod_wins = 0
        prod_losses = 0
        obs_wins = 0
        obs_losses = 0
        agreement_pnl = 0.0
        disc_prod_pnl = 0.0
        disc_obs_pnl = 0.0

        for decision_id, outcome in self._outcomes.items():
            d = self._find_decision(decision_id)
            if d is None:
                continue

            pnl = outcome["pnl"]
            prod_total_pnl += pnl

            obs_pnl = self._estimate_observer_pnl(
                d["production_action"],
                d["observer_recommendation"],
                pnl,
            )
            obs_total_pnl += obs_pnl

            # Production win / loss
            if outcome["success"]:
                prod_wins += 1
            else:
                prod_losses += 1

            # Observer win / loss (did the observer's advice lead to a
            # good outcome?)
            if self._observer_would_succeed(
                d["observer_recommendation"], outcome["success"],
            ):
                obs_wins += 1
            else:
                obs_losses += 1

            if d["disagreement"]:
                disc_prod_pnl += pnl
                disc_obs_pnl += obs_pnl
            else:
                agreement_pnl += pnl  # same for both when agreed

        resolved = len(self._outcomes)
        return {
            "production_total_pnl": round(prod_total_pnl, 4),
            "observer_total_pnl": round(obs_total_pnl, 4),
            "production_win_rate": (
                round(prod_wins / resolved, 4) if resolved > 0 else 0.0
            ),
            "observer_win_rate": (
                round(obs_wins / resolved, 4) if resolved > 0 else 0.0
            ),
            "pnl_delta": round(obs_total_pnl - prod_total_pnl, 4),
            "agreement_pnl": round(agreement_pnl, 4),
            "disagreement_pnl_production": round(disc_prod_pnl, 4),
            "disagreement_pnl_observer": round(disc_obs_pnl, 4),
        }

    def stats(self) -> dict:
        """Return basic shadow-execution statistics.

        Returns
        -------
        dict
            ``decisions`` (total shadowed), ``resolved`` (outcomes recorded),
            ``disagreements``, ``disagreement_rate``.
        """
        total = len(self._decisions)
        resolved = len(self._outcomes)
        disagreements = sum(1 for d in self._decisions if d["disagreement"])
        return {
            "decisions": total,
            "resolved": resolved,
            "disagreements": disagreements,
            "disagreement_rate": (
                round(disagreements / total, 4) if total > 0 else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_observer_pnl(self, production_action: str,
                                observer_rec: str,
                                actual_pnl: float) -> float:
        """Estimate what PnL the observer *would* have achieved.

        * Agreement — observer would have done the same thing → same PnL.
        * Disagreement, observer said EXECUTE → observer would have traded
          (assume same result since we cannot know the counterfactual price).
        * Disagreement, observer said HESITATE/AVOID/REVIEW → observer
          would NOT have traded → PnL is 0.
        """
        prod_rec = _production_to_rec(production_action)
        if prod_rec == observer_rec:
            return actual_pnl  # same decision
        if observer_rec == "EXECUTE":
            return actual_pnl  # would have traded, same outcome
        return 0.0  # avoided the trade

    @staticmethod
    def _observer_would_succeed(observer_rec: str, success: bool) -> bool:
        """Would the observer count this outcome as a success given their
        recommendation?

        * If the observer said EXECUTE → success means the trade won.
        * If the observer said anything else (avoid / hesitate / review)
          → success means the trade lost (they correctly stayed out).
        """
        if observer_rec == "EXECUTE":
            return success
        return not success

    def _find_decision(self, decision_id: str) -> Optional[dict]:
        for d in self._decisions:
            if d["decision_id"] == decision_id:
                return d
        return None

    def _empty_matrix(self) -> dict:
        return {
            "total_decisions": 0,
            "agreements": 0,
            "disagreements": 0,
            "disagreement_rate": 0.0,
            "production_precision": 0.0,
            "observer_precision": 0.0,
            "matrix": {
                rec: {"agreed": 0, "disagreed": 0,
                      "production_pnl": 0.0, "observer_pnl": 0.0}
                for rec in RECOMMENDATIONS
            },
        }
