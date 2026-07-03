"""
E7  -  Staged Promotion Engine.

Manages the lifecycle of a module's promotion from observer through
graduated execution influence stages.

Stages:
    OBSERVER  -> No execution influence (default)
    PAPER     -> Shadow paper trading alongside production
    P5        -> 5% execution influence
    P10       -> 10% execution influence
    P25       -> 25% execution influence
    P50       -> 50% execution influence
    P100      -> 100% execution influence

Run:  python -c "from validation.staged_promotion import StagedPromotion"
"""

import time
from typing import Dict, Optional


STAGES = ["OBSERVER", "PAPER", "P5", "P10", "P25", "P50", "P100"]

# Influence weights by stage (for reference)
STAGE_INFLUENCE = {
    "OBSERVER": 0.00,
    "PAPER": 0.00,
    "P5": 0.05,
    "P10": 0.10,
    "P25": 0.25,
    "P50": 0.50,
    "P100": 1.00,
}


# ---------------------------------------------------------------------------
# Per-module entry criteria definitions
# ---------------------------------------------------------------------------

# Each transition maps to a list of (criterion_key, checker_fn, required_display)
# checker_fn(state, metrics) -> {"pass": bool, "current": value, "required": value}

def _criteria_observer_to_paper(state: dict, metrics: dict) -> dict:
    """OBSERVER -> PAPER: E3 composite >= 0.50 AND E6 all checks pass."""
    composite = metrics.get("composite", 0.0)
    e6_pass = metrics.get("e6_all_pass", False)
    return {
        "min_composite": {
            "pass": composite >= 0.50,
            "current": composite,
            "required": 0.50,
        },
        "e6_all_pass": {
            "pass": bool(e6_pass),
            "current": bool(e6_pass),
            "required": True,
        },
    }


def _criteria_paper_to_p5(state: dict, metrics: dict) -> dict:
    """PAPER -> P5: 100+ observations, disagreement_rate < 0.30."""
    obs = state.get("observations", 0)
    disagreement = metrics.get("disagreement_rate", 1.0)
    return {
        "min_observations": {
            "pass": obs >= 100,
            "current": obs,
            "required": 100,
        },
        "max_disagreement": {
            "pass": disagreement < 0.30,
            "current": disagreement,
            "required": 0.30,
        },
    }


def _criteria_p5_to_p10(state: dict, metrics: dict) -> dict:
    """P5 -> P10: 200+ observations, no rollback in last 100 observations."""
    obs = state.get("observations", 0)
    obs_since_rb = state.get("observations_since_rollback", 0)
    return {
        "min_observations": {
            "pass": obs >= 200,
            "current": obs,
            "required": 200,
        },
        "rollback_free_observations": {
            "pass": obs_since_rb >= 100,
            "current": obs_since_rb,
            "required": 100,
        },
    }


def _criteria_p10_to_p25(state: dict, metrics: dict) -> dict:
    """P10 -> P25: 500+ observations, false_veto_rate < 0.20."""
    obs = state.get("observations", 0)
    false_veto = metrics.get("false_veto_rate", 1.0)
    return {
        "min_observations": {
            "pass": obs >= 500,
            "current": obs,
            "required": 500,
        },
        "max_false_veto_rate": {
            "pass": false_veto < 0.20,
            "current": false_veto,
            "required": 0.20,
        },
    }


def _criteria_p25_to_p50(state: dict, metrics: dict) -> dict:
    """P25 -> P50: 1000+ observations, drawdown_change < 0.02."""
    obs = state.get("observations", 0)
    dd_change = metrics.get("drawdown_change", 1.0)
    return {
        "min_observations": {
            "pass": obs >= 1000,
            "current": obs,
            "required": 1000,
        },
        "max_drawdown_change": {
            "pass": dd_change < 0.02,
            "current": dd_change,
            "required": 0.02,
        },
    }


def _criteria_p50_to_p100(state: dict, metrics: dict) -> dict:
    """P50 -> P100: 2000+ observations, all metrics stable for 500 obs."""
    obs = state.get("observations", 0)
    stable_obs = metrics.get("stable_observations", 0)
    return {
        "min_observations": {
            "pass": obs >= 2000,
            "current": obs,
            "required": 2000,
        },
        "min_stable_observations": {
            "pass": stable_obs >= 500,
            "current": stable_obs,
            "required": 500,
        },
    }


# Map (from_stage, to_stage) -> criteria function
ENTRY_CRITERIA = {
    ("OBSERVER", "PAPER"): _criteria_observer_to_paper,
    ("PAPER", "P5"):       _criteria_paper_to_p5,
    ("P5", "P10"):         _criteria_p5_to_p10,
    ("P10", "P25"):        _criteria_p10_to_p25,
    ("P25", "P50"):        _criteria_p25_to_p50,
    ("P50", "P100"):       _criteria_p50_to_p100,
}


# ---------------------------------------------------------------------------
# Staged Promotion Engine
# ---------------------------------------------------------------------------

class StagedPromotion:
    """Stateful staged promotion engine for module lifecycle management."""

    def __init__(self):
        self._modules: Dict[str, dict] = {}
        self._history: Dict[str, list] = {}
        self._review_history: Dict[str, list] = {}

    # ------------------------------------------------------------------
    # 1. register
    # ------------------------------------------------------------------

    def register(self, module_name: str) -> dict:
        """Register a module for staged promotion. Starts at OBSERVER.

        Parameters
        ----------
        module_name : str
            Module identifier (e.g. "D1", "D2", ...).

        Returns
        -------
        dict
            The new module state dict.
        """
        if module_name in self._modules:
            raise ValueError(
                f"Module '{module_name}' is already registered."
            )

        now = int(time.time())
        state = {
            "module": module_name,
            "stage": "OBSERVER",
            "stage_since": now,
            "observations": 0,
            "total_observations": 0,
            "rollbacks": 0,
            "last_rollback_reason": "",
            "metrics_snapshot": {},
            "observations_since_rollback": 0,
        }
        self._modules[module_name] = state
        self._history[module_name] = []
        self._review_history[module_name] = []
        return dict(state)

    # ------------------------------------------------------------------
    # 2. advance
    # ------------------------------------------------------------------

    def advance(self, module_name: str, metrics: dict) -> dict:
        """Check if module meets entry criteria for the next stage.

        Parameters
        ----------
        module_name : str
            Registered module identifier.
        metrics : dict
            Current performance metrics. Expected keys vary by transition:
            - composite, e6_all_pass (OBSERVER -> PAPER)
            - disagreement_rate (PAPER -> P5)
            - false_veto_rate (P10 -> P25)
            - drawdown_change (P25 -> P50)
            - stable_observations (P50 -> P100)

        Returns
        -------
        dict
            {
                "module": str,
                "stage_advanced": bool,
                "from_stage": str,
                "to_stage": str,
                "reason": str,
                "entry_criteria": { ... },
                "metrics_at_entry": dict,
            }
        """
        if module_name not in self._modules:
            return {
                "module": module_name,
                "stage_advanced": False,
                "from_stage": None,
                "to_stage": None,
                "reason": f"Module '{module_name}' is not registered.",
                "entry_criteria": {},
                "metrics_at_entry": {},
            }

        state = self._modules[module_name]
        current_stage = state["stage"]
        next_stage = self._next_stage(current_stage)

        if next_stage is None:
            return {
                "module": module_name,
                "stage_advanced": False,
                "from_stage": current_stage,
                "to_stage": None,
                "reason": f"Module is already at max stage: {current_stage}",
                "entry_criteria": {},
                "metrics_at_entry": {},
            }

        # Resolve criteria checker
        criteria_fn = ENTRY_CRITERIA.get((current_stage, next_stage))
        if criteria_fn is None:
            return {
                "module": module_name,
                "stage_advanced": False,
                "from_stage": current_stage,
                "to_stage": next_stage,
                "reason": f"No criteria defined for {current_stage} -> {next_stage}",
                "entry_criteria": {},
                "metrics_at_entry": {},
            }

        entry_criteria = criteria_fn(state, metrics)
        all_pass = all(v["pass"] for v in entry_criteria.values())

        if all_pass:
            now = int(time.time())
            transition = {
                "from": current_stage,
                "to": next_stage,
                "timestamp": now,
                "reason": "Entry criteria met",
                "metrics_at_entry": dict(metrics),
            }
            self._history[module_name].append(transition)

            state["stage"] = next_stage
            state["stage_since"] = now
            state["observations"] = 0
            state["metrics_snapshot"] = dict(metrics)

            return {
                "module": module_name,
                "stage_advanced": True,
                "from_stage": current_stage,
                "to_stage": next_stage,
                "reason": "All entry criteria met",
                "entry_criteria": entry_criteria,
                "metrics_at_entry": dict(metrics),
            }
        else:
            failed = [k for k, v in entry_criteria.items() if not v["pass"]]
            return {
                "module": module_name,
                "stage_advanced": False,
                "from_stage": current_stage,
                "to_stage": next_stage,
                "reason": "Entry criteria not met: "
                          f"{', '.join(failed)}",
                "entry_criteria": entry_criteria,
                "metrics_at_entry": {},
            }

    # ------------------------------------------------------------------
    # 3. rollback
    # ------------------------------------------------------------------

    def rollback(self, module_name: str, reason: str) -> dict:
        """Roll back one stage.

        Records reason, increments rollback counter.
        Cannot go below OBSERVER.

        Parameters
        ----------
        module_name : str
            Registered module identifier.
        reason : str
            Explanation for the rollback.

        Returns
        -------
        dict
            {
                "module": str,
                "rollback_successful": bool,
                "from_stage": str,
                "to_stage": str,
                "reason": str,
            }
        """
        if module_name not in self._modules:
            return {
                "module": module_name,
                "rollback_successful": False,
                "from_stage": None,
                "to_stage": None,
                "reason": f"Module '{module_name}' is not registered.",
            }

        state = self._modules[module_name]
        current_stage = state["stage"]
        prev_stage = self._prev_stage(current_stage)

        if prev_stage is None:
            return {
                "module": module_name,
                "rollback_successful": False,
                "from_stage": current_stage,
                "to_stage": None,
                "reason": "Cannot rollback below OBSERVER.",
            }

        now = int(time.time())
        transition = {
            "from": current_stage,
            "to": prev_stage,
            "timestamp": now,
            "reason": f"Rollback: {reason}",
            "metrics_at_entry": dict(state.get("metrics_snapshot", {})),
        }
        self._history[module_name].append(transition)

        old_stage = current_stage
        state["stage"] = prev_stage
        state["stage_since"] = now
        state["observations"] = 0
        state["rollbacks"] += 1
        state["last_rollback_reason"] = reason
        state["observations_since_rollback"] = 0

        # Reset consecutive review flags
        self._review_history[module_name] = []

        return {
            "module": module_name,
            "rollback_successful": True,
            "from_stage": old_stage,
            "to_stage": prev_stage,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # 4. auto_review
    # ------------------------------------------------------------------

    def auto_review(self, module_name: str,
                    current_metrics: dict) -> dict:
        """Check if rollback is needed based on current metrics.

        Rollback criteria:
        - Drawdown increase > 0.10 (relative to entry)
        - False veto rate > 0.40
        - Disagreement_rate > 0.50 for 3 consecutive reviews
        - Composite score drop > 0.20

        Parameters
        ----------
        module_name : str
            Registered module identifier.
        current_metrics : dict
            Current performance metrics. Expected keys:
            drawdown, false_veto_rate, disagreement_rate, composite.

        Returns
        -------
        dict
            {
                "module": str,
                "rollback_needed": bool,
                "reason": str,
                "triggered_rules": list,
            }
        """
        if module_name not in self._modules:
            return {
                "module": module_name,
                "rollback_needed": False,
                "reason": f"Module '{module_name}' is not registered.",
                "triggered_rules": [],
            }

        state = self._modules[module_name]
        snapshot = state.get("metrics_snapshot", {})
        triggered = []

        # --- 4a. Drawdown increase > 0.10 ---
        if "drawdown" in current_metrics and "drawdown" in snapshot:
            drawdown_increase = (current_metrics["drawdown"]
                                 - snapshot["drawdown"])
            if drawdown_increase > 0.10:
                triggered.append({
                    "rule": "drawdown_increase",
                    "detail": (
                        f"Drawdown increased by {drawdown_increase:.4f} "
                        f"(threshold: 0.10)"
                    ),
                    "current": current_metrics["drawdown"],
                    "entry": snapshot["drawdown"],
                })

        # --- 4b. False veto rate > 0.40 ---
        false_veto = current_metrics.get("false_veto_rate", 0.0)
        if false_veto > 0.40:
            triggered.append({
                "rule": "false_veto_rate",
                "detail": f"False veto rate {false_veto:.4f} > 0.40",
                "current": false_veto,
                "threshold": 0.40,
            })

        # --- 4c. Disagreement_rate > 0.50 for 3 consecutive reviews ---
        disagreement = current_metrics.get("disagreement_rate", 0.0)
        review_flags = self._review_history.setdefault(module_name, [])
        review_flags.append(disagreement > 0.50)
        if len(review_flags) > 3:
            # Keep only the last 3
            self._review_history[module_name] = review_flags[-3:]
            review_flags = self._review_history[module_name]

        if (len(review_flags) == 3
                and all(review_flags)):
            triggered.append({
                "rule": "high_disagreement_consecutive",
                "detail": "Disagreement rate > 0.50 for 3 consecutive reviews",
                "current": disagreement,
                "threshold": 0.50,
                "consecutive_count": 3,
            })

        # --- 4d. Composite score drop > 0.20 ---
        if "composite" in current_metrics and "composite" in snapshot:
            composite_drop = (snapshot["composite"]
                              - current_metrics["composite"])
            if composite_drop > 0.20:
                triggered.append({
                    "rule": "composite_drop",
                    "detail": (
                        f"Composite score dropped by {composite_drop:.4f} "
                        f"(threshold: 0.20)"
                    ),
                    "entry_composite": snapshot["composite"],
                    "current_composite": current_metrics["composite"],
                })

        if triggered:
            return {
                "module": module_name,
                "rollback_needed": True,
                "reason": "Rollback triggered by: "
                          f"{', '.join(t['rule'] for t in triggered)}",
                "triggered_rules": triggered,
            }
        else:
            return {
                "module": module_name,
                "rollback_needed": False,
                "reason": "All metrics within acceptable bounds.",
                "triggered_rules": [],
            }

    # ------------------------------------------------------------------
    # 5. status
    # ------------------------------------------------------------------

    def status(self, module_name: str) -> dict:
        """Current state of a module.

        Returns
        -------
        dict
            Full module state including ``registered`` bool.
        """
        if module_name not in self._modules:
            return {"module": module_name, "registered": False}
        return {
            "registered": True,
            **dict(self._modules[module_name]),
        }

    # ------------------------------------------------------------------
    # 6. all_status
    # ------------------------------------------------------------------

    def all_status(self) -> dict:
        """Status of all registered modules, with stage distribution.

        Returns
        -------
        dict
            {
                "modules": { name: state, ... },
                "stage_distribution": { stage: count, ... },
                "total_modules": int,
            }
        """
        distribution = {s: 0 for s in STAGES}
        modules_info = {}
        for name, state in self._modules.items():
            stage = state["stage"]
            distribution[stage] = distribution.get(stage, 0) + 1
            modules_info[name] = dict(state)

        return {
            "modules": modules_info,
            "stage_distribution": distribution,
            "total_modules": len(self._modules),
        }

    # ------------------------------------------------------------------
    # 7. stage_progression
    # ------------------------------------------------------------------

    def stage_progression(self, module_name: str) -> list:
        """Full history of stage transitions for a module.

        Returns
        -------
        list of dict
            Each entry: {from, to, timestamp, reason, metrics_at_entry}
        """
        if module_name not in self._modules:
            return []
        return list(self._history.get(module_name, []))

    # ------------------------------------------------------------------
    # 8. stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Aggregate statistics.

        Returns
        -------
        dict
            {
                "registered": int,
                "stage_distribution": {stage: count},
                "total_rollbacks": int,
                "average_observations": float,
            }
        """
        n = len(self._modules)
        if n == 0:
            return {
                "registered": 0,
                "stage_distribution": {s: 0 for s in STAGES},
                "total_rollbacks": 0,
                "average_observations": 0.0,
            }

        distribution = {s: 0 for s in STAGES}
        total_rb = 0
        total_obs = 0
        for state in self._modules.values():
            stage = state["stage"]
            distribution[stage] = distribution.get(stage, 0) + 1
            total_rb += state.get("rollbacks", 0)
            total_obs += state.get("total_observations", 0)

        return {
            "registered": n,
            "stage_distribution": distribution,
            "total_rollbacks": total_rb,
            "average_observations": (
                round(total_obs / n, 2) if n > 0 else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Utility: record a decision observation
    # ------------------------------------------------------------------

    def record_observation(self, module_name: str) -> dict:
        """Record a single decision observation for a module.

        Increments ``observations`` (current stage),
        ``total_observations`` (lifetime), and
        ``observations_since_rollback``.

        Parameters
        ----------
        module_name : str
            Registered module identifier.

        Returns
        -------
        dict
            {"module": str, "success": bool, "observations": int}
        """
        if module_name not in self._modules:
            return {
                "module": module_name,
                "success": False,
                "observations": -1,
            }

        state = self._modules[module_name]
        state["observations"] += 1
        state["total_observations"] += 1
        state["observations_since_rollback"] += 1
        return {
            "module": module_name,
            "success": True,
            "observations": state["observations"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_stage(self, current: str) -> Optional[str]:
        idx = STAGES.index(current)
        if idx >= len(STAGES) - 1:
            return None
        return STAGES[idx + 1]

    def _prev_stage(self, current: str) -> Optional[str]:
        idx = STAGES.index(current)
        if idx <= 0:
            return None
        return STAGES[idx - 1]
