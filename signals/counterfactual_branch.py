import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math

logger = logging.getLogger("proxima_demo")

BRANCH_TYPES = ["REAL", "HALF", "DOUBLE", "WAIT5", "WAIT20", "NONE"]


@dataclass
class CounterfactualBranch:
    branch_id: str
    parent_id: int
    action: str
    size_factor: float
    delay_ticks: int
    direction: int
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    label: Optional[int] = None
    pnl: Optional[float] = None
    resolved: bool = False
    entered: bool = False


class CounterfactualBranchEngine:
    def __init__(self):
        self._branches: Dict[int, List[CounterfactualBranch]] = defaultdict(list)
        self._delay_counters: Dict[str, int] = defaultdict(int)

    def register(self, thesis_id: int, direction: int, entry_price: float):
        specs = [
            ("REAL", 1.0, 0, direction),
            ("HALF", 0.5, 0, direction),
            ("DOUBLE", 2.0, 0, direction),
            ("WAIT5", 1.0, 5, direction),
            ("WAIT20", 1.0, 20, direction),
            ("NONE", 0.0, 0, 0),
        ]
        for action, size, delay, direc in specs:
            branch = CounterfactualBranch(
                branch_id=f"{thesis_id}_{action}",
                parent_id=thesis_id,
                action=action,
                size_factor=size,
                delay_ticks=delay,
                direction=direc,
                entry_price=entry_price if delay == 0 else None,
                entered=(delay == 0 and direc != 0),
            )
            self._branches[thesis_id].append(branch)
        logger.info(f"[COUNTERFACTUAL] registered thesis={thesis_id} "
                    f"dir={direction:+d} entry={entry_price}")

    def tick(self, symbol: str, price: float):
        for tid, branches in self._branches.items():
            for branch in branches:
                if branch.entered or branch.action == "NONE":
                    continue
                if branch.entry_price is not None:
                    continue
                self._delay_counters[branch.branch_id] += 1
                if self._delay_counters[branch.branch_id] >= branch.delay_ticks:
                    branch.entry_price = price
                    branch.entered = True
                    logger.info(f"[COUNTERFACTUAL] {branch.action} entered "
                                f"thesis={tid} at {price} "
                                f"(delayed {branch.delay_ticks} ticks)")

    def resolve(self, thesis_id: int, exit_price: float):
        branches = self._branches.get(thesis_id, [])
        if not branches:
            return
        for branch in branches:
            if branch.action == "NONE":
                branch.pnl = 0.0
                branch.label = 0
                branch.resolved = True
                continue
            effective_entry = branch.entry_price
            if effective_entry is None:
                branch.pnl = 0.0
                branch.label = 0
                branch.resolved = True
                continue
            raw_pnl = (exit_price - effective_entry) if branch.direction > 0 else \
                      (effective_entry - exit_price)
            branch.pnl = raw_pnl * branch.size_factor
            branch.label = 1 if branch.pnl > 0 else 0
            branch.exit_price = exit_price
            branch.resolved = True
        logger.info(f"[COUNTERFACTUAL] resolve thesis={thesis_id} "
                    f"exit={exit_price}")

    def get_branches(self, thesis_id: int) -> List[CounterfactualBranch]:
        return list(self._branches.get(thesis_id, []))

    def decision_score(self, thesis_id: int) -> Optional[float]:
        branches = self._get_counterfactuals(thesis_id)
        if not branches:
            return None
        real = self._find_branch(thesis_id, "REAL")
        if real is None or real.pnl is None:
            return None
        best = max(b.pnl for b in branches if b.pnl is not None)
        return real.pnl - best

    def decision_rank(self, thesis_id: int) -> Optional[Tuple[int, int]]:
        cf = self._get_counterfactuals(thesis_id) + [self._find_branch(thesis_id, "REAL")]
        active = [b for b in cf if b is not None and b.pnl is not None]
        if len(active) < 2:
            return None
        sorted_b = sorted(active, key=lambda b: b.pnl, reverse=True)
        for i, b in enumerate(sorted_b):
            if b.action == "REAL":
                return (i + 1, len(active))
        return None

    def regret(self, thesis_id: int) -> Optional[float]:
        branches = self._get_counterfactuals(thesis_id)
        if not branches:
            return None
        real = self._find_branch(thesis_id, "REAL")
        if real is None or real.pnl is None:
            return None
        best = max(b.pnl for b in branches if b.pnl is not None)
        return best - real.pnl

    def stability(self, thesis_id: int) -> Optional[float]:
        all_b = self._branches.get(thesis_id, [])
        pnls = [b.pnl for b in all_b if b.pnl is not None]
        if len(pnls) < 2:
            return None
        min_p = min(pnls)
        max_p = max(pnls)
        if abs(max_p - min_p) < 0.001:
            return 0.0
        midpoint = (max_p + min_p) / 2
        mad = sum(abs(p - midpoint) for p in pnls) / len(pnls)
        return round(mad / max(abs(midpoint), 0.001), 4)

    def summary(self, thesis_id: int) -> Optional[dict]:
        branches = self._branches.get(thesis_id, [])
        if not branches:
            return None
        rank_info = self.decision_rank(thesis_id)
        return {
            "best_branch": max(branches, key=lambda b: b.pnl or -1e9).action if branches else None,
            "real_rank": rank_info[0] if rank_info else None,
            "regret": self.regret(thesis_id),
            "stability": self.stability(thesis_id),
            "decision_score": self.decision_score(thesis_id),
        }

    def _find_branch(self, thesis_id: int, action: str) -> Optional[CounterfactualBranch]:
        for b in self._branches.get(thesis_id, []):
            if b.action == action:
                return b
        return None

    def _get_counterfactuals(self, thesis_id: int) -> List[CounterfactualBranch]:
        return [b for b in self._branches.get(thesis_id, [])
                if b.action in ("WAIT5", "WAIT20", "NONE")]

    def stats(self) -> dict:
        all_branches = sum(len(v) for v in self._branches.values())
        resolved = sum(1 for v in self._branches.values()
                      for b in v if b.resolved)
        scores = []
        ranks = []
        regrets = []
        stabilities = []
        for tid in self._branches:
            ds = self.decision_score(tid)
            if ds is not None:
                scores.append(ds)
            rnk = self.decision_rank(tid)
            if rnk:
                ranks.append(rnk[0])
            reg = self.regret(tid)
            if reg is not None:
                regrets.append(reg)
            stab = self.stability(tid)
            if stab is not None:
                stabilities.append(stab)
        robust = sum(1 for s in stabilities if s < 1.0)
        fragile = sum(1 for s in stabilities if s >= 1.0)
        return {
            "parents": len(self._branches),
            "total_branches": all_branches,
            "resolved": resolved,
            "mean_decision_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "mean_rank": round(sum(ranks) / len(ranks), 2) if ranks else 0.0,
            "mean_regret": round(sum(regrets) / len(regrets), 3) if regrets else 0.0,
            "robust_count": robust,
            "fragile_count": fragile,
            "robust_fraction": round(robust / max(len(stabilities), 1), 2),
        }
