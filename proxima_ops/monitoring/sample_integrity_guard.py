"""
SAMPLE INTEGRITY GUARD — Phase 4 Deliverable

Suppresses conclusive classifications when sample size is insufficient.

Phases:
  trades < 30:   EARLY_VALIDATION
  trades < 100:  COLLECTING_EVIDENCE
  trades < 300:  INTERMEDIATE_VALIDATION
  trades >= 300: FULL_VALIDATION
"""

import os
import math

SUPPRESSED_CLASSIFICATIONS = [
    "LIVE_DEPLOYABLE",
    "ALPHA_DECAYING",
    "ALPHA_CONFIRMED",
    "RESEARCH_ARTIFACT",
    "PRODUCTION_READY",
]

SUPPRESSION_REPLACEMENT = "INSUFFICIENT_EVIDENCE"


class SampleIntegrityGuard:
    def __init__(self, n_trades: int = 0):
        self.n_trades = n_trades

    @property
    def phase(self) -> str:
        if self.n_trades < 30:
            return "EARLY_VALIDATION"
        elif self.n_trades < 100:
            return "COLLECTING_EVIDENCE"
        elif self.n_trades < 300:
            return "INTERMEDIATE_VALIDATION"
        else:
            return "FULL_VALIDATION"

    @property
    def phase_num(self) -> int:
        return {"EARLY_VALIDATION": 1, "COLLECTING_EVIDENCE": 2,
                "INTERMEDIATE_VALIDATION": 3, "FULL_VALIDATION": 4}[self.phase]

    @property
    def can_classify(self) -> bool:
        return self.phase == "FULL_VALIDATION"

    def guard(self, classification: str) -> str:
        if classification in SUPPRESSED_CLASSIFICATIONS and not self.can_classify:
            return SUPPRESSION_REPLACEMENT
        return classification

    def ci_width(self, n_success: int = None) -> float:
        """95% Wilson confidence interval width for a proportion."""
        if self.n_trades == 0:
            return 1.0
        p = (n_success / self.n_trades) if n_success is not None else 0.5
        z = 1.96
        denominator = 1 + z**2 / self.n_trades
        centre = (p + z**2 / (2 * self.n_trades)) / denominator
        margin = z * math.sqrt((p * (1 - p) / self.n_trades + z**2 / (4 * self.n_trades**2))) / denominator
        return margin * 2  # total width

    def summary(self) -> str:
        lines = []
        lines.append(f"Validation Phase: {self.phase} (Phase {self.phase_num}/4)")
        lines.append(f"Total trades: {self.n_trades}")
        lines.append(f"Conclusive classification allowed: {self.can_classify}")
        lines.append(f"95% CI width (worst case): +/-{1.96*0.5/math.sqrt(max(self.n_trades,1)):.1%}")
        return "\n".join(lines)


def demo():
    guard = SampleIntegrityGuard(6)
    print(guard.summary())
    print()
    for cls in SUPPRESSED_CLASSIFICATIONS:
        print(f"  {cls:30s} -> {guard.guard(cls)}")


if __name__ == "__main__":
    demo()
