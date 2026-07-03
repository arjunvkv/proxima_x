from __future__ import annotations

from typing import Any

import numpy as np

from research.mechanism_discovery.base import BaseMechanism, MechanismScore


class MechanismScorer:

    def __init__(self) -> None:
        pass

    def score_mechanism(
        self,
        mechanism: BaseMechanism,
        mechanism_result: dict[str, Any],
        validator_results: dict[str, Any],
    ) -> MechanismScore:
        information_gain = self._extract_value(validator_results, "information_gain")
        sid = self._extract_value(validator_results, "sid")
        sir = self._extract_value(validator_results, "sir")
        persistence = self._extract_value(validator_results, "persistence", "persistence_score")

        failed_tests = self._count_failed_tests(validator_results)
        robustness = 1.0 / (1.0 + failed_tests)

        num_parameters = max(len(mechanism_result), 1)
        simplicity = 1.0 / (1.0 + num_parameters)

        max_sim = self._extract_similarity(mechanism_result, validator_results)
        novelty = 1.0 - max_sim if max_sim is not None else 0.5

        cross_asset_score = self._extract_value(validator_results, "cross_asset_score")
        cross_regime_score = self._extract_value(validator_results, "cross_regime_score")
        oos_score = self._extract_value(validator_results, "oos_score")

        details: dict[str, Any] = {
            "num_parameters": num_parameters,
            "failed_tests": failed_tests,
        }

        return MechanismScore(
            name=mechanism.name,
            category=mechanism.category,
            information_gain=information_gain,
            sid=sid,
            sir=sir,
            persistence=persistence,
            robustness=robustness,
            cross_asset_score=cross_asset_score,
            cross_regime_score=cross_regime_score,
            oos_score=oos_score,
            simplicity=simplicity,
            novelty=novelty,
            details=details,
        )

    def rank_mechanisms(self, scores: list[MechanismScore]) -> list[MechanismScore]:
        return sorted(scores, key=lambda s: s.composite_score, reverse=True)

    def get_top_mechanisms(
        self, scores: list[MechanismScore], top_k: int = 5
    ) -> list[MechanismScore]:
        ranked = self.rank_mechanisms(scores)
        return ranked[:top_k]

    def get_surviving_mechanisms(
        self, scores: list[MechanismScore]
    ) -> list[MechanismScore]:
        return [s for s in scores if s.survives]

    def score_summary(self, scores: list[MechanismScore]) -> dict[str, Any]:
        if not scores:
            return {
                "n_total": 0,
                "n_surviving": 0,
                "n_failed": 0,
                "top_mechanism": None,
                "worst_mechanism": None,
                "mean_score": 0.0,
                "std_score": 0.0,
            }

        n_total = len(scores)
        surviving = self.get_surviving_mechanisms(scores)
        n_surviving = len(surviving)
        n_failed = n_total - n_surviving

        ranked = self.rank_mechanisms(scores)
        composite_vals = np.array([s.composite_score for s in scores], dtype=np.float64)

        return {
            "n_total": n_total,
            "n_surviving": n_surviving,
            "n_failed": n_failed,
            "top_mechanism": ranked[0].name if ranked else None,
            "worst_mechanism": ranked[-1].name if ranked else None,
            "mean_score": float(np.mean(composite_vals)),
            "std_score": float(np.std(composite_vals)),
        }

    def generate_scoring_report(self, scores: list[MechanismScore]) -> str:
        ranked = self.rank_mechanisms(scores)
        summary = self.score_summary(scores)

        lines: list[str] = []
        lines.append("# Mechanism Scoring Report")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Mechanisms**: {summary['n_total']}")
        lines.append(f"- **Surviving**: {summary['n_surviving']}")
        lines.append(f"- **Failed**: {summary['n_failed']}")
        lines.append(f"- **Mean Composite Score**: {summary['mean_score']:.4f}")
        lines.append(f"- **Std Composite Score**: {summary['std_score']:.4f}")
        lines.append(f"- **Top Mechanism**: {summary['top_mechanism']}")
        lines.append(f"- **Worst Mechanism**: {summary['worst_mechanism']}")
        lines.append("")

        lines.append("## Ranked Mechanisms")
        lines.append("")
        lines.append(
            "| Rank | Name | Category | Composite | Info Gain | SID | SIR | Persist | Robust | Simple | Novelty | Survives |"
        )
        lines.append(
            "|------|------|----------|-----------|-----------|-----|-----|---------|--------|--------|---------|----------|"
        )

        for rank, score in enumerate(ranked, start=1):
            survives_mark = "Y" if score.survives else "N"
            lines.append(
                f"| {rank} | {score.name} | {score.category} "
                f"| {score.composite_score:.4f} | {score.information_gain:.4f} "
                f"| {score.sid:.4f} | {score.sir:.4f} "
                f"| {score.persistence:.4f} | {score.robustness:.4f} "
                f"| {score.simplicity:.4f} | {score.novelty:.4f} "
                f"| {survives_mark} |"
            )

        lines.append("")

        surviving = self.get_surviving_mechanisms(scores)
        if surviving:
            lines.append("## Surviving Mechanisms")
            lines.append("")
            for s in surviving:
                lines.append(
                    f"- **{s.name}** ({s.category}): composite={s.composite_score:.4f}, "
                    f"IG={s.information_gain:.4f}, SID={s.sid:.4f}, SIR={s.sir:.4f}"
                )
            lines.append("")

        failed = [s for s in scores if not s.survives]
        if failed:
            lines.append("## Failed Mechanisms")
            lines.append("")
            for s in failed:
                lines.append(
                    f"- **{s.name}** ({s.category}): composite={s.composite_score:.4f}"
                )
            lines.append("")

        return "\n".join(lines)

    def _extract_value(
        self, d: dict, *keys: str
    ) -> float:
        for key in keys:
            for k, v in d.items():
                if isinstance(k, str):
                    if isinstance(v, dict) and key in v:
                        val = v[key]
                        if isinstance(val, (int, float)):
                            return float(val)
                    if k == key and isinstance(v, (int, float)):
                        return float(v)
            if isinstance(key, str) and key in d and isinstance(d[key], (int, float)):
                return float(d[key])
        return 0.0

    def _count_failed_tests(self, validator_results: dict[str, Any]) -> int:
        failed = 0
        for k, v in validator_results.items():
            if isinstance(v, dict) and "passed" in v:
                if not v["passed"]:
                    failed += 1
            elif isinstance(v, bool) and k != "all":
                if not v:
                    failed += 1
        return failed

    def _extract_similarity(
        self,
        mechanism_result: dict[str, Any],
        validator_results: dict[str, Any],
    ) -> float | None:
        for container in (mechanism_result, validator_results):
            for k, v in container.items():
                if isinstance(k, str) and "similarity" in k.lower() and isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        if isinstance(sk, str) and "similarity" in sk.lower() and isinstance(sv, (int, float)):
                            return float(sv)
        return None
