import math
import statistics
from ...core.fsv_schema import FundamentalStateVector
from .fundamental_ranker import FundamentalRanker
from .macro_alignment import MacroAlignmentEngine


class FundamentalComparator:

    def __init__(self) -> None:
        self.ranker: FundamentalRanker = FundamentalRanker()
        self.alignment: MacroAlignmentEngine = MacroAlignmentEngine()

    def compare_symbols(
        self,
        symbols: list[str],
        fsves: dict[str, FundamentalStateVector],
        directions: dict[str, int],
    ) -> dict:
        candidates: dict[str, tuple[FundamentalStateVector, int]] = {}
        for sym in symbols:
            candidates[sym] = (fsves[sym], directions[sym])

        ranked: list[dict] = self.ranker.rank_symbols(candidates)
        ranking_vector: dict[str, float] = self.ranker.get_ranking_vector(ranked)

        aligned_data: dict[str, dict] = {}
        for sym in symbols:
            aligned_data[sym] = self.alignment.full_evaluation(
                sym, fsves[sym], directions[sym], fsves
            )

        divergences: dict[str, float] = {}
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a: str = symbols[i]
                sym_b: str = symbols[j]
                key: str = f"{sym_a}_{sym_b}"
                fsv_a: FundamentalStateVector = fsves[sym_a]
                fsv_b: FundamentalStateVector = fsves[sym_b]
                feat_a: list[float] = fsv_a.to_feature_vector()
                feat_b: list[float] = fsv_b.to_feature_vector()
                divergence: float = math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(feat_a, feat_b))
                )
                divergences[key] = round(divergence, 4)

        environment: str = self._detect_environment(aligned_data)

        compared_data: dict = {
            "ranked_symbols": [r["symbol"] for r in ranked],
            "ranking_vector": ranking_vector,
            "alignment": aligned_data,
            "divergences": divergences,
            "environment": environment,
        }
        compared_data["recommendation"] = self.generate_recommendation(compared_data)

        return compared_data

    def detect_strongest_alignment(self, aligned_data: dict) -> tuple[str, float]:
        best_sym: str = ""
        best_strength: float = -1.0
        for sym, data in aligned_data.items():
            strength: float = abs(data.get("alignment_score", 0.0))
            if strength > best_strength:
                best_strength = strength
                best_sym = sym
        return best_sym, best_strength

    def detect_weakest_contradiction(
        self, ranked: list[dict], aligned_data: dict
    ) -> tuple[str, float]:
        contradictions: dict[str, float] = {}
        for r in ranked:
            sym: str = r["symbol"]
            fund_score: float = r.get("fundamental_score", 0.0)
            align_score: float = aligned_data.get(sym, {}).get(
                "alignment_score", 0.0
            )
            contradictions[sym] = abs(fund_score - align_score)
        best_sym: str = min(contradictions, key=contradictions.get)
        return best_sym, contradictions[best_sym]

    def compute_confidence(
        self,
        symbol: str,
        ranked: list[dict],
        aligned_data: dict,
        divergences: dict,
    ) -> float:
        rank_pos: int = -1
        for i, r in enumerate(ranked):
            if r["symbol"] == symbol:
                rank_pos = i
                break

        if rank_pos == 0:
            rank_factor: float = 0.4
        elif rank_pos == 1:
            rank_factor: float = 0.25
        elif rank_pos == 2:
            rank_factor: float = 0.1
        else:
            rank_factor: float = 0.0

        align_strength: float = abs(
            aligned_data.get(symbol, {}).get("alignment_score", 0.0)
        )

        total_div: float = 0.0
        count_div: int = 0
        for key, val in divergences.items():
            if symbol in key:
                total_div += val
                count_div += 1
        avg_divergence: float = total_div / count_div if count_div > 0 else 0.0
        normalized_div: float = min(1.0, avg_divergence / 3.0)

        confidence: float = (
            rank_factor + 0.3 * align_strength + 0.3 * normalized_div
        )
        return max(0.0, min(1.0, confidence))

    def generate_recommendation(self, compared_data: dict) -> dict:
        ranked_symbols: list[str] = compared_data.get("ranked_symbols", [])
        ranking_vector: dict = compared_data.get("ranking_vector", {})
        aligned_data: dict = compared_data.get("alignment", {})
        divergences: dict = compared_data.get("divergences", {})

        if not ranked_symbols:
            return {
                "best_symbol": "",
                "confidence": 0.0,
                "reason": "no symbols to compare",
                "runner_up": "",
                "ranking_spread": 0.0,
            }

        best_sym: str = ranked_symbols[0]
        runner_up: str = ranked_symbols[1] if len(ranked_symbols) > 1 else best_sym

        ranking_spread: float = 0.0
        if len(ranked_symbols) >= 2:
            ranking_spread = (
                ranking_vector.get(best_sym, 0.0)
                - ranking_vector.get(runner_up, 0.0)
            )
            ranking_spread = max(0.0, ranking_spread)

        strongest_sym, strongest_strength = self.detect_strongest_alignment(
            aligned_data
        )
        weakest_sym, _ = self.detect_weakest_contradiction(
            [
                {
                    "symbol": s,
                    "fundamental_score": ranking_vector.get(s, 0.0),
                }
                for s in ranked_symbols
            ],
            aligned_data,
        )

        ranked_for_confidence: list[dict] = [
            {"symbol": s, "fundamental_score": ranking_vector.get(s, 0.0)}
            for s in ranked_symbols
        ]
        confidence: float = self.compute_confidence(
            best_sym, ranked_for_confidence, aligned_data, divergences
        )

        reasons: list[str] = []
        if strongest_sym == best_sym:
            reasons.append(
                f"{best_sym} has the strongest macro alignment ({strongest_strength:.3f})"
            )
        else:
            reasons.append(
                f"{best_sym} leads rank despite {strongest_sym} having stronger alignment"
            )
        if weakest_sym == best_sym:
            reasons.append(
                "lowest contradiction between fundamental score and alignment"
            )
        reasons.append(f"ranking spread of {ranking_spread:.3f} over {runner_up}")

        return {
            "best_symbol": best_sym,
            "confidence": round(confidence, 4),
            "reason": "; ".join(reasons),
            "runner_up": runner_up,
            "ranking_spread": round(ranking_spread, 4),
        }

    def _detect_environment(self, aligned_data: dict) -> str:
        scores: list[float] = [
            data.get("alignment_score", 0.0) for data in aligned_data.values()
        ]

        if not scores:
            return "neutral"

        positive: int = sum(1 for s in scores if s > 0.1)
        negative: int = sum(1 for s in scores if s < -0.1)

        if positive >= 2 and negative == 0:
            return "risk_on"
        elif negative >= 2 and positive == 0:
            return "risk_off"
        elif positive >= 1 and negative >= 1:
            return "mixed"
        else:
            return "neutral"
