import time
from collections import Counter
from itertools import combinations
from typing import Optional
from config.settings import (
    DRS_LAMBDA_DECAY, DRS_SLOT_INERTIA,
    MAX_POSITIONS, MAX_CURRENCY_FACTOR_EXPOSURE,
    DRS_CANDIDATE_POOL_SIZE
)
from data.models import DirectionHypothesis, PaperPosition


class DRS:
    def __init__(self):
        self._held_positions: list[PaperPosition] = []
        self._drs_scores: dict[str, float] = {}
        self.last_selection_trace: dict = {}
        self._replacement_margin = 0.15  # DRS_REPLACE_MARGIN

    def set_positions(self, positions: list[PaperPosition]) -> None:
        self._held_positions = positions

    def _currency_vector(self, symbol: str, direction: float) -> Counter:
        base, quote = self._get_currencies(symbol)
        mult = 1 if direction > 0 else -1
        v: Counter = Counter()
        v[base] += mult
        v[quote] -= mult
        return v

    def _net_currency_exposure(self) -> Counter:
        exposure: Counter = Counter()
        for pos in self._held_positions:
            direction = 1 if pos.direction == "BUY" else -1
            vector = self._currency_vector(pos.symbol, direction)
            for c, v in vector.items():
                exposure[c] += v
        return exposure

    def _exceeds_currency_limit(self, exposure: Counter) -> bool:
        return any(abs(v) > MAX_CURRENCY_FACTOR_EXPOSURE for v in exposure.values())

    def _factor_overlap_penalty(self, hypothesis: DirectionHypothesis,
                                 current_exposure: Counter) -> float:
        vector = self._currency_vector(hypothesis.symbol, hypothesis.direction)
        overlap = sum(abs(v) for c, v in vector.items() if c in current_exposure)
        return min(overlap * 0.25, 1.0)

    def rank(self, hypotheses: list[DirectionHypothesis],
             narrative_quality: dict[str, float] | None = None) -> list[DirectionHypothesis]:
        if not hypotheses:
            return []

        scored = []
        base_exposure = self._net_currency_exposure()
        for h in hypotheses:
            if narrative_quality and h.symbol in narrative_quality:
                conviction_weight = 0.30
                confidence_weight = 0.25
                quality_weight = 0.20
                diversification_weight = 0.15
                narrative_weight = 0.10
            else:
                conviction_weight = 0.35
                confidence_weight = 0.25
                quality_weight = 0.20
                diversification_weight = 0.20
                narrative_weight = 0.0

            strength_diff = abs(h.base_strength - h.quote_strength)
            conviction_score = min(strength_diff * 5000, 1.0)
            confidence_score = h.confidence
            quality_score = min(h.confidence, 1.0)

            raw_div = self._diversification_score(h, base_exposure)
            penalty = self._factor_overlap_penalty(h, base_exposure)
            diversification_score = max(raw_div - penalty, 0.0)

            nq_score = narrative_quality.get(h.symbol, 0.5) if narrative_quality else 0.5

            raw_score = (
                conviction_weight * conviction_score +
                confidence_weight * confidence_score +
                quality_weight * quality_score +
                diversification_weight * diversification_score +
                narrative_weight * nq_score
            )

            age_cycles = self._position_age(h.symbol)
            if age_cycles > 0:
                drs_entry = self._drs_scores.get(h.symbol, 0.0)
                decay = DRS_LAMBDA_DECAY ** age_cycles
                decayed_entry = drs_entry * decay
                current_factor = 1.0 - decay
            else:
                decayed_entry = 0.0
                current_factor = 1.0

            slot_inertia = self._get_slot_inertia(h.symbol)
            final_score = (decayed_entry + raw_score * current_factor) * slot_inertia

            h.drs_score = final_score
            scored.append((final_score, h))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in scored]

    def select(self, ranked: list[DirectionHypothesis],
               open_count: int = 0) -> list[DirectionHypothesis]:
        slots_needed = min(MAX_POSITIONS - open_count, 3)
        trace = {
            "open_count": open_count,
            "slots_available": slots_needed,
            "ranked_count": len(ranked),
            "candidate_count": 0,
            "tested_combinations": 0,
            "valid_combinations": 0,
            "best_score": None,
            "rejections": [],
            "selected": [],
        }
        self.last_selection_trace = trace

        if slots_needed <= 0 or not ranked:
            return []

        live_exposure = self._net_currency_exposure()
        pool = ranked[:DRS_CANDIDATE_POOL_SIZE]
        trace["candidate_count"] = len(pool)
        best_combo = None
        best_score = -1.0

        r_max = min(slots_needed, len(pool))
        for r in range(r_max, 0, -1):
            for combo in combinations(pool, r):
                trace["tested_combinations"] += 1
                sim_exposure = Counter()
                sim_exposure.update(live_exposure)
                score = 0.0
                valid = True
                symbols_seen = set()

                for h in combo:
                    if h.symbol in symbols_seen:
                        valid = False
                        break
                    symbols_seen.add(h.symbol)
                    vec = self._currency_vector(h.symbol, h.direction)
                    for c, v in vec.items():
                        if v != 0:
                            existing = sim_exposure.get(c, 0)
                            if existing != 0 and (1 if v > 0 else -1) != (1 if existing > 0 else -1):
                                valid = False
                                break
                    if not valid:
                        trace["rejections"].append({
                            "symbols": [h.symbol for h in combo],
                            "reason": "MIXED_SIGN"
                        })
                        break
                    for c, v in vec.items():
                        sim_exposure[c] += v
                    if self._exceeds_currency_limit(sim_exposure):
                        valid = False
                        trace["rejections"].append({
                            "symbols": [h.symbol for h in combo],
                            "reason": "CURRENCY_LIMIT"
                        })
                        break
                    score += h.drs_score

                if valid:
                    trace["valid_combinations"] += 1
                    if score > best_score:
                        best_score = score
                        best_combo = combo
            if best_combo:
                break

        selected = list(best_combo) if best_combo else []
        trace["best_score"] = best_score
        trace["selected"] = [h.symbol for h in selected]

        for h in selected:
            self._drs_scores[h.symbol] = h.drs_score

        for pos in self._held_positions:
            if pos.symbol not in self._drs_scores:
                if not hasattr(pos, 'legacy_entry'):
                    pos.legacy_entry = True

        return selected

    def _diversification_score(self, hypothesis: DirectionHypothesis,
                                current_exposure: Optional[Counter] = None) -> float:
        base, quote = self._get_currencies(hypothesis.symbol)
        if current_exposure is not None and len(current_exposure) > 0:
            base_curr = abs(current_exposure.get(base, 0))
            quote_curr = abs(current_exposure.get(quote, 0))
            max_hit = max(base_curr, quote_curr)
            return max(0.0, 1.0 - max_hit * 0.3)
        return 1.0

    def _get_currencies(self, symbol: str):
        if len(symbol) == 6:
            return symbol[:3], symbol[3:]
        return symbol[:3], symbol[3:6]

    def _position_age(self, symbol: str) -> int:
        for pos in self._held_positions:
            if pos.symbol == symbol:
                return int((time.time() - pos.entry_time) / 30)
        return 0

    def _get_slot_inertia(self, symbol: str) -> float:
        for i, pos in enumerate(self._held_positions):
            if pos.symbol == symbol:
                return DRS_SLOT_INERTIA[min(i, len(DRS_SLOT_INERTIA) - 1)]
        return 1.0

    def _is_held(self, symbol: str) -> bool:
        return any(pos.symbol == symbol for pos in self._held_positions)

    def replacement_candidates(self, ranked: list[DirectionHypothesis]) -> list[dict]:
        candidates = []
        ranked_symbols = {h.symbol: h.drs_score for h in ranked}
        for pos in self._held_positions:
            current_score = ranked_symbols.get(pos.symbol, 0.0)
            for h in ranked[:DRS_CANDIDATE_POOL_SIZE]:
                if h.symbol == pos.symbol:
                    continue
                if h.symbol in self._drs_scores:
                    continue
                if h.drs_score > current_score + self._replacement_margin:
                    candidates.append({
                        "replace": pos.symbol,
                        "current_score": round(current_score, 3),
                        "candidate": h.symbol,
                        "candidate_score": round(h.drs_score, 3),
                        "margin": round(h.drs_score - current_score, 3),
                    })
                    break
        return candidates

    def record_position(self, position: PaperPosition) -> None:
        self._drs_scores[position.symbol] = position.drs_entry

    def remove_position(self, symbol: str) -> None:
        self._drs_scores.pop(symbol, None)
