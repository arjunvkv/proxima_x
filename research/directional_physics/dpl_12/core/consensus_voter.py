import numpy as np


class ConsensusVoter:
    def __init__(self, min_cohorts=2, agreement_threshold=0.5, min_confidence=0.2):
        self.min_cohorts = min_cohorts
        self.agreement_threshold = agreement_threshold
        self.min_confidence = min_confidence

    def resolve(self, votes_dict):
        if len(votes_dict) < self.min_cohorts:
            return 0.0, 0.0, {}

        total_weight = 0.0
        weighted_sum = 0.0
        directions = []
        confidences = []
        weights = []

        for name, v in votes_dict.items():
            d = v["direction"]
            conf = v["confidence"]
            w = v["weight"]
            if conf < self.min_confidence:
                continue
            weighted_sum += d * conf * w
            total_weight += conf * w
            directions.append(d)
            confidences.append(conf)
            weights.append(w)

        if total_weight < 0.01:
            return 0.0, 0.0, {"n_cohorts": 0, "agreement": 0.0}

        consensus_dir = float(np.sign(weighted_sum))
        agreement = abs(weighted_sum) / total_weight

        if agreement < self.agreement_threshold:
            return 0.0, agreement, {"n_cohorts": int(len(directions)), "agreement": round(agreement, 4)}

        n_all = len(directions)
        n_majority = sum(1 for d in directions if d == consensus_dir)

        return consensus_dir, agreement, {
            "n_cohorts": n_all,
            "majority": n_majority,
            "agreement": round(agreement, 4),
            "weighted_sum": round(weighted_sum, 4),
            "directions": directions,
            "confidences": [round(c, 4) for c in confidences]
        }

    def batch_resolve(self, all_votes_by_t):
        signals = np.zeros(len(all_votes_by_t))
        agreements = np.zeros(len(all_votes_by_t))
        n_active = np.zeros(len(all_votes_by_t))
        for t, votes in enumerate(all_votes_by_t):
            d, agreement, meta = self.resolve(votes)
            signals[t] = d
            agreements[t] = agreement
            n_active[t] = meta.get("n_cohorts", 0) if isinstance(meta, dict) else 0
        return signals, agreements, n_active
