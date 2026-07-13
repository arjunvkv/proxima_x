import math


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def geometric_mean(values) -> float:
    valid = [v for v in values if v is not None and v > 0]
    if not valid:
        return 0.0
    product = 1.0
    for v in valid:
        product *= v
    return product ** (1.0 / len(valid))


class MaturityCalculator:
    def calculate(self, narrative) -> float:
        m = narrative.metrics

        foundation_components = [m.conviction, m.leadership_stability, m.cohesion]
        momentum_components = [m.velocity, m.acceleration, m.der_improvement, m.propagation]
        expression_components = [m.expression_score, m.opportunity_density]

        foundation = geometric_mean(foundation_components)
        momentum = geometric_mean(momentum_components) if any(v is not None for v in [m.velocity, m.acceleration, m.der_improvement, m.propagation]) else 0.5
        expression = geometric_mean(expression_components) if any(v is not None for v in [m.expression_score, m.opportunity_density]) else 0.5

        rns = foundation * momentum * expression
        age_factor = min(narrative.age / 50.0, 1.0)
        churn_penalty = (m.rank_churn or 0) * 0.5
        decay = narrative.metrics.der_improvement
        decay_penalty = (1.0 - (decay or 0.5)) * 0.3

        nmi = sigmoid(age_factor + rns - churn_penalty - decay_penalty)
        nmi = max(0.0, min(1.0, nmi))

        narrative.rns = rns
        narrative.nmi = nmi
        return nmi
