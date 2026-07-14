from .state import NarrativePhase, NarrativeState


def narrative_alignment(symbol: str, narrative: NarrativeState | None) -> float:
    """Overlay #1: Confidence modifier for trades aligned with dominant narrative.

    Returns multiplier in range [0.85, 1.15].
    Pure function — no side effects.
    """
    if narrative is None or not narrative.active:
        return 1.0

    leader = narrative.identity.leader
    if leader not in symbol:
        return 1.0

    phase = narrative.phase
    nmi = narrative.nmi

    if phase == NarrativePhase.EMERGING:
        return 1.05
    elif phase == NarrativePhase.GROWING:
        return 1.15
    elif phase == NarrativePhase.MATURE:
        return 1.05
    elif phase == NarrativePhase.DECAYING:
        return 0.90
    elif phase == NarrativePhase.EXHAUSTED:
        return 0.80

    return 1.0


def maturity_penalty(symbol: str, narrative: NarrativeState | None) -> float:
    """Overlay #2: Confidence penalty for late-stage narratives.

    Returns multiplier in range [0.70, 1.0].
    Applied after bar state alignment, before DRS ranking.
    Pure function — no side effects.
    """
    if narrative is None or not narrative.active:
        return 1.0

    leader = narrative.identity.leader
    if leader not in symbol:
        return 1.0

    nmi = narrative.nmi
    phase = narrative.phase
    age = narrative.age

    if age >= 5:
        return 0.70
    elif nmi > 0.90:
        return 0.75
    elif phase == NarrativePhase.DECAYING:
        return 0.85

    return 1.0


def narrative_quality(narrative: NarrativeState | None) -> float:
    """Overlay #3: Narrative quality score for DRS weighting.

    Returns score in range [0.0, 1.0].
    Pure function — no side effects.
    """
    if narrative is None or not narrative.active:
        return 0.5

    phase = narrative.phase
    nmi = narrative.nmi

    phase_scores = {
        NarrativePhase.EMERGING: 0.8,
        NarrativePhase.GROWING: 1.0,
        NarrativePhase.MATURE: 0.9,
        NarrativePhase.DECAYING: 0.6,
        NarrativePhase.EXHAUSTED: 0.2,
    }

    stability = 1.0 - narrative.metrics.rank_churn if narrative.metrics.rank_churn is not None else 0.5
    stability = max(0.0, min(1.0, stability))

    ps = phase_scores.get(phase, 0.5)
    return ps * nmi * stability


def narrative_health_score(narrative: NarrativeState | None) -> float:
    """Aggregate narrative health [0.0, 1.0] for dynamic profit targeting.
    Pure function — no side effects.
    """
    if narrative is None or not narrative.active:
        return 0.7

    phase_scores = {
        NarrativePhase.EMERGING: 0.7,
        NarrativePhase.GROWING: 1.0,
        NarrativePhase.MATURE: 0.75,
        NarrativePhase.DECAYING: 0.35,
        NarrativePhase.EXHAUSTED: 0.1,
    }
    pf = phase_scores.get(narrative.phase, 0.5)

    strength_momentum = 0.5
    if narrative.strength_delta is not None:
        strength_momentum = 0.5 + narrative.strength_delta * 250
    strength_momentum = max(0.0, min(1.0, strength_momentum))

    nmi_factor = 1.0 - narrative.nmi

    age_factor = max(0.0, 1.0 - narrative.age / 150)

    return max(0.05, min(1.0, pf * 0.45 + strength_momentum * 0.25 + nmi_factor * 0.20 + age_factor * 0.10))
