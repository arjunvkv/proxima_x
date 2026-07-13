def serialize_narrative(narrative) -> dict:
    if narrative is None:
        return {"active": False}

    m = narrative.metrics

    def _val(v, decimals=2):
        if v is None:
            return None
        return round(v, decimals)

    return {
        "active": narrative.active,
        "leader": narrative.identity.leader,
        "direction": narrative.identity.direction,
        "opponents": list(narrative.identity.opponents),
        "phase": narrative.phase.value,
        "nmi": round(narrative.nmi, 2),
        "rns": round(narrative.rns, 2),
        "age": narrative.age,
        "birth_cycle": narrative.birth_cycle,
        "current_strength": round(narrative.current_strength, 5),
        "strength_delta": round(narrative.strength_delta, 5),
        "peak_strength": round(narrative.peak_strength, 5),
        "last_event": narrative.last_event.value if narrative.last_event else None,
        "metrics": {
            "conviction": _val(m.conviction),
            "velocity": _val(m.velocity),
            "acceleration": _val(m.acceleration),
            "leadership_stability": _val(m.leadership_stability),
            "rank_churn": _val(m.rank_churn),
            "propagation": _val(m.propagation),
            "der_improvement": _val(m.der_improvement),
            "cohesion": _val(m.cohesion),
            "expression_score": _val(m.expression_score),
            "opportunity_density": _val(m.opportunity_density),
        },
        "expressions": narrative.expressions,
    }
