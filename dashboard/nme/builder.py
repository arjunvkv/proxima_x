from typing import Optional
from .model import NMEViewModel


def build_model(narrative_state: Optional[dict], market_data: Optional[dict] = None) -> NMEViewModel:
    model = NMEViewModel()
    if market_data:
        model.cycle = market_data.get("cycle", 0)
        model.tick_quality = market_data.get("tick_quality", 0)
        model.graph_quality = market_data.get("graph_quality", 0)
        model.reliability = market_data.get("reliability", {})

    if narrative_state is None or not narrative_state.get("active"):
        model.active = False
        model.phase = "BUILDING"
        model.nmi = 0.0
        model.leader = "--"
        if market_data:
            strengths = market_data.get("currency_strengths", {})
            if strengths:
                top = max(strengths, key=lambda c: abs(strengths[c]))
                if abs(strengths[top]) > 0:
                    model.leader = f"?{top}"
                    model.leader_strength = strengths[top]
                    model.direction = 1 if strengths[top] > 0 else -1
                    model.nmi = min(abs(strengths[top]) * 5000, 0.3)
                    sorted_ccy = sorted(strengths.items(), key=lambda x: abs(x[1]), reverse=True)
                    model.opponent_strengths = {}
                    for i, (c, v) in enumerate(sorted_ccy):
                        if c == top:
                            continue
                        if len(model.opponent_strengths) >= 2:
                            break
                        model.opponent_strengths[c] = v
                    model.research_layers = {
                        "WLS": min(max(abs(v) for v in strengths.values()) * 1000, 1.0),
                        "GRAPH": market_data.get("graph_quality", 0),
                    }
                    bursts = market_data.get("currency_bursts", {})
                    if bursts:
                        model.research_layers["BURST"] = max(abs(v) for v in bursts.values())
                    der = market_data.get("currency_der", {})
                    if der:
                        model.research_layers["DER"] = max(abs(v) for v in der.values())
        return model

    model.active = True
    model.leader = narrative_state.get("leader", "--")
    model.phase = narrative_state.get("phase", "--")
    model.nmi = narrative_state.get("nmi", 0.0)
    model.age = narrative_state.get("age", 0)
    model.last_event = narrative_state.get("last_event")
    model.leader_strength = narrative_state.get("current_strength", 0.0)
    model.leader_delta = narrative_state.get("strength_delta", 0.0)
    model.direction = narrative_state.get("direction", 0)

    opponents = narrative_state.get("opponents", [])
    if market_data:
        strengths = market_data.get("currency_strengths", {})
        model.opponent_strengths = {c: strengths.get(c, 0) for c in opponents if c in strengths}

    model.metrics = narrative_state.get("metrics", {})

    raw_expressions = narrative_state.get("expressions", [])
    if market_data:
        strengths = market_data.get("currency_strengths", {})
        bursts = market_data.get("currency_bursts", {})
        der = market_data.get("currency_der", {})
        model.expressions = []
        for pair in raw_expressions:
            base = pair[:3]
            quote = pair[3:]
            b_s = strengths.get(base, 0)
            q_s = strengths.get(quote, 0)
            spread = abs(b_s - q_s)
            pes = min(spread / 0.001, 1.0) if spread > 0 else 0
            model.expressions.append({
                "pair": pair,
                "pes": round(pes, 2),
                "der": round(der.get(pair, 0), 2),
                "burst": round(bursts.get(pair, 0), 2),
                "match": "strong" if pes > 0.6 else "weak",
            })

        if market_data:
            model.research_layers = {}
            if narrative_state.get("nmi") is not None:
                model.research_layers["NMI"] = narrative_state["nmi"]
            strengths = market_data.get("currency_strengths", {})
            if strengths:
                max_wls = max(abs(v) for v in strengths.values())
                model.research_layers["WLS"] = min(max_wls * 1000, 1.0)
        bursts = market_data.get("currency_bursts", {})
        if bursts:
            model.research_layers["BURST"] = max(abs(v) for v in bursts.values())
        der = market_data.get("currency_der", {})
        if der:
            model.research_layers["DER"] = max(abs(v) for v in der.values())
        model.research_layers["GRAPH"] = market_data.get("graph_quality", 0)
        reliabilities = market_data.get("reliability", {})
        if reliabilities:
            model.research_layers["C_REL"] = sum(reliabilities.values()) / max(len(reliabilities), 1)

    return model
