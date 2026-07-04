def system_objective(gt_signal, sy_signal, pnl_proxy, stre_state):
    edge_preservation = gt_signal.get("expected_move", 0) * gt_signal.get("p_cont", 0)
    gt_corr = stre_state.get("gt_corr", 0.0)
    sy_corr = stre_state.get("sy_corr", 0.0)
    stas = gt_corr - sy_corr
    sof = edge_preservation * max(0.0, (1.0 - abs(stas)))
    if stas < -0.1:
        sof *= 0.5
    return {
        "sof": round(sof, 6),
        "edge_preservation": round(edge_preservation, 6),
        "stas": round(stas, 4),
    }


def phase2_activation(stre_state):
    gt_corr = stre_state.get("gt_corr", 0.0)
    sy_corr = stre_state.get("sy_corr", 0.0)
    stas = gt_corr - sy_corr
    enabled = gt_corr > sy_corr and abs(stas) > 0.05
    return {
        "phase2_enabled": enabled,
        "gt_corr": gt_corr,
        "sy_corr": sy_corr,
        "stas": stas,
    }


def evaluate_system(gt_signal, sy_signal, pnl_proxy, stre_state):
    sof_result = system_objective(gt_signal, sy_signal, pnl_proxy, stre_state)
    phase2_gate = phase2_activation(stre_state)
    exec_eff = round(pnl_proxy * sof_result["edge_preservation"], 6)
    return {
        "SOF": sof_result["sof"],
        "edge_preservation": sof_result["edge_preservation"],
        "stas": sof_result["stas"],
        "execution_efficiency": exec_eff,
        "phase2_enabled": phase2_gate["phase2_enabled"],
    }
