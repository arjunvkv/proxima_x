def build_gt_vector(event):
    if not isinstance(event, dict):
        return [0] * 9
    return [
        float(event.get("oss_ev", 0)),
        float(event.get("oss_conf", 0)),
        float(event.get("entropy", 0)),
        float(event.get("ecdf_rank", 0)),
        float(event.get("at_rank", 0)),
        float(event.get("spread", 0)),
        float(event.get("research_p_cont", 0)),
        float(event.get("research_drift", 0)),
        float(event.get("exec_drift", 0)),
    ]


def build_sy_vector(state_entry):
    if not isinstance(state_entry, dict):
        return [0] * 3
    return [
        float(state_entry.get("suppression_delta", 0)),
        float(state_entry.get("lkg_similarity_score", 0)),
        float(state_entry.get("raw_conviction", 0.5)),
    ]
