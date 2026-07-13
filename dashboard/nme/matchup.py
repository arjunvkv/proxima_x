from .model import NMEViewModel
from .formatter import bar

W = 70

def render(model: NMEViewModel) -> str:
    lines = []
    lines.append(f"║{' ' * 68}║")
    lines.append(f"║  ◄ NARRATIVE MATCHUP{' ' * 47}║")
    lines.append(f"║{' ' * 68}║")

    leader = model.leader
    l_str = model.leader_strength
    l_bar = bar(abs(l_str), max_v=0.001, width=16)
    l_dir = "▲" if model.direction > 0 else "▼"
    if model.active:
        l_delta = f"+{model.leader_delta:.5f}" if model.leader_delta >= 0 else f"{model.leader_delta:.5f}"
        lines.append(f"║  {leader} {l_bar} {l_str:+.5f}  {l_dir}   Δ {l_delta:<10s}      ║")
    else:
        tag = "CANDIDATE" if leader.startswith("?") else "NO LEADER "
        lines.append(f"║  {leader[1:] if leader.startswith('?') else leader} {l_bar} {l_str:+.5f}  {l_dir}   {tag:<16s}      ║")

    if model.opponent_strengths:
        first_opp = list(model.opponent_strengths.items())[0]
        o_ccy, o_val = first_opp
        o_bar = bar(abs(o_val), max_v=0.001, width=16)
        o_dir = "▲" if o_val > 0 else "▼"
        lines.append(f"║  {o_ccy} {o_bar} {o_val:+.5f}  {o_dir}                     ║")
    else:
        lines.append(f"║{' ' * 68}║")

    lines.append(f"║{' ' * 68}║")
    return "\n".join(lines)
