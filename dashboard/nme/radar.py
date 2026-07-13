from .model import NMEViewModel

W = 70

def render(model: NMEViewModel) -> str:
    lines = []
    lines.append(f"║{' ' * 68}║")
    lines.append(f"║  ◄ CURRENCY RADAR{' ' * 50}║")
    lines.append(f"║{' ' * 68}║")

    all_currencies = {}
    if model.leader and model.leader != "--":
        all_currencies[model.leader] = model.leader_strength
    for ccy, val in model.opponent_strengths.items():
        if ccy != model.leader:
            all_currencies[ccy] = val

    sorted_ccy = sorted(all_currencies.items(), key=lambda x: abs(x[1]), reverse=True)

    if sorted_ccy:
        top = sorted_ccy[0]
        top_dir = "▲" if top[1] > 0 else "▼"
        top_text = f"    {top_dir} {top[0]} ({top[1]:+.5f}) {top_dir}"
        lines.append(f"║{top_text:^68}║")
        slash_text = "      /    \\"
        lines.append(f"║{slash_text:^68}║")

        if len(sorted_ccy) >= 2:
            left = sorted_ccy[1]
            l_dir = "▲" if left[1] > 0 else "▼"
            rest_text = f"  {left[0]} {l_dir}  ◄──────  {top[0]}"
            if len(sorted_ccy) >= 3:
                right = sorted_ccy[2]
                r_dir = "▲" if right[1] > 0 else "▼"
                rest_text += f"  ──────►  {r_dir} {right[0]}"
            lines.append(f"║{rest_text:^68}║")

            rest = sorted_ccy[3:]
            if rest:
                rest_str = "  ".join(f"{c} ({v:+.5f})" for c, v in rest)
                lines.append(f"║{'  ' + rest_str:^68}║")

    lines.append(f"║{' ' * 68}║")
    return "\n".join(lines)
