from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich import box

from .model import NMEViewModel
from .formatter import G, A, C, D, W, R


def render(model: NMEViewModel) -> Panel:
    all_currencies = {}
    if model.leader and model.leader != "--":
        all_currencies[model.leader] = model.leader_strength
    for ccy, val in model.opponent_strengths.items():
        if ccy != model.leader:
            all_currencies[ccy] = val

    sorted_ccy = sorted(all_currencies.items(), key=lambda x: abs(x[1]), reverse=True)

    tree = Tree("  Currency Radar", guide_style=D)

    if sorted_ccy:
        top_name, top_val = sorted_ccy[0]
        top_clr = G if top_val >= 0 else R
        top = tree.add(
            Text(top_name.replace("?", ""), style=f"bold {W}")
            + Text(" ▲" if top_val >= 0 else " ▼", style=f"bold {top_clr}")
            + Text(f" {top_val:+.5f}", style=W)
        )

        for ccy_name, ccy_val in sorted_ccy[1:]:
            ccy_clr = G if ccy_val >= 0 else R
            top.add(
                Text(ccy_name, style=W)
                + Text(" ▲" if ccy_val >= 0 else " ▼", style=f"bold {ccy_clr}")
                + Text(f" {ccy_val:+.5f}", style=W)
            )
    else:
        tree.add(Text("  -- no data --", style=D))

    return Panel(tree, box=box.ROUNDED, title="Radar", border_style=G, padding=(0, 1))
