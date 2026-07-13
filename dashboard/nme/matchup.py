from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich import box

from .model import NMEViewModel
from .formatter import pb, G, A, C, D, W, R


def render(model: NMEViewModel) -> Columns:
    def ccy_panel(name, val, delta, is_leader):
        clr = G if val >= 0 else R
        label = "LEADER" if is_leader else "OPPONENT"
        bclr = G if is_leader else R
        t = Table.grid(padding=0)
        t.add_column()
        t.add_row(
            Text(f" {name} ", style=f"bold {W}"),
            Text(f" {label}", style=bclr),
        )
        t.add_row(
            pb(abs(val), mx=0.001, w=16, clr=clr),
            Text(f" {val:+.5f} ", style=W),
            Text("▲" if val >= 0 else "▼", style=f"bold {clr}"),
        )
        if delta is not None:
            d_sym = "▲" if delta >= 0 else "▼"
            d_clr = G if delta >= 0 else R
            t.add_row(Text(f"  Δ {delta:+.5f} {d_sym}", style=d_clr))
        else:
            t.add_row(Text(""))
        return Panel(t, box=box.SQUARE, border_style=bclr, padding=(0, 1), width=41)

    panels = [ccy_panel(model.leader, model.leader_strength, model.leader_delta, True)]

    if model.opponent_strengths:
        first_opp = list(model.opponent_strengths.items())[0]
        panels.append(ccy_panel(first_opp[0], first_opp[1], None, False))
    else:
        panels.append(Panel(Text("  -- no opponent --", style=D),
                            box=box.SQUARE, border_style=R, padding=(0, 1), width=41))

    return Columns(panels, equal=True, expand=True)
