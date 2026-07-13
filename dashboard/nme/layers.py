from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .model import NMEViewModel
from .formatter import pb, A, D


def render(model: NMEViewModel) -> Panel:
    layers = model.research_layers
    t = Table.grid(padding=(0, 0))

    if layers:
        for _ in range(len(layers) * 2):
            t.add_column()
        row = []
        for name, val in layers.items():
            hc = {"NMI": "#10b981", "WLS": "#f59e0b", "BURST": "#ef4444",
                  "DER": "#f59e0b", "GRAPH": "#10b981", "C_REL": "#ef4444"}
            clr = hc.get(name, A)
            row.append(Text(f" {name}", style=D))
            row.append(pb(val, w=4, clr=clr))
        t.add_row(*row)
    else:
        t.add_column()
        t.add_row(Text(" -- no data --", style=D))

    return Panel(t, box=box.SQUARE, title=" Research Layers ", border_style=A, padding=(0, 1))
