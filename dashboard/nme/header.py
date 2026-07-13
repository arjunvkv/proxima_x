from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .model import NMEViewModel
from .formatter import pb, G, A, C, D, W, R


def render(model: NMEViewModel) -> Panel:
    status = "BUILD" if model.leader.startswith("?") else ("IDLE" if not model.active else "LIVE")
    status_clr = A if status == "BUILD" else (D if status == "IDLE" else G)

    t = Table.grid(padding=0)
    t.add_column()
    t.add_row(
        Text(" PROXIMA NME  ", style=f"bold {W}"),
        Text(f"CYCLE {model.cycle} ", style=C),
        Text("●", style=f"bold {status_clr}"),
        Text(f" {status}  ", style=status_clr),
        Text("NMI ", style=D),
        pb(model.nmi, w=5, clr=G),
        Text(f" {model.nmi:.2f} ", style=G),
        Text("▲ " if model.nmi > 0.02 else ("▼ " if model.nmi < -0.02 else "→ "), style=f"bold {G}" if model.nmi > 0.02 else (f"bold {R}" if model.nmi < -0.02 else D)),
        Text("AGE ", style=D),
        pb(min(model.age / 50, 1) if model.age > 0 else 0, w=4, clr=A),
        Text(f" {model.age}c" if model.age > 0 else " --", style=W),
    )
    return Panel(t, box=box.SQUARE, border_style=G, padding=(0, 1))
