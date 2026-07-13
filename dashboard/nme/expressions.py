from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .model import NMEViewModel
from .formatter import pb, G, A, D, W, R


def render(model: NMEViewModel) -> Panel:
    t = Table(box=box.MINIMAL, header_style=f"bold {D}", show_header=True, padding=(0, 1))
    t.add_column("PAIR", style=f"bold {W}")
    for h in ("PES", "DER", "BURST", "STATUS"):
        t.add_column()

    if model.expressions:
        for ex in model.expressions:
            pair = ex.get("pair", "--")
            pes = ex.get("pes")
            der = ex.get("der")
            burst = ex.get("burst")
            match = ex.get("match", "~")

            pes_v = pes if pes is not None else 0
            der_v = der if der is not None else 0
            burst_v = burst if burst is not None else 0

            is_strong = match == "strong"
            st = G if is_strong else A
            status_text = " STRONG" if is_strong else "  WEAK"

            t.add_row(
                pair,
                pb(pes_v, w=4, clr=G),
                pb(der_v, w=4, clr=A),
                pb(burst_v, w=4, clr=R),
                Text(status_text, style=f"bold {st}"),
            )
    else:
        t.add_row("--", Text("--", style=D), Text("--", style=D),
                  Text("--", style=D), Text("--", style=D))

    return Panel(t, box=box.SQUARE, title=" Expressions ", border_style=G, padding=(0, 1))
