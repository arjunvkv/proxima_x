from io import StringIO
from typing import Optional

from rich.console import Console

from .nme.builder import build_model
from .nme.header import render as render_header
from .nme.matchup import render as render_matchup
from .nme.radar import render as render_radar
from .nme.layers import render as render_layers
from .nme.vitals import render as render_vitals
from .nme.expressions import render as render_expressions
from .nme.formatter import D, G, A, C, W


class NMEDashboard:
    def __init__(self, width: int = 84):
        self.width = width

    def render(self, narrative_state: Optional[dict],
               market_data: Optional[dict] = None) -> str:
        model = build_model(narrative_state, market_data)

        buf = StringIO()
        c = Console(file=buf, width=self.width, force_terminal=True,
                    color_system="truecolor", legacy_windows=False,
                    highlight=False)

        c.print(render_header(model))
        c.print(render_matchup(model))
        c.print(render_radar(model))
        c.print(render_layers(model))
        c.print(render_vitals(model))
        c.print(render_expressions(model))

        # Footer
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich import box
        ft = Table.grid(padding=0)
        ft.add_column()
        ft.add_row(
            Text(" CONF ", style=f"dim {D}"),
            Text(f"{model.nmi:.2f}", style=f"bold {G}"),
            Text("  |  ", style=f"dim {D}"),
            Text("EXPOSURE ", style=f"dim {D}"), Text("--", style=A),
            Text("  |  ", style=f"dim {D}"),
            Text("OPP_REM ", style=f"dim {D}"), Text("--", style=A),
            Text("  |  ", style=f"dim {D}"),
            Text("RESEARCH ", style=f"dim {D}"),
            Text("ACTIVE" if model.active else ("BUILD" if model.leader.startswith("?") else "IDLE"),
                 style=f"bold {G}" if model.active else (f"bold {A}" if model.leader.startswith("?") else f"dim {D}")),
        )
        c.print(Panel(ft, box=box.SQUARE, border_style=G, padding=(0, 1)))

        return buf.getvalue()
