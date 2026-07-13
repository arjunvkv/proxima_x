from typing import Optional
from .nme.model import NMEViewModel
from .nme.builder import build_model
from .nme.header import render as render_header
from .nme.matchup import render as render_matchup
from .nme.radar import render as render_radar
from .nme.layers import render as render_layers
from .nme.vitals import render as render_vitals
from .nme.expressions import render as render_expressions


class NMEDashboard:
    def __init__(self, width: int = 70):
        self.width = width

    def render(self, narrative_state: Optional[dict], market_data: Optional[dict] = None) -> str:
        model = build_model(narrative_state, market_data)
        sep = f"╠{'═' * (self.width - 2)}╣"
        bottom = f"╠{'═' * (self.width - 2)}╣"
        status = "ACTIVE" if model.active else ("BUILDING" if model.leader.startswith("?") else "IDLE")
        footer_conf = f"║  CONF {self._val(model.nmi, 2)}  EXPOSURE --  OPP_REM --  RESEARCH {status:<9s}        ║"
        footer_end = f"╚{'═' * (self.width - 2)}╝"

        sections = [
            render_header(model),
            sep,
            render_matchup(model),
            sep,
            render_radar(model),
            sep,
            render_layers(model),
            sep,
            render_vitals(model),
            sep,
            render_expressions(model),
            bottom,
            footer_conf,
            footer_end,
        ]
        return "\n".join(sections)

    def _val(self, v, d=2):
        if v is None:
            return "--"
        return f"{v:.{d}f}"
