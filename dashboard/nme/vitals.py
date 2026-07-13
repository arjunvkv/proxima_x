from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .model import NMEViewModel
from .formatter import pb, G, A, C, D, W, R


def render(model: NMEViewModel) -> Panel:
    m = model.metrics

    field_order = [
        ("conviction", "GRW", G, "pct"),
        ("velocity", "VEL", A, "raw1"),
        ("acceleration", "ACC", A, "raw1"),
        ("leadership_stability", "STAB", G, "pct"),
        ("rank_churn", "CHRN", R, "pct"),
        ("propagation", "PROP", A, "raw2"),
        ("der_improvement", "DER", A, "raw2"),
        ("cohesion", "COH", G, "raw2"),
        ("expression_score", "PES", G, "raw2"),
        ("opportunity_density", "DENS", A, "raw2"),
    ]

    vitals = Table.grid(padding=(0, 0))
    for _ in range(4 * 3):
        vitals.add_column()

    def disp_val(key, style_fmt):
        v = m.get(key)
        if v is None:
            return "--"
        if style_fmt == "pct":
            return f"{v * 100:.0f}%"
        if style_fmt == "raw1":
            return f"{v:.1f}"
        return f"{v:.2f}"

    def metric_cells(key, label, clr, style_fmt):
        v = m.get(key)
        if v is None:
            return [Text(f" {label}", style=D), pb(0, w=4, clr=D),
                    Text("--", style=D), Text("  ", style=D)]
        return [
            Text(f" {label}", style=D),
            pb(v, w=4, clr=clr),
            Text(disp_val(key, style_fmt), style=f"bold {clr}"),
            Text("  ", style=D),
        ]

    rows_data = [
        field_order[0:4],
        field_order[4:8],
        field_order[8:10],
    ]

    for i, group in enumerate(rows_data):
        row = []
        for key, label, clr, style_fmt in group:
            row.extend(metric_cells(key, label, clr, style_fmt))
        while len(row) < 4 * 3:
            row.append(Text("", style=D))
        vitals.add_row(*row)

    # PHASE as separate row
    phase_row = [Text(" PHASE", style=D), Text("  ", style=D)] + [
        Text(model.phase, style=f"bold {C}"),
        Text("", style=D),
    ]
    while len(phase_row) < 4 * 3:
        phase_row.append(Text("", style=D))
    vitals.add_row(*phase_row)

    return Panel(vitals, box=box.SQUARE, title=" Narrative Vitals ", border_style=R, padding=(0, 1))
