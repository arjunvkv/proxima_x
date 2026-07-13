from .model import NMEViewModel
from .formatter import fmt_val, status_dot

W = 70

def render(model: NMEViewModel) -> str:
    lines = []
    lines.append(f"║{' ' * 68}║")
    lines.append(f"║  ◄ RESEARCH LAYERS{' ' * 48}║")
    lines.append(f"║{' ' * 68}║")

    layers = model.research_layers
    if layers:
        parts = []
        for name, val in layers.items():
            dot = status_dot(val)
            fv = fmt_val(val, 2)
            parts.append(f"{dot} {name} {fv}")
        line = "  ".join(parts)
        lines.append(f"║  {line:<66}║")
    else:
        lines.append(f"║  {'-- no data --':<66}║")

    lines.append(f"║{' ' * 68}║")
    return "\n".join(lines)
