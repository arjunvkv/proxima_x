from .model import NMEViewModel
from .formatter import fmt_val

W = 70

def render(model: NMEViewModel) -> str:
    lines = []
    lines.append(f"║{' ' * 68}║")
    lines.append(f"║  ◄ EXPRESSIONS{' ' * 52}║")
    lines.append(f"║{' ' * 68}║")

    if model.expressions:
        for ex in model.expressions:
            pair = ex.get("pair", "--")
            pes = ex.get("pes")
            der = ex.get("der")
            burst = ex.get("burst")
            match = ex.get("match", "~")
            pes_s = fmt_val(pes) if pes is not None else "--"
            der_s = fmt_val(der) if der is not None else "--"
            burst_s = fmt_val(burst) if burst is not None else "--"
            status = "✓ STRONG" if match == "strong" else "~ WEAK"
            lines.append(f"║   {pair:<6s}  PES {pes_s:<5s}  DER {der_s:<5s}  BURST {burst_s:<5s}  {status:<20s}║")
    else:
        lines.append(f"║   {'-- no expressions --':<66}║")

    lines.append(f"║{' ' * 68}║")
    return "\n".join(lines)
