from .model import NMEViewModel
from .formatter import bar, fmt_pct, fmt_val, fmt_delta

W = 70

def render(model: NMEViewModel) -> str:
    m = model.metrics
    lines = []
    lines.append(f"║{' ' * 68}║")
    lines.append(f"║  ◄ NARRATIVE VITALS{' ' * 47}║")
    lines.append(f"║{' ' * 68}║")

    age_v = model.age
    age_bar = bar(min(age_v / 50, 1) if age_v > 0 else 0, width=3)
    age_s = f"{age_v}c" if age_v > 0 else "--"

    grw = m.get("conviction")
    grw_bar = bar(grw, width=7)
    grw_s = fmt_pct(grw) if grw is not None else "--"
    grw_d = fmt_delta(grw)

    vel = m.get("velocity")
    vel_bar = bar(vel, width=4)
    vel_s = fmt_val(vel, 1) if vel is not None else "--"
    vel_d = fmt_delta(vel)

    row1 = f"   AGE  {age_bar}  {age_s:<4s}   GRW  {grw_bar}  {grw_s:<5s}{grw_d}   VEL  {vel_bar}  {vel_s:<5s}{vel_d}"
    lines.append(f"║{row1:<66}║")

    acc = m.get("acceleration")
    acc_bar = bar(acc, width=1)
    acc_s = fmt_val(acc, 1) if acc is not None else "--"
    acc_d = fmt_delta(acc)

    stab = m.get("leadership_stability")
    stab_bar = bar(stab, width=10)
    stab_s = fmt_pct(stab) if stab is not None else "--"
    stab_d = fmt_delta(stab)

    chrn = m.get("rank_churn")
    chrn_bar = bar(1 - chrn if chrn is not None else None, width=1)
    chrn_s = fmt_pct(chrn) if chrn is not None else "--"
    chrn_d = fmt_delta(chrn)

    row2 = f"   ACC  {acc_bar}  {acc_s:<5s}{acc_d}   STAB  {stab_bar}  {stab_s:<5s}{stab_d}   CHRN  {chrn_bar}  {chrn_s:<5s}{chrn_d}"
    lines.append(f"║{row2:<66}║")

    prop = m.get("propagation")
    prop_bar = bar(prop, width=8)
    prop_s = fmt_val(prop) if prop is not None else "--"
    prop_d = fmt_delta(prop)

    der = m.get("der_improvement")
    der_bar = bar(der, width=9)
    der_s = fmt_val(der) if der is not None else "--"
    der_d = fmt_delta(der)

    row3 = f"   PROP  {prop_bar}  {prop_s:<5s}{prop_d}   DER  {der_bar}  {der_s:<5s}{der_d}"
    lines.append(f"║{row3:<66}║")

    coh = m.get("cohesion")
    coh_bar = bar(coh, width=8)
    coh_s = fmt_val(coh) if coh is not None else "--"
    coh_d = fmt_delta(coh)

    pes = m.get("expression_score")
    pes_bar = bar(pes, width=9)
    pes_s = fmt_val(pes) if pes is not None else "--"
    pes_d = fmt_delta(pes)

    row4 = f"   COH  {coh_bar}  {coh_s:<5s}{coh_d}   PES  {pes_bar}  {pes_s:<5s}{pes_d}   PHASE  {model.phase}"
    lines.append(f"║{row4:<66}║")

    dens = m.get("opportunity_density")
    dens_bar = bar(dens, width=8)
    dens_s = fmt_val(dens) if dens is not None else "--"
    dens_d = fmt_delta(dens)

    row5 = f"   DENS  {dens_bar}  {dens_s:<5s}{dens_d}"
    lines.append(f"║{row5:<66}║")

    lines.append(f"║{' ' * 68}║")
    return "\n".join(lines)
