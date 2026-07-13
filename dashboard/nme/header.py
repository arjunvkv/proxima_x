from .model import NMEViewModel
from .formatter import trajectory_symbol

W = 70

def render(model: NMEViewModel) -> str:
    traj = trajectory_symbol(model.nmi)
    status = "BUILD" if model.leader.startswith("?") else ("IDLE" if not model.active else "LIVE ")
    line1 = f"║  PROXIMA NME{' ' * 18}CYCLE {model.cycle:<5d}          ● {status}  {model.nmi:<5.2f}  {traj}  ║"
    line2 = f"║{' ' * 68}║"
    return f"{line1}\n{line2}"
