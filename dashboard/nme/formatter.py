from rich.progress_bar import ProgressBar

G = "#10b981"
R = "#ef4444"
A = "#f59e0b"
C = "#06b6d4"
W = "#e2e8f0"
D = "#64748b"


def pb(val, mx=1.0, w=6, clr=G):
    if val is None:
        return ProgressBar(completed=0, total=1, width=w, complete_style=clr, style=D)
    return ProgressBar(completed=val, total=mx, width=w, complete_style=clr, style=D)
