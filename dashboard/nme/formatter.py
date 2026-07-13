def fmt_val(v, decimals=2):
    if v is None:
        return "--"
    return f"{v:.{decimals}f}"

def fmt_pct(v):
    if v is None:
        return "--"
    return f"{v*100:.0f}%"

def fmt_delta(v):
    if v is None:
        return " →"
    if v > 0:
        return " ▲"
    if v < 0:
        return " ▼"
    return " →"

def bar(v, max_v=1.0, width=10):
    if v is None or v <= 0:
        return " " * width
    ratio = min(v / max_v, 1.0)
    filled = max(1, int(ratio * width)) if ratio > 0 else 0
    filled = min(filled, width)
    return "▰" * filled + " " * (width - filled)

def status_dot(v, threshold_green=0.7, threshold_amber=0.4):
    if v is None:
        return "○"
    if v >= threshold_green:
        return "●"
    if v >= threshold_amber:
        return "◉"
    return "○"

def trajectory_symbol(v):
    if v is None:
        return "→"
    if v > 0.02:
        return "▲"
    if v < -0.02:
        return "▼"
    return "→"
