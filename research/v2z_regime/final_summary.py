"""Comprehensive trade count vs PnL summary."""
import re, numpy as np

path = "C:\\Users\\Arjun Sasi\\AppData\\Roaming\\MetaQuotes\\Tester\\D0E8209F77C8CF37AD8BF550E51FF075\\Agent-127.0.0.1-3000\\logs\\20260726.log"
with open(path, "rb") as f:
    t = f.read().decode("utf-16-le", errors="replace")
lines = t.split("\n")
fbs = [i for i,l in enumerate(lines) if "final balance" in l]

labels = [
    "Fwd EURAUD unfiltered",
    "Fwd EURAUD unfiltered",
    "Fwd EURAUD unfiltered",
    "Fwd EURAUD unfiltered",
    "OOS EURAUD unfiltered",
    "Fwd +body+sm (no tv)",
    "Fwd +tv (static filter)",
    "OOS +tv (static filter)",
    "GBPAUD unfiltered",
    "GBPAUD +body+sm",
    "Fwd dynamic regime",
    "Fwd vol2.0p",
    "OOS vol2.0p",
    "OOS vol2.0p (dup?)",
    "Fwd vol5.0p (best)",
    "OOS vol5.0p",
    "Fwd micro scalp M1",
    "Fwd micro scalp M5",
    "Fwd z=2.0 M1",
    "Fwd EURUSD z=3.5",
    "Fwd hybrid M5",
]
labels = labels[:len(fbs)]

print(f"{'Run':<4s} {'Config':<35s} {'Trades':<7s} {'Gross':<10s} {'Net@$7.50':<12s} {'/day':<7s}")
print("-"*80)
for ri in range(len(fbs)):
    lbl = labels[ri] if ri < len(labels) else f"Run {ri}"
    start = fbs[ri-1]+1 if ri > 0 else 0
    run = lines[start:fbs[ri]]
    
    # Count opens
    hy_opens = len([l for l in run if "HYBRID OPEN" in l])
    mi_opens = len([l for l in run if "MICRO OPEN" in l])
    z_opens = len([l for l in run if "OPEN EUR" in l or "OPEN EUR" in l])
    
    opens = hy_opens or mi_opens or z_opens
    bal = float(re.search(r"balance ([\d.]+)", lines[fbs[ri]]).group(1))
    gross = bal - 10000.0
    
    # Detect which type
    if hy_opens > 0:
        opens = hy_opens
        src = "HYBRID"
    elif mi_opens > 0:
        opens = mi_opens
        src = "MICRO"
    else:
        opens = z_opens
        src = "Z-SCORE"
    
    comm = opens * 7.5
    net = gross - comm
    days = 47 if "Fwd" in lbl else 59 if "OOS" in lbl else 47
    tpd = opens / days
    
    print(f"{ri:<4d} {lbl:<35s} {opens:<7d} {gross:+8.2f}  {net:+9.2f}   {tpd:.2f}")
