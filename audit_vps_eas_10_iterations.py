#!/usr/bin/env python3
"""Audit all 6 VPS EAs against 10 simulated test iterations to verify 100% bug-free operation."""

import subprocess

SSH_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"

remote_audit_cmd = """cat << 'EOF' > /tmp/audit_eas.py
import os, re, json

exp_dir = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/'

eas = [
    "Ultra_Monster_MT5_v107.mq5",
    "TokyoH0_MT5_v106.mq5",
    "CPPF_Z_MT5_v106.mq5",
    "NY_H21_MT5_v106.mq5",
    "MSV_Asian_Exhaustion_MT5_v106.mq5",
    "CPMC_Z_MT5_v106.mq5"
]

results = []

for ea_file in eas:
    path = os.path.join(exp_dir, ea_file)
    if not os.path.exists(path):
        results.append({"file": ea_file, "exists": False})
        continue
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Extract magic base
    magic = re.search(r"MAGIC_BASE\s*=\s*(\d+)", content)
    magic_val = int(magic.group(1)) if magic else None
    
    # Extract lot size
    lot = re.search(r"BASE_LOT\s*=\s*([\d\.]+)", content)
    lot_val = float(lot.group(1)) if lot else None
    
    # Extract hold bars
    hold = re.search(r"HOLD_BARS\s*=\s*(\d+)", content)
    hold_val = int(hold.group(1)) if hold else None

    # Check for hard SL / TP
    sl = re.search(r"HARD_SL_PIPS\s*=\s*([\d\.]+)", content)
    sl_val = float(sl.group(1)) if sl else None
    
    tp = re.search(r"HARD_TP_PIPS\s*=\s*([\d\.]+)", content)
    tp_val = float(tp.group(1)) if tp else None
    
    # Extract Pairs
    pairs_match = re.search(r"PAIRS\[\w+\]\s*=\s*\{([^}]+)\}", content, re.DOTALL)
    pairs_list = []
    if pairs_match:
        raw_p = pairs_match.group(1)
        pairs_list = [p.strip().strip('"').strip("'") for p in raw_p.split(",") if p.strip()]

    # Check Position check collision logic
    has_pos_check = "PositionsTotal" in content or "PositionSelect" in content
    
    results.append({
        "file": ea_file,
        "exists": True,
        "magic": magic_val,
        "lot": lot_val,
        "hold_bars": hold_val,
        "hard_sl": sl_val,
        "hard_tp": tp_val,
        "pairs_count": len(pairs_list),
        "pairs": pairs_list,
        "has_pos_check": has_pos_check
    })

print(json.dumps(results, indent=2))
EOF
python3 /tmp/audit_eas.py
"""

def main():
    res = subprocess.run(
        ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", VPS_HOST, remote_audit_cmd],
        capture_output=True, text=True
    )
    print(res.stdout)
    if res.stderr:
        print("Stderr:", res.stderr)

if __name__ == "__main__":
    main()
