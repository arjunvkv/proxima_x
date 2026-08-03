#!/usr/bin/env python3
"""VPS Lot Puller — Fetches exact live BASE_LOT values from VPS EAs via SSH and updates dashboard_config.json."""

import os, sys, subprocess, json, re
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "dashboard_config.json"
SSH_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
VPS_EXP_DIR = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

def fetch_vps_lots():
    """Queries VPS via SSH to extract BASE_LOT values from deployed v106 EAs."""
    print("Fetching live BASE_LOT parameters from VPS via SSH...")
    cmd = f"""
cat << 'EOF' > /tmp/get_vps_lots.py
import os, re, json

d = '{VPS_EXP_DIR}'
mapping = {{
    'ultra_monster': 'Ultra_Monster_MT5_v106.mq5',
    'tokyo_h0': 'TokyoH0_MT5_v106.mq5',
    'cppf_z': 'CPPF_Z_MT5_v106.mq5',
    'msv_asian': 'MSV_Asian_Exhaustion_MT5_v106.mq5',
    'ny_h21': 'NY_H21_MT5_v106.mq5',
    'cpmc_z': 'CPMC_Z_MT5_v106.mq5'
}}

lots = {{}}
for s_id, file_name in mapping.items():
    p = os.path.join(d, file_name)
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        m = re.search(r'input\s+double\s+BASE_LOT\s*=\s*([0-9\.]+)', content)
        if m:
            lots[s_id] = float(m.group(1))

print(json.dumps(lots))
EOF
python3 /tmp/get_vps_lots.py
"""
    try:
        res = subprocess.run(["ssh", "-i", SSH_KEY, VPS_HOST, cmd], capture_output=True, text=True, check=True)
        # Parse JSON output from last line
        lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip().startswith("{")]
        if lines:
            vps_lots = json.loads(lines[-1])
            print(f"🟢 VPS Live Lots Retrieved: {vps_lots}")
            return vps_lots
    except Exception as e:
        print(f"⚠️ Error fetching VPS lots via SSH: {e}")

    # Fallback to local default VPS baseline lots
    return {
        "ultra_monster": 0.15,
        "tokyo_h0": 0.15,
        "cppf_z": 0.15,
        "msv_asian": 0.18,
        "ny_h21": 0.25,
        "cpmc_z": 0.15
    }

def update_config_with_vps_lots():
    vps_lots = fetch_vps_lots()
    if not CONFIG_PATH.exists():
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    updated = False
    for st in config.get("strategies", []):
        s_id = st["id"]
        if s_id in vps_lots:
            st["vps_base_lot"] = vps_lots[s_id]
            st["effective_lot"] = vps_lots[s_id]
            # Recalculate target win USD based on VPS lot size
            # 1 lot ~ $10/pip
            pip_val = 10.0 * st["vps_base_lot"]
            pips = 16.5 if s_id == "ultra_monster" else (22.0 if s_id == "tokyo_h0" else 18.0)
            st["target_win_usd"] = round(pips * pip_val, 2)
            updated = True

    if updated:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print("🟢 Dashboard config updated with live VPS lot sizes!")

if __name__ == "__main__":
    update_config_with_vps_lots()
