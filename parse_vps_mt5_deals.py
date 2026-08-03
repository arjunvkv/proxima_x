#!/usr/bin/env python3
"""Parse actual live VPS MT5 executed trade deals and PnLs from terminal logs."""

import subprocess

SSH_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"

remote_code = r"""
import os, glob

logs_dir = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/logs'
mql5_logs = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/logs'

print("=== ACTUAL VPS MT5 LIVE TRADES & DEALS AUDIT ===")

# Search terminal trade logs
log_files = sorted(glob.glob(os.path.join(logs_dir, '*.log')))
for fpath in log_files[-3:]:
    fname = os.path.basename(fpath)
    print("--- Terminal Log:", fname, "---")
    try:
        with open(fpath, 'r', encoding='utf-16', errors='ignore') as f:
            lines = f.readlines()
    except:
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
    deals = [l.strip() for l in lines if 'deal #' in l or 'order #' in l or 'failed' in l]
    for d in deals[-20:]:
        print("  ", d)

print("\n=== MQL5 EXPERT TRADES & EXITS ===")
m_files = sorted(glob.glob(os.path.join(mql5_logs, '*.log')))
for fpath in m_files[-3:]:
    fname = os.path.basename(fpath)
    print("--- MQL5 Log:", fname, "---")
    try:
        with open(fpath, 'r', encoding='utf-16', errors='ignore') as f:
            lines = f.readlines()
    except:
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
    m_deals = [l.strip() for l in lines if any(k in l for k in ['ENTRY', 'CLOSE', 'SKIP', 'SUCCESS', 'failed'])]
    for md in m_deals[-20:]:
        print("  ", md)
"""

def main():
    # Write directly to remote file via SSH cat
    ssh_cmd = f"cat << 'EOF' > /tmp/parse_deals.py\n{remote_code}\nEOF\npython3 /tmp/parse_deals.py"
    res = subprocess.run(["ssh", "-i", SSH_KEY, VPS_HOST, ssh_cmd], capture_output=True, text=True)
    print(res.stdout)

if __name__ == "__main__":
    main()
