#!/usr/bin/env python3
"""Find all MT5 logs, reports, and trade history across the VPS Wine filesystem."""

import subprocess

SSH_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"

remote_script = """cat << 'EOF' > /tmp/find_mt5_logs.py
import os, glob

wine_dir = '/home/ubuntu/.wine/drive_c/'
print("Searching for MT5 log files in .wine...")
all_logs = []
for root, dirs, files in os.walk(wine_dir):
    for file in files:
        if file.endswith(('.log', '.htm', '.html', '.txt')) and any(k in root.lower() for k in ['logs', 'reports', 'history']):
            all_logs.append(os.path.join(root, file))

print(f"Found {len(all_logs)} log/report files:")
for path in all_logs[:15]:
    print("  File:", path)
    try:
        size = os.path.getsize(path)
        print(f"    Size: {size} bytes")
        with open(path, 'r', encoding='utf-16', errors='ignore') as f:
            lines = f.readlines()
            if not lines:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f2:
                    lines = f2.readlines()
        for l in lines[-5:]:
            print("     ", l.strip()[:100])
    except Exception as e:
        print("    Error reading:", e)

EOF
python3 /tmp/find_mt5_logs.py
"""

def main():
    res = subprocess.run(["ssh", "-i", SSH_KEY, VPS_HOST, remote_script], capture_output=True, text=True)
    print(res.stdout)

if __name__ == "__main__":
    main()
