import subprocess

key_path = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"

remote_code = """import os, glob

exp_dir = os.path.expanduser('~/.wine_fundednext/drive_c/Program Files/FundedNext MT5 Terminal/MQL5/Experts')
ex5_files = glob.glob(os.path.join(exp_dir, '*.ex5'))
print('=== 1. COMPILED EX5 FILES IN FUNDEDNEXT VPS ===')
for f in ex5_files:
    print('  ', os.path.basename(f), '-->', os.path.getsize(f), 'bytes')

log_dir = os.path.expanduser('~/.wine_fundednext/drive_c/Program Files/FundedNext MT5 Terminal/MQL5/Logs')
log_files = sorted(glob.glob(os.path.join(log_dir, '*.log')), reverse=True)
print('\\n=== 2. LATEST LIVE MT5 EXPERT LOG ===')
if log_files:
    latest = log_files[0]
    print('  Log File:', os.path.basename(latest))
    try:
        content = open(latest, encoding='utf-16le').read()
        lines = content.splitlines()[-20:]
        for l in lines:
            print('  ', l)
    except Exception as e:
        print('  Error reading log:', e)
else:
    print('  No MQL5 logs found yet (MT5 initialized cleanly).')

ps = os.popen('ps aux | grep -i terminal').read()
print('\\n=== 3. LIVE RUNNING MT5 PROCESSES ===')
for line in ps.splitlines():
    if 'grep' not in line:
        print('  ', line)
"""

cmd = ["ssh", "-i", key_path, "ubuntu@140.245.234.92", f"python3 -c \"{remote_code}\""]
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
