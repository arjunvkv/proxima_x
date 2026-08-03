import base64
import subprocess
import os

key_path = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
mq5_path = r"paper_trade\mt5_backtest\NY_H21_MT5.mq5"

content = open(mq5_path, "rb").read()
b64_str = base64.b64encode(content).decode("ascii")

remote_code = f"import base64, os; path=os.path.expanduser('~/.wine_fundednext/drive_c/Program Files/FundedNext MT5 Terminal/MQL5/Experts/NY_H21_MT5.mq5'); open(path, 'wb').write(base64.b64decode('{b64_str}')); print('SUCCESS:', os.path.getsize(path))"

cmd = ["ssh", "-i", key_path, "ubuntu@140.245.234.92", f"python3 -c \"{remote_code}\""]
res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
