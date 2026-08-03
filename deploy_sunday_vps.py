import base64
import subprocess
import os

key_path = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
mq5_path = r"paper_trade\mt5_backtest\Sunday_H22_MT5.mq5"
ex5_path = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts\Sunday_H22_MT5.ex5"

mq5_b64 = base64.b64encode(open(mq5_path, "rb").read()).decode("ascii")
ex5_b64 = base64.b64encode(open(ex5_path, "rb").read()).decode("ascii")

remote_code = f"""import base64, os
p_mq5 = os.path.expanduser('~/.wine_fundednext/drive_c/Program Files/FundedNext MT5 Terminal/MQL5/Experts/Sunday_H22_MT5.mq5')
p_ex5 = os.path.expanduser('~/.wine_fundednext/drive_c/Program Files/FundedNext MT5 Terminal/MQL5/Experts/Sunday_H22_MT5.ex5')
open(p_mq5, 'wb').write(base64.b64decode('{mq5_b64}'))
open(p_ex5, 'wb').write(base64.b64decode('{ex5_b64}'))
print('MQ5 SUCCESS:', os.path.getsize(p_mq5))
print('EX5 SUCCESS:', os.path.getsize(p_ex5))
"""

cmd = ["ssh", "-i", key_path, "ubuntu@140.245.234.92", f"python3 -c \"{remote_code}\""]
res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
