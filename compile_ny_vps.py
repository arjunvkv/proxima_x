import subprocess

key_path = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
remote_cmd = 'DISPLAY=:1 WINEPREFIX=~/.wine_fundednext wine "/home/ubuntu/.wine_fundednext/drive_c/Program Files/FundedNext MT5 Terminal/metaeditor64.exe" /compile:"C:\\Program Files\\FundedNext MT5 Terminal\\MQL5\\Experts\\NY_H21_MT5.mq5" && ls -la "/home/ubuntu/.wine_fundednext/drive_c/Program Files/FundedNext MT5 Terminal/MQL5/Experts/NY_H21_MT5.ex5"'

cmd = ["ssh", "-i", key_path, "ubuntu@140.245.234.92", remote_cmd]
res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
