import subprocess

key_path = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
me_local = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"

cmd = ["scp", "-i", key_path, me_local, "ubuntu@140.245.234.92:~/.wine_fundednext/drive_c/Program Files/FundedNext MT5 Terminal/metaeditor64.exe"]
res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
