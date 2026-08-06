import os, subprocess, shutil, time

METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"

mq5_path = os.path.join(APPDATA_EXP, "TokyoH0_MT5_v8.mq5")
ex5_path = os.path.join(APPDATA_EXP, "TokyoH0_MT5_v8.ex5")

print(f"Testing direct MetaEditor execution for {mq5_path}...")
cmd = [METAEDITOR, f"/compile:{mq5_path}"]
subprocess.run(cmd, check=False)
time.sleep(0.5)

if os.path.exists(ex5_path):
    print(f"  🟢 SUCCESS! Compiled file size: {os.path.getsize(ex5_path):,} bytes")
else:
    print("  ❌ Compilation failed")
