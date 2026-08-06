import os
import subprocess
import shutil
import time

METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_SCR = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Scripts"
LOCAL_DIR   = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
VPS_KEY     = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST    = "ubuntu@140.245.234.92"
VPS_SCR     = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Scripts/"

def main():
    print("=" * 100)
    print("PROXIMA X — COMPILING & DEPLOYING MASTER LAUNCHER SCRIPT")
    print("=" * 100)

    os.makedirs(APPDATA_SCR, exist_ok=True)
    src_mq5 = os.path.join(LOCAL_DIR, 'Proxima_v8_Master_Launcher.mq5')
    appdata_mq5 = os.path.join(APPDATA_SCR, 'Proxima_v8_Master_Launcher.mq5')
    appdata_ex5 = os.path.join(APPDATA_SCR, 'Proxima_v8_Master_Launcher.ex5')
    local_ex5 = os.path.join(LOCAL_DIR, 'Proxima_v8_Master_Launcher.ex5')

    shutil.copy(src_mq5, appdata_mq5)

    print("Compiling Proxima_v8_Master_Launcher.mq5 in Scripts folder...")
    subprocess.run([METAEDITOR, f"/compile:{appdata_mq5}", "/log"], check=False)
    time.sleep(4.5)

    if os.path.exists(appdata_ex5):
        size = os.path.getsize(appdata_ex5)
        print(f"🟢 SUCCESS! Compiled Master Launcher EX5 Size: {size:,} bytes")
        shutil.copy(appdata_ex5, local_ex5)
        subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-i", VPS_KEY, VPS_HOST, f"mkdir -p '{VPS_SCR}'"], check=False)
        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-i", VPS_KEY, appdata_ex5, f"{VPS_HOST}:{VPS_SCR}"], check=False)
        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-i", VPS_KEY, appdata_mq5, f"{VPS_HOST}:{VPS_SCR}"], check=False)
        print("🟢 Master Launcher deployed to VPS MQL5/Scripts/!")
    else:
        print("❌ Master Launcher compilation failed")

if __name__ == "__main__":
    main()
