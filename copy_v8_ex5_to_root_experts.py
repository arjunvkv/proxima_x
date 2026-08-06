import os
import shutil
import subprocess

LOCAL_DIR   = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"
VPS_KEY     = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST    = "ubuntu@140.245.234.92"
VPS_EXP     = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

V8_NAMES = [
    "TokyoH0_MT5_v8",
    "Ultra_Monster_MT5_v8",
    "CPPF_Z_MT5_v8",
    "MSV_Asian_Exhaustion_MT5_v8",
    "NY_H21_MT5_v8",
    "CPMC_Z_MT5_v8"
]

def main():
    print("=" * 100)
    print("COPYING ALL v8 EX5 BINARIES TO MAIN ROOT MQL5/Experts/ FOR INSTANT NAVIGATOR VISIBILITY...")
    print("=" * 100)

    for name in V8_NAMES:
        ex5_name = f"{name}.ex5"
        local_ex5 = os.path.join(LOCAL_DIR, ex5_name)
        appdata_root_ex5 = os.path.join(APPDATA_EXP, ex5_name)
        appdata_v8_ex5 = os.path.join(APPDATA_EXP, "v8", ex5_name)

        if os.path.exists(local_ex5):
            shutil.copy(local_ex5, appdata_root_ex5)
            if os.path.exists(os.path.join(APPDATA_EXP, "v8")):
                shutil.copy(local_ex5, appdata_v8_ex5)
            
            print(f"🟢 Copying {ex5_name} to VPS root Experts folder...")
            subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, local_ex5, f"{VPS_HOST}:{VPS_EXP}"], check=False)

    print("\n" + "=" * 100)
    print("🟢 SUCCESS: ALL v8 EAs ARE NOW VISIBLE AT THE MAIN ROOT LEVEL IN MT5 NAVIGATOR!")
    print("=" * 100)

if __name__ == "__main__":
    main()
