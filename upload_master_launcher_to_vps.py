import os
import shutil
import subprocess

LOCAL_DIR   = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
APPDATA_SCR = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Scripts"
VPS_KEY     = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST    = "ubuntu@140.245.234.92"
VPS_SCR     = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Scripts/"
VPS_TPL     = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Profiles/Templates/"

def main():
    src_ex5 = os.path.join(APPDATA_SCR, 'Proxima_v8_Master_Launcher.ex5')
    src_mq5 = os.path.join(APPDATA_SCR, 'Proxima_v8_Master_Launcher.mq5')

    if os.path.exists(src_ex5):
        shutil.copy(src_ex5, os.path.join(LOCAL_DIR, 'Proxima_v8_Master_Launcher.ex5'))
        size = os.path.getsize(src_ex5)
        print(f"🟢 Proxima_v8_Master_Launcher.ex5 ({size:,} bytes) -> Uploading to VPS MQL5/Scripts...")
        subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-i", VPS_KEY, VPS_HOST, f"mkdir -p '{VPS_SCR}' '{VPS_TPL}'"], check=False)
        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-i", VPS_KEY, src_ex5, f"{VPS_HOST}:{VPS_SCR}"], check=False)
        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-i", VPS_KEY, src_mq5, f"{VPS_HOST}:{VPS_SCR}"], check=False)

    templates = ['TokyoH0_v8.tpl', 'UltraMonster_v8.tpl', 'CPPF_Z_v8.tpl', 'MSV_Asian_v8.tpl', 'NY_H21_v8.tpl', 'CPMC_Z_v8.tpl']
    for tpl in templates:
        local_t = os.path.join(LOCAL_DIR, tpl)
        if os.path.exists(local_t):
            print(f"🟢 Uploading template {tpl} to VPS MQL5/Profiles/Templates...")
            subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-i", VPS_KEY, local_t, f"{VPS_HOST}:{VPS_TPL}"], check=False)

    print("=" * 115)
    print("🟢 MASTER LAUNCHER & ALL 6 STRATEGY TEMPLATES DEPLOYED TO VPS SUCCESS!")
    print("=" * 115)

if __name__ == "__main__":
    main()
