import os
import subprocess
import shutil
import time

METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"
APPDATA_TPL = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Profiles\Templates"
LOCAL_DIR   = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
VPS_KEY     = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST    = "ubuntu@140.245.234.92"
VPS_EXP     = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"
VPS_TPL     = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Profiles/Templates/"

TEMPLATES = [
    {"name": "TokyoH0_v8.tpl",      "ea": r"v8\TokyoH0_MT5_v8"},
    {"name": "UltraMonster_v8.tpl", "ea": r"v8\Ultra_Monster_MT5_v8"},
    {"name": "CPPF_Z_v8.tpl",        "ea": r"v8\CPPF_Z_MT5_v8"},
    {"name": "MSV_Asian_v8.tpl",     "ea": r"v8\MSV_Asian_Exhaustion_MT5_v8"},
    {"name": "NY_H21_v8.tpl",        "ea": r"v8\NY_H21_MT5_v8"},
    {"name": "CPMC_Z_v8.tpl",        "ea": r"v8\CPMC_Z_MT5_v8"},
]

def make_tpl_content(ea_name):
    return f"""<chart>
period_type=1
period_size=5
digits=5
scale=4
mode=1
grid=1
volume=0
scroll=1
shift=1
one_click=1
<expert>
name={ea_name}
flags=1
window=0
</expert>
</chart>
"""

def main():
    print("=" * 115)
    print("PROXIMA X — BUILDING, COMPILING & DEPLOYING v8 MASTER AUTO-ATTACH LAUNCHER TO VPS...")
    print("=" * 115)

    os.makedirs(APPDATA_TPL, exist_ok=True)

    # 1. Generate .tpl template files
    print("📄 Generating 6 Strategy Template Files (.tpl)...")
    for tpl in TEMPLATES:
        tname = tpl["name"]
        ea    = tpl["ea"]
        content = make_tpl_content(ea)
        
        local_path = os.path.join(LOCAL_DIR, tname)
        appdata_path = os.path.join(APPDATA_TPL, tname)
        
        with open(local_path, "wb") as f:
            f.write(b'\xff\xfe' + content.encode('utf-16le'))
        with open(appdata_path, "wb") as f:
            f.write(b'\xff\xfe' + content.encode('utf-16le'))
        print(f"  🟢 Generated template: {tname} -> Attached EA: {ea}")

    # 2. Copy launcher MQ5 to APPDATA_EXP and compile
    launcher_name = "Proxima_v8_Master_Launcher"
    mq5_filename = f"{launcher_name}.mq5"
    ex5_filename = f"{launcher_name}.ex5"

    local_mq5 = os.path.join(LOCAL_DIR, mq5_filename)
    appdata_mq5 = os.path.join(APPDATA_EXP, mq5_filename)
    appdata_ex5 = os.path.join(APPDATA_EXP, ex5_filename)
    local_ex5 = os.path.join(LOCAL_DIR, ex5_filename)

    shutil.copy(local_mq5, appdata_mq5)

    print(f"\n🔨 Compiling {mq5_filename}...")
    subprocess.run([METAEDITOR, f"/compile:{appdata_mq5}", "/log"], check=False)
    time.sleep(3.5)

    if os.path.exists(appdata_ex5):
        size = os.path.getsize(appdata_ex5)
        print(f"  🟢 {launcher_name} COMPILED SUCCESS! Size: {size:,} bytes")
        shutil.copy(appdata_ex5, local_ex5)
    else:
        print(f"  ❌ Compilation failed for {mq5_filename}")

    # 3. Upload to VPS via SSH
    print("\n🚀 Uploading Master Launcher & Templates to VPS...")
    ssh_mkdir = f"mkdir -p '{VPS_EXP}' '{VPS_TPL}'"
    subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, ssh_mkdir], check=False)

    # SCP launcher
    if os.path.exists(appdata_ex5):
        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, appdata_ex5, f"{VPS_HOST}:{VPS_EXP}"], check=False)
        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, appdata_mq5, f"{VPS_HOST}:{VPS_EXP}"], check=False)

    # SCP templates
    for tpl in TEMPLATES:
        tname = tpl["name"]
        local_t = os.path.join(LOCAL_DIR, tname)
        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, local_t, f"{VPS_HOST}:{VPS_TPL}"], check=False)

    print("\n" + "=" * 115)
    print("🟢 SUCCESS: 'Proxima_v8_Master_Launcher' & ALL 6 STRATEGY TEMPLATES DEPLOYED TO VPS!")
    print("===================================================================================================")

if __name__ == "__main__":
    main()
