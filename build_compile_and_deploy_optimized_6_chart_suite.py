import os
import subprocess
import shutil
import tarfile

METAEDITOR  = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_SCR = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Scripts"
APPDATA_TPL = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Profiles\Templates"
LOCAL_DIR   = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
VPS_KEY     = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST    = "ubuntu@140.245.234.92"
VPS_SCR     = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Scripts/"
VPS_TPL     = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Profiles/Templates/"
MQL5_CHARTS = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Profiles/Charts"
ROOT_CHARTS = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/Profiles/Charts"

STRATEGIES_6 = [
    {"symbol": "EURUSD", "ea": r"v8\TokyoH0_MT5_v8",            "strat": "Tokyo H0 (v8)", "lot": "1.00L"},
    {"symbol": "GBPUSD", "ea": r"v8\Ultra_Monster_MT5_v8",      "strat": "Ultra Monster (v8)", "lot": "1.20L"},
    {"symbol": "EURAUD", "ea": r"v8\CPPF_Z_MT5_v8",             "strat": "CPPF Z (v8)", "lot": "1.40L"},
    {"symbol": "USDJPY", "ea": r"v8\MSV_Asian_Exhaustion_MT5_v8", "strat": "MSV Asian (v8)", "lot": "1.00L"},
    {"symbol": "EURJPY", "ea": r"v8\NY_H21_MT5_v8",             "strat": "NY H21 (v8)", "lot": "1.50L"},
    {"symbol": "GBPAUD", "ea": r"v8\CPMC_Z_MT5_v8",            "strat": "CPMC Z (v8)", "lot": "1.40L"},
]

def make_chr_content(symbol, ea_name, chart_id):
    return f"""<chart>
id={chart_id}
symbol={symbol}
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
    print("PROXIMA X — BUILDING & DEPLOYING OPTIMIZED 6-CHART MASTER SUITE TO VPS...")
    print("=" * 115)

    # 1. Copy launcher to APPDATA_SCR and compile via cmd /c
    os.makedirs(APPDATA_SCR, exist_ok=True)
    src_mq5 = os.path.join(LOCAL_DIR, 'Proxima_v8_Master_Launcher.mq5')
    appdata_mq5 = os.path.join(APPDATA_SCR, 'Proxima_v8_Master_Launcher.mq5')
    appdata_ex5 = os.path.join(APPDATA_SCR, 'Proxima_v8_Master_Launcher.ex5')
    local_ex5 = os.path.join(LOCAL_DIR, 'Proxima_v8_Master_Launcher.ex5')

    shutil.copy(src_mq5, appdata_mq5)

    print("🔨 Compiling optimized 6-chart Proxima_v8_Master_Launcher.mq5...")
    cmd_str = f'"{METAEDITOR}" /compile:"{appdata_mq5}" /log'
    subprocess.run(["cmd.exe", "/c", cmd_str], capture_output=True, text=True)

    if os.path.exists(appdata_ex5):
        size = os.path.getsize(appdata_ex5)
        print(f"  🟢 Master Launcher EX5 Compiled SUCCESS! Size: {size:,} bytes")
        shutil.copy(appdata_ex5, local_ex5)
    else:
        print("  ❌ Master Launcher compilation failed")

    # 2. Build 6-chart profile directory
    local_prof_dir = r"C:\Trading\Agentic_Trading\proxima_x\temp_profile_6_charts"
    if os.path.exists(local_prof_dir):
        shutil.rmtree(local_prof_dir)
    os.makedirs(local_prof_dir, exist_ok=True)

    print("\n📊 Generating 6-Chart Profile Structure:")
    for idx, item in enumerate(STRATEGIES_6, start=1):
        filename = f"chart{idx:02d}.chr"
        chart_id = 10000000000000000 + idx
        chr_text = make_chr_content(item["symbol"], item["ea"], chart_id)
        
        file_path = os.path.join(local_prof_dir, filename)
        with open(file_path, "wb") as f:
            f.write(b'\xff\xfe' + chr_text.encode('utf-16le'))
        print(f"  • Chart {idx}/6: [{item['symbol']:<7} M5] -> Strategy: {item['strat']:<20} ({item['lot']}) | EA: {item['ea']}")

    # 3. Create TAR package
    tar_path = r"C:\Trading\Agentic_Trading\proxima_x\opt_6_charts.tar"
    with tarfile.open(tar_path, "w") as tar:
        for f in os.listdir(local_prof_dir):
            tar.add(os.path.join(local_prof_dir, f), arcname=f)

    # 4. Upload Launcher, Templates, and 6-Chart Profile Tar to VPS
    print("\n🚀 Uploading 6-Chart Master Suite & Profiles to VPS...")
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, appdata_ex5, f"{VPS_HOST}:{VPS_SCR}"], check=False)
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, appdata_mq5, f"{VPS_HOST}:{VPS_SCR}"], check=False)
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, tar_path, f"{VPS_HOST}:/tmp/opt_6_charts.tar"], check=False)

    remote_extract = (
        f"mkdir -p '{MQL5_CHARTS}/Proxima_v8' '{MQL5_CHARTS}/Default' '{ROOT_CHARTS}/Proxima_v8' '{ROOT_CHARTS}/Default' && "
        f"rm -rf '{MQL5_CHARTS}/Proxima_v8'/* '{MQL5_CHARTS}/Default'/* '{ROOT_CHARTS}/Proxima_v8'/* '{ROOT_CHARTS}/Default'/* && "
        f"tar -xf /tmp/opt_6_charts.tar -C '{MQL5_CHARTS}/Proxima_v8/' && "
        f"tar -xf /tmp/opt_6_charts.tar -C '{MQL5_CHARTS}/Default/' && "
        f"tar -xf /tmp/opt_6_charts.tar -C '{ROOT_CHARTS}/Proxima_v8/' && "
        f"tar -xf /tmp/opt_6_charts.tar -C '{ROOT_CHARTS}/Default/' && "
        f"rm /tmp/opt_6_charts.tar && "
        f"echo '=== Active VPS Profile Charts ===' && ls -1 '{MQL5_CHARTS}/Proxima_v8/'"
    )

    res = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"bash -c \"{remote_extract}\""], capture_output=True, text=True)

    print(res.stdout)
    print("=" * 115)
    print("🟢 SUCCESS: OPTIMIZED 6-CHART MASTER LAUNCHER & PROFILES DEPLOYED TO VPS!")
    print("===================================================================================================")

if __name__ == "__main__":
    main()
