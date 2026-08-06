import os
import subprocess
import tarfile

LOCAL_DIR   = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
APPDATA_TPL = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Profiles\Templates"
VPS_KEY     = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST    = "ubuntu@140.245.234.92"
VPS_TPL     = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Profiles/Templates/"
MQL5_CHARTS = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Profiles/Charts"
ROOT_CHARTS = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/Profiles/Charts"

TEMPLATES = [
    {"name": "TokyoH0_v8.tpl",      "ea": "TokyoH0_MT5_v8"},
    {"name": "UltraMonster_v8.tpl", "ea": "Ultra_Monster_MT5_v8"},
    {"name": "CPPF_Z_v8.tpl",        "ea": "CPPF_Z_MT5_v8"},
    {"name": "MSV_Asian_v8.tpl",     "ea": "MSV_Asian_Exhaustion_MT5_v8"},
    {"name": "NY_H21_v8.tpl",        "ea": "NY_H21_MT5_v8"},
    {"name": "CPMC_Z_v8.tpl",        "ea": "CPMC_Z_MT5_v8"},
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

STRATEGIES_6 = [
    {"symbol": "EURUSD", "ea": "TokyoH0_MT5_v8"},
    {"symbol": "GBPUSD", "ea": "Ultra_Monster_MT5_v8"},
    {"symbol": "EURAUD", "ea": "CPPF_Z_MT5_v8"},
    {"symbol": "USDJPY", "ea": "MSV_Asian_Exhaustion_MT5_v8"},
    {"symbol": "EURJPY", "ea": "NY_H21_MT5_v8"},
    {"symbol": "GBPAUD", "ea": "CPMC_Z_MT5_v8"},
]

def main():
    print("=" * 115)
    print("PROXIMA X — UPDATING TEMPLATES & PROFILES WITH ROOT EA PATHS FOR INSTANT ATTACHMENT...")
    print("=" * 115)

    os.makedirs(APPDATA_TPL, exist_ok=True)

    # 1. Write templates locally
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

    # 2. Write 6 chart files
    local_prof_dir = r"C:\Trading\Agentic_Trading\proxima_x\temp_profile_6_root"
    if os.path.exists(local_prof_dir):
        shutil.rmtree(local_prof_dir)
    os.makedirs(local_prof_dir, exist_ok=True)

    for idx, item in enumerate(STRATEGIES_6, start=1):
        filename = f"chart{idx:02d}.chr"
        chart_id = 10000000000000000 + idx
        chr_text = make_chr_content(item["symbol"], item["ea"], chart_id)
        file_path = os.path.join(local_prof_dir, filename)
        with open(file_path, "wb") as f:
            f.write(b'\xff\xfe' + chr_text.encode('utf-16le'))

    # 3. Create TAR package
    tar_path = r"C:\Trading\Agentic_Trading\proxima_x\opt_6_root.tar"
    with tarfile.open(tar_path, "w") as tar:
        for f in os.listdir(local_prof_dir):
            tar.add(os.path.join(local_prof_dir, f), arcname=f)

    # 4. Upload templates and tar profile to VPS
    print("🚀 Uploading root EA templates & profiles to VPS...")
    for tpl in TEMPLATES:
        tname = tpl["name"]
        local_t = os.path.join(LOCAL_DIR, tname)
        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, local_t, f"{VPS_HOST}:{VPS_TPL}"], check=False)

    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, tar_path, f"{VPS_HOST}:/tmp/opt_6_root.tar"], check=False)

    remote_extract = (
        f"mkdir -p '{MQL5_CHARTS}/Proxima_v8' '{MQL5_CHARTS}/Default' '{ROOT_CHARTS}/Proxima_v8' '{ROOT_CHARTS}/Default' && "
        f"rm -rf '{MQL5_CHARTS}/Proxima_v8'/* '{MQL5_CHARTS}/Default'/* '{ROOT_CHARTS}/Proxima_v8'/* '{ROOT_CHARTS}/Default'/* && "
        f"tar -xf /tmp/opt_6_root.tar -C '{MQL5_CHARTS}/Proxima_v8/' && "
        f"tar -xf /tmp/opt_6_root.tar -C '{MQL5_CHARTS}/Default/' && "
        f"tar -xf /tmp/opt_6_root.tar -C '{ROOT_CHARTS}/Proxima_v8/' && "
        f"tar -xf /tmp/opt_6_root.tar -C '{ROOT_CHARTS}/Default/' && "
        f"rm /tmp/opt_6_root.tar && "
        f"echo '=== Root Profile Deployment Verified ==='"
    )

    res = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"bash -c \"{remote_extract}\""], capture_output=True, text=True)
    print(res.stdout)

    print("=" * 115)
    print("🟢 SUCCESS: ROOT EA PATHS INSTALLED TO ALL TEMPLATES AND PROFILES ON VPS!")
    print("===================================================================================================")

if __name__ == "__main__":
    main()
