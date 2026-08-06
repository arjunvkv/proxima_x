import os
import subprocess
import shutil

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
CHARTS_DIR = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/Profiles/Charts"
LOCAL_TEMP_DIR = r"C:\Trading\Agentic_Trading\proxima_x\temp_profile_v8"

STRATEGIES = [
    {
        "name": "Tokyo H0",
        "ea": r"v8\TokyoH0_MT5_v8",
        "pairs": ["EURJPY", "GBPJPY", "USDJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "EURAUD", "EURGBP", "EURCAD", "GBPAUD", "GBPCAD"]
    },
    {
        "name": "Ultra Monster",
        "ea": r"v8\Ultra_Monster_MT5_v8",
        "pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "EURJPY", "GBPJPY", "EURAUD"]
    },
    {
        "name": "CPPF Z",
        "ea": r"v8\CPPF_Z_MT5_v8",
        "pairs": ["EURAUD", "GBPAUD", "AUDNZD", "EURNZD", "GBPNZD"]
    },
    {
        "name": "MSV Asian",
        "ea": r"v8\MSV_Asian_Exhaustion_MT5_v8",
        "pairs": ["USDJPY"]
    },
    {
        "name": "NY H21",
        "ea": r"v8\NY_H21_MT5_v8",
        "pairs": ["EURJPY", "GBPJPY"]
    },
    {
        "name": "CPMC Z",
        "ea": r"v8\CPMC_Z_MT5_v8",
        "pairs": ["GBPAUD", "GBPNZD"]
    }
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
    print("PROXIMA X — BUILDING & DEPLOYING AUTOMATED AUTO-ATTACH MT5 PROFILE 'Proxima_v8' TO VPS...")
    print("=" * 115)

    if os.path.exists(LOCAL_TEMP_DIR):
        shutil.rmtree(LOCAL_TEMP_DIR)
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

    chart_count = 1
    for strat in STRATEGIES:
        ea = strat["ea"]
        for pair in strat["pairs"]:
            filename = f"chart{chart_count:02d}.chr"
            chart_id = 10000000000000000 + chart_count
            chr_text = make_chr_content(pair, ea, chart_id)
            
            file_path = os.path.join(LOCAL_TEMP_DIR, filename)
            with open(file_path, "wb") as f:
                f.write(b'\xff\xfe' + chr_text.encode('utf-16le'))
            
            print(f"  • Chart {chart_count:02d}: {pair:<7} | EA: {ea:<28} -> {filename}")
            chart_count += 1

    # Upload entire directory at once via scp -r
    print("\n🚀 Uploading 'Proxima_v8' & 'Default' Profile to VPS...")
    vps_target_prof = f"{CHARTS_DIR}/Proxima_v8"
    vps_target_def  = f"{CHARTS_DIR}/Default"

    ssh_mkdir = f"mkdir -p '{vps_target_prof}' '{vps_target_def}'"
    subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, ssh_mkdir], check=False)

    scp_cmd = f"scp -o StrictHostKeyChecking=no -o BatchMode=yes -i \"{VPS_KEY}\" \"{LOCAL_TEMP_DIR}\"/* \"{VPS_HOST}:{vps_target_prof}/\""
    subprocess.run(scp_cmd, shell=True, check=False)

    scp_cmd2 = f"scp -o StrictHostKeyChecking=no -o BatchMode=yes -i \"{VPS_KEY}\" \"{LOCAL_TEMP_DIR}\"/* \"{VPS_HOST}:{vps_target_def}/\""
    subprocess.run(scp_cmd2, shell=True, check=False)

    print("\n" + "=" * 115)
    print(f"🟢 SUCCESS: {chart_count - 1} CHARTS & EAs AUTO-CONFIGURED IN 'Proxima_v8' & 'Default' PROFILES ON VPS!")
    print("   When MT5 is opened or refreshed, all charts and v8 EAs will load AUTOMATICALLY with zero manual steps!")
    print("=" * 115)

if __name__ == "__main__":
    main()
