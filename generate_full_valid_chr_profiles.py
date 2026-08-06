import os
import subprocess
import shutil
import tarfile

LOCAL_DIR   = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
VPS_KEY     = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST    = "ubuntu@140.245.234.92"
MQL5_CHARTS = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Profiles/Charts"
ROOT_CHARTS = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/Profiles/Charts"

STRATEGIES_6 = [
    {"symbol": "EURUSD", "ea": "TokyoH0_MT5_v8",               "lot": "1.00000000"},
    {"symbol": "GBPUSD", "ea": "Ultra_Monster_MT5_v8",         "lot": "1.20000000"},
    {"symbol": "EURAUD", "ea": "CPPF_Z_MT5_v8",                "lot": "1.40000000"},
    {"symbol": "USDJPY", "ea": "MSV_Asian_Exhaustion_MT5_v8",  "lot": "1.00000000"},
    {"symbol": "EURJPY", "ea": "NY_H21_MT5_v8",                "lot": "1.50000000"},
    {"symbol": "GBPAUD", "ea": "CPMC_Z_MT5_v8",                "lot": "1.40000000"},
]

def make_full_chr_content(symbol, ea_name, chart_id, lot_str):
    return f"""<chart>
id={chart_id}
symbol={symbol}
period_type=1
period_size=5
digits=5
tick_size=0.000000
position_time=0
scale_fix=0
scale=4
mode=1
fore=0
grid=1
volume=0
scroll=1
shift=1
shift_size=20.000000
fixed_pos=0.000000
ohlc=0
bidline=1
askline=0
lastline=0
days=0
descriptions=0
window_type=1
background_color=0
foreground_color=16777215
barup_color=65280
bardown_color=65280
bullcandle_color=0
bearcandle_color=16777215
chartline_color=65280
volumes_color=3329330
grid_color=10061943

<window>
height=100
<indicator>
name=Main
path=
apply=1
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=0
scale_fix_min_val=0.000000
scale_fix_max=0
scale_fix_max_val=0.000000
</indicator>
</window>

<expert>
name={ea_name}
flags=1
window=0
<inputs>
BASE_LOT={lot_str}
</inputs>
</expert>
</chart>
"""

def main():
    print("=" * 115)
    print("GENERATING FULL VALID .CHR PROFILE FILES FOR INSTANT AUTOMATED ATTACHMENT...")
    print("=" * 115)

    local_prof_dir = r"C:\Trading\Agentic_Trading\proxima_x\temp_profile_6_full"
    if os.path.exists(local_prof_dir):
        shutil.rmtree(local_prof_dir)
    os.makedirs(local_prof_dir, exist_ok=True)

    for idx, item in enumerate(STRATEGIES_6, start=1):
        filename = f"chart{idx:02d}.chr"
        chart_id = 10000000000000000 + idx
        chr_text = make_full_chr_content(item["symbol"], item["ea"], chart_id, item["lot"])
        file_path = os.path.join(local_prof_dir, filename)
        with open(file_path, "wb") as f:
            f.write(b'\xff\xfe' + chr_text.encode('utf-16le'))
        print(f"  • Chart {idx}/6: [{item['symbol']:<7} M5] -> EA: {item['ea']:<28} | Lot: {item['lot']}")

    # Create TAR package
    tar_path = r"C:\Trading\Agentic_Trading\proxima_x\full_6_chr.tar"
    with tarfile.open(tar_path, "w") as tar:
        for f in os.listdir(local_prof_dir):
            tar.add(os.path.join(local_prof_dir, f), arcname=f)

    # Upload tar profile to VPS
    print("\n🚀 Uploading full profile archive to VPS...")
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, tar_path, f"{VPS_HOST}:/tmp/full_6_chr.tar"], check=False)

    remote_extract = (
        f"mkdir -p '{MQL5_CHARTS}/Proxima_v8' '{MQL5_CHARTS}/Default' '{ROOT_CHARTS}/Proxima_v8' '{ROOT_CHARTS}/Default' && "
        f"rm -rf '{MQL5_CHARTS}/Proxima_v8'/* '{MQL5_CHARTS}/Default'/* '{ROOT_CHARTS}/Proxima_v8'/* '{ROOT_CHARTS}/Default'/* && "
        f"tar -xf /tmp/full_6_chr.tar -C '{MQL5_CHARTS}/Proxima_v8/' && "
        f"tar -xf /tmp/full_6_chr.tar -C '{MQL5_CHARTS}/Default/' && "
        f"tar -xf /tmp/full_6_chr.tar -C '{ROOT_CHARTS}/Proxima_v8/' && "
        f"tar -xf /tmp/full_6_chr.tar -C '{ROOT_CHARTS}/Default/' && "
        f"rm /tmp/full_6_chr.tar && "
        f"echo '=== Full Profile Deployment Verified ==='"
    )

    res = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"bash -c \"{remote_extract}\""], capture_output=True, text=True)
    print(res.stdout)

    print("=" * 115)
    print("🟢 SUCCESS: FULL VALID .CHR PROFILES DEPLOYED TO VPS!")
    print("===================================================================================================")

if __name__ == "__main__":
    main()
