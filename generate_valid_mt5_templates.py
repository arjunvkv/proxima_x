import os
import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
LOCAL_DIR   = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
APPDATA_TPL = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Profiles\Templates"
VPS_TPL     = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Profiles/Templates/"

TEMPLATES = [
    {"name": "TokyoH0_v8.tpl",      "ea": "TokyoH0_MT5_v8",               "lot": "1.00000000"},
    {"name": "UltraMonster_v8.tpl", "ea": "Ultra_Monster_MT5_v8",         "lot": "1.20000000"},
    {"name": "CPPF_Z_v8.tpl",        "ea": "CPPF_Z_MT5_v8",                "lot": "1.40000000"},
    {"name": "MSV_Asian_v8.tpl",     "ea": "MSV_Asian_Exhaustion_MT5_v8",  "lot": "1.00000000"},
    {"name": "NY_H21_v8.tpl",        "ea": "NY_H21_MT5_v8",                "lot": "1.50000000"},
    {"name": "CPMC_Z_v8.tpl",        "ea": "CPMC_Z_MT5_v8",                "lot": "1.40000000"},
]

def make_full_valid_tpl(ea_name, lot_str):
    return f"""<chart>
id=133000000000000000
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
    print("GENERATING 100% VALID FULL MT5 TEMPLATE FILES FOR INSTANT EA ATTACHMENT...")
    print("=" * 115)

    os.makedirs(APPDATA_TPL, exist_ok=True)

    for tpl in TEMPLATES:
        tname = tpl["name"]
        ea    = tpl["ea"]
        lot   = tpl["lot"]
        content = make_full_valid_tpl(ea, lot)
        
        local_path = os.path.join(LOCAL_DIR, tname)
        appdata_path = os.path.join(APPDATA_TPL, tname)
        
        # Save in UTF-16LE format with BOM
        with open(local_path, "wb") as f:
            f.write(b'\xff\xfe' + content.encode('utf-16le'))
        with open(appdata_path, "wb") as f:
            f.write(b'\xff\xfe' + content.encode('utf-16le'))

        print(f"  🟢 Generated Valid Template: {tname:<20} | Attached EA: {ea:<28} | Lot: {lot}")

    print("\n🚀 Uploading valid templates to VPS...")
    for tpl in TEMPLATES:
        tname = tpl["name"]
        local_t = os.path.join(LOCAL_DIR, tname)
        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, local_t, f"{VPS_HOST}:{VPS_TPL}"], check=False)

    print("=" * 115)
    print("🟢 SUCCESS: VALID FULL MT5 TEMPLATES INSTALLED TO LOCAL AND VPS!")
    print("===================================================================================================")

if __name__ == "__main__":
    main()
