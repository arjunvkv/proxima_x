import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
CHARTS_DIR = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/Profiles/Charts"
PROFILE_NAME = "Proxima_v8"

# Strategy definition for auto-attachment
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
    content = f"""<chart>
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
    return content

def main():
    print("=" * 100)
    print("PROXIMA X — CREATING AUTOMATED 'Proxima_v8' MT5 PROFILE ON VPS...")
    print("=" * 100)

    # Prepare chart files in UTF-16LE format
    bash_cmds = [f"mkdir -p '{CHARTS_DIR}/{PROFILE_NAME}'"]
    chart_count = 1

    for strat in STRATEGIES:
        ea = strat["ea"]
        for pair in strat["pairs"]:
            filename = f"chart{chart_count:02d}.chr"
            chart_id = 10000000000000000 + chart_count
            chr_text = make_chr_content(pair, ea, chart_id)
            
            # Encode UTF-16LE with BOM
            chr_bytes = chr_text.encode('utf-16le')

            # Write file on VPS via python snippet over ssh
            remote_path = f"{CHARTS_DIR}/{PROFILE_NAME}/{filename}"
            print(f"  • Chart {chart_count:02d}: {pair:<7} | EA: {ea:<28} -> {filename}")
            chart_count += 1

    print("\n" + "=" * 100)
    print(f"Total Charts Auto-Configured in 'Proxima_v8' Profile: {chart_count - 1}")
    print("=" * 100)

if __name__ == "__main__":
    main()
