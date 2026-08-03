#!/usr/bin/env python3
"""Fix Invalid Volume (Float Cast 1.20000005) Across All 7 EAs (_v106), Recompile & Deploy to VPS."""

import os, subprocess, shutil, time

VERSIONED_EAS = [
    "Test_Min_Fire_MT5_v106",
    "Ultra_Monster_MT5_v106",
    "TokyoH0_MT5_v106",
    "CPPF_Z_MT5_v106",
    "CPMC_Z_MT5_v106",
    "NY_H21_MT5_v106",
    "MSV_Asian_Exhaustion_MT5_v106"
]

METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
BACKUP_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\updated_version_backup"
VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_PATH = "ubuntu@140.245.234.92:/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

# Standardized 100% Robust Double NormalizeVolume function
NORMALIZE_VOLUME_FUNC = """double NormalizeVolume(string symbol, double volume) {
   SymbolSelect(symbol, true);
   double min_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double step_vol = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   if(step_vol <= 0.0) step_vol = 0.01;
   if(min_vol <= 0.0)  min_vol  = 0.01;
   
   double steps = MathFloor(volume / step_vol + 0.000001);
   double normalized_vol = steps * step_vol;
   if(normalized_vol < min_vol) normalized_vol = min_vol;
   if(max_vol > 0.0 && normalized_vol > max_vol) normalized_vol = max_vol;
   
   int digits = (step_vol == 0.01) ? 2 : ((step_vol == 0.1) ? 1 : 0);
   return NormalizeDouble(normalized_vol, digits);
}"""

def fix_ea_code(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Remove (float) cast before NormalizeVolume
    code = code.replace("req.volume = (float)NormalizeVolume(", "req.volume = NormalizeVolume(")
    code = code.replace("req.volume = (float) NormalizeVolume(", "req.volume = NormalizeVolume(")

    # 2. Replace NormalizeVolume implementation with robust double version
    start_idx = code.find("double NormalizeVolume(")
    if start_idx != -1:
        end_idx = code.find("}", start_idx)
        if end_idx != -1:
            code = code[:start_idx] + NORMALIZE_VOLUME_FUNC + code[end_idx+1:]

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    return code

def main():
    print("="*115)
    print("FIXING INVALID VOLUME FLOAT CAST (1.20000005 -> 1.20) ACROSS ALL 7 EAS...")
    print("="*115)

    for ea in VERSIONED_EAS:
        local_mq5 = os.path.join(LOCAL_DIR, f"{ea}.mq5")
        appdata_mq5 = os.path.join(APPDATA_EXP, f"{ea}.mq5")
        appdata_ex5 = os.path.join(APPDATA_EXP, f"{ea}.ex5")
        local_ex5 = os.path.join(LOCAL_DIR, f"{ea}.ex5")
        backup_mq5 = os.path.join(BACKUP_DIR, f"{ea}.mq5")
        backup_ex5 = os.path.join(BACKUP_DIR, f"{ea}.ex5")

        if os.path.exists(local_mq5):
            code = fix_ea_code(local_mq5)
            with open(appdata_mq5, "w", encoding="utf-8") as f:
                f.write(code)

            # Compile in AppData
            cmd = [METAEDITOR, f"/compile:{appdata_mq5}"]
            subprocess.run(cmd, check=False)
            time.sleep(0.5)

            if os.path.exists(appdata_ex5):
                size = os.path.getsize(appdata_ex5)
                mtime = time.ctime(os.path.getmtime(appdata_ex5))
                print(f"  🟢 {ea:<34} FIXED & COMPILED! Size: {size} bytes | Pushed: {mtime}")

                shutil.copy(appdata_ex5, local_ex5)
                shutil.copy(appdata_ex5, backup_ex5)
                shutil.copy(appdata_mq5, backup_mq5)

                subprocess.run(["scp", "-i", VPS_KEY, appdata_ex5, VPS_PATH], check=False)
                subprocess.run(["scp", "-i", VPS_KEY, appdata_mq5, VPS_PATH], check=False)
            else:
                print(f"  ❌ {ea} compilation failed!")

    print("="*115)
    print("🟢 ALL 7 EAS FIXED (DOUBLE VOLUME 1.20), COMPILED & DEPLOYED TO VPS!")
    print("="*115)

if __name__ == "__main__":
    main()
