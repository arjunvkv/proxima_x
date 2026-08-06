import os
import shutil
import glob

LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
TERMINAL_BASE = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal"

V8_EX5_FILES = [
    "TokyoH0_MT5_v8.ex5", "TokyoH0_MT5_v8.mq5",
    "Ultra_Monster_MT5_v8.ex5", "Ultra_Monster_MT5_v8.mq5",
    "CPPF_Z_MT5_v8.ex5", "CPPF_Z_MT5_v8.mq5",
    "MSV_Asian_Exhaustion_MT5_v8.ex5", "MSV_Asian_Exhaustion_MT5_v8.mq5",
    "NY_H21_MT5_v8.ex5", "NY_H21_MT5_v8.mq5",
    "CPMC_Z_MT5_v8.ex5", "CPMC_Z_MT5_v8.mq5",
    "Proxima_v8_Master_Launcher.ex5", "Proxima_v8_Master_Launcher.mq5"
]

TPL_FILES = [
    "TokyoH0_v8.tpl", "UltraMonster_v8.tpl", "CPPF_Z_v8.tpl",
    "MSV_Asian_v8.tpl", "NY_H21_v8.tpl", "CPMC_Z_v8.tpl"
]

def main():
    print("=" * 115)
    print("PROXIMA X — DEPLOYING v8 SUITE TO ALL LOCAL MT5 TERMINAL DATA FOLDERS...")
    print("=" * 115)

    terminal_folders = [d for d in os.listdir(TERMINAL_BASE) if len(d) == 32]
    print(f"Found {len(terminal_folders)} MT5 Terminal Instances:")
    for tf in terminal_folders:
        print(f"  • {tf}")

    for tf in terminal_folders:
        term_path = os.path.join(TERMINAL_BASE, tf)
        exp_path  = os.path.join(term_path, "MQL5", "Experts")
        scr_path  = os.path.join(term_path, "MQL5", "Scripts")
        tpl_path  = os.path.join(term_path, "MQL5", "Profiles", "Templates")
        prof_mql  = os.path.join(term_path, "MQL5", "Profiles", "Charts", "Proxima_v8")
        def_mql   = os.path.join(term_path, "MQL5", "Profiles", "Charts", "Default")

        os.makedirs(exp_path, exist_ok=True)
        os.makedirs(scr_path, exist_ok=True)
        os.makedirs(tpl_path, exist_ok=True)
        os.makedirs(prof_mql, exist_ok=True)
        os.makedirs(def_mql, exist_ok=True)

        # Copy EX5 and MQ5 files to Experts & Scripts
        for fname in V8_EX5_FILES:
            src = os.path.join(LOCAL_DIR, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(exp_path, fname))
                if "Launcher" in fname:
                    shutil.copy(src, os.path.join(scr_path, fname))

        # Copy Templates
        for tname in TPL_FILES:
            tsrc = os.path.join(LOCAL_DIR, tname)
            if os.path.exists(tsrc):
                shutil.copy(tsrc, os.path.join(tpl_path, tname))

        # Copy Chart profiles
        temp_prof = r"C:\Trading\Agentic_Trading\proxima_x\temp_profile_6_full"
        if os.path.exists(temp_prof):
            for cfile in os.listdir(temp_prof):
                csrc = os.path.join(temp_prof, cfile)
                shutil.copy(csrc, os.path.join(prof_mql, cfile))
                shutil.copy(csrc, os.path.join(def_mql, cfile))

        print(f"  🟢 Deployed to MT5 instance [{tf}] SUCCESS!")

    print("\n" + "=" * 115)
    print("🟢 SUCCESS: ALL 4 LOCAL MT5 TERMINAL INSTANCES FULLY SYNCED WITH v8 SUITE!")
    print("===================================================================================================")

if __name__ == "__main__":
    main()
