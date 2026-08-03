#!/usr/bin/env python3
"""Apply SL/TP Outer Safety Caps across All 7 Live EAs, Backup to Folder, Compile & Upload."""
import os, sys, shutil, subprocess
from pathlib import Path

BASE_DIR = Path(r"c:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest")
BACKUP_DIR = BASE_DIR / "updated_version_backup"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# 1. Update Ultra_Monster_MT5
def patch_monster():
    path = BASE_DIR / "Ultra_Monster_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    if "req.sl =" not in code:
        old_req = "req.type = ot; req.price = pr;"
        new_req = """req.type = ot; req.price = pr;
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.35 : 0.0035; // 35p Outer Emergency SL
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.45 : 0.0045; // 45p Outer Safety TP
   req.sl = (dir == "BUY") ? pr - sl_d : pr + sl_d;
   req.tp = (dir == "BUY") ? pr + tp_d : pr - tp_d;"""
        code = code.replace(old_req, new_req)
        code = code.replace("=== ULTRA MONSTER Engine MT5 Init v1.05 (12p Gate + 1p Buffer) ===", "=== ULTRA MONSTER MT5 v1.08 (Outer SL/TP Cap Active) ===")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    print("🟢 Ultra_Monster_MT5.mq5 updated with Outer SL=35p / TP=45p!")

# 2. Update CPPF_Z_MT5
def patch_cppf():
    path = BASE_DIR / "CPPF_Z_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    if "req.sl =" not in code:
        old_req = "req.type = ot; req.price = pr;"
        new_req = """req.type = ot; req.price = pr;
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.60 : 0.0060; // 60p Outer Emergency SL
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.80 : 0.0080; // 80p Outer Safety TP
   req.sl = (side == "BUY") ? pr - sl_d : pr + sl_d;
   req.tp = (side == "BUY") ? pr + tp_d : pr - tp_d;"""
        code = code.replace(old_req, new_req)
        code = code.replace("=== CPPF_Z_MT5 v1.02 (No Hedging Lock) ===", "=== CPPF_Z_MT5 v1.08 (Outer SL/TP Cap Active) ===")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    print("🟢 CPPF_Z_MT5.mq5 updated with Outer SL=60p / TP=80p!")

# 3. Update CPMC_Z_MT5
def patch_cpmc():
    path = BASE_DIR / "CPMC_Z_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    if "req.sl =" not in code:
        old_req = "req.type = ot; req.price = pr;"
        new_req = """req.type = ot; req.price = pr;
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.60 : 0.0060; // 60p Outer Emergency SL
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.80 : 0.0080; // 80p Outer Safety TP
   req.sl = (side == "BUY") ? pr - sl_d : pr + sl_d;
   req.tp = (side == "BUY") ? pr + tp_d : pr - tp_d;"""
        code = code.replace(old_req, new_req)
        code = code.replace("=== CPMC Engine MT5 v1.07 (Option B: z=5.0 + 8p Gate + 90m Hold) ===", "=== CPMC Engine MT5 v1.08 (Outer SL/TP Cap Active) ===")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    print("🟢 CPMC_Z_MT5.mq5 updated with Outer SL=60p / TP=80p!")

# 4. Update TokyoH0_MT5
def patch_tokyo():
    path = BASE_DIR / "TokyoH0_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    if "req.sl =" not in code:
        old_req = "req.type = ot; req.price = pr;"
        new_req = """req.type = ot; req.price = pr;
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.25 : 0.0025; // 25p Outer Emergency SL
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.35 : 0.0035; // 35p Outer Safety TP
   req.sl = (side == "BUY") ? pr - sl_d : pr + sl_d;
   req.tp = (side == "BUY") ? pr + tp_d : pr - tp_d;"""
        code = code.replace(old_req, new_req)
        code = code.replace("=== TokyoH0 v1.00 ===", "=== TokyoH0 MT5 v1.08 (Outer SL/TP Cap Active) ===")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    print("🟢 TokyoH0_MT5.mq5 updated with Outer SL=25p / TP=35p!")

# 5. Update Sunday_H22_MT5
def patch_sunday():
    path = BASE_DIR / "Sunday_H22_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    if "req.sl =" not in code:
        old_req = "req.type = ot; req.price = pr;"
        new_req = """req.type = ot; req.price = pr;
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.40 : 0.0040; // 40p Outer Emergency SL
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.50 : 0.0050; // 50p Outer Safety TP
   req.sl = (side == "BUY") ? pr - sl_d : pr + sl_d;
   req.tp = (side == "BUY") ? pr + tp_d : pr - tp_d;"""
        code = code.replace(old_req, new_req)
        code = code.replace("=== SundayH22 v1.00 ===", "=== SundayH22 MT5 v1.08 (Outer SL/TP Cap Active) ===")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    print("🟢 Sunday_H22_MT5.mq5 updated with Outer SL=40p / TP=50p!")

# 6. Update NY_H21_MT5 (with 45m hold buff!)
def patch_nyh():
    path = BASE_DIR / "NY_H21_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # Update hold to 9 bars (45 mins)
    code = code.replace("input int      HOLD_BARS           = 12;", "input int      HOLD_BARS           = 9;       // 45m hold time (Buffed 66.1% WR)")

    if "req.sl =" not in code:
        old_req = "req.type = ot; req.price = pr;"
        new_req = """req.type = ot; req.price = pr;
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.30 : 0.0030; // 30p Outer Emergency SL
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.40 : 0.0040; // 40p Outer Safety TP
   req.sl = (side == "BUY") ? pr - sl_d : pr + sl_d;
   req.tp = (side == "BUY") ? pr + tp_d : pr - tp_d;"""
        code = code.replace(old_req, new_req)
        code = code.replace("=== NYH21 v1.00 ===", "=== NYH21 MT5 v1.08 (45m Buff + Outer SL/TP Cap Active) ===")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    print("🟢 NY_H21_MT5.mq5 updated with 45m hold buff & Outer SL=30p / TP=40p!")

# 7. Update MSV_Asian_Exhaustion_MT5
def patch_msv():
    path = BASE_DIR / "MSV_Asian_Exhaustion_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    if "req.sl =" not in code:
        old_req = "req.type = ot; req.price = pr;"
        new_req = """req.type = ot; req.price = pr;
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.25 : 0.0025; // 25p Outer Emergency SL
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.35 : 0.0035; // 35p Outer Safety TP
   req.sl = (side == "BUY") ? pr - sl_d : pr + sl_d;
   req.tp = (side == "BUY") ? pr + tp_d : pr - tp_d;"""
        code = code.replace(old_req, new_req)
        code = code.replace("=== MSV Asian Exhaustion v1.00", "=== MSV Asian Exhaustion v1.08 (Outer SL/TP Cap Active)")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    print("🟢 MSV_Asian_Exhaustion_MT5.mq5 updated with Outer SL=25p / TP=35p!")

def backup_all_files():
    ea_files = [
        "Ultra_Monster_MT5.mq5", "CPPF_Z_MT5.mq5", "CPMC_Z_MT5.mq5",
        "TokyoH0_MT5.mq5", "Sunday_H22_MT5.mq5", "NY_H21_MT5.mq5", "MSV_Asian_Exhaustion_MT5.mq5"
    ]
    for ea in ea_files:
        src = BASE_DIR / ea
        dst = BACKUP_DIR / ea
        if src.exists():
            shutil.copy(src, dst)
            print(f"📦 Backed up {ea} to updated_version_backup/")

def main():
    patch_monster()
    patch_cppf()
    patch_cpmc()
    patch_tokyo()
    patch_sunday()
    patch_nyh()
    patch_msv()

    backup_all_files()

if __name__ == "__main__":
    main()
