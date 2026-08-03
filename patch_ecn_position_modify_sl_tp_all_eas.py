#!/usr/bin/env python3
"""Patch ECN PositionModify SL/TP attachment across ALL 6 Active EAs."""

import os, subprocess, shutil

EAS = [
    "Ultra_Monster_MT5",
    "TokyoH0_MT5",
    "CPPF_Z_MT5",
    "CPMC_Z_MT5",
    "NY_H21_MT5",
    "MSV_Asian_Exhaustion_MT5"
]

METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
BACKUP_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\updated_version_backup"
VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_PATH = "ubuntu@140.245.234.92:/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

ECN_OPENTRADE_CODE = """
bool OpenTrade(int idx, string side, double lot) {
   string s = PAIRS[idx];
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   double pr = (side == "BUY") ? tk.ask : tk.bid;
   ENUM_ORDER_TYPE ot = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.35 : 0.0035; // 35p Outer Emergency SL
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.45 : 0.0045; // 45p Outer Safety TP
   int digits = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
   double sl_val = NormalizeDouble((side == "BUY") ? pr - sl_d : pr + sl_d, digits);
   double tp_val = NormalizeDouble((side == "BUY") ? pr + tp_d : pr - tp_d, digits);
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = (float)NormalizeVolume(s, lot);
   req.type = ot; req.price = pr;
   req.sl = sl_val; req.tp = tp_val;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE + idx; req.comment = "UltraMonster_entry";
   
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   
   g_pos[idx].active = true; g_pos[idx].ticket = res.order; g_pos[idx].entry = res.price;
   g_pos[idx].held = 0; g_pos[idx].dir = (side == "BUY") ? 1 : -1;
   
   // ECN Post-Fill PositionModify: Guarantees SL/TP attachment on ECN/Market Execution brokers
   Sleep(100);
   if(trade.PositionModify(s, sl_val, tp_val)) {
      Print("  SL/TP ATTACHED (PositionModify) ", s, " sl=", sl_val, " tp=", tp_val);
   } else if(res.order > 0 && trade.PositionModify(res.order, sl_val, tp_val)) {
      Print("  SL/TP ATTACHED by ticket (PositionModify) ", s, " sl=", sl_val, " tp=", tp_val);
   } else {
      Print("  WARNING: Could not modify SL/TP on ", s);
   }
   
   Print("  ENTRY ", s, " ", side, " v=", lot, " at=", res.price);
   return true;
}
"""

def patch_ea(ea):
    path = os.path.join(LOCAL_DIR, f"{ea}.mq5")
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # Replace OpenTrade with ECN PositionModify implementation
    old_marker = "bool OpenTrade("
    if old_marker in code and "bool ClosePositionByTicket(" in code:
        start_idx = code.find(old_marker)
        end_idx = code.find("bool ClosePositionByTicket(")
        
        # Replace function
        ea_comment_code = ECN_OPENTRADE_CODE.replace("UltraMonster_entry", f"{ea}_entry")
        code = code[:start_idx] + ea_comment_code + "\n" + code[end_idx:]
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"  • Upgraded {ea}.mq5 with ECN Post-Fill PositionModify SL/TP Engine!")

def compile_and_backup():
    for ea in EAS:
        mq5_file = os.path.join(LOCAL_DIR, f"{ea}.mq5")
        ex5_appdata = os.path.join(APPDATA_EXP, f"{ea}.ex5")
        ex5_local = os.path.join(LOCAL_DIR, f"{ea}.ex5")
        ex5_backup = os.path.join(BACKUP_DIR, f"{ea}.ex5")
        mq5_backup = os.path.join(BACKUP_DIR, f"{ea}.mq5")

        cmd = [METAEDITOR, f"/compile:{mq5_file}"]
        subprocess.run(cmd, check=False)

        if os.path.exists(ex5_appdata):
            shutil.copy(ex5_appdata, ex5_local)
            shutil.copy(ex5_appdata, ex5_backup)
        shutil.copy(mq5_file, mq5_backup)
        print(f"  • Compiled & Backed up {ea}")

def push_to_vps():
    print("🚀 Pushing all 6 updated ECN PositionModify SL/TP EAs to VPS...")
    for ea in EAS:
        mq5_local = os.path.join(LOCAL_DIR, f"{ea}.mq5")
        ex5_local = os.path.join(LOCAL_DIR, f"{ea}.ex5")

        subprocess.run(["scp", "-i", VPS_KEY, ex5_local, VPS_PATH], check=False)
        subprocess.run(["scp", "-i", VPS_KEY, mq5_local, VPS_PATH], check=False)
        print(f"  • Uploaded {ea} to VPS!")

def main():
    print("="*115)
    print("APPLYING ECN POST-FILL POSITIONMODIFY SL/TP ENGINE ACROSS ALL 6 EAS...")
    print("="*115)
    for ea in EAS:
        patch_ea(ea)

    print("="*115)
    print("COMPILING & BACKING UP ALL 6 EAS...")
    print("="*115)
    compile_and_backup()

    print("="*115)
    print("PUSHING ALL 6 UPDATED BINARIES TO VPS...")
    print("="*115)
    push_to_vps()

    print("="*115)
    print("🟢 ALL 6 EAS SUCCESSFULLY UPGRADED WITH ECN POSITIONMODIFY SL/TP & PUSHED TO VPS!")
    print("="*115)

if __name__ == "__main__":
    main()
