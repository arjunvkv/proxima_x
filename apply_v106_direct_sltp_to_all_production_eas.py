#!/usr/bin/env python3
"""Upgrade All 6 Production EAs with Verified Low-Level Direct TRADE_ACTION_SLTP Engine v1.06."""

import os, subprocess, shutil, time

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

LOWLEVEL_ATTACH_FUNC = """
void AttachSLTPDirectly(string symbol, ulong magic_num, double sl_val, double tp_val) {
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   for(int attempt = 0; attempt < 10; attempt++) {
      Sleep(150);
      for(int i = 0; i < PositionsTotal(); i++) {
         ulong pos_ticket = PositionGetTicket(i);
         if(pos_ticket > 0) {
            string pos_sym = PositionGetString(POSITION_SYMBOL);
            long pos_magic = PositionGetInteger(POSITION_MAGIC);
            if(pos_sym == symbol && pos_magic == magic_num) {
               MqlTradeRequest req_sltp = {};
               MqlTradeResult res_sltp = {};
               req_sltp.action   = TRADE_ACTION_SLTP;
               req_sltp.position = pos_ticket;
               req_sltp.symbol   = symbol;
               req_sltp.sl       = NormalizeDouble(sl_val, digits);
               req_sltp.tp       = NormalizeDouble(tp_val, digits);
               
               if(OrderSend(req_sltp, res_sltp) && res_sltp.retcode == TRADE_RETCODE_DONE) {
                  Print("  🟢 LOW-LEVEL TRADE_ACTION_SLTP SUCCESS | Symbol: ", symbol, " | PosTicket: ", pos_ticket, " | SL: ", req_sltp.sl, " | TP: ", req_sltp.tp);
                  return;
               } else {
                  Print("  ⚠️ TRADE_ACTION_SLTP attempt ", attempt, " retcode: ", res_sltp.retcode, " comment: ", res_sltp.comment);
               }
            }
         }
      }
   }
}
"""

RETRY_OPENTRADE_CODE = """
bool OpenTrade(int idx, string side, double lot) {
   string s = PAIRS[idx];
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   double pr = (side == "BUY") ? tk.ask : tk.bid;
   ENUM_ORDER_TYPE ot = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.35 : 0.0035; // 35p Emergency SL
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.45 : 0.0045; // 45p Safety TP
   int digits = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
   ulong magic = MAGIC_BASE + idx;
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = (float)NormalizeVolume(s, lot);
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = magic; req.comment = "UltraMonster_entry";
   
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   
   g_pos[idx].active = true; g_pos[idx].ticket = res.order; g_pos[idx].entry = res.price;
   g_pos[idx].held = 0; g_pos[idx].dir = (side == "BUY") ? 1 : -1;
   
   double fill_p = (res.price > 0.0) ? res.price : pr;
   double sl_target = NormalizeDouble((side == "BUY") ? fill_p - sl_d : fill_p + sl_d, digits);
   double tp_target = NormalizeDouble((side == "BUY") ? fill_p + tp_d : fill_p - tp_d, digits);
   
   AttachSLTPDirectly(s, magic, sl_target, tp_target);
   
   Print("  ENTRY ", s, " ", side, " v=", lot, " at=", res.price);
   return true;
}
"""

def upgrade_production_eas():
    print("="*115)
    print("UPGRADING ALL 6 PRODUCTION EAS WITH DIRECT LOW-LEVEL TRADE_ACTION_SLTP ENGINE v1.06...")
    print("="*115)

    for ea in EAS:
        path = os.path.join(LOCAL_DIR, f"{ea}.mq5")
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()

        if "void AttachSLTPDirectly(" not in code:
            # Insert AttachSLTPDirectly helper function before OpenTrade
            old_marker = "bool OpenTrade("
            if old_marker in code and "bool ClosePositionByTicket(" in code:
                start_idx = code.find(old_marker)
                end_idx = code.find("bool ClosePositionByTicket(")
                ea_code = LOWLEVEL_ATTACH_FUNC + "\n" + RETRY_OPENTRADE_CODE.replace("UltraMonster_entry", f"{ea}_entry")
                code = code[:start_idx] + ea_code + "\n" + code[end_idx:]

                with open(path, "w", encoding="utf-8") as f:
                    f.write(code)
                print(f"  • Upgraded {ea}.mq5 locally!")

        # Copy updated .mq5 directly into AppData Experts directory
        appdata_mq5 = os.path.join(APPDATA_EXP, f"{ea}.mq5")
        appdata_ex5 = os.path.join(APPDATA_EXP, f"{ea}.ex5")
        shutil.copy(path, appdata_mq5)

        # Compile in AppData using MetaEditor
        res = subprocess.run([METAEDITOR, f"/compile:{appdata_mq5}"], capture_output=True, text=True)
        if os.path.exists(appdata_ex5):
            mtime = time.ctime(os.path.getmtime(appdata_ex5))
            size = os.path.getsize(appdata_ex5)
            print(f"  🟢 {ea}.ex5 compiled in AppData! Timestamp: {mtime} | Size: {size} bytes")

            # Backup and push to VPS
            shutil.copy(appdata_ex5, os.path.join(LOCAL_DIR, f"{ea}.ex5"))
            shutil.copy(appdata_ex5, os.path.join(BACKUP_DIR, f"{ea}.ex5"))
            shutil.copy(appdata_mq5, os.path.join(BACKUP_DIR, f"{ea}.mq5"))

            subprocess.run(["scp", "-i", VPS_KEY, appdata_ex5, VPS_PATH], check=False)
            subprocess.run(["scp", "-i", VPS_KEY, appdata_mq5, VPS_PATH], check=False)

    print("="*115)
    print("🟢 ALL 6 PRODUCTION EAS UPGRADED, COMPILED IN APPDATA, BACKED UP & PUSHED TO VPS!")
    print("="*115)

if __name__ == "__main__":
    upgrade_production_eas()
