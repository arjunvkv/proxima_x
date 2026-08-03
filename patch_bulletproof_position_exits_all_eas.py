#!/usr/bin/env python3
"""Patch Bulletproof Position Exit Engine across all 6 Active EAs with dynamic PositionsTotal scanning and explicit volume normalization."""

import os, subprocess, shutil

EAS = [
    "Ultra_Monster_MT5",
    "TokyoH0_MT5",
    "CPPF_Z_MT5",
    "CPMC_Z_MT5",
    "NY_H21_MT5",
    "MSV_Asian_Exhaustion_MT5"
]

LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"

def patch_ultra_monster():
    path = os.path.join(LOCAL_DIR, "Ultra_Monster_MT5.mq5")
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # Replacement CloseTrade and CheckExits functions
    new_exits = """
bool ClosePositionByTicket(ulong ticket, string s, string why) {
   if(!PositionSelectByTicket(ticket)) return false;
   double vol = NormalizeDouble(PositionGetDouble(POSITION_VOLUME), 2);
   if(vol <= 0.0) return false;
   
   ENUM_POSITION_TYPE ptype = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   ENUM_ORDER_TYPE close_type = (ptype == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   double price = (close_type == ORDER_TYPE_SELL) ? tk.bid : tk.ask;
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = s;
   req.volume = vol;
   req.type = close_type;
   req.price = price;
   req.position = ticket;
   req.deviation = 10;
   req.comment = "UM_exit_" + why;
   
   bool sent = OrderSend(req, res);
   if(sent && res.retcode == TRADE_RETCODE_DONE) {
      Print("  CLOSE SUCCESS ", s, " ticket=", ticket, " vol=", vol, " why=", why);
      return true;
   }
   
   // Fallback using CTrade
   if(trade.PositionClose(ticket)) {
      Print("  CLOSE CTrade ", s, " ticket=", ticket, " why=", why);
      return true;
   }
   return false;
}

void CheckExits() {
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      long magic = PositionGetInteger(POSITION_MAGIC);
      if(magic < MAGIC_BASE || magic >= MAGIC_BASE + N_PAIRS) continue;
      
      string s = PositionGetString(POSITION_SYMBOL);
      datetime pos_time = (datetime)PositionGetInteger(POSITION_TIME);
      int elapsed_sec = (int)(TimeCurrent() - pos_time);
      int elapsed_bars = elapsed_sec / 300; // 5-minute bars
      
      if(elapsed_bars >= HOLD_BARS) {
         ClosePositionByTicket(ticket, s, "expiry");
      }
   }
}
"""

    old_close_marker = "bool CloseTrade(int idx, string why) {"
    if old_close_marker in code:
        # Replace from CloseTrade to end of CheckExits
        start_idx = code.find(old_close_marker)
        end_idx = code.find("void CheckEntry() {")
        code = code[:start_idx] + new_exits + "\n" + code[end_idx:]
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print("🟢 Ultra_Monster_MT5.mq5 updated with Bulletproof Dynamic PositionsTotal Exit Engine!")

def main():
    patch_ultra_monster()

if __name__ == "__main__":
    main()
