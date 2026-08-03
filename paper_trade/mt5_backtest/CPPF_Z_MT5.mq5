#include <Trade\Trade.mqh>
CTrade trade;
//+------------------------------------------------------------------+
//|                                                   CPPF_Z_MT5.mq5 |
//|          Cross-Pair Volatility Dislocation Strategy (Z >= 5.0)   |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.00"
#property strict

input double   Z_THRESH            = 6.0;     // Z-score threshold (6-Sigma shock)
input int      LOOKBACK_BARS       = 200;     // 200 M5 bars rolling window
input int      RET_PERIOD          = 3;       // 3-bar (15-min) return
input int      HOLD_BARS           = 18;      // 90m hold time
input double   BASE_LOT            = 2.5;     // Base Lot Size
input ulong    MAGIC_BASE          = 202623;  // Magic Base
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips

#define N_CPPF 5
string CPPF_PAIRS[N_CPPF] = {"EURNZD", "AUDNZD", "GBPNZD", "GBPAUD", "EURAUD"};

struct PositionInfo {
   bool     active;
   ulong    ticket;
   double   entry;
   int      held;
   string   side;
};

PositionInfo g_t[N_CPPF];
datetime     g_last_bar = 0;
int          g_bar = 0;


double NormalizeVolume(string symbol, double volume) {
   SymbolSelect(symbol, true);
   double min_vol = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_vol = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double step_vol = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   if(step_vol <= 0.0) step_vol = 0.01;
   
   double normalized_vol = MathRound(volume / step_vol) * step_vol;
   if(normalized_vol < min_vol) normalized_vol = min_vol;
   if(normalized_vol > max_vol) normalized_vol = max_vol;
   
   int digits = (step_vol == 0.01) ? 2 : ((step_vol == 0.1) ? 1 : 0);
   return NormalizeDouble(normalized_vol, digits);
}

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

int OnInit() {
   Print("=== CPPF Z v1.00  z=", Z_THRESH, " lookback=", LOOKBACK_BARS, " hold=", HOLD_BARS, " ===");
   for(int i=0; i<N_CPPF; i++) { g_t[i].active=false; g_t[i].ticket=0; g_t[i].held=0; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {
   for(int i=0; i<N_CPPF; i++) { g_t[i].active=false; g_t[i].ticket=0; }
}

bool Open(int p, string side, double lot) {
   string s = CPPF_PAIRS[p];
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   // Global check across all EAs: Do not open if position on pair already exists in terminal
   for(int pos_i=PositionsTotal()-1; pos_i>=0; pos_i--) {
      if(PositionGetSymbol(pos_i) == s) {
         Print("⚠️ Blocked conflicting trade on ", s, " (Position already exists in terminal!)");
         return false;
      }
   }
   double pr = (side == "BUY") ? tk.ask : tk.bid;
   ENUM_ORDER_TYPE ot = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = (float)NormalizeVolume(s, lot);
   req.type = ot; req.price = pr;
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.60 : 0.0060; // 60p Outer Emergency SL
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.80 : 0.0080; // 80p Outer Safety TP
   req.sl = (side == "BUY") ? pr - sl_d : pr + sl_d;
   req.tp = (side == "BUY") ? pr + tp_d : pr - tp_d;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE + p; req.comment = "CPPF_Z_entry";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_t[p].active = true; g_t[p].ticket = res.order; g_t[p].entry = res.price;
   g_t[p].held = 0; g_t[p].side = side;
   Print("  ENTRY ", s, " ", side, " v=", lot, " at=", res.price);
   return true;
}





bool ClosePositionByTicket(ulong ticket, string s, string why) {
   if(!PositionSelectByTicket(ticket)) return false;
   double raw_vol = PositionGetDouble(POSITION_VOLUME);
   double vol = NormalizeVolume(s, raw_vol);
   if(vol <= 0.0) return false;
   
   // Primary Exit: CTrade PositionClose Partial with exact normalized volume step
   if(trade.PositionClosePartial(ticket, vol)) {
      Print("  CLOSE SUCCESS (Partial) ", s, " ticket=", ticket, " vol=", vol, " why=", why);
      return true;
   }
   if(trade.PositionClose(ticket)) {
      Print("  CLOSE SUCCESS (CTrade) ", s, " ticket=", ticket, " vol=", vol, " why=", why);
      return true;
   }
   if(trade.PositionClose(s)) {
      Print("  CLOSE SUCCESS (Symbol) ", s, " why=", why);
      return true;
   }
   return false;
}

void CheckEntry() {
   // --- SUNDAY OPEN 60-MINUTE FREEZE (Block 00:00 to 01:00 MT5 Server Time / 22:00 to 23:00 UTC) ---
   MqlDateTime _dt_fr; TimeCurrent(_dt_fr);
   if(_dt_fr.day_of_week == 0 && _dt_fr.hour == 0) return;

   for(int i=0; i<N_CPPF; i++) {
      if(g_t[i].active) continue;
      string s = CPPF_PAIRS[i];
      
      // Calculate 3-bar return history over LOOKBACK_BARS window
      double returns[];
      ArrayResize(returns, LOOKBACK_BARS);
      double sum = 0.0;
      
      for(int k=0; k<LOOKBACK_BARS; k++) {
         double c_now = iClose(s, PERIOD_M5, k);
         double c_prev = iClose(s, PERIOD_M5, k + RET_PERIOD);
         if(c_now <= 0 || c_prev <= 0) continue;
         returns[k] = MathLog(c_now / c_prev);
         sum += returns[k];
      }
      
      double mean = sum / LOOKBACK_BARS;
      double sq_sum = 0.0;
      for(int k=0; k<LOOKBACK_BARS; k++) {
         sq_sum += (returns[k] - mean) * (returns[k] - mean);
      }
      double std_dev = MathSqrt(sq_sum / LOOKBACK_BARS);
      if(std_dev <= 0) continue;
      
      double cur_ret = returns[0];
      double z_score = (cur_ret - mean) / std_dev;
      
      if(z_score <= -Z_THRESH) {
         Open(i, "BUY", BASE_LOT);
      } else if(z_score >= Z_THRESH) {
         Open(i, "SELL", BASE_LOT);
      }
   }
}

void OnTick() {
   datetime t[1];
   if(CopyTime(_Symbol, PERIOD_M5, 0, 1, t) != 1) return;
   if(t[0] == g_last_bar) return;
   g_last_bar = t[0]; g_bar++;
   CheckExits();
   CheckEntry();
}