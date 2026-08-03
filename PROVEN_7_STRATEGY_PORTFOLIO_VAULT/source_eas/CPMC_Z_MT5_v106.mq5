#include <Trade\Trade.mqh>
CTrade trade;

#property copyright "Proxima Trading"
#property version   "1.06"
#property strict

input double   Z_THRESH            = 4.5;     // Z-score threshold (>= 4.5 for high conviction)
input int      LOOKBACK_BARS       = 200;     // Rolling window bars
input int      HOLD_BARS           = 9;       // 45m hold time
input double   BASE_LOT            = 0.15;    // Base Lot Size for $6k account
input ulong    MAGIC_BASE          = 202626;  // Magic Base
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips

#define N_UNIV 2
string UNIV[N_UNIV] = {"GBPAUD", "GBPNZD"};

struct PositionInfo {
   bool     active;
   ulong    ticket;
   double   entry;
   int      held;
   int      dir; // 1 = BUY, -1 = SELL
};

PositionInfo g_t[N_UNIV];
datetime     g_last_bar = 0;

double NormalizeVolume(string symbol, double volume) {
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
               }
            }
         }
      }
   }
}

int OnInit() {
   Print("=== CPMC Momentum Continuation v1.06 z=", Z_THRESH, " hold=", HOLD_BARS, " ===");
   for(int i=0; i<N_UNIV; i++) { g_t[i].active=false; g_t[i].ticket=0; g_t[i].held=0; g_t[i].dir=0; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {}

bool Open(int p, string side, double lot) {
   string s = UNIV[p];
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   
   for(int pos_i=PositionsTotal()-1; pos_i>=0; pos_i--) {
      if(PositionGetSymbol(pos_i) == s) {
         Print("⚠️ Blocked conflicting trade on ", s, " (Position already exists in terminal!)");
         return false;
      }
   }
   double pr = (side == "BUY") ? tk.ask : tk.bid;
   ENUM_ORDER_TYPE ot = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.35 : 0.0035;
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.45 : 0.0045;
   int digits = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
   ulong magic = MAGIC_BASE + p;
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = NormalizeVolume(s, lot);
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = magic; req.comment = "CPMC_Mom_entry";
   
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_t[p].active = true; g_t[p].ticket = res.order; g_t[p].entry = res.price;
   g_t[p].held = 0; g_t[p].dir = (side == "BUY") ? 1 : -1;
   
   double fill_p = (res.price > 0.0) ? res.price : pr;
   double sl_target = NormalizeDouble((side == "BUY") ? fill_p - sl_d : fill_p + sl_d, digits);
   double tp_target = NormalizeDouble((side == "BUY") ? fill_p + tp_d : fill_p - tp_d, digits);
   
   AttachSLTPDirectly(s, magic, sl_target, tp_target);
   
   Print("  ENTRY ", s, " ", side, " v=", lot, " at=", res.price);
   return true;
}

bool ClosePositionByTicket(ulong ticket, string s, string why) {
   if(!PositionSelectByTicket(ticket)) return false;
   double raw_vol = PositionGetDouble(POSITION_VOLUME);
   double vol = NormalizeVolume(s, raw_vol);
   if(vol <= 0.0) return false;
   
   if(trade.PositionClosePartial(ticket, vol)) {
      Print("  CLOSE SUCCESS (Partial) ", s, " ticket=", ticket, " vol=", vol, " why=", why);
      return true;
   }
   if(trade.PositionClose(ticket)) {
      Print("  CLOSE SUCCESS (CTrade) ", s, " ticket=", ticket, " vol=", vol, " why=", why);
      return true;
   }
   return false;
}

void CheckExits() {
   for(int i=0; i<N_UNIV; i++) {
      if(!g_t[i].active) continue;
      string s = UNIV[i];
      g_t[i].held++;
      
      if(g_t[i].held >= HOLD_BARS) {
         if(ClosePositionByTicket(g_t[i].ticket, s, "HOLD_EXPIRED")) {
            g_t[i].active = false; g_t[i].ticket = 0;
         }
      }
   }
}

void CheckEntry() {
   MqlDateTime _dt_fr; TimeCurrent(_dt_fr);
   if(_dt_fr.day_of_week == 0 && _dt_fr.hour == 0) return;

   for(int i=0; i<N_UNIV; i++) {
      if(g_t[i].active) continue;
      string s = UNIV[i];
      
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      if(CopyRates(s, PERIOD_M5, 0, LOOKBACK_BARS + 10, rates) < LOOKBACK_BARS) continue;
      
      double sum = 0, sum_sq = 0;
      int n = LOOKBACK_BARS;
      for(int k=1; k<=n; k++) {
         double ret = (rates[k-1].close - rates[k].close) / rates[k].close;
         sum += ret; sum_sq += ret * ret;
      }
      double mean = sum / n;
      double var = (sum_sq / n) - (mean * mean);
      double std = (var > 0) ? MathSqrt(var) : 0.0001;
      
      double curr_ret = (rates[0].close - rates[1].close) / rates[1].close;
      double z = (curr_ret - mean) / std;
      
      if(z >= Z_THRESH) Open(i, "BUY", BASE_LOT);
      else if(z <= -Z_THRESH) Open(i, "SELL", BASE_LOT);
   }
}

void OnTick() {
   MqlDateTime dt; TimeCurrent(dt);
   datetime cur_bar = TimeCurrent() - (dt.sec % 300);
   if(cur_bar != g_last_bar) {
      g_last_bar = cur_bar;
      CheckExits();
      CheckEntry();
   }
}
