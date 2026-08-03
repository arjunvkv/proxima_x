#include <Trade\Trade.mqh>
CTrade trade;

#property copyright "Proxima Trading"
#property version   "1.06"
#property strict

input double   BASE_LOT            = 0.15;    // Base Lot Size
input int      HOLD_BARS           = 3;       // 15m hold time (3 M5 bars)
input double   MIN_RANGE_PIPS      = 6.0;     // Minimum 1-hour range pips (6.0p)
input ulong    MAGIC_BASE          = 202600;  // Magic Base
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips

#define N_PAIRS 9
string PAIRS[N_PAIRS] = {
   "EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD",
   "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"
};

struct PosState {
   bool     active;
   ulong    ticket;
   double   entry;
   int      held;
   int      dir; // 1 = BUY, -1 = SELL
};

PosState g_pos[N_PAIRS];
datetime g_last_bar = 0;

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
   Print("=== 🔥 REAL ULTRA MONSTER (Rolling ORB v1.06) Init | Pairs: 9 | MinRange: ", MIN_RANGE_PIPS, "p | Hold: ", HOLD_BARS, " bars ===");
   for(int i=0; i<N_PAIRS; i++) {
      g_pos[i].active = false;
      g_pos[i].ticket = 0;
      g_pos[i].held = 0;
      g_pos[i].dir = 0;
   }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {}

bool OpenTrade(int idx, string side, double lot) {
   string s = PAIRS[idx];
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
   ulong magic = MAGIC_BASE + idx;
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = NormalizeVolume(s, lot);
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = magic; req.comment = "UltraMonster_ORB";
   
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   
   g_pos[idx].active = true; g_pos[idx].ticket = res.order; g_pos[idx].entry = res.price;
   g_pos[idx].held = 0; g_pos[idx].dir = (side == "BUY") ? 1 : -1;
   
   double fill_p = (res.price > 0.0) ? res.price : pr;
   double sl_target = NormalizeDouble((side == "BUY") ? fill_p - sl_d : fill_p + sl_d, digits);
   double tp_target = NormalizeDouble((side == "BUY") ? fill_p + tp_d : fill_p - tp_d, digits);
   
   AttachSLTPDirectly(s, magic, sl_target, tp_target);
   
   Print("  🔥 ULTRA MONSTER BREAKOUT ENTRY ", s, " ", side, " v=", lot, " at=", res.price);
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
   for(int i=0; i<N_PAIRS; i++) {
      if(!g_pos[i].active) continue;
      string s = PAIRS[i];
      g_pos[i].held++;
      
      if(g_pos[i].held >= HOLD_BARS) {
         if(ClosePositionByTicket(g_pos[i].ticket, s, "HOLD_EXPIRED")) {
            g_pos[i].active = false; g_pos[i].ticket = 0;
         }
      }
   }
}

void CheckEntry() {
   MqlDateTime dt; TimeCurrent(dt);
   if(dt.day_of_week == 0 && dt.hour == 0) return;
   
   // Trigger on 00 and 30 minute M5 bars
   if(dt.min != 0 && dt.min != 30) return;
   
   for(int i=0; i<N_PAIRS; i++) {
      if(g_pos[i].active) continue;
      string s = PAIRS[i];
      
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      // Copy last 13 M5 bars (bar 0 is current bar, bars 1..12 cover previous 60 minutes)
      if(CopyRates(s, PERIOD_M5, 0, 13, rates) < 13) continue;
      
      double h_prev = rates[1].high;
      double l_prev = rates[1].low;
      for(int k=2; k<=12; k++) {
         if(rates[k].high > h_prev) h_prev = rates[k].high;
         if(rates[k].low < l_prev)  l_prev = rates[k].low;
      }
      
      double mult = (StringFind(s, "JPY") >= 0) ? 100.0 : 10000.0;
      double range_pips = (h_prev - l_prev) * mult;
      
      if(range_pips < MIN_RANGE_PIPS) continue;
      
      double c_now = rates[0].close;
      if(c_now > h_prev) {
         OpenTrade(i, "BUY", BASE_LOT);
      }
      else if(c_now < l_prev) {
         OpenTrade(i, "SELL", BASE_LOT);
      }
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
