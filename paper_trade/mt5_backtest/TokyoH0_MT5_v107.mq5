//+------------------------------------------------------------------+
//|                                TokyoH0_MT5_v107.mq5              |
//|   🔥 TOKYO H0 Engine v107 — 18-Pair UTC Midnight Mean Reversion |
//|   v107: Magic Base 202630 (Zero Collision), Clean Positions      |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.07"
#property strict

#include <Trade\Trade.mqh>
CTrade trade;

input int      LOOKBACK_BARS      = 6;       // 30-min Lookback
input int      HOLD_BARS          = 12;      // 60-min Hold Time
input int      TOP_N              = 5;       // Top 5 Pairs
input int      SESSION_HOUR       = 0;       // 00:00 UTC Session (gated on TimeGMT)
input double   BASE_LOT           = 0.15;    // Lot Size
input ulong    MAGIC_BASE         = 202630;  // Magic Base (Non-overlapping)
input double   SLIPPAGE_PIPS      = 1.0;     // Slippage Pips
input double   HARD_SL_PIPS       = 50.0;    // Hard SL Crash Guard
input double   HARD_TP_PIPS       = 80.0;    // Hard TP Crash Guard

#define N_PAIRS 18
string PAIRS[N_PAIRS] = {
   "EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY",
   "GBPJPY","EURAUD","EURNZD","GBPAUD","GBPNZD",
   "GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP",
   "EURCHF","USDCHF","AUDJPY"
};

struct TradeState {
   ulong  ticket;
   bool   active;
   double entry;
   int    held;
};

TradeState g_t[N_PAIRS];
datetime   g_last_bar = 0;

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
                  return;
               }
            }
         }
      }
   }
}

int OnInit() {
   Print("=== Tokyo H0 v1.07 lb=", LOOKBACK_BARS, " hold=", HOLD_BARS, " top_n=", TOP_N, " MagicBase=", MAGIC_BASE, " ===");
   for(int i=0; i<N_PAIRS; i++) { g_t[i].active=false; g_t[i].ticket=0; g_t[i].held=0; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {}

bool OpenTrade(int p, double lot) {
   string s = PAIRS[p];
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   
   for(int pos_i=PositionsTotal()-1; pos_i>=0; pos_i--) {
      if(PositionGetSymbol(pos_i) == s) return false;
   }
   double pr = tk.ask;
   double pip = (StringFind(s, "JPY") >= 0) ? 0.01 : 0.0001;
   double sl_d = HARD_SL_PIPS * pip;
   double tp_d = HARD_TP_PIPS * pip;
   int digits = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
   ulong magic = MAGIC_BASE + p;
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = NormalizeVolume(s, lot);
   req.type = ORDER_TYPE_BUY; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = magic; req.comment = "TokyoH0_v107";
   
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_t[p].active = true; g_t[p].ticket = res.order; g_t[p].entry = res.price; g_t[p].held = 0;
   
   double fill_p = (res.price > 0.0) ? res.price : pr;
   double sl_target = NormalizeDouble(fill_p - sl_d, digits);
   double tp_target = NormalizeDouble(fill_p + tp_d, digits);
   
   AttachSLTPDirectly(s, magic, sl_target, tp_target);
   
   Print("  🟢 ENTRY TokyoH0 ", s, " @", res.price, " lot=", lot);
   return true;
}

bool ClosePositionByTicket(ulong ticket, string s, string why) {
   if(!PositionSelectByTicket(ticket)) return false;
   double raw_vol = PositionGetDouble(POSITION_VOLUME);
   double vol = NormalizeVolume(s, raw_vol);
   if(vol <= 0.0) return false;
   
   if(trade.PositionClosePartial(ticket, vol)) {
      return true;
   }
   if(trade.PositionClose(ticket)) {
      return true;
   }
   return false;
}

void CheckExits() {
   for(int i=0; i<N_PAIRS; i++) {
      if(!g_t[i].active) continue;
      string s = PAIRS[i];
      if(!PositionSelect(s)) {
         g_t[i].active = false;
         g_t[i].ticket = 0;
         continue;
      }
      g_t[i].held++;
      
      if(g_t[i].held >= HOLD_BARS) {
         if(ClosePositionByTicket(g_t[i].ticket, s, "HOLD_EXPIRED")) {
            g_t[i].active = false; g_t[i].ticket = 0;
         }
      }
   }
}

void CheckEntry() {
   MqlDateTime dt; TimeToStruct(TimeGMT(), dt);
   if(dt.hour != SESSION_HOUR || dt.min > 5) return;
   
   double returns[N_PAIRS];
   int ids[N_PAIRS];
   
   for(int i=0; i<N_PAIRS; i++) {
      ids[i] = i;
      returns[i] = 0.0;
      string s = PAIRS[i];
      
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      if(CopyRates(s, PERIOD_M5, 0, LOOKBACK_BARS + 2, rates) >= LOOKBACK_BARS) {
         returns[i] = (rates[0].close - rates[LOOKBACK_BARS].close) / rates[LOOKBACK_BARS].close;
      }
   }
   
   for(int i=0; i<N_PAIRS-1; i++) {
      for(int j=i+1; j<N_PAIRS; j++) {
         if(returns[j] < returns[i]) {
            double tr = returns[i]; returns[i] = returns[j]; returns[j] = tr;
            int ti = ids[i]; ids[i] = ids[j]; ids[j] = ti;
         }
      }
   }
   
   int opened = 0;
   for(int k=0; k<N_PAIRS && opened < TOP_N; k++) {
      int p = ids[k];
      if(!g_t[p].active) {
         if(OpenTrade(p, BASE_LOT)) opened++;
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
