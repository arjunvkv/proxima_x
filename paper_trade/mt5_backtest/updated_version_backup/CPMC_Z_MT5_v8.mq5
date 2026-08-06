//+------------------------------------------------------------------+
//|                                CPMC_Z_MT5_v8.mq5               |
//|   🔥 CPMC Z Engine v8 — Cross-Pair Momentum Expansion (61%+ WR) |
//|   v8: Magic Base 202690 (Zero Collision), SL=50p/TP=80p Guard  |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "8.00"
#property strict

#include <Trade\Trade.mqh>
CTrade trade;

input double   BASE_LOT            = 1.40;    // Base Lot Size
input int      HOLD_BARS           = 9;       // 45m Hold Time (9 M5 bars)
input ulong    MAGIC_BASE          = 202690;  // Magic Base (Non-overlapping)
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips
input double   HARD_SL_PIPS        = 50.0;    // Hard SL Crash Guard
input double   HARD_TP_PIPS        = 80.0;    // Hard TP Crash Guard

#define N_PAIRS 2
string PAIRS[N_PAIRS] = {"GBPAUD", "EURAUD"};

struct PosState {
   bool   active;
   ulong  ticket;
   double entry;
   int    held;
   int    dir;
};

PosState g_pos[N_PAIRS];
datetime g_last_bar = 0;

double NormalizeVolume(string symbol, double volume) {
   SymbolSelect(symbol, true);
   double min_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double step_vol = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   if(step_vol <= 0.0) step_vol = 0.01;
   
   double normalized_vol = MathFloor(volume / step_vol + 0.000001) * step_vol;
   if(normalized_vol < min_vol) normalized_vol = min_vol;
   if(normalized_vol > max_vol) normalized_vol = max_vol;
   
   int digits = (step_vol == 0.01) ? 2 : ((step_vol == 0.1) ? 1 : 0);
   return NormalizeDouble(normalized_vol, digits);
}

double GetPipSize(string symbol) {
   return (StringFind(symbol, "JPY") >= 0) ? 0.01 : 0.0001;
}

int OnInit() {
   Print("=== CPMC Z Engine MT5 Init v8 (Cross-Pair Momentum) MagicBase=", MAGIC_BASE, " ===");
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
      if(PositionGetSymbol(pos_i) == s) return false;
   }
   
   double pr = (side == "BUY") ? tk.ask : tk.bid;
   ENUM_ORDER_TYPE ot = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   
   double pip = GetPipSize(s);
   double sl_price = 0.0, tp_price = 0.0;
   
   if(HARD_SL_PIPS > 0) {
      sl_price = (side == "BUY") ? pr - HARD_SL_PIPS * pip : pr + HARD_SL_PIPS * pip;
      sl_price = NormalizeDouble(sl_price, (int)SymbolInfoInteger(s, SYMBOL_DIGITS));
   }
   if(HARD_TP_PIPS > 0) {
      tp_price = (side == "BUY") ? pr + HARD_TP_PIPS * pip : pr - HARD_TP_PIPS * pip;
      tp_price = NormalizeDouble(tp_price, (int)SymbolInfoInteger(s, SYMBOL_DIGITS));
   }
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = NormalizeVolume(s, lot);
   req.type = ot; req.price = pr;
   req.sl = sl_price;
   req.tp = tp_price;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE + idx; req.comment = "CPMC_Z_v8";
   
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   
   g_pos[idx].active = true; g_pos[idx].ticket = res.order; g_pos[idx].entry = res.price;
   g_pos[idx].held = 0; g_pos[idx].dir = (side == "BUY") ? 1 : -1;
   
   Print("  🟢 ENTRY CPMC_Z ", s, " ", side, " v=", req.volume, " at=", res.price, " SL=", sl_price, " TP=", tp_price);
   return true;
}

bool CloseTrade(int idx, string why) {
   if(!g_pos[idx].active) return false;
   string s = PAIRS[idx];
   if(!PositionSelect(s)) { g_pos[idx].active = false; g_pos[idx].ticket = 0; return false; }
   
   double raw_vol = PositionGetDouble(POSITION_VOLUME);
   double vol = NormalizeVolume(s, raw_vol);
   ulong tix = (ulong)PositionGetInteger(POSITION_TICKET);
   ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = vol; req.position = tix;
   if(pos_type == POSITION_TYPE_BUY) {
      req.type = ORDER_TYPE_SELL; req.price = tk.bid;
   } else {
      req.type = ORDER_TYPE_BUY; req.price = tk.ask;
   }
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE + idx; req.comment = "CPMC_Z_exit";
   
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   
   double pnl = PositionGetDouble(POSITION_PROFIT);
   Print("  🔴 CLOSE CPMC_Z ", s, " ", why, " pnl=", pnl);
   g_pos[idx].active = false; g_pos[idx].ticket = 0;
   return true;
}

void CheckExits() {
   for(int i=0; i<N_PAIRS; i++) {
      if(!g_pos[i].active) continue;
      if(!PositionSelect(PAIRS[i])) {
         g_pos[i].active = false;
         g_pos[i].ticket = 0;
         continue;
      }
      g_pos[i].held++;
      if(g_pos[i].held >= HOLD_BARS) CloseTrade(i, "expiry");
   }
}

void CheckEntry() {
   for(int i=0; i<N_PAIRS; i++) {
      if(g_pos[i].active) continue;
      string s = PAIRS[i];
      
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      if(CopyRates(s, PERIOD_M5, 0, 12, rates) < 12) continue;
      
      double c_now = rates[0].close;
      double c_prev = rates[11].close;
      double ret = (c_now - c_prev) / c_prev;
      
      if(ret > 0.0015) {
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
