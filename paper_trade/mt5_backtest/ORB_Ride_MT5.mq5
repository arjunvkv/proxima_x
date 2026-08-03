//+------------------------------------------------------------------+
//|                                    ORB_Ride_MT5.mq5              |
//|    ORB Breakout Ride (#4) — London Open Momentum Expansion EA    |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.00"
#property strict

input int      HOLD_BARS           = 12;      // 60m hold time
input double   BASE_LOT            = 0.15;    // Base Lot Size for $6k account
input ulong    MAGIC_BASE          = 202644;  // Magic Base
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips

#define N_UNIV 5
string UNIV[N_UNIV] = {"EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD"};

struct PosState {
   bool   active;
   ulong  ticket;
   double entry;
   int    held;
   int    dir;
};

PosState g_pos[N_UNIV];
datetime g_last_bar = 0;

int OnInit() {
   Print("=== ORB Breakout Ride (#4) EA Init v1.02 ===");
   for(int i=0; i<N_UNIV; i++) { g_pos[i].active=false; g_pos[i].ticket=0; g_pos[i].held=0; g_pos[i].dir=0; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {}

bool OpenTrade(int idx, string side, double lot) {
   string s = UNIV[idx];
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   double pr = (side == "BUY") ? tk.ask : tk.bid;
   ENUM_ORDER_TYPE ot = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = lot;
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE + idx; req.comment = "ORB_Ride_entry";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_pos[idx].active = true; g_pos[idx].ticket = res.order; g_pos[idx].entry = res.price;
   g_pos[idx].held = 0; g_pos[idx].dir = (side == "BUY") ? 1 : -1;
   Print("  ENTRY ", s, " ", side, " v=", lot, " at=", res.price);
   return true;
}

bool CloseTrade(int idx, string why) {
   if(!g_pos[idx].active) return false;
   string s = UNIV[idx];
   if(!PositionSelect(s)) { g_pos[idx].active = false; g_pos[idx].ticket = 0; return false; }
   double vol = NormalizeDouble(PositionGetDouble(POSITION_VOLUME), 2);
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
   req.magic = MAGIC_BASE + idx; req.comment = "ORB_Ride_exit";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   double pnl = PositionGetDouble(POSITION_PROFIT);
   Print("  CLOSE ", s, " ", why, " pnl=", pnl);
   g_pos[idx].active = false; g_pos[idx].ticket = 0;
   return true;
}

void CheckExits() {
   for(int i=0; i<N_UNIV; i++) {
      if(!g_pos[i].active) continue;
      g_pos[i].held++;
      if(g_pos[i].held >= HOLD_BARS) CloseTrade(i, "expiry");
   }
}

void CheckEntry() {
   MqlDateTime dt;
   TimeCurrent(dt);

   // Convert MT5 Server time (UTC+3) to UTC hour (10 MT5 Server = 07 UTC London Open)
   if(dt.hour >= 10 && dt.hour <= 12) {
      for(int i=0; i<N_UNIV; i++) {
         if(g_pos[i].active) continue;
         string s = UNIV[i];
         
         // 30-min Opening Range (06:30 to 07:00 UTC -> 6 bars back)
         double orb_high = -1.0;
         double orb_low = 999999.0;
         for(int k=1; k<=6; k++) {
            double h = iHigh(s, PERIOD_M5, k);
            double l = iLow(s, PERIOD_M5, k);
            if(h > orb_high) orb_high = h;
            if(l < orb_low) orb_low = l;
         }
         double c_now = iClose(s, PERIOD_M5, 0);
         if(c_now <= 0 || orb_high <= 0) continue;

         // Breakout Ride: BUY if price breaks above ORB High, SELL if below ORB Low
         if(c_now > orb_high) {
            OpenTrade(i, "BUY", BASE_LOT);
         } else if(c_now < orb_low) {
            OpenTrade(i, "SELL", BASE_LOT);
         }
      }
   }
}

void OnTick() {
   datetime t[1];
   if(CopyTime(_Symbol, PERIOD_M5, 0, 1, t) != 1) return;
   if(t[0] == g_last_bar) return;
   g_last_bar = t[0];
   CheckExits();
   CheckEntry();
}
