#define N_PAIRS 3
string PAIRS[N_PAIRS] = {"EURJPY", "GBPJPY", "USDJPY"};

input int      LOOKBACK_BARS       = 12;      // 60m lookback
input int      HOLD_BARS           = 12;      // 60m hold time
input int      SESSION_HOUR        = 21;      // 21:00 UTC
input double   BASE_LOT            = 0.20;    // 0.20 Lot per pair for $6k account
input double   TRAILING_TRIG_PIPS  = 15.0;    // Trailing trigger (+15 pips)
input double   TRAILING_STEP_PIPS  = 10.0;    // Trailing distance (+10 pips)
input ulong    MAGIC_BASE          = 202621;  // Magic Base
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips

struct PositionInfo {
   bool     active;
   ulong    ticket;
   double   entry;
   int      held;
   datetime bar;
   bool     trailed;
};

PositionInfo g_t[N_PAIRS];
datetime     g_last_bar = 0;
int          g_bar = 0;

int OnInit() {
   Print("=== NY H21 v1.02  lb=", LOOKBACK_BARS, " hold=", HOLD_BARS, " lot=", BASE_LOT, " ===");
   for(int i=0; i<N_PAIRS; i++) { g_t[i].active=false; g_t[i].ticket=0; g_t[i].held=0; g_t[i].trailed=false; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {
   for(int i=0; i<N_PAIRS; i++) { g_t[i].active=false; g_t[i].ticket=0; }
}

bool Open(int p, string side, double lot) {
   string s = PAIRS[p];
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   double pr = (side == "BUY") ? tk.ask : tk.bid;
   ENUM_ORDER_TYPE ot = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = lot;
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE + p; req.comment = "NY_H21_entry";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_t[p].active = true; g_t[p].ticket = res.order; g_t[p].entry = res.price;
   g_t[p].held = 0; g_t[p].bar = g_last_bar; g_t[p].trailed = false;
   Print("  ENTRY ", s, " ", side, " v=", lot, " at=", res.price);
   return true;
}

bool Close(int p, string why) {
   if(!g_t[p].active) return false;
   string s = PAIRS[p];
   if(!PositionSelect(s)) { g_t[p].active = false; g_t[p].ticket = 0; return false; }
   double vol = (float)PositionGetDouble(POSITION_VOLUME);
   ulong tix = (ulong)PositionGetInteger(POSITION_TICKET);
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = vol;
   req.type = ORDER_TYPE_SELL; req.price = tk.bid;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE + p; req.comment = "NY_H21_exit"; req.position = tix;
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   double pnl = (tk.bid - g_t[p].entry) * vol * SymbolInfoDouble(s, SYMBOL_TRADE_CONTRACT_SIZE);
   if(StringFind(s, "JPY") >= 0) { MqlTick u; if(SymbolInfoTick("USDJPY", u) && u.bid > 0) pnl /= u.bid; }
   Print("  CLOSE ", s, " ", why, " pnl=", pnl);
   g_t[p].active = false; g_t[p].ticket = 0;
   return true;
}

void CheckExits() {
   for(int i=0; i<N_PAIRS; i++) {
      if(!g_t[i].active) continue;
      string s = PAIRS[i];
      MqlTick tk;
      if(SymbolInfoTick(s, tk)) {
         double pt = SymbolInfoDouble(s, SYMBOL_POINT);
         double gain_pips = (tk.bid - g_t[i].entry) / (pt * 10.0);
         if(gain_pips >= TRAILING_TRIG_PIPS && !g_t[i].trailed) {
            g_t[i].trailed = true;
            Print("  TRAILING LOCKED ", s, " gain_pips=", gain_pips);
         }
         if(g_t[i].trailed && gain_pips <= TRAILING_STEP_PIPS) {
            Close(i, "trailing_lock");
            continue;
         }
      }
      g_t[i].held++;
      if(g_t[i].held >= HOLD_BARS) Close(i, "expiry");
   }
}

#define N_UNIV 18
string UNIV[N_UNIV] = {"EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD","GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF","USDCHF","AUDJPY"};

void CheckEntry() {
   int active_cnt = 0;
   for(int i=0; i<N_PAIRS; i++) if(g_t[i].active) active_cnt++;
   if(active_cnt > 0) return;
   MqlDateTime dt; TimeToStruct(g_last_bar, dt);
   int h = dt.hour;
   if(h != SESSION_HOUR || (dt.min != 0 && dt.min != 5)) return;

   int decl_cnt = 0;
   double min_ret = 0.0;
   for(int i=0; i<N_UNIV; i++) {
      double cur = iClose(UNIV[i], PERIOD_M5, 0);
      double prv = iClose(UNIV[i], PERIOD_M5, LOOKBACK_BARS);
      if(cur <= 0 || prv <= 0) continue;
      double ret = MathLog(cur / prv);
      if(ret < 0) decl_cnt++;
   }
   if(decl_cnt < 8) return;

   for(int i=0; i<N_PAIRS; i++) {
      double cur = iClose(PAIRS[i], PERIOD_M5, 0);
      double prv = iClose(PAIRS[i], PERIOD_M5, LOOKBACK_BARS);
      if(cur <= 0 || prv <= 0) continue;
      double ret = MathLog(cur / prv);
      if(ret < -0.0001) {
         Open(i, "BUY", BASE_LOT);
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
