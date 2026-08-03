#include <Trade\Trade.mqh>
CTrade trade;
//+------------------------------------------------------------------+
//| TokyoH0_MT5.mq5 - v1.05 Auto-Sync SymbolSelect for All 18 Pairs  |
//| Best config: lb=6, hold=12, top_n=5 (30min lookback, 60min hold) |
//+------------------------------------------------------------------+
#property version "1.05"
#property description "Tokyo H0 - Enter LONG at UTC midnight on most-declined FX pairs"

input int      LOOKBACK_BARS      = 6;
input int      HOLD_BARS          = 12;
input int      TOP_N              = 5;
input int      SESSION_HOUR       = 0;
input int      MIN_PAIRS          = 3;       // 3 Min Valid Pairs (Fixed Live Gate)
input double   GAP_THRESHOLD_PCT  = 0.5;
input double   BASE_LOT           = 0.15;
input double   MAX_SPREAD_PIPS    = 5.0;
input int      MAGIC_BASE         = 202607;
input double   SLIPPAGE_PIPS      = 1.0;

string PAIRS[] = {
   "EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY",
   "GBPJPY","EURAUD","EURNZD","GBPAUD","GBPNZD",
   "GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP",
   "EURCHF","USDCHF","AUDJPY"
};
#define N_PAIRS 18
#define EXTRA_BARS 100

struct Series {
   datetime time[];
   double   close[];
   double   open[];
   int      total;
   int      idx;
};
Series g_s[N_PAIRS];

double g_vbuf[N_PAIRS][50] = {};
int g_vn[N_PAIRS] = {};
int g_vh[N_PAIRS] = {};

struct Trade {
   long   ticket;
   bool   active;
   double entry;
   int    held;
   int    bar;
};
Trade g_t[N_PAIRS];
int g_act = 0;

datetime g_last_bar = 0;
int g_bar = 0;
datetime g_last_entry = 0;

double Pt(string s) { return StringFind(s,"JPY")>=0 ? 0.01 : 0.0001; }

bool LoadRange(int i, datetime from, datetime thru) {
   string s = PAIRS[i];
   SymbolSelect(s, true);
   double buf[];
   ArrayResize(buf, EXTRA_BARS+50);
   int c = CopyClose(s, PERIOD_M5, from, thru, buf);
   if(c <= 2) return false;
   ArrayResize(g_s[i].close, c);
   ArrayCopy(g_s[i].close, buf, 0, 0, c);
   ArrayResize(g_s[i].time, c);
   ArrayResize(g_s[i].open, c);
   if(CopyTime(s, PERIOD_M5, from, thru, g_s[i].time) != c) return false;
   if(CopyOpen(s, PERIOD_M5, from, thru, g_s[i].open) != c) return false;
   g_s[i].total = c;
   g_s[i].idx = c-1;
   return true;
}

void SyncIdx() {
   for(int i=0; i<N_PAIRS; i++) {
      if(g_s[i].total <= 0) continue;
      int ci = g_s[i].idx;
      while(ci < g_s[i].total-1 && g_s[i].time[ci] < g_last_bar) ci++;
      while(ci > 0 && g_s[i].time[ci] > g_last_bar) ci--;
      if(ci == g_s[i].total-1 && g_s[i].time[ci] < g_last_bar) {
         if(LoadRange(i, g_last_bar - 3600*96, g_last_bar + 3600)) {
            ci = g_s[i].idx;
         }
      }
      g_s[i].idx = ci;
   }
}

double GetC(int i, int shift) {
   int target_idx = g_s[i].idx - shift;
   if(target_idx < 0 || target_idx >= g_s[i].total) return -1.0;
   return g_s[i].close[target_idx];
}

double Vol(int p, double ret) {
   int n = g_vn[p];
   g_vbuf[p][n % 50] = MathAbs(ret);
   g_vn[p]++;
   int sz = MathMin(g_vn[p], 50);
   double sum = 0;
   for(int k=0; k<sz; k++) sum += g_vbuf[p][k];
   return sum / sz;
}

void Sort(double &r[], int &id[], int n) {
   for(int i=0; i<n-1; i++) {
      for(int j=i+1; j<n; j++) {
         if(r[j] < r[i]) {
            double tr = r[i]; r[i] = r[j]; r[j] = tr;
            int ti = id[i]; id[i] = id[j]; id[j] = ti;
         }
      }
   }
}

bool NewBar() {
   datetime t[1];
   if(CopyTime(_Symbol, PERIOD_M5, 0, 1, t) != 1) return false;
   if(t[0] == g_last_bar) return false;
   g_last_bar = t[0];
   g_bar++;
   return true;
}

bool Open(int p, double lot) {
   string s = PAIRS[p];
   SymbolSelect(s, true);
   MqlTick tk;
   if(!SymbolInfoTick(s,tk)) { Print("  FAIL ", s, ": no tick"); return false; }
   if(tk.ask<=0) { Print("  FAIL ", s, ": ask=", tk.ask); return false; }
   double sl = tk.ask - 500.0*Pt(s)*10.0;
   MqlTradeRequest req={}; MqlTradeResult res={};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = s; req.volume = (float)NormalizeVolume(s, lot);
   req.type = ORDER_TYPE_BUY; req.price = tk.ask; req.sl = sl;
   req.deviation = (ulong)(SLIPPAGE_PIPS*10);
   req.magic = MAGIC_BASE+p; req.comment = "TokyoH0";
   bool sent = OrderSend(req,res);
   if(!sent) { Print("  FAIL ", s, ": send fail rc=", res.retcode, " desc=", res.comment); return false; }
   if(res.retcode!=TRADE_RETCODE_DONE) { Print("  FAIL ", s, ": rc=", res.retcode, " (want DONE)"); return false; }
   g_t[p].active = true; g_t[p].ticket = res.order;
   g_t[p].entry = res.price; g_t[p].held = 0; g_t[p].bar = g_bar;
   g_act++;
   Print("  OPEN ", s, " @", res.price, " lot=", lot);
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

   int active_cnt = 0;
   for(int i=0; i<N_PAIRS; i++) if(g_t[i].active) active_cnt++;
   if(active_cnt > 0) return;
   MqlDateTime dt; TimeToStruct(g_last_bar, dt);
   int h = dt.hour;
   if(h!=SESSION_HOUR || dt.min > 15) return; // Allow 00:00 to 00:15 UTC entry window
   MqlDateTime td; TimeToStruct(g_last_bar, td);
   td.hour=0; td.min=0; td.sec=0;
   datetime today = StructToTime(td);
   if(today == g_last_entry) return;

   int lb = LOOKBACK_BARS; if(lb<2) lb=2;
   double r[]; int id[];
   ArrayResize(r, N_PAIRS); ArrayResize(id, N_PAIRS);
   int vc=0;
   for(int i=0;i<N_PAIRS;i++) {
      if(g_s[i].total <= lb+2) continue;
      double cur = GetC(i,0), prv = GetC(i,lb);
      if(cur<=0 || prv<=0) continue;
      double ret = MathLog(cur/prv);
      double pv = GetC(i,1);
      if(pv>0) { double gp=MathAbs(cur-pv)/pv*100.0; if(gp>=GAP_THRESHOLD_PCT) continue; }
      int cb = MathMin(3,lb/2);
      if(cb>=2) { double ps=GetC(i,cb); if(ps>0 && MathLog(cur/ps)>0) continue; }
      r[vc] = ret; id[vc] = i; vc++;
   }
   if(vc < MIN_PAIRS) { Print("  SKIP ", vc, " < ", MIN_PAIRS); return; }
   Sort(r, id, vc);
   int te = MathMin(TOP_N, vc);
   Print("  ENTRY v=", vc, " top", te, " bar=", g_last_bar);
   int en=0;
   for(int i=0;i<te;i++) {
      if(r[i]>=0) break;
      int p = id[i];
      double vol = Vol(p, r[i]);
      double margin = MathAbs(r[i]) / MathMax(vol, 1e-10);
      double conf = MathMin(0.95, margin * 0.15);
      if(conf < 0.01) { Print("  SKIP ", PAIRS[p], " conf=", conf); continue; }
      if(Open(p, BASE_LOT)) en++;
   }
   Print("  Entered ", en, "/", te);
   g_last_entry = today;
}

void CheckExits() {
   for(int i=0;i<N_PAIRS;i++) {
      if(!g_t[i].active) continue;
      g_t[i].held++;
      if(g_t[i].held >= HOLD_BARS) Close(i, "expiry");
   }
}


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
   Print("=== Tokyo H0 v1.06  lb=", LOOKBACK_BARS, " hold=", HOLD_BARS, " n=", TOP_N, " ===");
   if(!SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE)) return INIT_FAILED;
   for(int i=0;i<N_PAIRS;i++) {
      g_t[i].active=false; g_t[i].ticket=0; g_vn[i]=0; g_vh[i]=0;
      SymbolSelect(PAIRS[i], true);
   }
   for(int i=0;i<N_PAIRS;i++) for(int j=0;j<50;j++) g_vbuf[i][j]=0.0;
   Print("Loading ", N_PAIRS, " pairs...");
   for(int attempt=1; attempt<=3; attempt++) {
      for(int i=0;i<N_PAIRS;i++) {
         if(g_s[i].total <= 0) LoadRange(i, TimeCurrent()-3600*96, TimeCurrent()+3600*24*300);
      }
      int c=0; for(int i=0;i<N_PAIRS;i++) if(g_s[i].total>0) c++;
      if(c >= N_PAIRS) {
         Print(c, "/", N_PAIRS, " fully loaded and ready!");
         break;
      }
      if(attempt < 3) Sleep(1000);
   }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {
   for(int i=0;i<N_PAIRS;i++) { g_t[i].active=false; g_t[i].ticket=0; }
   g_act=0;
}

void OnTick() {
   if(!NewBar()) return;
   SyncIdx();
   CheckExits();
   CheckEntry();
}