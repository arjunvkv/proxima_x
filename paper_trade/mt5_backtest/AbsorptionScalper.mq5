#property copyright "Proxima"
#property version "1.00"
#property description "Absorption Scalping Engine"
#property description "Tick-level flow absorption detection with fixed-hold exit."
#property description "No trailing stops, no z-scores, no bar-level logic."

input group "=== Strategy Parameters ==="
input double   InpImpulsePips     = 3.0;        // Impulse threshold (pips)
input int      InpImpulseWindowMs = 20000;       // Impulse detection window (ms)
input int      InpMinImpulseTicks = 10;         // Minimum ticks for impulse (noise filter)
input bool     InpUseFIR          = true;        // Use FIR EMA absorption filter
input double   InpFIR_Alpha       = 0.30;        // FIR EMA adaptation rate (0=slow, 1=fast)
input int      InpMaxBarSec        = 0;          // Only trade in first N sec of bar (0=off)
input int      InpHoldSec         = 60;          // Hold duration (seconds)
input int      InpCooldownSec     = 5;           // Cooldown after close (seconds)
input double   InpSL_Pips         = 15.0;        // Safety stop (pips)

input group "=== Risk ==="
input double   InpBaseLot         = 0.10;        // Lot size
input double   InpMaxSpreadPips   = 2.0;         // Max spread (pips)
input int      InpMagic           = 20260731;    // Magic number
input int      InpMaxDailyTrades  = 100;         // Daily trade limit
input int      InpSessionStart    = 7;           // Session start hour UTC
input int      InpSessionEnd      = 18;          // Session end hour UTC

double   gPipSize;
double   gPipPoint;
MqlTick  gTick;

bool     gImpActive    = false;
double   gImpStartBid  = 0.0;
double   gImpPeakBid   = 0.0;
double   gImpNadirBid  = 0.0;
int      gImpTicks     = 0;
int      gImpDir       = 0;
datetime gImpStartTime = 0;

double   gFirEMA   = -1.0;
int      gFirPass  = 0;
int      gFirFail  = 0;

bool     gPosActive     = false;
int      gPosDir        = 0;
double   gPosEntryPrice = 0.0;
double   gPosSL         = 0.0;
datetime gPosEntryTime  = 0;
ulong    gPosTicket     = 0;

int      gDayTrades = 0;
string   gDayStr    = "";
int      gTimerSec    = 0;
int      gCooldownSec = 0;
bool     gFirstTick   = true;

int      gDbgImpulseStarts = 0;
int      gDbgImpulseCancels = 0;
int      gDbgImpulseEvaluated = 0;
int      gDbgEntries = 0;

int OnInit() {
   if (!SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE)) {
      Print(_Symbol, " not tradeable"); return INIT_FAILED;
   }
   gPipPoint = 10.0 * _Point;
   gPipSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   Print("AbsorptionScalper on ", _Symbol, " pip=", gPipPoint,
         " tick_size=", gPipSize, " lot=", InpBaseLot,
         " impulse=", InpImpulsePips, "pip hold=", InpHoldSec, "s");

   EventSetTimer(1);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   EventKillTimer();
}

double GetSpreadPips() {
   long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   return (double)spread * tick_size / gPipPoint;
}

bool SessionCheck() {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;
   if (InpSessionStart < InpSessionEnd)
      return (h >= InpSessionStart && h < InpSessionEnd);
   else
      return (h >= InpSessionStart || h < InpSessionEnd);
}

void UpdateDayState() {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   string s = StringFormat("%04d%02d%02d", dt.year, dt.mon, dt.day);
   if (s != gDayStr) {
      gDayStr = s; gDayTrades = 0;
   }
}

double BidPipsFrom(double ref) {
   return (gTick.bid - ref) / gPipPoint;
}

void StartImpulse() {
   gImpActive    = true;
   gImpStartBid  = gTick.bid;
   gImpPeakBid   = gTick.bid;
   gImpNadirBid  = gTick.bid;
   gImpTicks     = 0;
   gImpStartTime = TimeCurrent();
}

void EndImpulse() {
   gImpActive = false;
   gImpStartBid = gTick.bid;
}

void UpdateFirEMA(double fir) {
   if (gFirEMA < 0.0) {
      gFirEMA = fir;
   } else {
      gFirEMA = (1.0 - InpFIR_Alpha) * gFirEMA + InpFIR_Alpha * fir;
   }
}

void EvaluateImpulse() {
   double move = 0.0;
   double fir  = 0.0;
   if (gImpDir > 0) move = gImpPeakBid - gImpStartBid;
   else             move = gImpStartBid - gImpNadirBid;
   if (move <= 0.0 || gImpTicks <= 0) { EndImpulse(); return; }
   if (gImpTicks < InpMinImpulseTicks) {
      gDbgImpulseCancels++;
      EndImpulse();
      return;
   }
   fir = move / (double)gImpTicks;
   UpdateFirEMA(fir);
   int dir = (gImpDir > 0) ? -1 : 1;
   bool trade = true;
   if (InpUseFIR) {
      trade = (fir <= gFirEMA);
      if (trade) gFirPass++; else gFirFail++;
   }
   if (trade) {
      gDbgEntries++;
      EnterTrade(dir);
   }
   EndImpulse();
}

void EnterTrade(int direction) {
   if (gPosActive) return;
   UpdateDayState();
   if (gDayTrades >= InpMaxDailyTrades) return;
   if (!SessionCheck()) return;
   if (GetSpreadPips() > InpMaxSpreadPips) return;
   if (InpMaxBarSec > 0) {
      int bar_sec = (int)(TimeCurrent() % 60);
      if (bar_sec > InpMaxBarSec) { gDbgImpulseCancels++; return; }
   }
   double price = (direction > 0) ? gTick.ask : gTick.bid;
   double sl = (direction > 0) ?
      (price - InpSL_Pips * gPipPoint) :
      (price + InpSL_Pips * gPipPoint);
   ENUM_ORDER_TYPE cmd = (direction > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   MqlTradeRequest req = {};
   MqlTradeResult   res = {};
   req.action   = TRADE_ACTION_DEAL;
   req.symbol   = _Symbol;
   req.volume   = InpBaseLot;
   req.type     = cmd;
   req.price    = price;
   req.sl       = sl;
   req.deviation = 100;
   req.magic    = InpMagic;
   req.comment  = "Absorb";

   if (!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) {
      Print("Entry fail: retcode=", res.retcode, " err=", GetLastError(), " dir=",
            (direction > 0 ? "BUY" : "SELL"), " price=", price, " sl=", sl);
      return;
   }
   gPosActive     = true;
   gPosDir        = direction;
   gPosEntryPrice = res.price;
   gPosSL         = sl;
   gPosEntryTime  = TimeCurrent();
   gPosTicket     = res.order;
   gDayTrades++;
   gTimerSec = 0;
   Print("OPEN dir=", (direction > 0 ? "BUY" : "SELL"),
         " entry=", gPosEntryPrice, " sl=", sl,
         " spread=", GetSpreadPips(), " trade#", gDayTrades);
}

void ClosePosition(string reason) {
   if (!gPosActive) return;
   MqlTick tick;
   if (!SymbolInfoTick(_Symbol, tick)) return;
   ENUM_ORDER_TYPE cmd = (gPosDir > 0) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   double price = (gPosDir > 0) ? tick.bid : tick.ask;

   MqlTradeRequest req = {};
   MqlTradeResult   res = {};
   req.action   = TRADE_ACTION_DEAL;
   req.symbol   = _Symbol;
   req.volume   = InpBaseLot;
   req.type     = cmd;
   req.price    = price;
   req.deviation = 100;
   req.magic    = InpMagic;
   req.position = gPosTicket;
   req.comment  = "Absorb_cl";

   if (!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) {
      Print("Close fail: retcode=", res.retcode, " err=", GetLastError());
      return;
   }
   double contract = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   double raw = (gPosDir > 0) ?
      (res.price - gPosEntryPrice) * InpBaseLot * contract :
      (gPosEntryPrice - res.price) * InpBaseLot * contract;
   Print("CLOSE rsn=", reason, " exit=", res.price, " raw=", raw);
   gPosActive     = false;
   gPosDir        = 0;
   gPosEntryPrice = 0.0;
   gPosSL         = 0.0;
   gPosEntryTime  = 0;
   gPosTicket     = 0;
   gCooldownSec = InpCooldownSec;
}

void OnTick() {
   if (!SymbolInfoTick(_Symbol, gTick)) return;

   if (gFirstTick) {
      gFirstTick    = false;
      gImpStartBid  = gTick.bid;
      gImpPeakBid   = gTick.bid;
      gImpNadirBid  = gTick.bid;
      return;
   }

   if (gPosActive) {
      if (gPosDir > 0 && gTick.bid <= gPosSL) ClosePosition("stop_loss");
      else if (gPosDir < 0 && gTick.ask >= gPosSL) ClosePosition("stop_loss");
      return;
   }

   if (gCooldownSec > 0) return;

   if (!gImpActive) {
      double move = gTick.bid - gImpStartBid;
      if (MathAbs(move) >= InpImpulsePips * gPipPoint) {
         gDbgImpulseStarts++;
         StartImpulse();
         gImpDir = (move > 0) ? 1 : -1;
      }
      return;
   }

   gImpTicks++;
   bool new_peak  = (gTick.bid > gImpPeakBid);
   bool new_nadir = (gTick.bid < gImpNadirBid);
   if (new_peak)  gImpPeakBid  = gTick.bid;
   if (new_nadir) gImpNadirBid = gTick.bid;

   double total_move = (gImpDir > 0) ?
      (gImpPeakBid - gImpStartBid) :
      (gImpStartBid - gImpNadirBid);

   double retrace = (gImpDir > 0) ?
      (gImpPeakBid - gTick.bid) :
      (gTick.bid - gImpNadirBid);

   if (retrace > total_move * 0.80 && gImpTicks > 5) {
      gDbgImpulseCancels++;
      EndImpulse();
      return;
   }

   int elapsed_ms = (int)((TimeCurrent() - gImpStartTime) * 1000);
   if (elapsed_ms >= InpImpulseWindowMs) {
      gDbgImpulseEvaluated++;
      EvaluateImpulse();
   }
}

void OnTimer() {
   if (gCooldownSec > 0) gCooldownSec--;
   if (gPosActive) {
      gTimerSec++;
      if (gTimerSec >= InpHoldSec) {
         ClosePosition("hold_expiry");
      }
   }
   static int dbg_counter = 0;
   dbg_counter++;
   if (dbg_counter % 300 == 0) {
      Print("DBG impulses=", gDbgImpulseStarts, " cancelled=", gDbgImpulseCancels,
            " evaluated=", gDbgImpulseEvaluated, " entries=", gDbgEntries,
            " fir_p/f=", gFirPass, "/", gFirFail, " pos=", gPosActive);
   }
}
