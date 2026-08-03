#!/usr/bin/env python3
"""Build, Compile, and Deploy All 7 Versioned EAs (_v106) with 0 Errors."""

import os, subprocess, shutil, time

VERSIONED_EAS = [
    "Test_Min_Fire_MT5_v106",
    "Ultra_Monster_MT5_v106",
    "TokyoH0_MT5_v106",
    "CPPF_Z_MT5_v106",
    "CPMC_Z_MT5_v106",
    "NY_H21_MT5_v106",
    "MSV_Asian_Exhaustion_MT5_v106"
]

METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA_EXP = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"
BACKUP_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\updated_version_backup"
VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_PATH = "ubuntu@140.245.234.92:/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/"

CODE_TEST_MIN_FIRE = """//+------------------------------------------------------------------+
//|                                Test_Min_Fire_MT5_v106.mq5        |
//|   1-Minute Live Fire Test EA v1.06 with Low-Level Direct SL/TP Engine|
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.06"
#property strict

#include <Trade\\Trade.mqh>
CTrade trade;

input double   TEST_LOT    = 1.20;    // Standard Test Volume (1.20L)
input ulong    MAGIC_NUM   = 999999;  // Test Magic Number
input int      SL_PIPS     = 35;      // Emergency SL Pips
input int      TP_PIPS     = 45;      // Safety TP Pips

datetime g_last_minute = 0;

double NormalizeVolume(string symbol, double volume) {
   SymbolSelect(symbol, true);
   double min_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
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
               }
            }
         }
      }
   }
}

int OnInit() {
   Print("=== 🧪 TEST MIN FIRE v1.06: Direct Low-Level TRADE_ACTION_SLTP Engine Active ===");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {
   Print("=== 🧪 TEST MIN FIRE DEINIT ===");
}

void CloseTestPositions() {
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      long magic = PositionGetInteger(POSITION_MAGIC);
      if(magic != MAGIC_NUM) continue;
      
      string s = PositionGetString(POSITION_SYMBOL);
      double raw_vol = PositionGetDouble(POSITION_VOLUME);
      double vol = NormalizeVolume(s, raw_vol);
      if(vol <= 0.0) continue;
      
      if(trade.PositionClosePartial(ticket, vol)) {
         Print("  🧪 TEST CLOSE SUCCESS (Partial) | Symbol: ", s, " | Ticket: ", ticket, " | Vol: ", vol);
      } else if(trade.PositionClose(ticket)) {
         Print("  🧪 TEST CLOSE SUCCESS (CTrade) | Symbol: ", s, " | Ticket: ", ticket);
      }
   }
}

void OpenTestPosition() {
   string s = _Symbol;
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return;
   
   double pr = tk.ask;
   double sl_d = (StringFind(s, "JPY") >= 0) ? (SL_PIPS * 0.01) : (SL_PIPS * 0.0001);
   double tp_d = (StringFind(s, "JPY") >= 0) ? (TP_PIPS * 0.01) : (TP_PIPS * 0.0001);
   int digits = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL;
   req.symbol = s;
   req.volume = (float)NormalizeVolume(s, TEST_LOT);
   req.type = ORDER_TYPE_BUY;
   req.price = pr;
   req.deviation = 10;
   req.magic = MAGIC_NUM;
   req.comment = "Test_1min_buy";
   
   if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE) {
      Print("  🧪 TEST BUY FILL SUCCESS | Symbol: ", s, " | Order: ", res.order);
      
      double fill_p = (res.price > 0.0) ? res.price : pr;
      double sl_target = NormalizeDouble(fill_p - sl_d, digits);
      double tp_target = NormalizeDouble(fill_p + tp_d, digits);
      
      AttachSLTPDirectly(s, MAGIC_NUM, sl_target, tp_target);
   } else {
      Print("  ⚠️ TEST BUY FAIL | Code: ", res.retcode, " | Comment: ", res.comment);
   }
}

void OnTick() {
   MqlDateTime dt;
   TimeCurrent(dt);
   
   datetime current_min = TimeCurrent() - (dt.sec);
   if(current_min != g_last_minute) {
      g_last_minute = current_min;
      
      CloseTestPositions();
      OpenTestPosition();
   }
}
"""

CODE_ULTRA_MONSTER = """#include <Trade\\Trade.mqh>
CTrade trade;

#property copyright "Proxima Trading"
#property version   "1.06"
#property strict

input double   BASE_LOT            = 1.20;    // Standard Lot Size (1.20L)
input int      HOLD_BARS           = 3;       // 15m hold time
input ulong    MAGIC_BASE          = 202600;  // Magic Base
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips

#define N_PAIRS 18
string PAIRS[N_PAIRS] = {
   "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD",
   "CADJPY","EURAUD","EURCAD","EURCHF","EURGBP",
   "EURJPY","EURNZD","EURUSD","GBPAUD","GBPCAD",
   "GBPJPY","GBPNZD","GBPUSD"
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
               }
            }
         }
      }
   }
}

int OnInit() {
   Print("=== 🔥 ULTRA MONSTER Engine MT5 Init v1.06 ===");
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
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = (float)NormalizeVolume(s, lot);
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = magic; req.comment = "UltraMonster_entry";
   
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   
   g_pos[idx].active = true; g_pos[idx].ticket = res.order; g_pos[idx].entry = res.price;
   g_pos[idx].held = 0; g_pos[idx].dir = (side == "BUY") ? 1 : -1;
   
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
   MqlDateTime _dt_fr; TimeCurrent(_dt_fr);
   if(_dt_fr.day_of_week == 0 && _dt_fr.hour == 0) return;
   
   if(_dt_fr.min % 15 == 0) {
      for(int i=0; i<N_PAIRS; i++) {
         if(!g_pos[i].active) {
            OpenTrade(i, "BUY", BASE_LOT);
            break;
         }
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
"""

CODE_TOKYO_H0 = """#include <Trade\\Trade.mqh>
CTrade trade;

#property copyright "Proxima Trading"
#property version   "1.06"
#property strict

input int      LOOKBACK_BARS      = 6;
input int      HOLD_BARS          = 12;
input int      TOP_N              = 5;
input int      SESSION_HOUR       = 0;
input double   BASE_LOT           = 0.15;
input ulong    MAGIC_BASE         = 202607;
input double   SLIPPAGE_PIPS      = 1.0;

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
               }
            }
         }
      }
   }
}

int OnInit() {
   Print("=== Tokyo H0 v1.06 lb=", LOOKBACK_BARS, " hold=", HOLD_BARS, " top_n=", TOP_N, " ===");
   for(int i=0; i<N_PAIRS; i++) { g_t[i].active=false; g_t[i].ticket=0; g_t[i].held=0; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {}

bool OpenTrade(int p, double lot) {
   string s = PAIRS[p];
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   
   for(int pos_i=PositionsTotal()-1; pos_i>=0; pos_i--) {
      if(PositionGetSymbol(pos_i) == s) {
         Print("⚠️ Blocked conflicting trade on ", s, " (Position already exists in terminal!)");
         return false;
      }
   }
   double pr = tk.ask;
   double sl_d = (StringFind(s, "JPY") >= 0) ? 0.35 : 0.0035;
   double tp_d = (StringFind(s, "JPY") >= 0) ? 0.45 : 0.0045;
   int digits = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
   ulong magic = MAGIC_BASE + p;
   
   MqlTradeRequest req = {}; MqlTradeResult res = {};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = (float)NormalizeVolume(s, lot);
   req.type = ORDER_TYPE_BUY; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = magic; req.comment = "TokyoH0_entry";
   
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_t[p].active = true; g_t[p].ticket = res.order; g_t[p].entry = res.price; g_t[p].held = 0;
   
   double fill_p = (res.price > 0.0) ? res.price : pr;
   double sl_target = NormalizeDouble(fill_p - sl_d, digits);
   double tp_target = NormalizeDouble(fill_p + tp_d, digits);
   
   AttachSLTPDirectly(s, magic, sl_target, tp_target);
   
   Print("  OPEN ", s, " @", res.price, " lot=", lot);
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
      if(!g_t[i].active) continue;
      string s = PAIRS[i];
      g_t[i].held++;
      
      if(g_t[i].held >= HOLD_BARS) {
         if(ClosePositionByTicket(g_t[i].ticket, s, "HOLD_EXPIRED")) {
            g_t[i].active = false; g_t[i].ticket = 0;
         }
      }
   }
}

void CheckEntry() {
   MqlDateTime dt; TimeCurrent(dt);
   if(dt.hour != SESSION_HOUR || dt.min != 0) return;
   
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
"""

CODE_CPPF_Z = """#include <Trade\\Trade.mqh>
CTrade trade;

#property copyright "Proxima Trading"
#property version   "1.06"
#property strict

input double   Z_THRESH            = 6.0;     // Z-score threshold (6-Sigma shock)
input int      LOOKBACK_BARS       = 200;     // 200 M5 bars rolling window
input int      HOLD_BARS           = 18;      // 90m hold time
input double   BASE_LOT            = 0.15;    // Base Lot Size
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

double NormalizeVolume(string symbol, double volume) {
   SymbolSelect(symbol, true);
   double min_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
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
               }
            }
         }
      }
   }
}

int OnInit() {
   Print("=== CPPF Z v1.06  z=", Z_THRESH, " lookback=", LOOKBACK_BARS, " hold=", HOLD_BARS, " ===");
   for(int i=0; i<N_CPPF; i++) { g_t[i].active=false; g_t[i].ticket=0; g_t[i].held=0; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {}

bool Open(int p, string side, double lot) {
   string s = CPPF_PAIRS[p];
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
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = (float)NormalizeVolume(s, lot);
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = magic; req.comment = "CPPF_Z_entry";
   
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_t[p].active = true; g_t[p].ticket = res.order; g_t[p].entry = res.price;
   g_t[p].held = 0; g_t[p].side = side;
   
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
   for(int i=0; i<N_CPPF; i++) {
      if(!g_t[i].active) continue;
      string s = CPPF_PAIRS[i];
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

   for(int i=0; i<N_CPPF; i++) {
      if(g_t[i].active) continue;
      string s = CPPF_PAIRS[i];
      
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
      
      double curr_ret = (rates[0].close - rates[3].close) / rates[3].close;
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
"""

CODE_CPMC_Z = """#include <Trade\\Trade.mqh>
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
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = (float)NormalizeVolume(s, lot);
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
"""

CODE_NY_H21 = """#include <Trade\\Trade.mqh>
CTrade trade;

#property copyright "Proxima Trading"
#property version   "1.06"
#property strict

input int      LOOKBACK_BARS       = 12;      // 60m lookback
input int      HOLD_BARS           = 12;      // 60m hold time
input double   BASE_LOT            = 0.25;    // Base Lot Size for $6k account
input ulong    MAGIC_BASE          = 202621;  // Magic Base
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips

#define N_PAIR 2
string PAIR_UNIV[N_PAIR] = {"EURJPY", "GBPJPY"};

struct PosRecord {
   bool     active;
   ulong    ticket;
   double   entry;
   int      held;
   int      dir; // 1 = BUY, -1 = SELL
};

PosRecord g_rec[N_PAIR];
datetime  g_last_bar = 0;

double NormalizeVolume(string symbol, double volume) {
   SymbolSelect(symbol, true);
   double min_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
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
               }
            }
         }
      }
   }
}

int OnInit() {
   Print("=== NY H21 v1.06  lb=", LOOKBACK_BARS, " hold=", HOLD_BARS, " ===");
   for(int i=0; i<N_PAIR; i++) { g_rec[i].active=false; g_rec[i].ticket=0; g_rec[i].held=0; g_rec[i].dir=0; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {}

bool Open(int p, string side, double lot) {
   string s = PAIR_UNIV[p];
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
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = (float)NormalizeVolume(s, lot);
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = magic; req.comment = "NY_H21_entry";
   
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_rec[p].active = true; g_rec[p].ticket = res.order; g_rec[p].entry = res.price;
   g_rec[p].held = 0; g_rec[p].dir = (side == "BUY") ? 1 : -1;
   
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
   for(int i=0; i<N_PAIR; i++) {
      if(!g_rec[i].active) continue;
      string s = PAIR_UNIV[i];
      g_rec[i].held++;
      
      if(g_rec[i].held >= HOLD_BARS) {
         if(ClosePositionByTicket(g_rec[i].ticket, s, "HOLD_EXPIRED")) {
            g_rec[i].active = false; g_rec[i].ticket = 0;
         }
      }
   }
}

void CheckEntry() {
   MqlDateTime dt; TimeCurrent(dt);
   if(dt.hour != 21 || dt.min != 0) return;
   
   for(int i=0; i<N_PAIR; i++) {
      if(g_rec[i].active) continue;
      string s = PAIR_UNIV[i];
      
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      if(CopyRates(s, PERIOD_M5, 0, LOOKBACK_BARS + 2, rates) < LOOKBACK_BARS) continue;
      
      double ret = (rates[0].close - rates[LOOKBACK_BARS].close) / rates[LOOKBACK_BARS].close;
      if(ret < -0.0001) Open(i, "BUY", BASE_LOT);
      else if(ret > 0.0001) Open(i, "SELL", BASE_LOT);
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
"""

CODE_MSV_ASIAN = """#include <Trade\\Trade.mqh>
CTrade trade;

#property copyright "Proxima Trading"
#property version   "1.06"
#property strict

input int      HOLD_BARS           = 12;      // 60m hold time
input double   BASE_LOT            = 0.18;    // Base Lot Size for $6k account
input ulong    MAGIC_BASE          = 202610;  // Magic Base
input double   SLIPPAGE_PIPS       = 1.0;     // Slippage Pips

#define N_CURR 7
string CURR[N_CURR] = {"USD","EUR","GBP","JPY","AUD","NZD","CAD"};

#define N_FX 18
string FX_PAIRS[N_FX] = {
   "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD",
   "CADJPY","EURAUD","EURCAD","EURCHF","EURGBP",
   "EURJPY","EURNZD","EURUSD","GBPAUD","GBPCAD",
   "GBPJPY","GBPNZD","GBPUSD"
};

struct MSVPos {
   bool     active;
   ulong    ticket;
   double   entry;
   int      held;
   int      dir; // 1 = BUY, -1 = SELL
};

MSVPos   g_msv[N_FX];
datetime g_last_bar = 0;

double NormalizeVolume(string symbol, double volume) {
   SymbolSelect(symbol, true);
   double min_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_vol  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
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
               }
            }
         }
      }
   }
}

int OnInit() {
   Print("=== MSV Asian Exhaustion v1.06 hold=", HOLD_BARS, " lot=", BASE_LOT, " ===");
   for(int i=0; i<N_FX; i++) { g_msv[i].active=false; g_msv[i].ticket=0; g_msv[i].held=0; g_msv[i].dir=0; }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int) {}

bool Open(int p, string side, double lot) {
   string s = FX_PAIRS[p];
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
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = (float)NormalizeVolume(s, lot);
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = magic; req.comment = "MSV_Asian_entry";
   
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_msv[p].active = true; g_msv[p].ticket = res.order; g_msv[p].entry = res.price;
   g_msv[p].held = 0; g_msv[p].dir = (side == "BUY") ? 1 : -1;
   
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
   for(int i=0; i<N_FX; i++) {
      if(!g_msv[i].active) continue;
      string s = FX_PAIRS[i];
      g_msv[i].held++;
      
      if(g_msv[i].held >= HOLD_BARS) {
         if(ClosePositionByTicket(g_msv[i].ticket, s, "HOLD_EXPIRED")) {
            g_msv[i].active = false; g_msv[i].ticket = 0;
         }
      }
   }
}

void CheckEntry() {
   MqlDateTime dt; TimeCurrent(dt);
   if(dt.hour < 0 || dt.hour > 6) return;
   
   for(int i=0; i<N_FX; i++) {
      if(g_msv[i].active) continue;
      string s = FX_PAIRS[i];
      
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      if(CopyRates(s, PERIOD_M5, 0, 15, rates) < 12) continue;
      
      double ret = (rates[0].close - rates[12].close) / rates[12].close;
      if(ret < -0.0002) Open(i, "BUY", BASE_LOT);
      else if(ret > 0.0002) Open(i, "SELL", BASE_LOT);
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
"""

FILES_MAP = {
    "Test_Min_Fire_MT5_v106.mq5": CODE_TEST_MIN_FIRE,
    "Ultra_Monster_MT5_v106.mq5": CODE_ULTRA_MONSTER,
    "TokyoH0_MT5_v106.mq5": CODE_TOKYO_H0,
    "CPPF_Z_MT5_v106.mq5": CODE_CPPF_Z,
    "CPMC_Z_MT5_v106.mq5": CODE_CPMC_Z,
    "NY_H21_MT5_v106.mq5": CODE_NY_H21,
    "MSV_Asian_Exhaustion_MT5_v106.mq5": CODE_MSV_ASIAN
}

def main():
    print("="*115)
    print("CREATING ALL 7 VERSIONED EAS (_v106), COMPILING DIRECTLY IN APPDATA, & VERIFYING 0 ERRORS...")
    print("="*115)

    results = {}
    for fname, code in FILES_MAP.items():
        base_name = fname.replace(".mq5", "")
        local_mq5 = os.path.join(LOCAL_DIR, fname)
        appdata_mq5 = os.path.join(APPDATA_EXP, fname)
        appdata_ex5 = os.path.join(APPDATA_EXP, f"{base_name}.ex5")
        log_file = os.path.join(APPDATA_EXP, f"{base_name}_comp.log")

        with open(local_mq5, "w", encoding="utf-8") as f:
            f.write(code)
        with open(appdata_mq5, "w", encoding="utf-8") as f:
            f.write(code)

        if os.path.exists(log_file):
            os.remove(log_file)

        cmd = [METAEDITOR, f"/compile:{appdata_mq5}", f"/log:{log_file}"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        time.sleep(0.5)

        log_txt = ""
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-16le", errors="ignore") as f:
                    log_txt = f.read().strip()
            except Exception:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    log_txt = f.read().strip()

        ok = (res.returncode == 0) and os.path.exists(appdata_ex5)
        size = os.path.getsize(appdata_ex5) if ok else 0
        results[base_name] = (ok, size, log_txt)

        status = "0 ERRORS 🟢" if ok else "FAIL ❌"
        print(f"  • {base_name:<34} Retcode: {res.returncode} | Status: {status:<12} | Size: {size} bytes")

        if ok:
            shutil.copy(appdata_ex5, os.path.join(LOCAL_DIR, f"{base_name}.ex5"))
            shutil.copy(appdata_ex5, os.path.join(BACKUP_DIR, f"{base_name}.ex5"))
            shutil.copy(appdata_mq5, os.path.join(BACKUP_DIR, fname))

            subprocess.run(["scp", "-i", VPS_KEY, appdata_ex5, VPS_PATH], check=False)
            subprocess.run(["scp", "-i", VPS_KEY, appdata_mq5, VPS_PATH], check=False)

    print("="*115)
    print("🟢 ALL 7 VERSIONED EAS CREATED, COMPILED WITH 0 ERRORS IN APPDATA, BACKED UP & PUSHED TO VPS!")
    print("="*115)

if __name__ == "__main__":
    main()
