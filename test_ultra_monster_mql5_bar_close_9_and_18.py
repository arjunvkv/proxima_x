#!/usr/bin/env python3
"""Test Ultra Monster MQL5 EA on completed bar close rates[1] for both 9-pair and 18-pair universes."""

import os, subprocess, time, re
import pandas as pd

TERMINAL = r"C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"
METAEDITOR = r"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe"
APPDATA = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C"
LOCAL_DIR = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest"

PAIRS_9 = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]
PAIRS_18 = [
    "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD",
    "CADJPY","EURAUD","EURCAD","EURCHF","EURGBP",
    "EURJPY","EURNZD","EURUSD","GBPAUD","GBPCAD",
    "GBPJPY","GBPNZD","GBPUSD"
]

def make_mql5_ea(filename, pair_list):
    pairs_str = ", ".join([f'"{p}"' for p in pair_list])
    n_pairs = len(pair_list)
    code = f"""//+------------------------------------------------------------------+
//|                                {filename}.mq5                     |
//|   🔥 ULTRA MONSTER Engine — Bar Close Breakout (Rates[1])        |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.07"
#property strict

#include <Trade\\Trade.mqh>
CTrade trade;

input double   BASE_LOT            = 0.15;
input int      HOLD_BARS           = 3;       // 15m hold time
input double   MIN_RANGE_PIPS      = 6.0;     // 6.0 Pips Min Hourly Range
input ulong    MAGIC_BASE          = 202600;
input double   SLIPPAGE_PIPS       = 1.0;

#define N_PAIRS {n_pairs}
string PAIRS[N_PAIRS] = {{{pairs_str}}};

struct PosState {{
   bool   active;
   ulong  ticket;
   int    held;
}};

PosState g_pos[N_PAIRS];
datetime g_last_bar = 0;

double NormalizeVolume(string symbol, double volume) {{
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
}}

int OnInit() {{
   Print("=== 🔥 ULTRA MONSTER Engine Init ===");
   for(int i=0; i<N_PAIRS; i++) {{ g_pos[i].active=false; g_pos[i].ticket=0; g_pos[i].held=0; }}
   return INIT_SUCCEEDED;
}}

void OnDeinit(const int) {{}}

bool OpenTrade(int idx, string side, double lot) {{
   string s = PAIRS[idx];
   MqlTick tk;
   if(!SymbolInfoTick(s, tk)) return false;
   double pr = (side == "BUY") ? tk.ask : tk.bid;
   ENUM_ORDER_TYPE ot = (side == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   MqlTradeRequest req = {{}}; MqlTradeResult res = {{}};
   req.action = TRADE_ACTION_DEAL; req.symbol = s; req.volume = NormalizeVolume(s, lot);
   req.type = ot; req.price = pr;
   req.deviation = (ulong)(SLIPPAGE_PIPS * 10);
   req.magic = MAGIC_BASE + idx; req.comment = "UltraMonster_entry";
   if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) return false;
   g_pos[idx].active = true; g_pos[idx].ticket = res.order; g_pos[idx].held = 0;
   return true;
}}

void CheckExits() {{
   for(int i=0; i<N_PAIRS; i++) {{
      if(!g_pos[i].active) continue;
      string s = PAIRS[i];
      g_pos[i].held++;
      if(g_pos[i].held >= HOLD_BARS) {{
         if(PositionSelectByTicket(g_pos[i].ticket)) {{
            double vol = PositionGetDouble(POSITION_VOLUME);
            trade.PositionClose(g_pos[i].ticket);
            g_pos[i].active = false; g_pos[i].ticket = 0;
         }}
      }}
   }}
}}

void CheckEntry() {{
   MqlDateTime dt; TimeCurrent(dt);
   if(dt.min == 0 || dt.min == 30) {{
      for(int i=0; i<N_PAIRS; i++) {{
         if(g_pos[i].active) continue;
         string s = PAIRS[i];
         
         MqlRates rates[];
         ArraySetAsSeries(rates, true);
         // Copy 14 bars starting from bar 1 (completed bars)
         if(CopyRates(s, PERIOD_M5, 1, 14, rates) < 14) continue;

         double c_closed = rates[0].close; // rates[0] is Bar 1 (most recent completed bar)
         double h_prev = rates[1].high;
         double l_prev = rates[1].low;
         for(int k=2; k<=12; k++) {{
            if(rates[k].high > h_prev) h_prev = rates[k].high;
            if(rates[k].low < l_prev)  l_prev = rates[k].low;
         }}

         double mult = (StringFind(s, "JPY") >= 0) ? 100.0 : 10000.0;
         double range_pips = (h_prev - l_prev) * mult;
         if(range_pips < MIN_RANGE_PIPS) continue;

         if(c_closed > h_prev) {{
            OpenTrade(i, "BUY", BASE_LOT);
         }} else if(c_closed < l_prev) {{
            OpenTrade(i, "SELL", BASE_LOT);
         }}
      }}
   }}
}}

void OnTick() {{
   MqlDateTime dt; TimeCurrent(dt);
   datetime cur_bar = TimeCurrent() - (dt.sec % 300);
   if(cur_bar != g_last_bar) {{
      g_last_bar = cur_bar;
      CheckExits();
      CheckEntry();
   }}
}}
"""
    return code

def main():
    print("="*115)
    print("MQL5 ULTRA MONSTER AUDIT: 9-PAIR vs 18-PAIR UNIVERSE EXPLANATION & COMPILATION")
    print("="*115)

    code_9 = make_mql5_ea("Ultra_Monster_9Pair", PAIRS_9)
    code_18 = make_mql5_ea("Ultra_Monster_18Pair", PAIRS_18)

    p9_path = os.path.join(APPDATA, "MQL5", "Experts", "Ultra_Monster_9Pair.mq5")
    p18_path = os.path.join(APPDATA, "MQL5", "Experts", "Ultra_Monster_18Pair.mq5")

    with open(p9_path, "w", encoding="utf-8") as f: f.write(code_9)
    with open(p18_path, "w", encoding="utf-8") as f: f.write(code_18)

    subprocess.run([METAEDITOR, f"/compile:{p9_path}"], check=False)
    subprocess.run([METAEDITOR, f"/compile:{p18_path}"], check=False)
    time.sleep(3.0)

    p9_ex5 = os.path.join(APPDATA, "MQL5", "Experts", "Ultra_Monster_9Pair.ex5")
    p18_ex5 = os.path.join(APPDATA, "MQL5", "Experts", "Ultra_Monster_18Pair.ex5")

    print(f"9-Pair EA  (.ex5): {'EXISTS 🟢' if os.path.exists(p9_ex5) else 'FAILED ❌'}")
    print(f"18-Pair EA (.ex5): {'EXISTS 🟢' if os.path.exists(p18_ex5) else 'FAILED ❌'}")
    print("="*115)

if __name__ == "__main__":
    main()
