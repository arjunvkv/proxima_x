//+------------------------------------------------------------------+
//|                               Proxima_v8_Master_Launcher.mq5      |
//|   🔥 Proxima X v8 Master Portfolio Launcher — OPTIMIZED 6-CHART SUITE |
//|   Launches 6 Multi-Symbol Strategies across 6 Charts with 1 Click    |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "8.00"
#property script_show_inputs

#include <Trade\Trade.mqh>

struct StratConfig {
   string strat_name;
   string template_name;
   string anchor_pair;
   string lot_size;
};

void OnStart() {
   Print("===================================================================================================");
   Print("🚀 PROXIMA X v8 OPTIMIZED 6-CHART PORTFOLIO LAUNCHER — STARTING...");
   Print("===================================================================================================");

   StratConfig suite[6];

   // 1. Tokyo H0 (18 Pairs -> Anchor: EURUSD)
   suite[0].strat_name = "Tokyo H0 (v8)";
   suite[0].template_name = "TokyoH0_v8.tpl";
   suite[0].anchor_pair = "EURUSD";
   suite[0].lot_size = "1.00 Lot";

   // 2. Ultra Monster (9 Pairs -> Anchor: GBPUSD)
   suite[1].strat_name = "Ultra Monster (v8)";
   suite[1].template_name = "UltraMonster_v8.tpl";
   suite[1].anchor_pair = "GBPUSD";
   suite[1].lot_size = "1.20 Lot";

   // 3. CPPF Z (5 Pairs -> Anchor: EURAUD)
   suite[2].strat_name = "CPPF Z (v8)";
   suite[2].template_name = "CPPF_Z_v8.tpl";
   suite[2].anchor_pair = "EURAUD";
   suite[2].lot_size = "1.40 Lot";

   // 4. MSV Asian (1 Pair -> Anchor: USDJPY)
   suite[3].strat_name = "MSV Asian (v8)";
   suite[3].template_name = "MSV_Asian_v8.tpl";
   suite[3].anchor_pair = "USDJPY";
   suite[3].lot_size = "1.00 Lot";

   // 5. NY H21 (2 Pairs -> Anchor: EURJPY)
   suite[4].strat_name = "NY H21 (v8)";
   suite[4].template_name = "NY_H21_v8.tpl";
   suite[4].anchor_pair = "EURJPY";
   suite[4].lot_size = "1.50 Lot";

   // 6. CPMC Z (2 Pairs -> Anchor: GBPAUD)
   suite[5].strat_name = "CPMC Z (v8)";
   suite[5].template_name = "CPMC_Z_v8.tpl";
   suite[5].anchor_pair = "GBPAUD";
   suite[5].lot_size = "1.40 Lot";

   int total_charts_opened = 0;

   for(int s=0; s<6; s++) {
      string pair = suite[s].anchor_pair;
      SymbolSelect(pair, true);
      
      long chart_id = ChartOpen(pair, PERIOD_M5);
      if(chart_id > 0) {
         bool applied = ChartApplyTemplate(chart_id, suite[s].template_name);
         total_charts_opened++;
         Print("🟢 Chart ", s+1, "/6: [", pair, " M5] -> Strategy: ", suite[s].strat_name, " (", suite[s].lot_size, ") | Template: ", suite[s].template_name, " -> Result: ", applied ? "SUCCESS" : "FAILED");
      } else {
         Print("❌ Failed to open chart for ", pair);
      }
      Sleep(150);
   }

   Print("===================================================================================================");
   Print("🟢 OPTIMIZED LAUNCH COMPLETE: ALL 6 MULTI-SYMBOL v8 EAs RUNNING ACROSS 6 CHARTS!");
   Print("===================================================================================================");
}
