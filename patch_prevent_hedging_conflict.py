#!/usr/bin/env python3
"""Patch CPPF_Z_MT5.mq5 and CPMC_Z_MT5.mq5 to prevent conflicting hedged trades on same pair."""

def patch_ea(path, name):
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # Add a check inside OpenTrade to refuse opening if an opposite or active trade already exists on that pair
    old_check = "if(g_pos[idx].active) return;"
    new_check = """if(g_pos[idx].active) return;
   // Global check across all EAs: Do not open if position on pair already exists in terminal
   for(int i=PositionsTotal()-1; i>=0; i--) {
      if(PositionGetSymbol(i) == UNIV[idx]) {
         Print("⚠️ Blocked conflicting trade on ", UNIV[idx], " (Position already exists!)");
         return;
      }
   }"""

    if old_check in code and "Blocked conflicting trade" not in code:
        code = code.replace(old_check, new_check)
        code = code.replace(f"=== {name} Init ===", f"=== {name} Init v1.02 (No Hedging Conflict) ===")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"🟢 Successfully patched {name} to prevent conflicting hedged trades!")
    else:
        print(f"⚠️ {name} already patched or target line not found.")

def main():
    cppf_path = r"c:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\CPPF_Z_MT5.mq5"
    cpmc_path = r"c:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\CPMC_Z_MT5.mq5"

    patch_ea(cppf_path, "CPPF_Z_MT5")
    patch_ea(cpmc_path, "CPMC_Z_MT5")

if __name__ == "__main__":
    main()
