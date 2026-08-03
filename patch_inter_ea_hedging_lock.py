#!/usr/bin/env python3
"""Patch CPPF_Z_MT5.mq5 and CPMC_Z_MT5.mq5 with Global Inter-EA Position Lock Filter."""

def patch_file(path, name):
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    target = "if(!SymbolInfoTick(s, tk)) return false;"
    replacement = """if(!SymbolInfoTick(s, tk)) return false;
   // Global check across all EAs: Do not open if position on pair already exists in terminal
   for(int pos_i=PositionsTotal()-1; pos_i>=0; pos_i--) {
      if(PositionGetSymbol(pos_i) == s) {
         Print("⚠️ Blocked conflicting trade on ", s, " (Position already exists in terminal!)");
         return false;
      }
   }"""

    if target in code and "Blocked conflicting trade" not in code:
        code = code.replace(target, replacement)
        code = code.replace(f"=== {name}", f"=== {name} v1.02 (No Hedging Lock)")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"🟢 Successfully patched {name} to v1.02 (Global Inter-EA Position Lock active)!")
    else:
        print(f"⚠️ {name} already patched or target line not found.")

def main():
    patch_file(r"paper_trade\mt5_backtest\CPPF_Z_MT5.mq5", "CPPF_Z_MT5")
    patch_file(r"paper_trade\mt5_backtest\CPMC_Z_MT5.mq5", "CPMC_Z_MT5")

if __name__ == "__main__":
    main()
