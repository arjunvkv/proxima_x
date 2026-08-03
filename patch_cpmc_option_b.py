#!/usr/bin/env python3
"""Patch CPMC_Z_MT5.mq5 to Option B (Z=5.0, 8.0-Pip Volatility Gate, 90m Hold)."""

def main():
    path = r"paper_trade\mt5_backtest\CPMC_Z_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Update Z_THRESH default to 5.0
    code = code.replace("input double   Z_THRESH            = 4.5;", "input double   Z_THRESH            = 5.0;     // Z-score threshold (5-Sigma shock)")

    # 2. Add 8.0-Pip Volatility Gate check inside Open logic
    if "MIN_RANGE_PIPS" not in code:
        old_open = "bool Open(int p, string side, double lot) {"
        new_open = """input double   MIN_RANGE_PIPS      = 8.0;     // 8.0 Pips Min 15m Range Gate

bool Open(int p, string side, double lot) {
   string s = UNIV[p];
   double h1 = iHigh(s, PERIOD_M5, 1); double h2 = iHigh(s, PERIOD_M5, 2); double h3 = iHigh(s, PERIOD_M5, 3);
   double l1 = iLow(s, PERIOD_M5, 1);  double l2 = iLow(s, PERIOD_M5, 2);  double l3 = iLow(s, PERIOD_M5, 3);
   double max_h = MathMax(h1, MathMax(h2, h3));
   double min_l = MathMin(l1, MathMin(l2, l3));
   double r_pips = (StringFind(s, "JPY") >= 0) ? (max_h - min_l) * 100.0 : (max_h - min_l) * 10000.0;
   if(r_pips < MIN_RANGE_PIPS) return false;"""
        code = code.replace(old_open, new_open)

    code = code.replace("=== CPMC Engine MT5 v1.05 (90m Hold Optimization) ===", "=== CPMC Engine MT5 v1.07 (Option B: z=5.0 + 8p Gate + 90m Hold) ===")

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    print("🟢 CPMC_Z_MT5.mq5 successfully upgraded to Option B (z=5.0, 8.0p Gate, 90m Hold)!")

if __name__ == "__main__":
    main()
