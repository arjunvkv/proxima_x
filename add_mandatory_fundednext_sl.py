#!/usr/bin/env python3
"""Add Mandatory Hard Stop Loss on Order Send for FundedNext Compliance across all EAs."""

def patch_monster():
    path = r"paper_trade\mt5_backtest\Ultra_Monster_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # Ensure sl is specified in MqlTradeRequest
    if "req.sl =" not in code:
        old_req = "req.type = ot; req.price = pr;"
        new_req = """req.type = ot; req.price = pr;
   double sl_dist = (StringFind(s, "JPY") >= 0) ? 0.30 : 0.0030; // 30-pip Emergency SL for FundedNext
   req.sl = (dir == "BUY") ? pr - sl_dist : pr + sl_dist;"""
        code = code.replace(old_req, new_req)
        code = code.replace("=== ULTRA MONSTER Engine MT5 Init v1.05 (12p Gate + 1p Buffer) ===", "=== ULTRA MONSTER Engine MT5 v1.06 (FundedNext Mandatory SL Compliant) ===")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print("🟢 Ultra_Monster_MT5.mq5 successfully patched with 30-pip Mandatory SL for FundedNext!")

def patch_cppf():
    path = r"paper_trade\mt5_backtest\CPPF_Z_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    if "req.sl =" not in code:
        old_req = "req.type = ot; req.price = pr;"
        new_req = """req.type = ot; req.price = pr;
   double sl_dist = (StringFind(s, "JPY") >= 0) ? 0.50 : 0.0050; // 50-pip Emergency SL for FundedNext
   req.sl = (side == "BUY") ? pr - sl_dist : pr + sl_dist;"""
        code = code.replace(old_req, new_req)
        code = code.replace("=== CPPF_Z_MT5 v1.02 (No Hedging Lock) ===", "=== CPPF_Z_MT5 v1.03 (FundedNext Mandatory SL Compliant) ===")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print("🟢 CPPF_Z_MT5.mq5 successfully patched with 50-pip Mandatory SL for FundedNext!")

def patch_cpmc():
    path = r"paper_trade\mt5_backtest\CPMC_Z_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    if "req.sl =" not in code:
        old_req = "req.type = ot; req.price = pr;"
        new_req = """req.type = ot; req.price = pr;
   double sl_dist = (StringFind(s, "JPY") >= 0) ? 0.50 : 0.0050; // 50-pip Emergency SL for FundedNext
   req.sl = (side == "BUY") ? pr - sl_dist : pr + sl_dist;"""
        code = code.replace(old_req, new_req)
        code = code.replace("=== CPMC Engine MT5 v1.05 (90m Hold Optimization) ===", "=== CPMC Engine MT5 v1.06 (FundedNext Mandatory SL Compliant) ===")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print("🟢 CPMC_Z_MT5.mq5 successfully patched with 50-pip Mandatory SL for FundedNext!")

def main():
    patch_monster()
    patch_cppf()
    patch_cpmc()

if __name__ == "__main__":
    main()
