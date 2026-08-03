#!/usr/bin/env python3
"""Revert Stop Loss Patch across all EAs immediately."""

def revert_file(path, old_str, new_str, init_old, init_new):
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    if old_str in code:
        code = code.replace(old_str, new_str)
        code = code.replace(init_old, init_new)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"🟢 Successfully reverted {path}!")

def main():
    # Monster
    m_old = """req.type = ot; req.price = pr;
   double sl_dist = (StringFind(s, "JPY") >= 0) ? 0.30 : 0.0030; // 30-pip Emergency SL for FundedNext
   req.sl = (dir == "BUY") ? pr - sl_dist : pr + sl_dist;"""
    m_new = "req.type = ot; req.price = pr;"
    revert_file(r"paper_trade\mt5_backtest\Ultra_Monster_MT5.mq5", m_old, m_new, 
                "=== ULTRA MONSTER Engine MT5 v1.06 (FundedNext Mandatory SL Compliant) ===", 
                "=== ULTRA MONSTER Engine MT5 Init v1.05 (12p Gate + 1p Buffer) ===")

    # CPPF
    cppf_old = """req.type = ot; req.price = pr;
   double sl_dist = (StringFind(s, "JPY") >= 0) ? 0.50 : 0.0050; // 50-pip Emergency SL for FundedNext
   req.sl = (side == "BUY") ? pr - sl_dist : pr + sl_dist;"""
    cppf_new = "req.type = ot; req.price = pr;"
    revert_file(r"paper_trade\mt5_backtest\CPPF_Z_MT5.mq5", cppf_old, cppf_new, 
                "=== CPPF_Z_MT5 v1.03 (FundedNext Mandatory SL Compliant) ===", 
                "=== CPPF_Z_MT5 v1.02 (No Hedging Lock) ===")

    # CPMC
    cpmc_old = """req.type = ot; req.price = pr;
   double sl_dist = (StringFind(s, "JPY") >= 0) ? 0.50 : 0.0050; // 50-pip Emergency SL for FundedNext
   req.sl = (side == "BUY") ? pr - sl_dist : pr + sl_dist;"""
    cpmc_new = "req.type = ot; req.price = pr;"
    revert_file(r"paper_trade\mt5_backtest\CPMC_Z_MT5.mq5", cpmc_old, cpmc_new, 
                "=== CPMC Engine MT5 v1.06 (FundedNext Mandatory SL Compliant) ===", 
                "=== CPMC Engine MT5 v1.05 (90m Hold Optimization) ===")

if __name__ == "__main__":
    main()
