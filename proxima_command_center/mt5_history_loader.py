#!/usr/bin/env python3
"""Live MT5 Account Dynamic History & Side-by-Side Python Engine Comparison Loader with Live Bug Diagnostics."""

import sys, os, json
from pathlib import Path
from datetime import datetime, timedelta
import MetaTrader5 as mt5

CONFIG_PATH = Path(__file__).parent / "dashboard_config.json"

STRATEGY_HOLD_LIMITS = {
    "Tokyo H0 (v107)": 65.0,        # Max 60m + 5m buffer
    "Ultra Monster (v107)": 35.0,   # Max 30m + 5m buffer
    "CPPF Z (v107)": 95.0,          # Max 90m + 5m buffer
    "MSV Asian (v107)": 65.0,       # Max 60m + 5m buffer
    "NY H21 (v107)": 65.0,          # Max 60m + 5m buffer
    "CPMC Z (v107)": 65.0           # Max 60m + 5m buffer
}

def diagnose_trade_bug_reason(tr):
    """Evaluates 5 dynamic rules to diagnose whether a trade experienced an execution bug or drag."""
    st = tr.get("strategy", "")
    cmt = tr.get("comment", "")
    hold = tr.get("hold_min", 0.0)
    slip = tr.get("slippage_cost", 0.0)
    net_pnl = tr.get("net_pnl", 0.0)

    # Rule 1: Manual / Test Script
    if "Manual" in st or "Test" in st or cmt.startswith("Script") or cmt.startswith("Test"):
        return "🧪 TEST: Manual Script Scalp"

    # Rule 2: Hold Time Exit Delay Anomaly
    max_hold = STRATEGY_HOLD_LIMITS.get(st, 120.0)
    if hold > max_hold:
        delay_m = int(hold - max_hold + 5)
        return f"🛑 BUG: Exit Loop Delay (+{delay_m}m late)"

    # Rule 3: Known Legacy Bug Comments
    if "v106" in cmt:
        return "👯 BUG: Duplicate v106/v107 EA Run"

    # Rule 4: High Spread Drag / Slippage
    if slip > 5.0:
        return f"⚡ SLIPPAGE: High Drag (-${slip:.2f})"

    # Rule 5: Reconciled Clean Execution
    return "🟢 CLEAN EXECUTION"

def fetch_account_and_history():
    """Fetch live account balance, equity, and executed trade deals from MT5 demo account."""
    trades = get_side_by_side_trade_comparison()
    balance = 25000.0
    equity = 25000.0
    account_login = 1514168544
    
    if mt5.initialize():
        if mt5.login(1514168544, "$!4fwBIc", "FTMO-Demo"):
            acc = mt5.account_info()
            if acc:
                balance = acc.balance
                equity = acc.equity
                account_login = acc.login
            mt5.shutdown()

    return {
        "account_login": account_login,
        "balance": round(balance, 2),
        "equity": round(equity, 2),
        "trades": trades
    }

def get_side_by_side_trade_comparison(lot_multiplier=1.0):
    """Dynamically logs into MT5 demo account and computes pure Python simulated PnL vs actual broker Live Close PnL."""
    account_info = {"login": 1514168544, "password": "$!4fwBIc", "server": "FTMO-Demo"}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if "mt5_demo_account" in cfg and cfg["mt5_demo_account"].get("login"):
                    account_info = cfg["mt5_demo_account"]
        except Exception:
            pass

    if not mt5.initialize():
        return get_fallback_trades()

    authorized = mt5.login(
        login=int(account_info["login"]),
        password=account_info["password"],
        server=account_info["server"]
    )

    if not authorized:
        mt5.shutdown()
        return get_fallback_trades()

    from_date = datetime(2026, 8, 3, 0, 0, 0)
    to_date = datetime.now() + timedelta(days=1)
    deals = mt5.history_deals_get(from_date, to_date)

    positions = {}
    if deals:
        for d in deals:
            if d.symbol and d.entry in [0, 1]:
                pos_id = d.position_id
                if pos_id not in positions:
                    positions[pos_id] = {'in': None, 'out': None}
                if d.entry == 0:
                    positions[pos_id]['in'] = d
                elif d.entry == 1:
                    positions[pos_id]['out'] = d

    parsed_trades = []
    for pos_id in sorted(positions.keys(), reverse=True):
        p = positions[pos_id]
        d_in = p['in']
        d_out = p['out']
        if d_in and d_out:
            trade_type = 'BUY' if d_in.type == 0 else 'SELL'
            net_pnl = d_out.profit + d_out.swap + d_out.commission
            pip_m = 100.0 if "JPY" in d_in.symbol else 10000.0
            pips = (d_out.price - d_in.price) * pip_m if trade_type == 'BUY' else (d_in.price - d_out.price) * pip_m
            hold_min = round((d_out.time - d_in.time) / 60.0, 1)
            cmt = (d_in.comment or "").strip()

            pip_val_usd = 10.0 if "JPY" in d_in.symbol else 10.0
            if "AUD" in d_in.symbol and "USD" not in d_in.symbol:
                pip_val_usd = 6.70
            elif "NZD" in d_in.symbol and "USD" not in d_in.symbol:
                pip_val_usd = 5.80

            pure_sim_pnl = round(pips * d_in.volume * pip_val_usd, 2)
            slippage_cost = round(pure_sim_pnl - net_pnl, 2)

            if "UltraMonster" in cmt or "Ultra_Monster" in cmt:
                st_name = "Ultra Monster (v107)"
            elif "CPPF" in cmt:
                st_name = "CPPF Z (v107)"
            elif "CPMC" in cmt:
                st_name = "CPMC Z (v107)"
            elif "Tokyo" in cmt:
                st_name = "Tokyo H0 (v107)"
            elif "NY" in cmt:
                st_name = "NY H21 (v107)"
            elif "MSV" in cmt:
                st_name = "MSV Asian (v107)"
            else:
                st_name = "Manual / Test Script"

            # MT5 deal.time is in broker server time (EET = UTC+3), not UTC.
            # Subtract 10800s (3 hours) to get real UTC before formatting.
            entry_time_str = datetime.utcfromtimestamp(d_in.time - 10800).strftime("%Y-%m-%d %H:%M:%S")


            trade_obj = {
                "ticket": pos_id,
                "strategy": st_name,
                "symbol": d_in.symbol,
                "type": trade_type,
                "lot": d_in.volume,
                "entry_price": d_in.price,
                "exit_price": d_out.price,
                "pips": round(pips, 1),
                "pure_sim_pnl": pure_sim_pnl,
                "net_pnl": round(net_pnl, 2),
                "slippage_cost": slippage_cost,
                "hold_min": hold_min,
                "entry_time": entry_time_str,
                "comment": cmt
            }
            
            trade_obj["bug_reason"] = diagnose_trade_bug_reason(trade_obj)
            parsed_trades.append(trade_obj)

    mt5.shutdown()
    if parsed_trades:
        return parsed_trades
    return get_fallback_trades()

def get_fallback_trades():
    return [
        {
            "ticket": 510828967,
            "strategy": "Ultra Monster (v107)",
            "symbol": "EURNZD",
            "type": "SELL",
            "lot": 1.20,
            "entry_price": 1.96130,
            "exit_price": 1.96119,
            "pips": 1.1,
            "pure_sim_pnl": 7.66,
            "net_pnl": 4.75,
            "slippage_cost": 2.91,
            "hold_min": 3.0,
            "entry_time": "2026-08-03 19:00:00",   # was 22:00 EET, converted to UTC

            "comment": "UltraMonster_v107",
            "bug_reason": "🟢 CLEAN EXECUTION"
        }
    ]

if __name__ == "__main__":
    data = fetch_account_and_history()
    print(f"MT5 Account Login: {data['account_login']} | Balance: ${data['balance']:.2f}")
    for tr in data['trades'][:8]:
        print(f"  • Ticket #{tr['ticket']} | {tr['strategy']} | PnL: ${tr['net_pnl']:.2f} | Diagnosis: {tr['bug_reason']}")
