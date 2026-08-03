#!/usr/bin/env python3
"""Parse all generated MQL5 Ultra Monster reports across all 9 pairs."""

import os, re, glob
import pandas as pd

APPDATA = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C"
PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]

def parse_report(pair):
    path = os.path.join(APPDATA, f"ultra_monster_{pair}_report.htm")
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-16le", errors="ignore") as f:
            html = f.read()
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text)

    def get_val(kw):
        m = re.search(f"{kw}[^0-9\-]*([\-0-9\.\%\$\s]+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else "0"

    net_pnl = get_val("Total Net Profit")
    gross_win = get_val("Gross Profit")
    gross_loss = get_val("Gross Loss")
    trades = get_val("Total Trades")
    pf = get_val("Profit Factor")
    payoff = get_val("Expected Payoff")

    return {
        "Symbol": pair,
        "Total Trades": trades,
        "Gross Profit": gross_win,
        "Gross Loss": gross_loss,
        "Net Realized PnL": net_pnl,
        "Profit Factor": pf,
        "Expected Payoff": payoff
    }

def main():
    print("="*115)
    print("MQL5 STRATEGY TESTER MULTI-SYMBOL AUDIT FOR ULTRA MONSTER")
    print("="*115)

    rows = []
    for p in PAIRS_ALL:
        res = parse_report(p)
        if res:
            rows.append(res)

    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

    print("="*115)

if __name__ == "__main__":
    main()
