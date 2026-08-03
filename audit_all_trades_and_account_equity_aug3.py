#!/usr/bin/env python3
"""Audit All Closed Trades and PnL Fired Today (Aug 3, 2026)."""
import glob, os

mql5_dir = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/logs/'
term_dir = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/logs/'

term_logs = sorted(glob.glob(term_dir + '*.log'))
mql_logs = sorted(glob.glob(mql5_dir + '*.log'))

def main():
    print("="*115)
    print("FULL TERMINAL DEALS & CLOSE PNL AUDIT FOR MONDAY AUG 3, 2026:")
    print("="*115)

    all_deals = []
    for path in term_logs[-2:]:
        with open(path, 'r', encoding='utf-16le', errors='ignore') as f:
            for line in f:
                if 'deal #' in line:
                    all_deals.append((os.path.basename(path), line.strip()))

    for fn, d in all_deals:
        print(f"[{fn}] {d}")

    print("="*115)
    print("MQL5 CLOSE LOGS WITH PROPORTIONAL PNL:")
    print("="*115)
    mql_closes = []
    for path in mql_logs[-2:]:
        with open(path, 'r', encoding='utf-16le', errors='ignore') as f:
            for line in f:
                if 'CLOSE' in line or 'pnl=' in line:
                    mql_closes.append((os.path.basename(path), line.strip()))

    for fn, c in mql_closes:
        print(f"[{fn}] {c}")
    print("="*115)

if __name__ == "__main__":
    main()
