#!/usr/bin/env python3
"""Audit exact BUY and SELL deals on JPY pairs and identify triggering EAs."""
import os, glob

term_dir = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/logs/'
mql5_dir = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/logs/'

term_logs = sorted(glob.glob(term_dir + '*.log'))
mql_logs = sorted(glob.glob(mql5_dir + '*.log'))

def main():
    print("="*115)
    print("OPPOSITE JPY DEALS & EA ATTRIBUTION AUDIT (AUG 2 - AUG 3):")
    print("="*115)

    all_jpy_deals = []
    for path in term_logs[-2:]:
        fn = os.path.basename(path)
        with open(path, 'r', encoding='utf-16le', errors='ignore') as f:
            for line in f:
                if 'JPY' in line and 'deal #' in line:
                    all_jpy_deals.append((fn, line.strip()))

    for fn, d in all_jpy_deals:
        print(f"[{fn}] {d}")

    print("="*115)
    print("MQL5 LOGS FOR JPY ENTRIES:")
    print("="*115)
    for path in mql_logs[-2:]:
        fn = os.path.basename(path)
        with open(path, 'r', encoding='utf-16le', errors='ignore') as f:
            for line in f:
                if 'JPY' in line and 'ENTRY' in line:
                    print(f"[{fn}] {line.strip()}")
    print("="*115)

if __name__ == "__main__":
    main()
