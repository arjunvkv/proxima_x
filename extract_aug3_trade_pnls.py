#!/usr/bin/env python3
"""Extract exact PnL and trade details for all trades taken today (Aug 3, 2026)."""
import glob, os

mql5_dir = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/logs/'
term_dir = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/logs/'

mql_logs = sorted(glob.glob(mql5_dir + '*.log'))
term_logs = sorted(glob.glob(term_dir + '*.log'))

def main():
    print("="*115)
    print("MONDAY MORNING LIVE TRADE EXECUTION & PNL AUDIT (AUG 3, 2026):")
    print("="*115)

    with open(mql_logs[-1], 'r', encoding='utf-16le', errors='ignore') as f:
        for line in f:
            if 'ENTRY' in line or 'CLOSE' in line:
                print(' ', line.strip())

    print("="*115)

if __name__ == "__main__":
    main()
