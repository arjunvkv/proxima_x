#!/usr/bin/env python3
"""Audit Sunday Open Spread Impact and Exact Deal PnLs."""
import os, glob

mql5_dir = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/logs/'
term_dir = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/logs/'

term_logs = sorted(glob.glob(term_dir + '*.log'))

def main():
    print("="*115)
    print("SUNDAY OPEN SPREAD & LOSS ANALYSIS (AUG 2 - AUG 3):")
    print("="*115)

    deals = []
    with open(term_logs[-1], 'r', encoding='utf-16le', errors='ignore') as f:
        for line in f:
            if 'deal #' in line:
                deals.append(line.strip())

    print(f"Total Deals Executed at Sunday Reopen: {len(deals)}")
    print("-"*115)
    for d in deals:
        print(" ", d)
    print("="*115)

if __name__ == "__main__":
    main()
