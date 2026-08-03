#!/usr/bin/env python3
"""Gather and categorize all non-Sunday_H22 trade loss stories and playouts."""
import glob, os, re

mql5_dir = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/logs/'
term_dir = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/logs/'

term_logs = sorted(glob.glob(term_dir + '*.log'))
mql_logs = sorted(glob.glob(mql5_dir + '*.log'))

def main():
    print("="*115)
    print("FORENSIC AUDIT OF ALL NON-SUNDAY_H22 TRADES (JUL 31 - AUG 3, 2026):")
    print("="*115)

    all_lines = []
    for path in term_logs[-4:]:
        fn = os.path.basename(path)
        with open(path, 'r', encoding='utf-16le', errors='ignore') as f:
            for line in f:
                if 'deal #' in line or 'market' in line or 'failed' in line:
                    all_lines.append((fn, line.strip()))

    for fn, l in all_lines:
        print(f"[{fn}] {l}")

    print("="*115)

if __name__ == "__main__":
    main()
