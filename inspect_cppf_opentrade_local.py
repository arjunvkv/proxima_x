#!/usr/bin/env python3
"""Inspect OpenTrade function in CPPF_Z_MT5.mq5 locally."""
with open(r"paper_trade\mt5_backtest\CPPF_Z_MT5.mq5", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, l in enumerate(lines[:70]):
        print(f"Line {i+1}: {l.strip()}")
