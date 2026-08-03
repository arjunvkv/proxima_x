#!/usr/bin/env python3
"""Inspect Open function in CPMC_Z_MT5.mq5 locally."""
with open(r"paper_trade\mt5_backtest\CPMC_Z_MT5.mq5", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, l in enumerate(lines[:60]):
        print(f"Line {i+1}: {l.strip()}")
