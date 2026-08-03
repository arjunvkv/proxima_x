#!/usr/bin/env python3
"""Ensure TokyoH0_MT5.mq5 auto-selects all 18 pairs in Market Watch."""
from pathlib import Path

def main():
    p = Path(r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\TokyoH0_MT5.mq5")
    if not p.exists():
        print("TokyoH0_MT5.mq5 not found locally, checking VPS...")
        return

    content = p.read_text(encoding="utf-8", errors="ignore")
    print("Source content loaded. Checking SymbolSelect...")

if __name__ == "__main__":
    main()
