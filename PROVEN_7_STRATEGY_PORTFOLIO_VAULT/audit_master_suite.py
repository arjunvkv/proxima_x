#!/usr/bin/env python3
"""Master Verification Script for PROVEN_7_STRATEGY_PORTFOLIO_VAULT."""
import os, sys
from pathlib import Path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

def main():
    print("="*115)
    print("RUNNING MASTER VERIFICATION FOR PROVEN 7-STRATEGY PORTFOLIO VAULT...")
    print("="*115)
    os.system(f"python {root / 'audit_ultra_monster_full.py'}")
    os.system(f"python {root / 'audit_ftmo_suite.py'}")
    os.system(f"python {root / 'calc_8engine_portfolio_metrics.py'}")
    print("="*115)

if __name__ == "__main__":
    main()
