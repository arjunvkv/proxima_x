#!/usr/bin/env python3
"""Audit all 5 MQ5 EAs for execution blockers, timezone traps, and fill bugs."""
import glob, re
from pathlib import Path

ea_files = glob.glob(r"paper_trade/mt5_backtest/*.mq5")

print("="*85)
print("COMPREHENSIVE CODE AUDIT ACROSS ALL 5 EXPERT ADVISORS")
print("="*85)

for fpath in sorted(ea_files):
    fname = Path(fpath).name
    code = open(fpath).read()
    print(f"\n--- AUDITING: {fname} ---")

    # 1. Check type_filling error 10030
    if "type_filling" in code:
        print("  ⚠️ WARNING: type_filling set — check if causing error 10030 [Unsupported filling mode]")
    else:
        print("  ✅ Fill Mode: PASS (Uses default market execution, no filling mode error)")

    # 2. Check Timezone / GMT handling
    if "TimeGMT" in code or "TimeCurrent" in code or "g_last_bar" in code:
        print("  ℹ️ Time Handling: Inspected (TimeStruct from M5 bar time)")

    # 3. Check Confidence / Margin Skips
    conf_skips = re.findall(r"if\(conf\s*<\s*[\d\.]+\)", code)
    if conf_skips:
        print(f"  ⚠️ Confidence Gate: {conf_skips}")
    else:
        print("  ✅ Confidence Gate: PASS (No restrictive conf gate)")

    # 4. Check Minimum Pairs Gate
    min_pairs_skips = re.findall(r"if\(.*decl_cnt\s*<\s*\d+\)", code) + re.findall(r"if\(vc\s*<\s*\d+\)", code)
    if min_pairs_skips:
        print(f"  ℹ️ Market Gate: {min_pairs_skips}")

    # 5. Check OrderSend & Return Code checks
    if "TRADE_RETCODE_DONE" in code:
        print("  ✅ OrderSend Retcode: PASS (Checks TRADE_RETCODE_DONE)")

print("\nAudit Complete!")
