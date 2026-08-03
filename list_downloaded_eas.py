#!/usr/bin/env python3
"""List all downloaded VPS EA files in local folder."""
import os, glob
from pathlib import Path

def main():
    folder = Path(r"c:\Trading\Agentic_Trading\proxima_x\vps_deployed_eas")
    files = sorted(list(folder.glob("*")))

    print("="*95)
    print("DOWNLOADED VPS EXPERT ADVISORS FOLDER: vps_deployed_eas")
    print(f"Path: {folder.resolve()}")
    print("="*95)

    rows = []
    for f in files:
        size_kb = f.stat().st_size / 1024.0
        file_type = "Compiled Binary (.ex5)" if f.suffix == ".ex5" else "MQL5 Source Code (.mq5)"
        rows.append({
            "File Name": f.name,
            "File Size": f"{size_kb:.1f} KB ({f.stat().st_size} bytes)",
            "Type": file_type
        })

    import pandas as pd
    print(pd.DataFrame(rows).to_string(index=False))
    print("="*95)

if __name__ == "__main__":
    main()
