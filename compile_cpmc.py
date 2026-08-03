#!/usr/bin/env python3
import shutil, subprocess, os

src = r"c:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\CPMC_Z_MT5.mq5"
dst = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts\CPMC_Z_MT5.mq5"

shutil.copy(src, dst)
print("Copied updated CPMC_Z_MT5.mq5 to MetaQuotes folder.")

comp_cmd = r'"C:\Program Files\FundedNext MT5 Terminal\MetaEditor64.exe" /compile:"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts\CPMC_Z_MT5.mq5"'
subprocess.run(comp_cmd, shell=True)
print("Compiled EX5 file.")
