@echo off
setlocal

set CLI=C:\Users\Arjun Sasi\AppData\Local\Spotware\cTrader\71dd452b763c6040bbae13b68c9ca250\ctrader-cli.exe
set ALGO=C:\Trading\Agentic_Trading\proxima_x\TokyoH0_snapshot.algo
set PWD=C:\Users\Arjun Sasi\Documents\cAlgo\Sources\Robots\ctrader-cli.pwd
set LOG=C:\Trading\Agentic_Trading\proxima_x\backtest_7m_18p.log
set JSON=C:\Trading\Agentic_Trading\proxima_x\backtest_7m_18p.json
set HTML=C:\Trading\Agentic_Trading\proxima_x\backtest_7m_18p.html

echo [%DATE% %TIME%] Starting 7-month backtest... > "%LOG%"

"%CLI%" backtest "%ALGO%" --ctid=arjun.vkv.97@gmail.com --pwd-file="%PWD%" --account=17163016 --symbol=EURUSD --period=m5 --start=01/12/2025 --end=01/07/2026 --data-mode=m1 --full-access --report="%HTML%" --report-json="%JSON%" >> "%LOG%" 2>&1

echo [%DATE% %TIME%] Done. >> "%LOG%"
