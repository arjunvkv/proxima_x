$ErrorActionPreference = "Continue"
$cli = "C:\Users\Arjun Sasi\AppData\Local\Spotware\cTrader\71dd452b763c6040bbae13b68c9ca250\ctrader-cli.exe"
$algo = "C:\Users\Arjun Sasi\Documents\cAlgo\Sources\Robots\TokyoH0.algo"
$log = "C:\Trading\Agentic_Trading\proxima_x\backtest_7m_18p.log"
$json = "C:\Trading\Agentic_Trading\proxima_x\backtest_7m_18p.json"
$html = "C:\Trading\Agentic_Trading\proxima_x\backtest_7m_18p.html"

& $cli backtest "$algo" --ctid=arjun.vkv.97@gmail.com --pwd-file="C:\Users\Arjun Sasi\Documents\cAlgo\Sources\Robots\ctrader-cli.pwd" --account=17163016 --symbol=EURUSD --period=m5 --start=01/12/2025 --end=01/07/2026 --data-mode=m1 --full-access --report="$html" --report-json="$json" 2>&1 | Out-File "$log" -Encoding ASCII
