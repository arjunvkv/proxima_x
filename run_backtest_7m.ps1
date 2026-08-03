$cli = "C:\Users\Arjun Sasi\AppData\Local\Spotware\cTrader\71dd452b763c6040bbae13b68c9ca250\ctrader-cli.exe"
$algo = "C:\Users\Arjun Sasi\Documents\cAlgo\Sources\Robots\TokyoH0.algo"
$log = "C:\Trading\Agentic_Trading\proxima_x\backtest_7m_18p.log"
$json = "C:\Trading\Agentic_Trading\proxima_x\backtest_7m_18p.json"
$html = "C:\Trading\Agentic_Trading\proxima_x\backtest_7m_18p.html"

$cmd = "backtest `"$algo`" --ctid=arjun.vkv.97@gmail.com --pwd-file=`"C:\Users\Arjun Sasi\Documents\cAlgo\Sources\Robots\ctrader-cli.pwd`" --account=17163016 --symbol=EURUSD --period=m5 --start=01/12/2025 --end=01/07/2026 --data-mode=m1 --full-access --report=`"$html`" --report-json=`"$json`""

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting 7-month backtest..."
$proc = Start-Process -FilePath $cli -ArgumentList $cmd -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
$proc.Id | Out-File "C:\Trading\Agentic_Trading\proxima_x\backtest_7m.pid"
Write-Host "PID: $($proc.Id)"
Write-Host "Started. Tail with: Get-Content '$log' -Tail 50 -Wait"
