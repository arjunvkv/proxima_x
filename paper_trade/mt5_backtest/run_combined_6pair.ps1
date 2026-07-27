param([string]$FromDate="2026.04.01",[string]$ToDate="2026.05.31",[int]$Deposit=25000)

$ErrorActionPreference = "Stop"
$dataDir = "C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075"
$terminal = "C:\Program Files\MetaTrader 5\terminal64.exe"
$configDir = "C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\bt_configs"
$reportDir = "C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\bt_reports"
$eaName = "V2z_CPPF"
$setDir = "$dataDir\MQL5\Profiles\Tester"

$p = @{
    Z_THRESHOLD="2.5"; STOP_A="4.0"; TRIG_A="1.5"; GAP_A="0.08"
    MAX_HOLD_BARS="54"; ATR_PERIOD="20"; Z_WINDOW="50"
    BASE_LOT="1.0"; MAX_DAILY_LOSS="1250.0"; MAX_SPREAD_PIPS="5.0"
    MAGIC_NUMBER="202411"; MAX_TRADES_DAY="100"
    LIMIT_ENTRY_ATR="0.0"; ATR_GATE_PCT="0.0"
    TRADE_START_HOUR="0"; TRADE_END_HOUR="7"
    SPREAD_MULT_MAX="0.0"; LOT_SCALE_MIN_Z="0.0"; LOT_SCALE_MAX="2.0"
    MIN_GAP_PIPS="0.5"; TAKE_PROFIT_ATR="0.0"
}

$symbols = @("AUDNZD","EURAUD","EURNZD","GBPAUD","GBPCAD","GBPNZD")
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$results = @()

foreach ($sym in $symbols) {
    Write-Host "`n=== $sym ==="

    $setLines = @()
    foreach ($k in $p.Keys) { $setLines += "$k=$($p[$k])" }
    $setLines -join "`r`n" | Set-Content ($setDir + "\V2z_CPPF.set") -Encoding ASCII -Force

    Get-Process -Name "terminal64" -ErrorAction SilentlyContinue | ForEach-Object { taskkill /F /PID $_.Id 2>$null }
    Start-Sleep -Seconds 2

    $rptName = "6p_$sym"
    $iniFile = "$configDir\temp_$sym.ini"
    @"
[Common]
Login=
Password=
Server=

[Tester]
Expert=$eaName
ExpertParameters=V2z_CPPF.set
Symbol=$sym
Period=M1
Model=0
FromDate=$FromDate
ToDate=$ToDate
Deposit=$Deposit
Leverage=100
Optimization=0
ShutdownTerminal=1
ReplaceReport=1
Report=$rptName
"@ | Set-Content $iniFile -Encoding UTF8

    $proc = Start-Process -FilePath $terminal -ArgumentList "/config:`"$iniFile`"" -WindowStyle Hidden -PassThru
    $maxWait = 120; $elapsed = 0
    do { Start-Sleep 5; $elapsed += 5
        $still = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    } while ($still -and $elapsed -lt $maxWait)
    if ($still) { taskkill /F /PID $proc.Id 2>$null; Write-Host "  TIMEOUT" }
    else { Write-Host "  done ${elapsed}s" }

    $reportSrc = "$dataDir\$rptName.htm"
    if (Test-Path $reportSrc) {
        Copy-Item $reportSrc "$reportDir\$rptName.htm" -Force
        Write-Host "  report saved"
    } else { Write-Warning "  NO REPORT" }
}

Write-Host "`n=== All 6 pairs done ==="
Write-Host "Analyze via: store_backtest_report for each 6p_*.htm"
