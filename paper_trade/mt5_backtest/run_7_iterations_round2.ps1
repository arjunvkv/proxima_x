param([string]$Symbol="AUDNZD",[string]$FromDate="2026.04.01",[string]$ToDate="2026.05.31",[int]$Deposit=25000)

$ErrorActionPreference = "Stop"
$dataDir = "C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075"
$terminal = "C:\Program Files\MetaTrader 5\terminal64.exe"
$configDir = "C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\bt_configs"
$reportDir = "C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\bt_reports"
$eaName = "V2z_CPPF"
$setDir = "$dataDir\MQL5\Profiles\Tester"

$base = @{
    Z_THRESHOLD="2.5"; STOP_A="1.5"; TRIG_A="0.5"; GAP_A="0.03"
    MAX_HOLD_BARS="54"; ATR_PERIOD="20"; Z_WINDOW="50"
    BASE_LOT="1.0"; MAX_DAILY_LOSS="1250.0"; MAX_SPREAD_PIPS="5.0"
    MAGIC_NUMBER="202411"; MAX_TRADES_DAY="100"
    LIMIT_ENTRY_ATR="0.0"; ATR_GATE_PCT="0.0"
    TRADE_START_HOUR="0"; TRADE_END_HOUR="24"
    SPREAD_MULT_MAX="0.0"; LOT_SCALE_MIN_Z="0.0"; LOT_SCALE_MAX="2.0"
    MIN_GAP_PIPS="0.5"; TAKE_PROFIT_ATR="0.0"
}

$iters = @(
    @{n="01_tightstop"; d="Tight stop 1.0/0.3/0.02 + tick-trail"; o=@{STOP_A="1.0"; TRIG_A="0.3"; GAP_A="0.02"}}
    @{n="02_widestop"; d="Wide stop 4.0/1.5/0.08 + time 0-7 + tick"; o=@{STOP_A="4.0"; TRIG_A="1.5"; GAP_A="0.08"; TRADE_END_HOUR="7"}}
    @{n="03_z30"; d="Z-thresh 3.0 + time 0-7 + tick"; o=@{Z_THRESHOLD="3.0"; TRADE_END_HOUR="7"}}
    @{n="04_z30tight"; d="Z30 + tight stop 1.0/0.3/0.02 + time 0-7 + tick"; o=@{Z_THRESHOLD="3.0"; STOP_A="1.0"; TRIG_A="0.3"; GAP_A="0.02"; TRADE_END_HOUR="7"}}
    @{n="05_tp"; d="Take profit 1.0×ATR + tick"; o=@{TAKE_PROFIT_ATR="1.0"}}
    @{n="06_tp_time"; d="TP 1.0×ATR + time 0-7 + tick"; o=@{TAKE_PROFIT_ATR="1.0"; TRADE_END_HOUR="7"}}
    @{n="07_all"; d="ALL: Z30+tight+tp+time+limit+tick"; o=@{Z_THRESHOLD="3.0"; STOP_A="1.0"; TRIG_A="0.3"; GAP_A="0.02"; TRADE_END_HOUR="7"; TAKE_PROFIT_ATR="1.0"; LIMIT_ENTRY_ATR="0.05"}}
)

New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$results = @()

foreach ($it in $iters) {
    $itName = $Symbol + "_" + $it["n"]
    Write-Host "`n=== $($it['d']) ==="

    $p = $base.Clone()
    foreach ($kv in $it["o"].GetEnumerator()) { $p[$kv.Key] = $kv.Value }

    $setLines = @()
    foreach ($k in $p.Keys) { $setLines += "$k=$($p[$k])" }
    $setLines -join "`r`n" | Set-Content ($setDir + "\V2z_CPPF.set") -Encoding ASCII -Force

    Get-Process -Name "terminal64" -ErrorAction SilentlyContinue | ForEach-Object { taskkill /F /PID $_.Id 2>$null }
    Start-Sleep -Seconds 2

    $iniFile = "$configDir\temp_$Symbol.ini"
    @"
[Common]
Login=
Password=
Server=

[Tester]
Expert=$eaName
ExpertParameters=V2z_CPPF.set
Symbol=$Symbol
Period=M1
Model=0
FromDate=$FromDate
ToDate=$ToDate
Deposit=$Deposit
Leverage=100
Optimization=0
ShutdownTerminal=1
ReplaceReport=1
Report=$itName
"@ | Set-Content $iniFile -Encoding UTF8

    $proc = Start-Process -FilePath $terminal -ArgumentList "/config:`"$iniFile`"" -WindowStyle Hidden -PassThru
    $maxWait = 120; $elapsed = 0
    do { Start-Sleep 5; $elapsed += 5
        $still = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    } while ($still -and $elapsed -lt $maxWait)

    if ($still) { taskkill /F /PID $proc.Id 2>$null; Write-Host "  TIMEOUT" }
    else { Write-Host "  done ${elapsed}s" }

    $reportSrc = "$dataDir\$itName.htm"
    if (Test-Path $reportSrc) {
        Copy-Item $reportSrc "$reportDir\$itName.htm" -Force
        Write-Host "  report saved"
    } else { Write-Warning "  NO REPORT" }

    $bal = "?"
    $al = "C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\logs\20260725.log"
    if (Test-Path $al) {
        $m = (Get-Content $al -Encoding UTF8 | Select-String "final balance ([\d.]+) USD")
        if ($m) { $bal = $m.Matches.Groups[1].Value }
    }
    $results += [PSCustomObject]@{Iteration=$it["n"]; Desc=$it["d"]; Final=[double]$bal; PnL=[double]$bal-25000}
}

Write-Host "`n=== RESULTS ROUND 2 ==="
$results | Format-Table Iteration, Desc, Final, PnL -AutoSize
$results | Export-Csv ($reportDir + "\iteration_results_r2.csv") -NoTypeInformation
Write-Host "Done"
