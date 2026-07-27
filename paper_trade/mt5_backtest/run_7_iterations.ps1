param(
    [string]$Symbol = "AUDNZD",
    [string]$FromDate = "2026.04.01",
    [string]$ToDate = "2026.05.31",
    [int]$Deposit = 25000,
    [switch]$SkipWipe
)

$ErrorActionPreference = "Stop"
$dataDir = "C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075"
$terminal = "C:\Program Files\MetaTrader 5\terminal64.exe"
$configDir = "C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\bt_configs"
$reportDir = "C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\bt_reports"
$eaName = "V2z_CPPF"
$setDir = "$dataDir\MQL5\Profiles\Tester"

# Baseline parameters
$baseParams = @{
    Z_THRESHOLD      = "2.5"
    STOP_A           = "1.5"
    TRIG_A           = "0.5"
    GAP_A            = "0.03"
    MAX_HOLD_BARS    = "54"
    ATR_PERIOD       = "20"
    Z_WINDOW         = "50"
    BASE_LOT         = "1.0"
    MAX_DAILY_LOSS   = "1250.0"
    MAX_SPREAD_PIPS  = "5.0"
    MAGIC_NUMBER     = "202411"
    MAX_TRADES_DAY   = "100"
    LIMIT_ENTRY_ATR  = "0.0"
    ATR_GATE_PCT     = "0.0"
    TRADE_START_HOUR = "0"
    TRADE_END_HOUR   = "24"
    SPREAD_MULT_MAX  = "0.0"
    LOT_SCALE_MIN_Z  = "0.0"
    LOT_SCALE_MAX    = "2.0"
}

$iterations = @(
    @{name="00_baseline"; desc="Baseline 1.5/0.5/0.03 z=2.5"; overrides=@{}},
    @{name="01_widen_stop"; desc="Stop 3.0/1.0/0.05"; overrides=@{STOP_A="3.0"; TRIG_A="1.0"; GAP_A="0.05"}},
    @{name="02_limit_entry"; desc="Limit entry 0.05×ATR"; overrides=@{LIMIT_ENTRY_ATR="0.05"}},
    @{name="03_atr_gate"; desc="ATR gate 25th pct"; overrides=@{ATR_GATE_PCT="0.25"}},
    @{name="04_time_gate"; desc="Time gate 0-7 UTC"; overrides=@{TRADE_START_HOUR="0"; TRADE_END_HOUR="7"}},
    @{name="05_z_thresh_20"; desc="Z-thresh 2.0"; overrides=@{Z_THRESHOLD="2.0"}},
    @{name="06_dynamic_lot"; desc="Dynamic lot z≥2.5→3x"; overrides=@{LOT_SCALE_MIN_Z="2.5"; LOT_SCALE_MAX="3.0"}},
    @{name="07_spread_filter"; desc="Spread filter 2×median"; overrides=@{SPREAD_MULT_MAX="2.0"}}
)

New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

$results = @()

foreach ($it in $iterations) {
    $itName = $Symbol + "_" + $it["name"]
    Write-Host "`n=== " + $it["desc"] + " ==="

    # Build set content
    $params = $baseParams.Clone()
    foreach ($kv in $it["overrides"].GetEnumerator()) {
        $params[$kv.Key] = $kv.Value
    }

    $setContent = @"
Z_THRESHOLD=$($params.Z_THRESHOLD)
STOP_A=$($params.STOP_A)
TRIG_A=$($params.TRIG_A)
GAP_A=$($params.GAP_A)
MAX_HOLD_BARS=$($params.MAX_HOLD_BARS)
ATR_PERIOD=$($params.ATR_PERIOD)
Z_WINDOW=$($params.Z_WINDOW)
BASE_LOT=$($params.BASE_LOT)
MAX_DAILY_LOSS=$($params.MAX_DAILY_LOSS)
MAX_SPREAD_PIPS=$($params.MAX_SPREAD_PIPS)
MAGIC_NUMBER=$($params.MAGIC_NUMBER)
MAX_TRADES_DAY=$($params.MAX_TRADES_DAY)
LIMIT_ENTRY_ATR=$($params.LIMIT_ENTRY_ATR)
ATR_GATE_PCT=$($params.ATR_GATE_PCT)
TRADE_START_HOUR=$($params.TRADE_START_HOUR)
TRADE_END_HOUR=$($params.TRADE_END_HOUR)
SPREAD_MULT_MAX=$($params.SPREAD_MULT_MAX)
LOT_SCALE_MIN_Z=$($params.LOT_SCALE_MIN_Z)
LOT_SCALE_MAX=$($params.LOT_SCALE_MAX)
"@

    $setContent | Set-Content ($setDir + "\V2z_CPPF.set") -Encoding ASCII -Force

    # Kill terminal
    Get-Process -Name "terminal64" -ErrorAction SilentlyContinue | ForEach-Object { taskkill /F /PID $_.Id 2>$null }
    Start-Sleep -Seconds 2

    if (-not $SkipWipe) {
        Get-ChildItem $dataDir -Directory | Where-Object { $_.Name -notin @("MQL5","config","origin","Accounts") } | ForEach-Object {
            Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
        Get-ChildItem $dataDir -File | Where-Object { $_.Name -ne "origin.txt" } | ForEach-Object {
            Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
        }
        New-Item -ItemType Directory -Force -Path "$dataDir\Tester" | Out-Null
        New-Item -ItemType Directory -Force -Path "$dataDir\logs" | Out-Null

        $iniPath = "$dataDir\config\terminal.ini"
        if (Test-Path $iniPath) {
            $content = Get-Content $iniPath -Raw
            $content = $content -replace '(?s)\[Tester\].*?(?=\[|$)', ''
            $content | Set-Content $iniPath -Encoding UTF8
        }
    }

    # Create config INI
    $iniFile = $configDir + "\temp_" + $Symbol + ".ini"
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

    $maxWait = 600
    $elapsed = 0
    do {
        Start-Sleep -Seconds 5
        $elapsed += 5
        $stillRunning = Get-Process -Name "terminal64" -ErrorAction SilentlyContinue | Where-Object { $_.Id -eq $proc.Id }
    } while ($stillRunning -and $elapsed -lt $maxWait)

    if ($stillRunning) {
        taskkill /F /PID $proc.Id 2>$null
        Write-Host "  TIMEOUT after ${maxWait}s"
    } else {
        Write-Host "  completed in ${elapsed}s"
    }

    # Copy report
    $reportSrc = $dataDir + "\" + $itName + ".htm"
    if (Test-Path $reportSrc) {
        Copy-Item $reportSrc ($reportDir + "\" + $itName + ".htm") -Force
        Write-Host "  report saved"
    } else {
        Write-Warning "  report not found at $reportSrc"
    }

    # Parse balance from log
    $balance = "unknown"
    $agentLog = "C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\logs\20260725.log"
    if (Test-Path $agentLog) {
        $agentContent = Get-Content $agentLog -Encoding UTF8
        $match = $agentContent | Select-String -Pattern "final balance ([\d.]+) USD"
        if ($match) {
            $balance = $match.Matches.Groups[1].Value
        }
        $agentContent | Select-String -Pattern "Z_THRESHOLD|started with inputs|final balance" | Select-Object -Last 3 | ForEach-Object { Write-Host "  $_" }
    }

    $results += [PSCustomObject]@{
        Iteration = $it["name"]
        Description = $it["desc"]
        FinalBalance = [double]$balance
        PnL = [double]$balance - 25000
    }
}

Write-Host "`n=== RESULTS ==="
$results | Format-Table Iteration, Description, FinalBalance, PnL -AutoSize
$results | Export-Csv ($reportDir + "\iteration_results.csv") -NoTypeInformation

Write-Host "`n=== All 7 iterations done ==="
