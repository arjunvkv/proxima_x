param(
    [string[]]$Symbols = @("GBPNZD"),
    [string]$ZThreshold = "1.5",
    [string]$FromDate = "2025.01.01",
    [string]$ToDate = "2025.06.01",
    [int]$Deposit = 25000,
    [int]$Leverage = 100,
    [switch]$SkipWipe
)

$ErrorActionPreference = "Stop"
$dataDir = "C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075"
$terminal = "C:\Program Files\MetaTrader 5\terminal64.exe"
$configDir = "C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\bt_configs"
$reportDir = "C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\bt_reports"
$eaName = "V2z_CPPF"
$setName = "V2z_CPPF.set"
$setPath = "$dataDir\MQL5\Profiles\Tester\$setName"

# Create set file (NO BOM - ASCII encoding is critical!)
$setContent = @"
Z_THRESHOLD=$ZThreshold
STOP_A=1.5
TRIG_A=0.5
GAP_A=0.03
MAX_HOLD_BARS=54
ATR_PERIOD=20
Z_WINDOW=50
BASE_LOT=1.0
MAX_DAILY_LOSS=1250.0
MAX_SPREAD_PIPS=5.0
MAGIC_NUMBER=202411
MAX_TRADES_DAY=100
"@
$setContent | Set-Content $setPath -Encoding ASCII -Force

# Verify no BOM
$bytes = Get-Content $setPath -Encoding Byte -Raw
if ($bytes[0] -eq 239 -and $bytes[1] -eq 187 -and $bytes[2] -eq 191) {
    Write-Error "BOM still present in set file!"; exit 1
}

# Ensure report dir exists
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

foreach ($sym in $Symbols) {
    Write-Host "`n=== $sym Z=$ZThreshold ==="
    $reportName = "${sym}_z${ZThreshold}"
    $iniFile = "$configDir\temp_${sym}.ini"

    # Kill any running terminal
    Get-Process -Name "terminal64" -ErrorAction SilentlyContinue | ForEach-Object { taskkill /F /PID $_.Id 2>$null }
    Start-Sleep -Seconds 2

    if (-not $SkipWipe) {
        # Full wipe of terminal state (keeps MQL5, config, origin, Accounts)
        Get-ChildItem $dataDir -Directory | Where-Object { $_.Name -notin @("MQL5","config","origin","Accounts") } | ForEach-Object {
            Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
        Get-ChildItem $dataDir -File | Where-Object { $_.Name -ne "origin.txt" } | ForEach-Object {
            Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
        }
        New-Item -ItemType Directory -Force -Path "$dataDir\Tester" | Out-Null
        New-Item -ItemType Directory -Force -Path "$dataDir\logs" | Out-Null

        # Strip [Tester] from terminal.ini
        $iniPath = "$dataDir\config\terminal.ini"
        $content = Get-Content $iniPath -Raw
        $content = $content -replace '(?s)\[Tester\].*?(?=\[|$)', ''
        $content | Set-Content $iniPath -Encoding UTF8
    }

    # Create config INI
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
Leverage=$Leverage
Optimization=0
ShutdownTerminal=1
ReplaceReport=1
Report=$reportName
"@ | Set-Content $iniFile -Encoding UTF8

    # Launch terminal
    $proc = Start-Process -FilePath $terminal -ArgumentList "/config:`"$iniFile`"" -WindowStyle Hidden -PassThru

    # Wait for terminal to exit (ShutdownTerminal=1)
    $maxWait = 600
    $elapsed = 0
    do {
        Start-Sleep -Seconds 5
        $elapsed += 5
        $stillRunning = Get-Process -Name "terminal64" -ErrorAction SilentlyContinue | Where-Object { $_.Id -eq $proc.Id }
    } while ($stillRunning -and $elapsed -lt $maxWait)

    if ($stillRunning) {
        Write-Warning "${sym}: terminal still running after ${maxWait}s - killing"
        taskkill /F /PID $proc.Id 2>$null
    } else {
        Write-Host "${sym}: completed in ${elapsed}s"
    }

    # Copy report to our folder
    $reportSrc = "$dataDir\$reportName.htm"
    if (Test-Path $reportSrc) {
        Copy-Item $reportSrc "$reportDir\$reportName.htm" -Force
        Copy-Item "$dataDir\$reportName.png" "$reportDir\$reportName.png" -Force -ErrorAction SilentlyContinue
        Write-Host "${sym}: report saved"
    } else {
        Write-Warning "${sym}: report not found at $reportSrc"
    }

    # Parse result from terminal log
    $logFile = "$dataDir\logs\20260725.log"
    if (Test-Path $logFile) {
        $logContent = Get-Content $logFile -Encoding UTF8
        $logContent | Select-String -Pattern "last test passed|final balance" | ForEach-Object { Write-Host "  $_" }
    }

    # Check agent log for Z_THRESHOLD
    $agentLog = "C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\logs\20260725.log"
    if (Test-Path $agentLog) {
        $agentContent = Get-Content $agentLog -Encoding UTF8
        $agentContent | Select-String -Pattern "Z_THRESHOLD|started with inputs|final balance" | Select-Object -Last 3 | ForEach-Object { Write-Host "  $_" }
    }
}

Write-Host "`n=== All done ==="
