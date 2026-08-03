$mt5Data = "C:\Program Files\FundedNext MT5 Terminal"
$outDir = "$env:APPDATA\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C"
$filesDir = "$outDir\MQL5\Files\custom_data"

Write-Host "MT5 data: $outDir"
Write-Host "Files dir: $filesDir"

if (!(Test-Path $filesDir)) {
    New-Item -ItemType Directory -Path $filesDir -Force | Out-Null
    Write-Host "Created: $filesDir"
}

$src = "C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\custom_data"
Copy-Item "$src\FN_EURJPY.csv" "$filesDir\" -Force
Copy-Item "$src\FN_EURUSD.csv" "$filesDir\" -Force
Copy-Item "$src\FN_GBPJPY.csv" "$filesDir\" -Force
Write-Host "Copied CSV files to $filesDir"
Get-ChildItem $filesDir
