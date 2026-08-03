# Copy TokyoH0 cBot to cAlgo Sources
$src = "C:\Trading\Agentic_Trading\proxima_x\ctrader_bots\TokyoH0\TokyoH0.cs"
$dst = "$env:USERPROFILE\Documents\cAlgo\Sources\Robots\TokyoH0.cs"
Copy-Item $src $dst -Force
Write-Host "Deployed TokyoH0.cs -> $dst"

# Launch cTrader
$ct = "C:\Users\Arjun Sasi\AppData\Local\Spotware\cTrader\71dd452b763c6040bbae13b68c9ca250\cTrader.exe"
if (Test-Path $ct) {
    Start-Process $ct
    Write-Host "Launched cTrader"
} else {
    Write-Host "cTrader not found at $ct"
}
