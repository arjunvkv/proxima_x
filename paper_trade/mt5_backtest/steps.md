# MT5 CLI Backtest — Step by Step

## Prerequisites

1. **MetaTrader 5** installed at `C:\Program Files\MetaTrader 5\terminal64.exe`
2. **MQL5 EA source** at `MQL5\Experts\V2z_CPPF.mq5` (compiles clean: 0 err, 0 warn)
3. **Python 3.10+** with `mt5linux` or raw socket if using `run_mt5_bt.py`
4. **Historical data** downloaded in MT5 for all pairs (M1, at least 6 months)

## Step 1: Write the EA

```mql5
// V2z_CPPF.mq5 — V2+z Mean Reversion on M1 bars
// Matches Python hfdf_m1 logic exactly:
//   - BUF_SZ = Z_WINDOW + 2 (52 closes, 51 returns)
//   - z = (cur_ret - mean(50 prior)) / std(50 prior)
//   - Entry on bar close (IsNewBar detection)
//   - ATR(20) trailing stop via TRADE_ACTION_SLTP
//   - Trail-before-stop: TG*ATR activation, GP*ATR offset
```

**Critical correct patterns** (learned from errors):

```mql5
// ✓ MqlDateTime: use dt.mon, dt.day, dt.year (NOT dt.month)
MqlDateTime dt;
TimeToStruct(TimeCurrent(), dt);
string today = IntegerToString(dt.year) + StringFormat("%02d%02d", dt.mon, dt.day);

// ✓ ENUM_ORDER_TYPE, not int (avoids implicit enum conversion warnings)
ENUM_ORDER_TYPE order_type = (direction > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

// ✓ MqlTradeRequest/MqlTradeResult aggregate init
MqlTradeRequest req = {};
MqlTradeResult res = {};

// ✓ OrderSend two-arg form (still valid in build 6037)
if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) { ... }

// ✓ Dynamic arrays with input-var sizes (use ArrayResize, NOT #define)
ArrayResize(g_close_buf, Z_WINDOW + 2);
```

## Step 2: Compile the EA

**Manual**: Open MetaEditor (F4 in MT5), open `V2z_CPPF.mq5`, press F7.

**Automated** (terminal startup): Place `.mq5` in `MQL5\Experts\`, restart terminal.
The built-in compiler scans all `.mq5` files on startup.

**CLI compile** (get detailed error output):
```powershell
$log = "$env:TEMP\compile_out.txt"
& "C:\Program Files\MetaTrader 5\MetaEditor64.exe" /compile:"path\to\EA.mq5" /log:"$log"
Get-Content $log
```

**WARNING**: The log file `metaeditor.log` in `MQL5\Logs\` only stores **summary**
lines (e.g., "1 errors, 2 warnings"). Use the CLI `/log:` flag to get line-level
error messages with file/line/column.

## Step 3: Create EA Parameter File (.set)

### ⚠️ CRITICAL: UTF-8 BOM KILLS THE FILE

`.set` files **MUST be saved WITHOUT UTF-8 BOM**. A BOM causes MT5's parser
to silently reject the entire file — the EA then runs with compiled-in defaults
with NO error or warning.

**Detection**:
```powershell
# Returns 239 187 191 if BOM present, 90 95 84 if clean
$bytes = Get-Content "V2z_CPPF.set" -Encoding Byte -Raw
$bytes[0..3]
```

**Write correctly**:
```powershell
# ✗ BAD — adds BOM:
Set-Content V2z_CPPF.set -Encoding UTF8 -Value "Z_THRESHOLD=2.5..."

# ✓ GOOD — clean ASCII, no BOM:
Set-Content V2z_CPPF.set -Encoding ASCII -Value "Z_THRESHOLD=2.5..."
```

### File Content

```ini
Z_THRESHOLD=2.5
STOP_A=3.0
TRIG_A=0.5
GAP_A=0.1
MAX_HOLD_BARS=54
ATR_PERIOD=20
Z_WINDOW=50
BASE_LOT=0.01
MAX_DAILY_LOSS=1250.0
MAX_SPREAD_PIPS=5.0
MAGIC_NUMBER=202411
MAX_TRADES_DAY=100
```

### File Placement

Save as **`<EAName>.set`** in **`MQL5\Profiles\Tester\`**:

| File | Path |
|------|------|
| Set file | `MQL5\Profiles\Tester\V2z_CPPF.set` |

**Auto-discovery**: MT5 automatically loads a `.set` file whose name matches
the EA name from `MQL5\Profiles\Tester\`. No `ExpertParameters` key is
needed in the INI config.

## Step 4: Create MT5 Tester Config (.ini)

### ⚠️ CRITICAL: Must Include `[Common]` Section

The INI config **MUST** start with a `[Common]` section (with empty credentials)
BEFORE the `[Tester]` section. Without it, the terminal may silently skip
launching the tester agent.

Also, ensure the terminal data directory is **fully wiped** before each run
(stale cache causes the terminal to ignore the `/config:` directive).

```ini
[Common]
Login=
Password=
Server=

[Tester]
Expert=V2z_CPPF
Symbol=GBPNZD
Period=M1
Model=0
; Model=0 = Every Tick, 1 = 1 Minute OHLC, 2 = Open prices only
FromDate=2025.01.01
ToDate=2025.06.01
Deposit=25000
Leverage=100
Optimization=0
ShutdownTerminal=1
ReplaceReport=1
Report=GBPNZD_z2.5
```

**Key differences from old INI template**:
- `[Common]` section added (required for auto-start reliability)
- `ShutdownTerminal=1` auto-exits terminal after test (not `ShutdownSeconds`)
- `Report=` specifies report name (saved to data directory root)
- No `ExpertParameters` key (auto-discovery via filename match)
- No `ExecutionMode` key (redundant with `Model=0`)

Place config `.ini` in `bt_configs/` (the `.set` file stays in `MQL5\Profiles\Tester\`).

## Step 5: Run the Backtest

### Prerequisite: Wipe Terminal Data Directory

The terminal caches state aggressively. After the first backtest, subsequent
runs with different configs will be IGNORED unless you wipe the data directory:

```powershell
$dataDir = "$env:APPDATA\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075"

# Delete everything except critical folders
Get-ChildItem $dataDir -Directory | Where-Object {
    $_.Name -notin @("MQL5","config","origin","Accounts")
} | Remove-Item -Recurse -Force

Get-ChildItem $dataDir -File | Where-Object { $_.Name -ne "origin.txt" } |
    Remove-Item -Force

# Strip [Tester] from terminal.ini (prevents stale config reuse)
$iniPath = "$dataDir\config\terminal.ini"
$content = Get-Content $iniPath -Raw
$content = $content -replace '(?s)\[Tester\].*?(?=\[|$)', ''
$content | Set-Content $iniPath -Encoding UTF8
```

### Launch the Backtest

```powershell
# Kill any running terminal
Get-Process -Name "terminal64" -ErrorAction SilentlyContinue | ForEach-Object {
    taskkill /F /PID $_.Id
}
Start-Sleep -Seconds 2

# Launch in backtest mode (hidden window)
$proc = Start-Process -FilePath "C:\Program Files\MetaTrader 5\terminal64.exe" `
    -ArgumentList "/config:`"bt_configs\GBPNZD.ini`"" -WindowStyle Hidden -PassThru

# Wait for terminal to exit (ShutdownTerminal=1)
$maxWait = 180  # ~35-45s per run is typical
$elapsed = 0
do {
    Start-Sleep -Seconds 5
    $elapsed += 5
    $stillRunning = Get-Process -Name "terminal64" -ErrorAction SilentlyContinue |
        Where-Object { $_.Id -eq $proc.Id }
} while ($stillRunning -and $elapsed -lt $maxWait)
```

**Important**: You CANNOT run `terminal64.exe` normally (with GUI) while
also launching it in backtest mode — they conflict. Always kill all
terminal64 processes first.

**How it works**: The terminal launches → metatester64.exe agent starts →
downloads history data → runs the EA on generated ticks → writes report →
terminal auto-exits (via `ShutdownTerminal=1`). Total time: ~35-45s per pair.

**Report location**: Written to `$dataDir\` as `*.htm` (UTF-16 LE encoded).
Copy to `bt_reports/` for archival:

## Step 6: Parse the Report

MT5 writes reports as `.htm` files encoded in **UTF-16 LE** (detectable by
BOM `FF FE 3C 00`). They are NOT regular HTML — they use `<td>` layout with
colspan attributes. Always open with `encoding='utf-16-le'`.

### Parsing in Python

```python
import re
with open('report.htm', 'r', encoding='utf-16-le') as f:
    content = f.read()

# Extract key statistics
results = {
    'Total Net Profit': re.search(
        r'Total Net Profit:</td>\s*<td[^>]*><b>([^<]+)</b></td>', content),
    'Gross Profit': re.search(
        r'Gross Profit:</td>\s*<td[^>]*><b>([^<]+)</b></td>', content),
    'Gross Loss': re.search(
        r'Gross Loss:</td>\s*<td[^>]*><b>([^<]+)</b></td>', content),
    'Profit Factor': re.search(
        r'Profit Factor:</td>\s*<td[^>]*><b>([^<]+)</b></td>', content),
    'Total Trades': re.search(
        r'Total Trades:</td>\s*<td[^>]*><b>(\d+)</b></td>', content),
    'Profit Trades': re.search(
        r'Profit Trades.*?<b>(\d+)', content),
    'Loss Trades': re.search(
        r'Loss Trades.*?<b>(\d+)', content),
}

# Extract trade-by-trade data from the deals table
# (table after "Orders" section contains each deal row)
```

## Step 7: Compare with Python

Run the Python numpy backtest on the same date range with:
- Same Z_WINDOW, STOP_A, TRIG_A, GAP_A, MAX_HOLD_BARS
- Add spread cost (5 pips per trade is typical for GBP-crosses)
- Compare: total trades, win rate, Sharpe, profit factor, max DD
- **Important**: Python (Dukascopy) and MT5 (broker) data will differ.
  Compare distributions, not exact numbers.

The difference = tick execution gap (data source + spread + slippage + fill latency).

## Step 8: Automation via PowerShell

The `run_v2z_backtest.ps1` script automates the entire pipeline:

```powershell
# Single pair
powershell -File run_v2z_backtest.ps1 -Symbols "GBPNZD" -ZThreshold "2.5"

# Multiple pairs
powershell -File run_v2z_backtest.ps1 -Symbols "GBPNZD,EURNZD,GBPAUD" -ZThreshold "1.5"

# Custom period
powershell -File run_v2z_backtest.ps1 -Symbols "GBPNZD" -ZThreshold "2.5" `
    -FromDate "2026.04.01" -ToDate "2026.06.29"
```

### What the script does:

1. Kills any running terminal64.exe
2. Wipes the terminal data directory (keeps MQL5, config, origin, Accounts)
3. Strips `[Tester]` from terminal.ini
4. Writes a BOM-less `V2z_CPPF.set` to `MQL5\Profiles\Tester\` (ASCII encoding)
5. Writes a temporary `.ini` config with `[Common]` + `[Tester]` sections
6. Launches terminal64.exe hidden with `/config:` pointing to the `.ini`
7. Polls every 5s for terminal exit (ShutdownTerminal=1, timeout after 180s)
8. Copies the `.htm` report to `bt_reports/{Symbol}_z{Z}.htm`
9. Checks agent logs for parameter confirmation

### Verification

After the run, check:
1. **Agent log** confirms parameters: `Get-Content $agentLog | Select-String "Z_THRESHOLD"`
2. **Report exists**: `Test-Path "bt_reports/GBPNZD_z2.5.htm"`
3. **Final balance** in log: `final balance 25000.00 USD`

### Known Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| 0 trades | BOM in .set file | Use -Encoding ASCII |
| Terminal won't launch | Stale cache + missing [Common] | Full wipe + [Common] in INI |
| Parameters not loaded | BOM or wrong file name | Verify byte 0 is NOT 239 |
| Terminal runs forever | INI not found or malformed | Check path, kill process |
