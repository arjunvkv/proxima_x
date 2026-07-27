# Problems Faced & Resolutions

## 1. Compile Failure — `metaeditor.log` Hides Error Details

**Symptom**: `metaeditor.log` shows only `1 errors, 2 warnings` with no line
numbers or error messages. For successful compiles it shows timing info
(`420 ms elapsed`) but for errors just shows the count.

**Root Cause**: The built-in terminal startup compiler logs only summary lines
(severity `2` means errors present). It does NOT include the actual errors in
the log file.

**Fix**: Use MetaEditor64.exe CLI with the `/log:` flag:
```powershell
& "C:\Program Files\MetaTrader 5\MetaEditor64.exe" /compile:"EA.mq5" /log:"$env:TEMP\compile_out.txt"
```
This gives full error output with file/line/column:
```
EA.mq5(70,37) : error 256: undeclared identifier 'month'
EA.mq5(139,15) : warning 42: implicit enum conversion
```

---

## 2. `MqlDateTime` Field Name: `mon`, NOT `month`

**Symptom**: `error 256: undeclared identifier 'month'` on line using `dt.month`.

**Root Cause**: The MqlDateTime struct in MQL5 uses abbreviated field names:
```c
struct MqlDateTime {
   int year;   // 1970-...
   int mon;    // 1-12        ← NOT "month"
   int day;    // 1-31
   int hour;   // 0-23
   int min;    // 0-59        ← NOT "minute"
   int sec;    // 0-58
   int day_of_week;  // 0-6 (Sunday=0)
   int day_of_year;  // 1-366
};
```

**Fix**: Replace all `dt.month` → `dt.mon`.

---

## 3. Enum Conversion Warnings

**Symptom**: `warning 42: implicit enum conversion` when assigning
`ORDER_TYPE_BUY` / `ORDER_TYPE_SELL` to `int`.

**Root Cause**: `ORDER_TYPE_BUY` is an `ENUM_ORDER_TYPE`, not an `int`.
Assigning it to `int` causes implicit conversion.

**Fix**: Declare as `ENUM_ORDER_TYPE`:
```mql5
// ✗ BAD:
int order_type = (direction > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

// ✓ GOOD:
ENUM_ORDER_TYPE order_type = (direction > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
```

---

## 4. Compiler Cache by File Name — Stale Error Reports

**Symptom**: After fixing all errors and recompiling, the terminal still reports
the old error count. Deleting and recreating the file with the same name still
reports old errors. Even creating a fresh file with different content shows the
same error count from cached result.

**Root Cause**: The MQL5 compiler caches compilation results keyed by file
**name** (not content hash). When the terminal restarts, it checks if a file
has changed by comparing its current state against the cache. If the cache
says "this filename produced 1 error", it REPORTS that cached result without
actually recompiling — even if the file content changed.

Evidence:
- Same exact code compiled as `V2z_isolation.mq5` → 0 errors
- Same exact code as `V2z_CPPF.mq5` → 1 error (cached from prior bad compile)

**Fix**: 
- Use a **new filename** when the cache appears stuck
- Or compile via **MetaEditor64.exe CLI** (which bypasses the cache)
- The terminal's startup compiler eventually clears the cache (after enough
  restarts with clean compilation)

---

## 5. `#define` with input Variables is Invalid

**Symptom**: `#define BUF_SZ (Z_WINDOW + 2)` causes compile errors.

**Root Cause**: `input` variables in MQL5 are runtime parameters, not
compile-time constants. The preprocessor (`#define`) operates at compile-time
and cannot evaluate `input` variable values.

**Fix**: Use a function or inline the expression:
```mql5
// ✗ BAD:
#define BUF_SZ (Z_WINDOW + 2)   // Z_WINDOW is input, not const

// ✓ GOOD (inline):
int bs = Z_WINDOW + 2;

// ✓ GOOD (function):
int CloseBufSize() { return Z_WINDOW + 2; }
```

---

## 6. `CopyClose` Start Index: 0 vs 1

**Issue**: `CopyClose(_Symbol, PERIOD_M1, 0, 1, closes)` gets the current
(in-progress) bar's close, which changes every tick. `CopyClose(..., 1, ...)`
gets the **completed** bar's close.

**Fix**: Use index `1` for the most recently completed bar, matching the
Python logic which computes z-score at bar close using the closed bar's
price.

---

## 7. `TimeCurrent(dt)` → `TimeToStruct(TimeCurrent(), dt)`

**Symptom**: `TimeCurrent(dt)` fails because `TimeCurrent` does not accept
a `MqlDateTime&` parameter in MQL5.

**Fix**:
```mql5
// ✗ BAD:
MqlDateTime dt;
TimeCurrent(dt);

// ✓ GOOD:
MqlDateTime dt;
TimeToStruct(TimeCurrent(), dt);
```

---

## 8. `PositionGetInteger` Return Type — Needs `(ulong)` Cast

**Symptom**: Assigning `PositionGetInteger(POSITION_TICKET)` directly to `ulong`
works, but the compiler may warn about implicit conversion from `long`.

**Fix**:
```mql5
ulong ticket = (ulong)PositionGetInteger(POSITION_TICKET);
```

---

## 9. OrderSend — `!OrderSend(req, res)` Returns `bool`

**MQL5 (build 6037)**: `OrderSend` still uses the two-argument form:
```mql5
bool OrderSend(MqlTradeRequest& request, MqlTradeResult& result);
```
This remains valid in build 6037. The newer single-argument form
(`ulong OrderSend(const MqlTradeRequest&)`) was introduced in very old builds
and the two-argument form has NOT been removed.

**Always check `result.retcode`**, not just the return value:
```mql5
if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE) {
   Print("OrderSend failed: ", res.retcode, " ", GetLastError());
}
```

---

## 10. MqlTradeRequest Aggregate Initialization

**Pattern** `MqlTradeRequest req = {};` is valid and widely used in MQL5
build 6037. It zero-initializes all struct fields.

**Alternative** that also works:
```mql5
MqlTradeRequest req;
ZeroMemory(req);
```

---

## 11. Buffer Sizing for Z-Score

**Bug (original code)**: Close buffer had `Z_WINDOW` elements, but the
z-score computation needs `Z_WINDOW+1` returns, which requires
`Z_WINDOW+2` closes (one extra for the `c[i+1] - c[i]` difference).

**Fix**: Buffer size = `Z_WINDOW + 2`:
```mql5
ArrayResize(g_close_buf, Z_WINDOW + 2);   // 52 closes
// 51 returns: rets[0..50] = g_close_buf[1..51] - g_close_buf[0..50]
// cur_ret = rets[50], prior 50 = rets[0..49]
```

---

## 12. Terminal Process Management

When running backtests via CLI:
1. **Kill all terminal64.exe** processes first (they conflict)
2. Launch with `Start-Process -WindowStyle Hidden`
3. Wait ~30-40s for startup + compilation + backtest
4. The `ShutdownSeconds=30` in .ini causes auto-close after backtest
5. Check for `.ex5` to confirm compilation, parse `.xml` for results

```powershell
Get-Process -Name "terminal64" | Stop-Process -Force
Start-Process -FilePath "C:\Program Files\MetaTrader 5\terminal64.exe" `
  -ArgumentList "/config:bt_configs\EURUSD.ini"
```

---

## 13. UTF-8 BOM in `.set` Files Causes Silent Parameter Ignore

**Symptom**: EA uses compiled-in default values despite a `.set` file being present.
Agent log shows the expected parameter values (e.g., `Z_THRESHOLD=2.5`) — but some
confirm independently via debug Print that the EA is actually running with defaults.

Example: `Z_THRESHOLD` set to `2.5` in `.set` file, but the EA prints
`Z_THRESHOLD=2.5` on startup and you can see the engine loads it.

**Root Cause**: `Set-Content -Encoding UTF8` in PowerShell adds a 3-byte UTF-8 BOM
(`239 187 191` / `0xEF 0xBB 0xBF`) at the start of the file. MT5's `.set` file
parser does NOT strip the BOM — it reads it as part of the first key name,
corrupting `\ufeffZ_THRESHOLD` instead of `Z_THRESHOLD`. The parser silently
rejects the entire file and the EA falls back to compiled-in defaults.

**Detection**:
```powershell
$bytes = Get-Content "V2z_CPPF.set" -Encoding Byte -Raw
if ($bytes[0] -eq 239) { "BOM DETECTED" }
```

**Fix**: Use `-Encoding ASCII` instead of `-Encoding UTF8`:
```powershell
# ✗ BAD — adds BOM:
Set-Content V2z_CPPF.set -Encoding UTF8 -Value "Z_THRESHOLD=2.5`n..."

# ✓ GOOD — clean ASCII, no BOM:
Set-Content V2z_CPPF.set -Encoding ASCII -Value "Z_THRESHOLD=2.5`n..."
```

**Verification**: After writing, confirm the file starts with `Z` not `\ufeffZ`:
```powershell
$b = Get-Content "V2z_CPPF.set" -Encoding Byte -Raw
$b[0..3]   # Should be [90, 95, 84, 72] = "Z_THR", NOT [239, 187, 191, 90]
```

**Why this is insidious**: The agent log (`Agent-127.0.0.1-3000\logs\*.log`)
prints input parameters as they appear AFTER parsing, so if the file is silently
rejected you'll see compiled defaults. In our case Z_THRESHOLD happened to be
already set to the same value in the EA (2.5), making it look like the file loaded.

---

## 14. Terminal Auto-Start Fails After First Backtest

**Symptom**: `terminal64.exe /config:temp_GBPNZD.ini` works correctly the first
time but subsequent launches with different configs do nothing — no terminal
window appears, no agent starts, no error message.

**Root Cause**: MT5's terminal caches state (history data, log files, cache
databases, chart templates) in the data directory
(`AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075`).
When a stale `terminal.ini` still contains a `[Tester]` section from a prior
run, the terminal ignores the new `/config:` directive and uses the cached
settings. Additionally, the `metatester64.exe` agent may still be bound to
port 3000 from the previous run.

**Fix — Full data directory wipe**:
1. Kill all `terminal64.exe` and `metatester64.exe` processes
2. Delete everything in the terminal data directory EXCEPT:
   - `MQL5/` (EA source, compiled ex5, profiles)
   - `config/` (terminal configs — but strip `[Tester]` from `terminal.ini`)
   - `origin/` (broker connection info)
   - `Accounts/` (login credentials)
3. Strip `[Tester]` section from `terminal.ini` before each launch
4. Include a `[Common]` section with empty Login/Password/Server in the INI config

**Working INI template**:
```ini
[Common]
Login=
Password=
Server=

[Tester]
Expert=V2z_CPPF
Symbol=GBPNZD
...
```

**Why [Common] is required**: Without it, the terminal may skip launching
the tester agent entirely. The `[Common]` section with empty credentials
tells the terminal "no login needed" (the tester handles login internally).

**Automated wipe function** (from `run_v2z_backtest.ps1`):
```powershell
Get-ChildItem $dataDir -Directory | Where-Object {
    $_.Name -notin @("MQL5","config","origin","Accounts")
} | Remove-Item -Recurse -Force

Get-ChildItem $dataDir -File | Where-Object { $_.Name -ne "origin.txt" } |
    Remove-Item -Force

# Strip [Tester] from terminal.ini
$iniPath = "$dataDir\config\terminal.ini"
$content = Get-Content $iniPath -Raw
$content = $content -replace '(?s)\[Tester\].*?(?=\[|$)', ''
$content | Set-Content $iniPath -Encoding UTF8
```

---

## 15. `.set` File Auto-Discovery — No `ExpertParameters` Key Needed

**Symptom**: When placing a `.set` file in `MQL5\Profiles\Tester\`, the EA
still uses default values. Adding `ExpertParameters=V2z_CPPF.set` to the INI
`[Tester]` section makes no difference.

**Root Cause**: MT5 auto-discovers `.set` files by **filename matching**.
If a file named `<EAName>.set` exists in `MQL5\Profiles\Tester\`, it is
automatically loaded for that EA. The `ExpertParameters` key in the INI
config is **redundant** (or may even cause conflicts). The real issue was
the UTF-8 BOM (Problem 13), not the missing INI key.

**Fix**:
1. Name the file exactly `<EAName>.set` (e.g., `V2z_CPPF.set`)
2. Place it in `MQL5\Profiles\Tester\`
3. Ensure **NO BOM** (use `-Encoding ASCII`)
4. No `ExpertParameters` key needed in the INI

**Verification**: The agent log will show each parameter value on startup:
```
CS  0  13:45:21.738  Tester  Z_THRESHOLD=2.5
CS  0  13:45:21.738  Tester  STOP_A=3.0
...
```
