import re

line = 'OQ\t0\t18:09:58.914\tCore 01\t2026.06.09 02:28:00   OPEN GBPUSD dir=1 entry=1.33383 sl=1.3336695000000003 atr=0.00005349999999996191 z=-4.4548720369387675 sprd=6.0'
print(f"Line: {repr(line[:80])}")

# Step by step
m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)', line)
print(f'Step 1 (timestamp): {m}')

m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+', line)
print(f'Step 2 (+Core): {m}')

m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t', line)
print(f'Step 3 (+trailing tab): {m}')

m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)', line)
print(f'Step 4 (+date): {m}')

if m:
    print(f'  Groups: {m.groups()}')

# Now try the full CLOSE pattern on a close line
close_line = 'KK\t0\t18:09:58.914\tCore 01\t2026.06.09 02:30:28   CLOSE GBPUSD rsn=stop pnl=-8.5 z=-4.4548720369387675 atr=0.00005349999999996191 sprd=6.0 held=2 bars'
print(f"\nCLOSE line: {repr(close_line[:80])}")

# Test full CLOSE regex
m = re.match(
    r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t'
    r'(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+'
    r'(CLOSE \w+ rsn=\w+ pnl=[-\d.]+.*)',
    close_line
)
print(f'CLOSE full regex: {m}')
if m:
    print(f'  Groups: {m.groups()}')
