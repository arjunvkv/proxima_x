import re

log_path = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\81A933A9AFC5DE3C23B15CAB19C63850\Tester\logs\20260727.log"
with open(log_path, encoding='utf-16') as f:
    raw = f.read()
lines = raw.replace('\r\n', '\n').split('\n')

print(f"Total lines: {len(lines)}")

# Find a CLOSE line
for i, l in enumerate(lines):
    if 'CLOSE GBPUSD' in l:
        print(f"Testing CLOSE line {i}: {repr(l[:150])}")
        
        # My exact regex pattern
        pat = r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(CLOSE \w+ rsn=\w+ pnl=[-\d.]+.*)'
        m = re.match(pat, l)
        if m:
            print(f"  MATCHED!")
            print(f"  log_ts={m.group(1)}")
            print(f"  date={m.group(2)}")
            print(f"  time={m.group(3)}")
            print(f"  msg={m.group(4)}")
        else:
            print(f"  NO MATCH - testing simpler patterns:")
        
        # Simpler tests
        m2 = re.match(r'^\w+\t\d+\t', l)
        print(f"    ^\\w+\\t\\d+\\t: {bool(m2)}")
        
        m3 = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)', l)
        print(f"    +timestamp: {bool(m3)}")
        if m3:
            print(f"    ts={m3.group(1)}")
        
        m4 = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore 01', l)
        print(f"    +Core 01: {bool(m4)}")
        
        m5 = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)', l)
        print(f"    +date: {bool(m5)}")
        if m5:
            print(f"    date={m5.group(2)}")
            
        m6 = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)', l)
        print(f"    +time: {bool(m6)}")
        
        m7 = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(CLOSE.*)', l)
        print(f"    +CLOSE: {bool(m7)}")
        if m7:
            print(f"    msg={m7.group(4)}")
        break
