"""Parse V2z_Pro trades from MT5 tester log."""
import re

with open(r'C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\Tester\logs\20260726.log', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Find V2z_Pro section
start = None
for i, line in enumerate(lines):
    if 'V2z_Pro.ex5 from 2026.06.08' in line and 'started' not in line and 'testing' in line:
        start = i
        break

if start is None:
    # Try again: find the test configuration header
    for i, line in enumerate(lines):
        if 'V2z_Pro.ex5' in line and 'testing of' in line:
            start = i
            break

if start is None:
    print("V2z_Pro section not found!")
    import sys; sys.exit(1)

print(f"V2z_Pro section starts at line {start+1}")

trades = []
for line in lines[start:]:
    if 'CLOSE' in line and 'EURAUD' in line:
        m = re.search(r'raw=([-\d.]+)', line)
        n = re.search(r'pnl=([-\d.]+)', line)
        if m and n:
            raw = float(m.group(1))
            net = float(n.group(1))
            rsn = 'stp' if 'stp' in line or 'stop' in line else 'exp' if 'expiry' in line else '?'
            trades.append((raw, net, rsn))

if not trades:
    print("No CLOSE trades found in V2z_Pro section!")
    # Debug: show some lines
    for line in lines[start:start+20]:
        print(repr(line[:200]))
    import sys; sys.exit(1)

print(f"\nTrades extracted: {len(trades)}")
for raw, net, rsn in trades:
    print(f"  raw=${raw:+>8.2f}  net=${net:+>8.2f}  rsn={rsn}")

wins = [t for t in trades if t[1] > 0]
losses = [t for t in trades if t[1] < 0]
print(f"\nSummary:")
print(f"  Total trades: {len(trades)}")
print(f"  WR: {len(wins)/len(trades)*100:.1f}%")
if wins:
    print(f"  Gross wins: ${sum(t[0] for t in wins):+.2f} avg=${sum(t[0] for t in wins)/len(wins):+.2f}")
if losses:
    print(f"  Gross losses: ${sum(t[0] for t in losses):+.2f} avg=${sum(t[0] for t in losses)/len(losses):+.2f}")
print(f"  Gross PnL: ${sum(t[0] for t in trades):+.2f}")
print(f"  Commission: ${len(trades)*3.75:.2f}")
print(f"  Net PnL: ${sum(t[1] for t in trades):+.2f}")
print(f"  Exit by stop: {sum(1 for t in trades if t[2]=='stp')}")
print(f"  Exit by expiry: {sum(1 for t in trades if t[2]=='exp')}")
