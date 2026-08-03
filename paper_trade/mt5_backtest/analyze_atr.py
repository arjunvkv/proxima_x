"""Verify ATR threshold hypothesis on all AUDUSD trades."""
import re
from collections import Counter

log_path = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\81A933A9AFC5DE3C23B15CAB19C63850\Tester\logs\20260727.log"

with open(log_path, encoding='utf-16') as f:
    raw = f.read()
lines = raw.replace('\r\n', '\n').split('\n')

trades = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    if 'OPEN ' in line and 'dir=' in line:
        continue

    m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(CLOSE \w+ rsn=\w+ pnl=[-\d.]+.*)', line)
    if m:
        cm = re.match(r'CLOSE (\w+) rsn=(\w+) pnl=([-\d.]+) z=([-\d.e+]+) atr=([\d.e-]+) sprd=([\d.]+) held=(\d+) bars?', m.group(4))
        if cm:
            trades.append({'symbol': cm.group(1), 'pnl': float(cm.group(3)), 'z_entry': float(cm.group(4)),
                         'atr': float(cm.group(5)), 'spread': float(cm.group(6)), 'held_bars': int(cm.group(7)),
                         'result': 'WIN' if float(cm.group(3)) > 0 else 'LOSS', 'log_ts': m.group(1)})
            continue

    m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(LOST \w+ dir=\d+ entry=[\d.]+.*)', line)
    if m:
        lm = re.match(r'LOST (\w+) dir=(\d+) entry=([\d.]+) stop=([\d.]+) pnl=([-\d.]+) atr=([\d.e-]+) z=([-\d.e+]+) sprd=([\d.]+) held=(\d+) bars?', m.group(4))
        if lm:
            trades.append({'symbol': lm.group(1), 'pnl': float(lm.group(5)), 'z_entry': float(lm.group(7)),
                         'atr': float(lm.group(6)), 'spread': float(lm.group(8)), 'held_bars': int(lm.group(9)),
                         'result': 'WIN' if float(lm.group(5)) > 0 else 'LOSS', 'log_ts': m.group(1)})

# AUDUSD ATR distribution
au_trades = [t for t in trades if t['symbol'] == 'AUDUSD']

print("=" * 70)
print("AUDUSD: ATR THRESHOLD SIMULATION")
print("=" * 70)
print(f"{'Min ATR':>8} | {'Trades':>6} | {'Wins':>4} | {'WR':>5} | {'PnL':>8} | {'AvgW':>6} | {'AvgL':>6}")
print("-" * 70)

thresholds = [0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00010, 0.00012, 0.00015]
for thresh in thresholds:
    filtered = [t for t in au_trades if t['atr'] >= thresh]
    if not filtered:
        continue
    wins = sum(1 for t in filtered if t['result'] == 'WIN')
    pnl = sum(t['pnl'] for t in filtered)
    avg_w = sum(t['pnl'] for t in filtered if t['result'] == 'WIN') / wins if wins > 0 else 0
    losses = [t for t in filtered if t['result'] == 'LOSS']
    avg_l = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
    print(f" ATR>={thresh:.5f} | {len(filtered):6d} | {wins:4d} | {wins/len(filtered)*100:4.0f}% | "
          f"${pnl:6.2f} | ${avg_w:5.2f} | ${avg_l:5.2f}")

# Also show combined ATR + spread + z filter
print()
print("=" * 70)
print("COMBINED: ATR>=0.00007 + spread<=5 + |z|<5.5")
print("=" * 70)
combined = [t for t in au_trades if t['atr'] >= 0.00007 and t['spread'] <= 5 and abs(t['z_entry']) < 5.5]
wins = sum(1 for t in combined if t['result'] == 'WIN')
pnl = sum(t['pnl'] for t in combined)
print(f"  Trades: {len(combined)}, Wins: {wins}, WR: {wins/len(combined)*100:.0f}%, PnL: ${pnl:.2f}")
for t in combined:
    print(f"  {t['result']:4s} | z={t['z_entry']:+6.2f} | sprd={t['spread']:.0f} | "
          f"atr={t['atr']:.6f} | held={t['held_bars']:2d} | pnl={t['pnl']:+6.1f}")

# Repeat for GBPUSD
print()
print("=" * 70)
print("GBPUSD: ATR THRESHOLD SIMULATION")
print("=" * 70)
gbp_trades = [t for t in trades if t['symbol'] == 'GBPUSD']
print(f"{'Min ATR':>8} | {'Trades':>6} | {'Wins':>4} | {'WR':>5} | {'PnL':>8} | {'AvgW':>6} | {'AvgL':>6}")
print("-" * 70)

for thresh in thresholds:
    filtered = [t for t in gbp_trades if t['atr'] >= thresh]
    if not filtered:
        continue
    wins = sum(1 for t in filtered if t['result'] == 'WIN')
    pnl = sum(t['pnl'] for t in filtered)
    avg_w = sum(t['pnl'] for t in filtered if t['result'] == 'WIN') / wins if wins > 0 else 0
    losses = [t for t in filtered if t['result'] == 'LOSS']
    avg_l = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
    print(f" ATR>={thresh:.5f} | {len(filtered):6d} | {wins:4d} | {wins/len(filtered)*100:4.0f}% | "
          f"${pnl:6.2f} | ${avg_w:5.2f} | ${avg_l:5.2f}")

print()
print("COMBINED: ATR>=0.00004 + spread<=5 on GBPUSD (all runs):")
combined_gbp = [t for t in gbp_trades if t['atr'] >= 0.00004 and t['spread'] <= 5]
wins = sum(1 for t in combined_gbp if t['result'] == 'WIN')
pnl = sum(t['pnl'] for t in combined_gbp)
print(f"  Trades: {len(combined_gbp)}, Wins: {wins}, WR: {wins/len(combined_gbp)*100:.0f}%, PnL: ${pnl:.2f}")

# Also show hour 04 specifically
print()
print("=" * 70)
print("HOUR 04 ANALYSIS (all pairs)")
print("=" * 70)
h4_trades = []
for line in lines:
    line = line.strip()
    m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(LOST .*)', line)
    if m and m.group(3).startswith('04:'):
        h4_trades.extend([t for t in trades if t['log_ts'] == m.group(1)])
        
# Actually, simpler: filter by hour
import re as re2
h4_list = [t for t in trades if t['time'].startswith('04:')]
h4_au = [t for t in h4_list if t['symbol'] == 'AUDUSD']
h4_gbp = [t for t in h4_list if t['symbol'] == 'GBPUSD']
print(f"AUDUSD at h=04: {len(h4_au)} trades, {sum(1 for t in h4_au if t['result']=='WIN')}W, "
      f"PnL=${sum(t['pnl'] for t in h4_au):.2f}")
print(f"GBPUSD at h=04: {len(h4_gbp)} trades, {sum(1 for t in h4_gbp if t['result']=='WIN')}W, "
      f"PnL=${sum(t['pnl'] for t in h4_gbp):.2f}")

# And the evening window analysis
print()
print("=" * 70)
print("HOUR 01-02 vs HOUR 03-04 comparison (AUDUSD)")
print("=" * 70)
h12 = [t for t in au_trades if t['time'].startswith(('01:', '02:'))]
h34 = [t for t in au_trades if t['time'].startswith(('03:', '04:'))]
for label, ts in [("h=01-02", h12), ("h=03-04", h34)]:
    wins = sum(1 for t in ts if t['result'] == 'WIN')
    pnl = sum(t['pnl'] for t in ts)
    print(f"{label}: {len(ts)} trades, {wins}W, WR={wins/len(ts)*100:.0f}%, PnL=${pnl:.2f}, "
          f"Avg ATR={sum(t['atr'] for t in ts)/len(ts):.6f}")
