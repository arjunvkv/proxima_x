"""Analyze what the stability gate actually filters on AUDUSD."""
import re
from collections import defaultdict

log_path = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\81A933A9AFC5DE3C23B15CAB19C63850\Tester\logs\20260727.log"

with open(log_path, encoding='utf-16') as f:
    raw = f.read()
lines = raw.replace('\r\n', '\n').split('\n')

# Parse all LOST/CLOSE events
trades = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    if 'OPEN ' in line and 'dir=' in line:
        continue

    # CLOSE
    m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(CLOSE \w+ rsn=\w+ pnl=[-\d.]+.*)', line)
    if m:
        msg = m.group(4)
        cm = re.match(r'CLOSE (\w+) rsn=(\w+) pnl=([-\d.]+) z=([-\d.e+]+) atr=([\d.e-]+) sprd=([\d.]+) held=(\d+) bars?', msg)
        if cm:
            trades.append({
                'type': 'CLOSE', 'symbol': cm.group(1), 'pnl': float(cm.group(3)),
                'z_entry': float(cm.group(4)), 'atr': float(cm.group(5)),
                'spread': float(cm.group(6)), 'held_bars': int(cm.group(7)),
                'log_ts': m.group(1), 'date': m.group(2), 'time': m.group(3),
                'result': 'WIN' if float(cm.group(3)) > 0 else 'LOSS',
            })
            continue

    # LOST
    m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(LOST \w+ dir=\d+ entry=[\d.]+.*)', line)
    if m:
        msg = m.group(4)
        lm = re.match(r'LOST (\w+) dir=(\d+) entry=([\d.]+) stop=([\d.]+) pnl=([-\d.]+) atr=([\d.e-]+) z=([-\d.e+]+) sprd=([\d.]+) held=(\d+) bars?', msg)
        if lm:
            trades.append({
                'type': 'LOST', 'symbol': lm.group(1), 'pnl': float(lm.group(5)),
                'z_entry': float(lm.group(7)), 'atr': float(lm.group(6)),
                'spread': float(lm.group(8)), 'held_bars': int(lm.group(9)),
                'log_ts': m.group(1), 'date': m.group(2), 'time': m.group(3),
                'result': 'WIN' if float(lm.group(5)) > 0 else 'LOSS',
            })

# Parse STABLE entry lines  
stable_entries = []
for line in lines:
    line = line.strip()
    m = re.match(r'^\w+\t\d+\t(\d+:\d+:\d+\.\d+)\tCore \d+\t(\d+\.\d+\.\d+)\s+(\d+:\d+:\d+)\s+(STABLE entry z=.*)', line)
    if m:
        msg = m.group(4)
        sm = re.match(r'STABLE entry z=([-\d.e+]+) z_cum=([-\d.e+]+) spd_z=([-\d.e+]+) vol_z=([-\d.e+]+) stab=([\d.e+]+)', msg)
        if sm:
            stable_entries.append({
                'log_ts': m.group(1), 'date': m.group(2), 'time': m.group(3),
                'z': float(sm.group(1)), 'z_cum': float(sm.group(2)),
                'spd_z': float(sm.group(3)), 'vol_z': float(sm.group(4)),
                'stab': float(sm.group(5)),
            })

# ===========================
# KEY ANALYSIS: AUDUSD baseline vs stable gate
# ===========================

# AUDUSD baseline run 18:19:12 = NO GATE (29 trades)
# AUDUSD stable run 18:44:44 = GATE 0.3 zcum=0 (10 trades)
# AUDUSD stable run 18:18:06 = GATE 0.5 zcum=5 (6 trades)

base_au = [t for t in trades if t['log_ts'].startswith('18:19:12') and t['symbol'] == 'AUDUSD']
stable_au = [t for t in trades if t['log_ts'].startswith('18:44:44') and t['symbol'] == 'AUDUSD']
strict_au = [t for t in trades if t['log_ts'].startswith('18:18:06') and t['symbol'] == 'AUDUSD']

print("=" * 90)
print("AUDUSD BASELINE (no gate): 29 trades")
print("=" * 90)
base_by_key = {}
for t in base_au:
    key = t['date'] + t['time']
    base_by_key[key] = t
    print(f"  {t['date']} {t['time']} z={t['z_entry']:+7.2f} sprd={t['spread']:.0f}  "
          f"atr={t['atr']:.6f} held={t['held_bars']:2d}  {t['result']:4s}  pnl={t['pnl']:+6.1f}")

base_pnl = sum(t['pnl'] for t in base_au)
base_w = sum(1 for t in base_au if t['result'] == 'WIN')
print(f"\n  TOTAL: {len(base_au)} trades, {base_w}W, WR={base_w/len(base_au)*100:.0f}%, PnL=${base_pnl:.2f}")

print()
print("=" * 90)
print("AUDUSD STABLE GATE 0.3 zcum=0 (nocum): 10 trades")  
print("=" * 90)
stable_by_key = {}
for t in stable_au:
    key = t['date'] + t['time']
    stable_by_key[key] = t
    print(f"  {t['date']} {t['time']} z={t['z_entry']:+7.2f} sprd={t['spread']:.0f}  "
          f"atr={t['atr']:.6f} held={t['held_bars']:2d}  {t['result']:4s}  pnl={t['pnl']:+6.1f}")

stable_pnl = sum(t['pnl'] for t in stable_au)
stable_w = sum(1 for t in stable_au if t['result'] == 'WIN')
print(f"\n  TOTAL: {len(stable_au)} trades, {stable_w}W, WR={stable_w/len(stable_au)*100:.0f}%, PnL=${stable_pnl:.2f}")

print()
print("=" * 90)
print("AUDUSD STRICT GATE 0.5 zcum=5: 6 trades")
print("=" * 90)
for t in strict_au:
    key = t['date'] + t['time']
    print(f"  {t['date']} {t['time']} z={t['z_entry']:+7.2f} sprd={t['spread']:.0f}  "
          f"atr={t['atr']:.6f} held={t['held_bars']:2d}  {t['result']:4s}  pnl={t['pnl']:+6.1f}")

# ===========================
# FILTERED OUT analysis
# ===========================
print()
print("=" * 90)
print("TRADES FILTERED OUT BY STABLE GATE 0.3 (nocum):")
print("(in baseline but NOT in stable gate run)")
print("=" * 90)
filtered = []
for key, t in sorted(base_by_key.items()):
    if key not in stable_by_key:
        filtered.append(t)
        print(f"  {t['date']} {t['time']} z={t['z_entry']:+7.2f} sprd={t['spread']:.0f}  "
              f"atr={t['atr']:.6f} held={t['held_bars']:2d}  {t['result']:4s}  pnl={t['pnl']:+6.1f}")

filt_w = sum(1 for t in filtered if t['result'] == 'WIN')
filt_pnl = sum(t['pnl'] for t in filtered)
print(f"\n  TOTAL filtered: {len(filtered)} trades, {filt_w}W, WR={filt_w/len(filtered)*100:.0f}%, PnL=${filt_pnl:.2f}")

# ===========================
# Compare characteristics between filtered vs passed
# ===========================
print()
print("=" * 90)
print("CHARACTERISTICS: Filtered vs Passed (AUDUSD baseline vs stable 0.3 nocum)")
print("=" * 90)

for group_name, trades_list in [("FILTERED OUT", filtered), ("PASSED GATE", stable_au)]:
    if not trades_list:
        continue
    avg_z = sum(abs(t['z_entry']) for t in trades_list) / len(trades_list)
    avg_spread = sum(t['spread'] for t in trades_list) / len(trades_list)
    avg_atr = sum(t['atr'] for t in trades_list) / len(trades_list)
    avg_held = sum(t['held_bars'] for t in trades_list) / len(trades_list)
    wins = sum(1 for t in trades_list if t['result'] == 'WIN')
    pnl_per = sum(t['pnl'] for t in trades_list) / len(trades_list)
    
    # Hour distribution
    hours = [int(t['time'].split(':')[0]) for t in trades_list]
    from collections import Counter
    hour_dist = Counter(hours)
    
    print(f"\n{group_name} ({len(trades_list)} trades):")
    print(f"  Avg |z|: {avg_z:.2f}  Avg spread: {avg_spread:.1f}  Avg ATR: {avg_atr:.6f}  Avg held: {avg_held:.1f}")
    print(f"  WR: {wins/len(trades_list)*100:.0f}%  PnL/trade: ${pnl_per:.2f}")
    print(f"  Hour dist: {dict(sorted(hour_dist.items()))}")

# ===========================
# WHAT MAKES A WINNER ON AUDUSD?
# ===========================
print()
print("=" * 90)
print("AUDUSD: WINNERS VS LOSERS CHARACTERISTICS (ALL RUNS COMBINED)")
print("=" * 90)

all_au = [t for t in trades if t['symbol'] == 'AUDUSD']
winners = [t for t in all_au if t['result'] == 'WIN']
losers = [t for t in all_au if t['result'] == 'LOSS']

for label, ts in [("WINNERS", winners), ("LOSERS", losers)]:
    print(f"\n{label} ({len(ts)} trades):")
    print(f"  Avg z={sum(abs(t['z_entry']) for t in ts)/len(ts):.2f}  "
          f"Avg spread={sum(t['spread'] for t in ts)/len(ts):.1f}  "
          f"Avg ATR={sum(t['atr'] for t in ts)/len(ts):.6f}  "
          f"Avg held={sum(t['held_bars'] for t in ts)/len(ts):.1f}")
    
    # Spread distribution
    sprd_count = Counter(t['spread'] for t in ts)
    print(f"  Spread dist: {dict(sorted(sprd_count.items()))}")
    
    # Hour distribution
    hour_count = Counter(int(t['time'].split(':')[0]) for t in ts)
    print(f"  Hour dist: {dict(sorted(hour_count.items()))}")
    
    # Held bars distribution
    held_count = Counter(t['held_bars'] for t in ts)
    print(f"  Held bars: {dict(sorted(held_count.items()))}")
    
    # z-score distribution
    z_dist = Counter(int(abs(t['z_entry'])) for t in ts)
    print(f"  |z| dist: {dict(sorted(z_dist.items()))}")

# ===========================
# KEY INSIGHT: What killed AUDUSD baseline?
# ===========================
print()
print("=" * 90)
print("KEY INSIGHT: AUDUSD loses by spread + held_bars combination")
print("=" * 90)
losers_by_spread = defaultdict(list)
for t in losers:
    losers_by_spread[t['spread']].append(t)

for sprd in sorted(losers_by_spread.keys()):
    ts = losers_by_spread[sprd]
    print(f"\n  sprd={int(sprd)} losses ({len(ts)}):")
    for t in ts:
        print(f"    {t['date']} {t['time']} z={t['z_entry']:+7.2f} held={t['held_bars']:2d} "
              f"atr={t['atr']:.6f} pnl={t['pnl']:+6.1f}")

# ===========================
# Compare to GBPUSD  
# ===========================
print()
print("=" * 90)
print("GBPUSD: WINNERS VS LOSERS CHARACTERISTICS")
print("=" * 90)
all_gbp = [t for t in trades if t['symbol'] == 'GBPUSD']
gbp_winners = [t for t in all_gbp if t['result'] == 'WIN']
gbp_losers = [t for t in all_gbp if t['result'] == 'LOSS']

for label, ts in [("WINNERS", gbp_winners), ("LOSERS", gbp_losers)]:
    print(f"\n{label} ({len(ts)} trades):")
    print(f"  Avg z={sum(abs(t['z_entry']) for t in ts)/len(ts):.2f}  "
          f"Avg spread={sum(t['spread'] for t in ts)/len(ts):.1f}  "
          f"Avg ATR={sum(t['atr'] for t in ts)/len(ts):.6f}  "
          f"Avg held={sum(t['held_bars'] for t in ts)/len(ts):.1f}")
    
    sprd_count = Counter(t['spread'] for t in ts)
    print(f"  Spread dist: {dict(sorted(sprd_count.items()))}")
    hour_count = Counter(int(t['time'].split(':')[0]) for t in ts)
    print(f"  Hour dist: {dict(sorted(hour_count.items()))}")
