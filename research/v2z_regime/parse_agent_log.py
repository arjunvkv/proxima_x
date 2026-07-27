"""Parse the agent log (tab-separated) to extract per-month trade data."""
import re, os
from collections import defaultdict

LOG = os.path.join(os.environ['APPDATA'],
    r'MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\logs\20260726.log')

# Format: CS\t0\ttimestamp\tsource\tmessage
# source can be "Tester" or "V2z_v2_Clean (EURAUD,M1)"
# message contains the actual data

RUN_START = re.compile(r'(\w+),M1: testing of .+? from (\d{4}\.\d{2}\.\d{2}) \d{2}:\d{2} to (\d{4}\.\d{2}\.\d{2}) \d{2}:\d{2}')
CLOSEDONE = re.compile(r'(\d{4}\.\d{2}\.\d{2}) \d{2}:\d{2}:\d{2}\s+CLOSEDONE (\S+) pnl=([-\d.]+)')
FINAL_BAL = re.compile(r'final balance ([\d.]+) USD')
Z_PATTERN = re.compile(r'Z_THRESHOLD=([\d.]+)')

runs = []
cur_run = None

with open(LOG, 'r', encoding='utf-16', errors='replace') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        source = parts[3]  # "Tester" or "V2z_v2_Clean (EURAUD,M1)"
        msg = parts[4]
        
        # Detect new run from Tester messages
        if source == 'Tester':
            m = RUN_START.match(msg)
            if m:
                cur_run = {
                    'pair': m.group(1),
                    'start': m.group(2),
                    'end': m.group(3),
                    'z': None,
                    'trades': []
                }
                runs.append(cur_run)
                continue
            m = Z_PATTERN.search(msg)
            if m and cur_run is not None:
                cur_run['z'] = float(m.group(1))
                continue
            m = FINAL_BAL.search(msg)
            if m and cur_run is not None:
                cur_run['final_balance'] = float(m.group(1))
                continue
        else:
            # Source is like "V2z_v2_Clean (EURAUD,M1)"
            m = CLOSEDONE.search(msg)
            if m and cur_run is not None:
                date_str = m.group(1)
                pair = m.group(2)
                pnl = float(m.group(3))
                month = date_str[:7]
                cur_run['trades'].append({'date': date_str, 'pair': pair, 'pnl': pnl, 'month': month})

print(f"Parsed {len(runs)} runs, {sum(len(r['trades']) for r in runs)} total trades")

# Group by (z, pair, month)
z_pair_month = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
pair_day = defaultdict(lambda: defaultdict(list))

for run in runs:
    z = run['z']
    for t in run['trades']:
        z_pair_month[z][t['pair']][t['month']].append(t['pnl'])
        pair_day[t['pair']][t['date']].append(t['pnl'])

# Per-run summary
print("\n" + "=" * 110)
print("RUN-BY-RUN SUMMARY")
print("=" * 110)
for i, run in enumerate(runs):
    nt = len(run['trades'])
    gross = sum(t['pnl'] for t in run['trades'])
    avg = gross / nt if nt > 0 else 0
    fb = run.get('final_balance', '?')
    print(f"  Run {i}: {run['pair']} z={run['z']} {run['start']}->{run['end']}  trades={nt:>4d}  pnl=${gross:>+8.2f}  avg=${avg:>+7.2f}  final=${fb}")

# Month-by-month for each z threshold
for z in sorted(z_pair_month.keys()):
    data = z_pair_month[z]
    pairs = sorted(data.keys())
    months = sorted(set(m for p in pairs for m in data[p]))
    
    print(f"\n" + "=" * 110)
    print(f"Z-THRESHOLD = {z}")
    print(f"Commission per trade (lot=0.75): Fusion $3.38, IC Markets $5.25")
    print("=" * 110)
    
    header = f"{'Pair':<10s}" + "".join(f"{m:>10s}" for m in months) + f"{'TOTAL':>10s}  {'N':>5s}  {'GROSS/TR':>8s}"
    print(header)
    print("-" * len(header))
    
    gt = 0
    gn = 0
    for pair in pairs:
        total = 0
        n = 0
        row = f"{pair:<10s}"
        for m in months:
            pnls = data[pair].get(m, [])
            s = sum(pnls)
            cnt = len(pnls)
            row += f"{s:>+10.2f}"
            total += s
            n += cnt
        row += f"{total:>+10.2f}  {n:>5d}  {total/max(n,1):>+8.2f}"
        print(row)
        gt += total
        gn += n
    print("-" * len(header))
    print(f"{'ALL':<10s}" + "".join(f"{'':>10s}" for _ in months) + f"{gt:>+10.2f}  {gn:>5d}  {gt/max(gn,1):>+8.2f}")
    
    # Commission scenarios
    for comm_label, comm_per_side in [("Fusion $2.25/lot", 2.25), ("IC Markets $3.50/lot", 3.50)]:
        # Determine lot size
        lot_size = 0.75  # default for z=3.5
        for run in runs:
            if run['z'] == z and len(run['trades']) > 0:
                # Check trade PnL magnitudes to infer lot
                avg_pnl = abs(sum(t['pnl'] for t in run['trades']) / len(run['trades']))
                if avg_pnl > 30:
                    lot_size = 1.0
                break
        
        comm_per_trade = lot_size * comm_per_side * 2
        net_all = gt - gn * comm_per_trade
        
        print(f"\n  {comm_label} (lot={lot_size}, ${comm_per_trade:.2f}/trade):")
        print(f"  Gross: ${gt:>+.2f}  Comm: ${gn * comm_per_trade:.2f}  Net: ${net_all:>+.2f}  Retained: {net_all/gt*100 if gt != 0 else 0:.1f}%")

# Per-pair statistics
print("\n" + "=" * 110)
print("PER-PAIR SUMMARY (all runs)")
print("=" * 110)
pair_totals = defaultdict(lambda: {'pnl': 0, 'n': 0, 'wins': 0})
for run in runs:
    for t in run['trades']:
        pair_totals[t['pair']]['pnl'] += t['pnl']
        pair_totals[t['pair']]['n'] += 1
        if t['pnl'] > 0:
            pair_totals[t['pair']]['wins'] += 1

for pair in sorted(pair_totals.keys()):
    d = pair_totals[pair]
    wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
    print(f"  {pair:<10s}  trades={d['n']:>5d}  gross=${d['pnl']:>+9.2f}  avg=${d['pnl']/max(d['n'],1):>+7.2f}  wr={wr:>5.1f}%")

# Daily max drawdown
print("\n" + "=" * 110)
print("FTMO COMPLIANCE: Max Drawdown per Pair (Gross PnL, no commission)")
print("=" * 110)

for pair in sorted(pair_day.keys()):
    days = pair_day[pair]
    day_list = []
    for d in sorted(days.keys()):
        daily_total = sum(days[d])
        day_list.append((d, daily_total))
    
    if not day_list:
        continue
    
    cum_pnl = 0
    peak = 0
    max_dd_abs = 0
    max_dd_pct = 0
    
    for date_str, dp in day_list:
        cum_pnl += dp
        if cum_pnl > peak:
            peak = cum_pnl
        if peak > 0:
            dd_abs = peak - cum_pnl
            dd_pct = dd_abs / 10000 * 100  # assuming $10k account
            if dd_abs > max_dd_abs:
                max_dd_abs = dd_abs
    
    total = sum(x[1] for x in day_list)
    worst_day = min(x[1] for x in day_list)
    worst_day_date = min(day_list, key=lambda x: x[1])[0]
    
    print(f"\n{pair}:")
    print(f"  Total gross PnL: ${total:+.2f}")
    print(f"  Max DD (peak-to-trough): ${max_dd_abs:.2f}")
    print(f"  Max DD on $10k: {max_dd_abs/10000*100:.2f}%")
    print(f"  Worst day: ${worst_day:+.2f} ({worst_day_date})")
    days_over_500 = sum(1 for _, dp in day_list if dp < -500)
    days_over_1000 = sum(1 for _, dp in day_list if dp < -1000)
    print(f"  Days with loss > $500: {days_over_500}")
    print(f"  Days with loss > $1000: {days_over_1000}")

# GBPNZD sensitivity analysis
print("\n" + "=" * 110)
print("GBPNZD EXCLUSION ANALYSIS (is removing it overfitting?)")
print("=" * 110)

for z in sorted(z_pair_month.keys()):
    data = z_pair_month[z]
    gt_w = 0; gn_w = 0; gt_wo = 0; gn_wo = 0
    for pair in data:
        for m in data[pair]:
            gross = sum(data[pair][m])
            cnt = len(data[pair][m])
            gt_w += gross; gn_w += cnt
            if pair != 'GBPNZD':
                gt_wo += gross; gn_wo += cnt
    
    lot_size = 0.75 if z == 3.5 else 1.0
    for comm_label, comm_per_side in [("Fusion $2.25", 2.25), ("IC Markets $3.50", 3.50)]:
        comm = lot_size * comm_per_side * 2
        net_w = gt_w - gn_w * comm
        net_wo = gt_wo - gn_wo * comm
        print(f"\n  z={z}, {comm_label}:")
        print(f"    With GBPNZD:    gross=${gt_w:.2f}  net=${net_w:.2f}  (n={gn_w})")
        print(f"    Without GBPNZD: gross=${gt_wo:.2f}  net=${net_wo:.2f}  (n={gn_wo})")
        improvement = net_wo - net_w
        print(f"    Improvement: ${improvement:+.2f}")
        if improvement > 0:
            print(f"    ✓ Removing GBPNZD helps by ${improvement:.2f}")
        else:
            print(f"    ✗ Removing GBPNZD hurts by ${abs(improvement):.2f}")
