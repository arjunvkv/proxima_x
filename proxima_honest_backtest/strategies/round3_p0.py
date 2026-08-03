#!/usr/bin/env python3
"""P0 analysis: 7 new anomalies."""
import sys, time, json, math
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
from data.providers.mt5_provider import MT5Provider

ALL_PAIRS = ['EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','NZDUSD','USDCHF',
             'EURJPY','GBPJPY','AUDJPY','EURGBP','EURAUD','GBPAUD','AUDNZD',
             'EURNZD','GBPNZD','GBPCAD','GBPCHF']
MONTHS = [(2026, m) for m in range(1, 8)]
provider = MT5Provider()

def load():
    raw = {}
    for p in ALL_PAIRS:
        frames = [provider.load_rates(p, y, m, 'm5') for y,m in MONTHS]
        frames = [f for f in frames if not f.empty]
        if frames:
            d = pd.concat(frames, ignore_index=True)
            d.sort_values('time', inplace=True)
            d.reset_index(drop=True, inplace=True)
            raw[p] = d
    return raw

def make_aligned(raw):
    pieces = []
    for pair, df in raw.items():
        sub = df.set_index('time')[['close','open','high','low','tick_volume']]
        sub.columns = [pair, f'{pair}_open', f'{pair}_high', f'{pair}_low', f'{pair}_volume']
        pieces.append(sub)
    aligned = pd.concat(pieces, axis=1, sort=True)
    aligned.sort_index(inplace=True)
    aligned.ffill(inplace=True); aligned.bfill(inplace=True)
    aligned.reset_index(inplace=True); aligned.rename(columns={'index':'time'}, inplace=True)
    aligned['hour'] = aligned['time'].dt.hour
    aligned['minute'] = aligned['time'].dt.minute
    aligned['ymd'] = aligned['time'].dt.strftime('%Y%m%d')
    return aligned.to_dict('records'), aligned

print('Loading...')
t0 = time.time()
raw = load()
records, aligned_df = make_aligned(raw)
print(f'Loaded {len(records)} bars in {time.time()-t0:.1f}s')
n = len(records)

results = {}

# ──────────────────────────────────────────────
# 1) LONDON-NY OVERLAP MOMENTUM (12:00 UTC)
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('1) LONDON-NY OVERLAP MOMENTUM (12:00 UTC)')
print('='*60)
pnl, wins, loss = [], 0, 0
for i, r in enumerate(records):
    if r['hour'] != 12 or r['minute'] != 0: continue
    rets = []
    for p in ALL_PAIRS:
        c = r.get(p)
        if c is None or np.isnan(c): continue
        pi = i - 6
        if pi < 0: continue
        pv = records[pi].get(p)
        if pv is None or np.isnan(pv) or pv <= 0: continue
        rets.append((p, (c - pv) / pv))
    if len(rets) < 8: continue
    rets.sort(key=lambda x: x[1], reverse=True)
    for pair, _ in rets[:3]:
        fi = i + 12
        if fi >= n: continue
        eo = r.get(f'{pair}_open', r.get(pair))
        fc = records[fi].get(pair)
        if any(v is None or np.isnan(v) for v in (eo, fc)): continue
        ret = (fc - eo) / eo
        pnl.append(ret); wins += ret > 0; loss += ret < 0
wr = wins/(wins+loss)*100 if wins+loss else 0
avg = np.mean(pnl)*100 if pnl else 0
results['1_london_ny_overlap'] = {'events': len(pnl), 'WR': f'{wr:.1f}%', 'avg_ret_pct': f'{avg:+.4f}'}
print(f'  Events={len(pnl)} WR={wr:.1f}% Avg={avg:+.4f}%')

# ──────────────────────────────────────────────
# 2) VOLATILITY REGIME EXPANSION
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('2) VOLATILITY REGIME EXPANSION')
print('='*60)
pair_pnl = defaultdict(list)
for pair in ALL_PAIRS:
    df = raw.get(pair)
    if df is None or len(df) < 500: continue
    c = df['close'].values; h = df['high'].values; l = df['low'].values; o = df['open'].values
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    atr12 = np.convolve(tr, np.ones(12)/12, 'valid')
    atr48 = np.convolve(tr, np.ones(48)/48, 'valid')
    min_l = min(len(atr12), len(atr48))
    ratio = atr12[:min_l] / (atr48[:min_l] + 1e-12)
    c_arr = c[48:]
    o_arr = o[48:]
    h_arr = h[48+12-1:]  # align for 60-min range
    l_arr = l[48+12-1:]
    c60 = c[48+12:]

    events = 0; w = 0; lcnt = 0
    for j in range(12, min_l-12):
        if ratio[j] < 1.5: continue
        hi60 = max(c_arr[j-12:j])
        lo60 = min(c_arr[j-12:j])
        px = o_arr[j]
        if px > hi60:
            fut = c60[j]
            ret = (fut - px)/px
            pair_pnl[pair].append(ret); w += ret > 0; lcnt += ret < 0; events += 1
        elif px < lo60:
            fut = c60[j]
            ret = (px - fut)/px
            pair_pnl[pair].append(ret); w += ret > 0; lcnt += ret < 0; events += 1
    if events:
        wr2 = w/(w+lcnt)*100
        avg2 = np.mean(pair_pnl[pair])*100
        print(f'  {pair}: ev={events} WR={wr2:.1f}% avg={avg2:+.4f}%')

all_vp = [x for v in pair_pnl.values() for x in v]
wr_vp = sum(1 for x in all_vp if x>0)/len(all_vp)*100 if all_vp else 0
avg_vp = np.mean(all_vp)*100 if all_vp else 0
results['2_vol_expansion'] = {'events': len(all_vp), 'WR': f'{wr_vp:.1f}%', 'avg_ret_pct': f'{avg_vp:+.4f}%'}
print(f'  TOTAL: ev={len(all_vp)} WR={wr_vp:.1f}% avg={avg_vp:+.4f}%')

# ──────────────────────────────────────────────
# 3) TOKYO OPEN MOMENTUM (00:00 UTC, SHORT decliners)
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('3) TOKYO OPEN MOMENTUM (00:00 UTC, SHORT decliners)')
print('='*60)
pnl3, w3, l3 = [], 0, 0
for i, r in enumerate(records):
    if r['hour'] != 0 or r['minute'] != 5: continue
    rets = []
    for p in ALL_PAIRS:
        c = r.get(p)
        if c is None or np.isnan(c): continue
        pi = i - 36
        if pi < 0: continue
        pv = records[pi].get(p)
        if pv is None or np.isnan(pv) or pv <= 0: continue
        rets.append((p, (c - pv) / pv))
    if len(rets) < 8: continue
    rets.sort(key=lambda x: x[1])
    for pair, _ in rets[:3]:
        fi = i + 12
        if fi >= n: continue
        eo = r.get(f'{pair}_open', r.get(pair))
        fc = records[fi].get(pair)
        if any(v is None or np.isnan(v) for v in (eo, fc)): continue
        ret = (eo - fc) / eo
        pnl3.append(ret); w3 += ret > 0; l3 += ret < 0
wr3 = w3/(w3+l3)*100 if w3+l3 else 0
avg3 = np.mean(pnl3)*100 if pnl3 else 0
results['3_tokyo_momentum'] = {'events': len(pnl3), 'WR': f'{wr3:.1f}%', 'avg_ret_pct': f'{avg3:+.4f}%'}
print(f'  Events={len(pnl3)} WR={wr3:.1f}% Avg={avg3:+.4f}%')

# ──────────────────────────────────────────────
# 4) POST-SPIKE EXHAUSTION
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('4) POST-SPIKE EXHAUSTION (wait 1 bar, then fade)')
print('='*60)
all_sp = []
for pair in ALL_PAIRS:
    df = raw.get(pair)
    if df is None or len(df) < 200: continue
    c = df['close'].values; o = df['open'].values
    rets = np.diff(c) / c[:-1]
    rmean = np.mean(rets[:200]); rstd = np.std(rets[:200])
    sp = []; w4=0; l4=0
    for j in range(201, len(rets)-2):
        r = rets[j]
        if abs(r) < 3 * rstd: continue
        # wait 1 bar, enter fade
        entry = o[j+1]
        exit_p = c[j+2]
        if r > 0:  # spike up - fade = SHORT
            ret = (entry - exit_p) / entry
        else:  # spike down - fade = LONG
            ret = (exit_p - entry) / entry
        sp.append(ret); w4 += ret > 0; l4 += ret < 0
    if sp:
        wr4 = w4/(w4+l4)*100
        avg4 = np.mean(sp)*100
        print(f'  {pair}: ev={len(sp)} WR={wr4:.1f}% avg={avg4:+.4f}%')
        all_sp.extend(sp)

wr4t = sum(1 for x in all_sp if x>0)/len(all_sp)*100 if all_sp else 0
avg4t = np.mean(all_sp)*100 if all_sp else 0
results['4_spike_exhaustion'] = {'events': len(all_sp), 'WR': f'{wr4t:.1f}%', 'avg_ret_pct': f'{avg4t:+.4f}%'}
print(f'  TOTAL: ev={len(all_sp)} WR={wr4t:.1f}% avg={avg4t:+.4f}%')

# ──────────────────────────────────────────────
# 5) CROSS-PAIR MODERATE Z-SCORE MOMENTUM
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('5) CROSS-PAIR MODERATE Z-SCORE MOMENTUM (z=2-3, ride shock)')
print('='*60)
cross_pairs = ['EURAUD','GBPAUD','AUDNZD','EURNZD','GBPNZD','GBPCAD']
zpnl = defaultdict(list)
for pair in cross_pairs:
    df = raw.get(pair)
    if df is None or len(df) < 210: continue
    c = df['close'].values; o = df['open'].values
    ret3 = np.full(len(c), np.nan)
    for j in range(3, len(c)):
        ret3[j] = (c[j] - c[j-3]) / c[j-3]
    rmean = np.nanmean(ret3[:200]); rstd = np.nanstd(ret3[:200])
    events = 0; w=0; l=0
    for j in range(200, len(c)-6):
        z = (ret3[j] - rmean) / (rstd + 1e-12)
        if abs(z) < 2.0 or abs(z) > 3.5: continue
        # Ride the shock direction (momentum)
        entry = o[j+3]
        exit_p = c[j+9]  # ~30 min hold
        if z > 0:  # positive shock - ride up
            ret = (exit_p - entry) / entry
        else:
            ret = (entry - exit_p) / entry  # SHORT
        zpnl[pair].append(ret); w += ret > 0; l += ret < 0; events += 1
    if events:
        wr5 = w/(w+l)*100
        avg5 = np.mean(zpnl[pair])*100
        print(f'  {pair}: ev={events} WR={wr5:.1f}% avg={avg5:+.4f}%')

all_zp = [x for v in zpnl.values() for x in v]
wr5t = sum(1 for x in all_zp if x>0)/len(all_zp)*100 if all_zp else 0
avg5t = np.mean(all_zp)*100 if all_zp else 0
results['5_cross_z_momentum'] = {'events': len(all_zp), 'WR': f'{wr5t:.1f}%', 'avg_ret_pct': f'{avg5t:+.4f}%'}
print(f'  TOTAL: ev={len(all_zp)} WR={wr5t:.1f}% avg={avg5t:+.4f}%')

# ──────────────────────────────────────────────
# 6) US SESSION BREAKDOWN (13:00-20:00 UTC)
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('6) US SESSION BREAKDOWN (13:00 UTC entry, ride 120 min)')
print('='*60)
pnl6, w6, l6 = [], 0, 0
for i, r in enumerate(records):
    if r['hour'] != 13 or r['minute'] != 0: continue
    rets = []
    for p in ALL_PAIRS:
        c = r.get(p)
        if c is None or np.isnan(c): continue
        pi = i - 12
        if pi < 0: continue
        pv = records[pi].get(p)
        if pv is None or np.isnan(pv) or pv <= 0: continue
        rets.append((p, (c - pv) / pv))
    if len(rets) < 8: continue
    rets.sort(key=lambda x: x[1], reverse=True)
    for pair, _ in rets[:3]:
        fi = i + 24
        if fi >= n: continue
        eo = r.get(f'{pair}_open', r.get(pair))
        fc = records[fi].get(pair)
        if any(v is None or np.isnan(v) for v in (eo, fc)): continue
        ret = (fc - eo) / eo
        pnl6.append(ret); w6 += ret > 0; l6 += ret < 0
wr6 = w6/(w6+l6)*100 if w6+l6 else 0
avg6 = np.mean(pnl6)*100 if pnl6 else 0
results['6_us_session'] = {'events': len(pnl6), 'WR': f'{wr6:.1f}%', 'avg_ret_pct': f'{avg6:+.4f}%'}
print(f'  Events={len(pnl6)} WR={wr6:.1f}% Avg={avg6:+.4f}%')

# ──────────────────────────────────────────────
# 7) MONTH-END FIXING (last 2 days of month)
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('7) MONTH-END FIXING')
print('='*60)
month_ends = set()
for y,m in MONTHS:
    for day in [28,29,30,31]:
        ds = f'{y}{m:02d}{day:02d}'
        month_ends.add(ds)

pnl7, w7, l7 = [], 0, 0
for i, r in enumerate(records):
    ymd = str(r['ymd'])
    if ymd not in month_ends: continue
    if r['hour'] != 20 or r['minute'] != 0: continue
    rets = []
    for p in ALL_PAIRS:
        c = r.get(p)
        if c is None or np.isnan(c): continue
        pi = i - 24
        if pi < 0: continue
        pv = records[pi].get(p)
        if pv is None or np.isnan(pv) or pv <= 0: continue
        rets.append((p, (c - pv) / pv))
    if len(rets) < 8: continue
    rets.sort(key=lambda x: x[1], reverse=True)
    for pair, _ in rets[:3]:
        fi = i + 12
        if fi >= n: continue
        eo = r.get(f'{pair}_open', r.get(pair))
        fc = records[fi].get(pair)
        if any(v is None or np.isnan(v) for v in (eo, fc)): continue
        ret = (fc - eo) / eo
        pnl7.append(ret); w7 += ret > 0; l7 += ret < 0
wr7 = w7/(w7+l7)*100 if w7+l7 else 0
avg7 = np.mean(pnl7)*100 if pnl7 else 0
results['7_month_end'] = {'events': len(pnl7), 'WR': f'{wr7:.1f}%', 'avg_ret_pct': f'{avg7:+.4f}%'}
print(f'  Events={len(pnl7)} WR={wr7:.1f}% Avg={avg7:+.4f}%')

# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('P0 SUMMARY')
print('='*60)
for k, v in sorted(results.items()):
    tag = 'LIVE' if float(v['WR'].rstrip('%')) > 58 and v['events'] > 30 else 'DEAD'
    ev_cnt = v['events']; wr_str = v['WR']; avg_str = v['avg_ret_pct']
    print(f'  {k:35s} | ev={ev_cnt:>4d} WR={wr_str:>6s} avg={avg_str:>8s}  {tag}')

out = Path(__file__).parent / 'round3_p0_results.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to {out}')