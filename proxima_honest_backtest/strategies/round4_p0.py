#!/usr/bin/env python3
"""P0 analysis: Round 4 — 7 new anomalies."""
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
    aligned['dow'] = aligned['time'].dt.dayofweek
    return aligned.to_dict('records'), aligned

print('Loading...')
t0 = time.time()
raw = load()
records, aligned_df = make_aligned(raw)
print(f'Loaded {len(records)} bars in {time.time()-t0:.1f}s')
n = len(records)

results = {}

# ──────────────────────────────────────────────
# 1) EURJPY TRIANGLE CATCH-UP
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('1) EURJPY TRIANGLE CATCH-UP')
print('='*60)
pnl1, w1, l1 = [], 0, 0
for i, r in enumerate(records):
    eurusd = r.get('EURUSD')
    usdjpy = r.get('USDJPY')
    eurjpy = r.get('EURJPY')
    if any(x is None or np.isnan(x) for x in (eurusd, usdjpy, eurjpy)): continue
    pi = i - 3
    if pi < 0: continue
    pv_es = records[pi].get('EURUSD')
    pv_uj = records[pi].get('USDJPY')
    pv_ej = records[pi].get('EURJPY')
    if any(x is None or np.isnan(x) for x in (pv_es, pv_uj, pv_ej)): continue
    ret_es = (eurusd - pv_es) / pv_es
    ret_uj = (usdjpy - pv_uj) / pv_uj
    ret_ej = (eurjpy - pv_ej) / pv_ej
    expected_ej = (1 + ret_es) * (1 + ret_uj) - 1
    if ret_es * ret_uj < 0: continue  # not aligned
    # EURJPY lags? |actual| < 0.5 * |expected|
    if abs(ret_ej) > 0.5 * abs(expected_ej): continue
    # trade catch-up
    direction = 1 if expected_ej > 0 else -1
    fi = i + 6
    if fi >= n: continue
    fc = records[fi].get('EURJPY')
    if fc is None or np.isnan(fc): continue
    actual_ret = (fc - eurjpy) / eurjpy * direction
    pnl1.append(actual_ret); w1 += actual_ret > 0; l1 += actual_ret < 0
wr1 = w1/(w1+l1)*100 if w1+l1 else 0
avg1 = np.mean(pnl1)*100 if pnl1 else 0
results['1_eurjpy_triangle'] = {'events': len(pnl1), 'WR': f'{wr1:.1f}%', 'avg_ret_pct': f'{avg1:+.4f}%'}
print(f'  Events={len(pnl1)} WR={wr1:.1f}% Avg={avg1:+.4f}%')

# ──────────────────────────────────────────────
# 2) VOLUME EXHAUSTION FADE
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('2) VOLUME EXHAUSTION FADE')
print('='*60)
all_ve = []
for pair in ALL_PAIRS:
    df = raw.get(pair)
    if df is None or len(df) < 200: continue
    c = df['close'].values; o = df['open'].values; v = df['tick_volume'].values.astype(float)
    hl = df['high'].values - df['low'].values
    tr = np.maximum(hl[1:], np.maximum(np.abs(df['high'].values[1:]-c[:-1]), np.abs(df['low'].values[1:]-c[:-1])))
    atr20 = np.convolve(tr, np.ones(20)/20, 'valid')
    vol_ma20 = np.convolve(v, np.ones(20)/20, 'valid')
    min_l = min(len(atr20), len(vol_ma20), len(c)-1)
    ve_pnl = []; w2_arr = []; l2_arr = []
    for j in range(20, min_l-6):
        if v[j] < 3 * vol_ma20[j-20]: continue
        bar_range = abs(c[j] - o[j])
        if bar_range < 1.5 * atr20[j-20]: continue
        direction = 1 if c[j] < o[j] else -1  # fade: if up bar -> SHORT
        entry = o[j+1]
        exit_p = c[j+6]
        ret = (exit_p - entry) / entry * direction
        ve_pnl.append(ret)
    if ve_pnl:
        wr2p = sum(1 for x in ve_pnl if x>0)/len(ve_pnl)*100
        avg2p = np.mean(ve_pnl)*100
        print(f'  {pair}: ev={len(ve_pnl)} WR={wr2p:.1f}% avg={avg2p:+.4f}%')
        all_ve.extend(ve_pnl)
wr2 = sum(1 for x in all_ve if x>0)/len(all_ve)*100 if all_ve else 0
avg2 = np.mean(all_ve)*100 if all_ve else 0
results['2_volume_exhaustion'] = {'events': len(all_ve), 'WR': f'{wr2:.1f}%', 'avg_ret_pct': f'{avg2:+.4f}%'}
print(f'  TOTAL: ev={len(all_ve)} WR={wr2:.1f}% avg={avg2:+.4f}%')

# ──────────────────────────────────────────────
# 3) EUROPEAN OPEN 07:00 MOMENTUM
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('3) EUROPEAN OPEN (07:00 UTC) MOMENTUM')
print('='*60)
pnl3, w3, l3 = [], 0, 0
for i, r in enumerate(records):
    if r['hour'] != 7 or r['minute'] != 0: continue
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
        pnl3.append(ret); w3 += ret > 0; l3 += ret < 0
wr3 = w3/(w3+l3)*100 if w3+l3 else 0
avg3 = np.mean(pnl3)*100 if pnl3 else 0
results['3_european_open'] = {'events': len(pnl3), 'WR': f'{wr3:.1f}%', 'avg_ret_pct': f'{avg3:+.4f}%'}
print(f'  Events={len(pnl3)} WR={wr3:.1f}% Avg={avg3:+.4f}%')

# ──────────────────────────────────────────────
# 4) WM FIX (15:00 UTC) REVERSION
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('4) WM FIX (15:00 UTC) REVERSION')
print('='*60)
pnl4, w4, l4 = [], 0, 0
for i, r in enumerate(records):
    if r['hour'] != 15 or r['minute'] != 0: continue
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
    # Fade top movers - pair with largest abs move
    rets.sort(key=lambda x: abs(x[1]), reverse=True)
    for pair, ret in rets[:5]:
        fi = i + 6
        if fi >= n: continue
        eo = r.get(f'{pair}_open', r.get(pair))
        fc = records[fi].get(pair)
        if any(v is None or np.isnan(v) for v in (eo, fc)): continue
        if ret > 0:
            ret4 = (eo - fc) / eo  # SHORT if moved up
        else:
            ret4 = (fc - eo) / eo  # LONG if moved down
        pnl4.append(ret4); w4 += ret4 > 0; l4 += ret4 < 0
wr4 = w4/(w4+l4)*100 if w4+l4 else 0
avg4 = np.mean(pnl4)*100 if pnl4 else 0
results['4_wm_fix'] = {'events': len(pnl4), 'WR': f'{wr4:.1f}%', 'avg_ret_pct': f'{avg4:+.4f}%'}
print(f'  Events={len(pnl4)} WR={wr4:.1f}% Avg={avg4:+.4f}%')

# ──────────────────────────────────────────────
# 5) TUESDAY REVERSAL
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('5) TUESDAY REVERSAL')
print('='*60)
# Group bars by day
day_bars = defaultdict(list)
for i, r in enumerate(records):
    day_bars[r['ymd']].append(i)

pnl5, w5, l5 = [], 0, 0
for ymd, idxs in day_bars.items():
    first = idxs[0]
    fr = records[first]
    if fr['dow'] != 1: continue  # Tuesday = 1
    monday_ymd = str(int(ymd) - 1)
    mon_idxs = day_bars.get(monday_ymd, [])
    if len(mon_idxs) < 12: continue
    mon_last_idx = mon_idxs[-1]
    mon_first_idx = mon_idxs[0]
    # Monday last 60 min best performers
    rets = []
    for p in ALL_PAIRS:
        start = records[mon_last_idx - 12].get(p)
        end = records[mon_last_idx].get(p)
        if any(x is None or np.isnan(x) for x in (start, end)): continue
        if start <= 0: continue
        rets.append((p, (end - start) / start))
    if len(rets) < 8: continue
    rets.sort(key=lambda x: x[1], reverse=True)
    # Tuesday: reverse top 3 Monday performers (fade)
    for pair, _ in rets[:3]:
        entry_idx = first
        if entry_idx + 12 >= n: continue
        eo = records[entry_idx].get(f'{pair}_open', records[entry_idx].get(pair))
        fc = records[entry_idx + 12].get(pair)
        if any(v is None or np.isnan(v) for v in (eo, fc)): continue
        ret = (eo - fc) / eo  # SHORT Monday's best
        pnl5.append(ret); w5 += ret > 0; l5 += ret < 0
wr5 = w5/(w5+l5)*100 if w5+l5 else 0
avg5 = np.mean(pnl5)*100 if pnl5 else 0
results['5_tuesday_reversal'] = {'events': len(pnl5), 'WR': f'{wr5:.1f}%', 'avg_ret_pct': f'{avg5:+.4f}%'}
print(f'  Events={len(pnl5)} WR={wr5:.1f}% Avg={avg5:+.4f}%')

# ──────────────────────────────────────────────
# 6) FRIDAY POSITION SQUARING (17:00 UTC)
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('6) FRIDAY POSITION SQUARING (17:00 UTC)')
print('='*60)
pnl6, w6, l6 = [], 0, 0
for i, r in enumerate(records):
    if r['dow'] != 4: continue  # Friday
    if r['hour'] != 17 or r['minute'] != 0: continue
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
        fi = i + 12
        if fi >= n: continue
        eo = r.get(f'{pair}_open', r.get(pair))
        fc = records[fi].get(pair)
        if any(v is None or np.isnan(v) for v in (eo, fc)): continue
        ret = (eo - fc) / eo  # SHORT Friday's best
        pnl6.append(ret); w6 += ret > 0; l6 += ret < 0
wr6 = w6/(w6+l6)*100 if w6+l6 else 0
avg6 = np.mean(pnl6)*100 if pnl6 else 0
results['6_friday_squaring'] = {'events': len(pnl6), 'WR': f'{wr6:.1f}%', 'avg_ret_pct': f'{avg6:+.4f}%'}
print(f'  Events={len(pnl6)} WR={wr6:.1f}% Avg={avg6:+.4f}%')

# ──────────────────────────────────────────────
# 7) BOLLINGER BAND WIDTH EXPANSION
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('7) BOLLINGER BAND WIDTH EXPANSION')
print('='*60)
all_bb = []
for pair in ALL_PAIRS:
    df = raw.get(pair)
    if df is None or len(df) < 200: continue
    c = df['close'].values; o = df['open'].values
    # Rolling std
    bbw = np.full(len(c), np.nan)
    for j in range(20, len(c)):
        s = np.std(c[j-19:j+1])
        bbw[j] = s / c[j]  # normalized width
    bbw_pct = np.full(len(c), np.nan)
    for j in range(40, len(c)):
        window = bbw[j-39:j+1]
        pct = np.sum(bbw[j] > window) / len(window)
        bbw_pct[j] = pct
    bb_pnl = []
    for j in range(41, len(c)-6):
        if bbw_pct[j] is None or np.isnan(bbw_pct[j]): continue
        if bbw_pct[j] > 0.75: continue  # already wide
        if bbw_pct[j] < 0.15: continue  # already compressed
        # about to expand from compressed
        fut = c[j+6]
        entry = o[j]
        ret = (fut - entry) / entry
        bb_pnl.append(ret)
    if bb_pnl:
        wr7p = sum(1 for x in bb_pnl if x>0)/len(bb_pnl)*100
        avg7p = np.mean(bb_pnl)*100
        print(f'  {pair}: ev={len(bb_pnl)} WR={wr7p:.1f}% avg={avg7p:+.4f}%')
        all_bb.extend(bb_pnl)
wr7 = sum(1 for x in all_bb if x>0)/len(all_bb)*100 if all_bb else 0
avg7 = np.mean(all_bb)*100 if all_bb else 0
results['7_bb_width_expansion'] = {'events': len(all_bb), 'WR': f'{wr7:.1f}%', 'avg_ret_pct': f'{avg7:+.4f}%'}
print(f'  TOTAL: ev={len(all_bb)} WR={wr7:.1f}% avg={avg7:+.4f}%')

# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('P0 SUMMARY')
print('='*60)
for k, v in sorted(results.items()):
    tag = 'LIVE' if float(v['WR'].rstrip('%')) > 58 and v['events'] > 30 else 'DEAD'
    if k == '3_european_open' and float(v['WR'].rstrip('%')) > 57: tag = 'LIVE'
    if k == '4_wm_fix' and float(v['WR'].rstrip('%')) > 55: tag = 'BORDERLINE'
    ev_cnt = v['events']; wr_str = v['WR']; avg_str = v['avg_ret_pct']
    print(f'  {k:30s} | ev={ev_cnt:>5d} WR={wr_str:>6s} avg={avg_str:>8s}  {tag}')

out = Path(__file__).parent / 'round4_p0_results.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to {out}')
