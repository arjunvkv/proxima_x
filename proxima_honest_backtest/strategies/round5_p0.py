"""P0 analysis: Round 5 — 7 new angles."""
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
            d.sort_values('time', inplace=True); d.reset_index(drop=True, inplace=True)
            raw[p] = d
    return raw

def align(raw):
    pieces = []
    for pair, df in raw.items():
        sub = df.set_index('time')[['close','open','high','low','tick_volume']]
        sub.columns = [pair, f'{pair}_open', f'{pair}_high', f'{pair}_low', f'{pair}_volume']
        pieces.append(sub)
    a = pd.concat(pieces, axis=1, sort=True)
    a.sort_index(inplace=True); a.ffill(inplace=True); a.bfill(inplace=True)
    a.reset_index(inplace=True)
    a['hour'] = a['time'].dt.hour; a['minute'] = a['time'].dt.minute
    a['ymd'] = a['time'].dt.strftime('%Y%m%d'); a['dow'] = a['time'].dt.dayofweek
    return a.to_dict('records'), a

print('Loading...'); t0 = time.time()
raw = load(); records, df = align(raw); n = len(records)
print(f'Loaded {n} bars in {time.time()-t0:.1f}s')
results = {}

# ─── 1) FRIDAY->MONDAY GAP RIDE ──────────────────────────────
print('\n' + '='*60)
print('1) FRIDAY->MONDAY GAP RIDE')
print('='*60)
pnl1, w1, l1 = [], 0, 0
for i, r in enumerate(records):
    if r['dow'] != 0 or r['hour'] != 0 or r['minute'] != 5:
        continue
    rets = []
    for p in ALL_PAIRS:
        c = r.get(p)
        if c is None or np.isnan(c):
            continue
        pi = i - 1
        while pi >= 0 and records[pi]['dow'] != 4:
            pi -= 1
        if pi < 0:
            continue
        pv = records[pi].get(p)
        if pv is None or np.isnan(pv) or pv <= 0:
            continue
        rets.append((p, (c - pv) / pv))
    if len(rets) < 8:
        continue
    rets.sort(key=lambda x: x[1], reverse=True)
    for pair, _ in rets[:3]:
        fi = i + 12
        if fi >= n:
            continue
        eo = r.get(f'{pair}_open', r.get(pair))
        fc = records[fi].get(pair)
        if any(v is None or np.isnan(v) for v in (eo, fc)):
            continue
        ret = (fc - eo) / eo
        pnl1.append(ret); w1 += ret > 0; l1 += ret < 0
wr1 = w1/(w1+l1)*100 if w1+l1 else 0
avg1 = np.mean(pnl1)*100 if pnl1 else 0
results['1_fri_mon_gap'] = {'events': len(pnl1), 'WR': f'{wr1:.1f}%', 'avg_ret_pct': f'{avg1:+.4f}%'}
print(f'  Events={len(pnl1)} WR={wr1:.1f}% Avg={avg1:+.4f}%')

# ─── 2) CONSECUTIVE BAR STREAK REVERSAL ──────────────────────
print('\n' + '='*60)
print('2) CONSECUTIVE BAR STREAK (5+ bars) FADE')
print('='*60)
all_sr = []
for pair in ALL_PAIRS:
    df = raw.get(pair)
    if df is None or len(df) < 200:
        continue
    c = df['close'].values
    o = df['open'].values
    directions = np.sign(np.diff(c))
    stre = np.zeros(len(directions))
    for j in range(1, len(directions)):
        if directions[j] == directions[j-1]:
            stre[j] = stre[j-1] + 1
        else:
            stre[j] = 1
    pnl = []
    for j in range(6, len(directions)-6):
        if stre[j] < 5:
            continue
        total_move = abs(c[j] - c[j-5]) / c[j-5]
        if total_move < 0.0003:
            continue
        entry = o[j+1]
        exit_p = c[j+6]
        if directions[j] > 0:
            ret = (entry - exit_p) / entry
        else:
            ret = (exit_p - entry) / entry
        pnl.append(ret)
    if pnl:
        wrp = sum(1 for x in pnl if x>0)/len(pnl)*100
        avgp = np.mean(pnl)*100
        print(f'  {pair}: ev={len(pnl)} WR={wrp:.1f}% avg={avgp:+.4f}%')
        all_sr.extend(pnl)
wr2 = sum(1 for x in all_sr if x>0)/len(all_sr)*100 if all_sr else 0
avg2 = np.mean(all_sr)*100 if all_sr else 0
results['2_streak_fade'] = {'events': len(all_sr), 'WR': f'{wr2:.1f}%', 'avg_ret_pct': f'{avg2:+.4f}%'}
print(f'  TOTAL: ev={len(all_sr)} WR={wr2:.1f}% avg={avg2:+.4f}%')

# ─── 3) CROSS-PAIR AUD BASKET CASCADE ────────────────────────
print('\n' + '='*60)
print('3) CROSS-PAIR AUD BASKET CASCADE')
print('='*60)
basket = ['EURNZD', 'GBPNZD', 'GBPAUD', 'GBPCAD']
leaders = ['EURAUD', 'GBPAUD']
pnl3, w3, l3 = [], 0, 0
for i, r in enumerate(records):
    pi = i - 3
    if pi < 0:
        continue
    for leader in leaders:
        lc = r.get(leader)
        lpv = records[pi].get(leader)
        if any(x is None or np.isnan(x) for x in (lc, lpv)):
            continue
        lret = (lc - lpv) / lpv
        if abs(lret) < 0.0008:
            continue
        dir_sig = 1 if lret > 0 else -1
        for target in basket:
            if target == leader:
                continue
            tc = r.get(target)
            tpv = records[pi].get(target)
            if any(x is None or np.isnan(x) for x in (tc, tpv)):
                continue
            tret = (tc - tpv) / tpv
            if abs(tret) > 0.3 * abs(lret):
                continue
            fi = i + 6
            if fi >= n:
                continue
            fc = records[fi].get(target)
            if fc is None or np.isnan(fc):
                continue
            ret = (fc - tc) / tc * dir_sig
            pnl3.append(ret); w3 += ret > 0; l3 += ret < 0
wr3 = w3/(w3+l3)*100 if w3+l3 else 0
avg3 = np.mean(pnl3)*100 if pnl3 else 0
results['3_cross_cascade'] = {'events': len(pnl3), 'WR': f'{wr3:.1f}%', 'avg_ret_pct': f'{avg3:+.4f}%'}
print(f'  Events={len(pnl3)} WR={wr3:.1f}% Avg={avg3:+.4f}%')

# ─── 4) ROLLING HOURLY ORB ──────────────────────────────────
print('\n' + '='*60)
print('4) ROLLING HOURLY ORB')
print('='*60)
pnl4, w4, l4 = [], 0, 0
curr_hour = None; orb_hi = {}; orb_lo = {}; orb_active = False
for i, r in enumerate(records):
    h = r['hour']
    m5 = r['minute']
    if curr_hour != h:
        curr_hour = h; orb_hi = {}; orb_lo = {}; orb_active = True
    if orb_active and m5 < 10:
        for p in ALL_PAIRS:
            hi = r.get(f'{p}_high', r.get(p))
            lo = r.get(f'{p}_low', r.get(p))
            if hi is not None and not np.isnan(hi):
                orb_hi[p] = max(orb_hi.get(p, 0), hi)
            if lo is not None and not np.isnan(lo):
                orb_lo[p] = min(orb_lo.get(p, float('inf')), lo)
    elif orb_active:
        for p in ALL_PAIRS:
            c = r.get(p)
            if c is None or np.isnan(c):
                continue
            hi = orb_hi.get(p)
            lo = orb_lo.get(p)
            if hi is None or lo is None:
                continue
            entry = r.get(f'{p}_open', c)
            if entry is None or np.isnan(entry):
                continue
            fi = i + 6
            if fi >= n:
                continue
            fc = records[fi].get(p)
            if fc is None or np.isnan(fc):
                continue
            if c > hi:
                ret = (fc - entry) / entry
            elif c < lo:
                ret = (entry - fc) / entry
            else:
                continue
            pnl4.append(ret); w4 += ret > 0; l4 += ret < 0
        orb_active = False
wr4 = w4/(w4+l4)*100 if w4+l4 else 0
avg4 = np.mean(pnl4)*100 if pnl4 else 0
results['4_hourly_orb'] = {'events': len(pnl4), 'WR': f'{wr4:.1f}%', 'avg_ret_pct': f'{avg4:+.4f}%'}
print(f'  Events={len(pnl4)} WR={wr4:.1f}% Avg={avg4:+.4f}%')

# ─── 5) INTRADAY SEASONALITY ──────────────────────────────
print('\n' + '='*60)
print('5) INTRADAY SEASONALITY (best hour per pair)')
print('='*60)
pair_hour_returns = defaultdict(lambda: defaultdict(list))
for i, r in enumerate(records):
    h = r['hour']; m5 = r['minute']
    if m5 != 0: continue
    fi = i + 12
    if fi >= n: continue
    for p in ALL_PAIRS:
        eo = r.get(f'{p}_open', r.get(p))
        fc = records[fi].get(p)
        if any(x is None or np.isnan(x) for x in (eo, fc)):
            continue
        ret = (fc - eo) / eo
        pair_hour_returns[p][h].append(ret)

seasonal_pnl = []; sw = 0; sl = 0
for pair in ALL_PAIRS:
    by_hr = pair_hour_returns[pair]
    scored = []
    for h, rets in by_hr.items():
        if len(rets) < 30: continue
        wr = sum(1 for x in rets if x>0)/len(rets)
        avg = np.mean(rets)
        scored.append((h, wr, avg, len(rets)))
    scored.sort(key=lambda x: x[1]*abs(x[2]), reverse=True)
    if not scored: continue
    best_h, best_wr, best_avg, n_best = scored[0]
    best_rets = by_hr[best_h]
    print(f'  {pair}: h={best_h:02d}:00 WR={best_wr:.1%} avg={best_avg*100:+.4f}% n={n_best}')
    seasonal_pnl.extend(best_rets)
    sw += sum(1 for x in best_rets if x>0)
    sl += sum(1 for x in best_rets if x<0)
wr5 = sw/(sw+sl)*100 if sw+sl else 0
avg5 = np.mean(seasonal_pnl)*100 if seasonal_pnl else 0
results['5_seasonality'] = {'events': len(seasonal_pnl), 'WR': f'{wr5:.1f}%', 'avg_ret_pct': f'{avg5:+.4f}%'}
print(f'  TOTAL: ev={len(seasonal_pnl)} WR={wr5:.1f}% avg={avg5:+.4f}%')

# ─── 6) RANGE CONTRACTION BREAKOUT ──────────────────────────
print('\n' + '='*60)
print('6) RANGE CONTRACTION BREAKOUT')
print('='*60)
all_rc = []
for pair in ALL_PAIRS:
    df = raw.get(pair)
    if df is None or len(df) < 200: continue
    c = df['close'].values; o = df['open'].values
    h = df['high'].values; l = df['low'].values
    rng = h - l
    range_ma = np.convolve(rng, np.ones(20)/20, 'valid')
    c_valid = c[19:]; o_valid = o[19:]
    pnl = []
    for j in range(40, len(range_ma)-6):
        if range_ma[j] > 0.5 * range_ma[j-20]: continue
        hi20 = max(c_valid[j-20:j])
        lo20 = min(c_valid[j-20:j])
        entry = o_valid[j]
        if entry is None or np.isnan(entry): continue
        fi = j + 6
        if fi >= len(c_valid): continue
        exit_p = c_valid[fi]
        if c_valid[j] > hi20:
            ret = (exit_p - entry) / entry
            pnl.append(ret)
        elif c_valid[j] < lo20:
            ret = (entry - exit_p) / entry
            pnl.append(ret)
    if pnl:
        wrp = sum(1 for x in pnl if x>0)/len(pnl)*100
        avgp = np.mean(pnl)*100
        print(f'  {pair}: ev={len(pnl)} WR={wrp:.1f}% avg={avgp:+.4f}%')
        all_rc.extend(pnl)
wr6 = sum(1 for x in all_rc if x>0)/len(all_rc)*100 if all_rc else 0
avg6 = np.mean(all_rc)*100 if all_rc else 0
results['6_range_contraction'] = {'events': len(all_rc), 'WR': f'{wr6:.1f}%', 'avg_ret_pct': f'{avg6:+.4f}%'}
print(f'  TOTAL: ev={len(all_rc)} WR={wr6:.1f}% avg={avg6:+.4f}%')

# ─── 7) H1 TREND ALIGNMENT ──────────────────────────────────
print('\n' + '='*60)
print('7) H1 TREND ALIGNMENT')
print('='*60)
pnl7, w7, l7 = [], 0, 0
for i, r in enumerate(records):
    h = r['hour']; m5 = r['minute']
    if m5 != 0: continue
    h1_dir = {}
    for p in ALL_PAIRS:
        pi = i - 12
        if pi < 0: continue
        pc = records[pi].get(p); cc = r.get(p)
        if any(x is None or np.isnan(x) for x in (pc, cc)): continue
        if pc <= 0: continue
        ret = (cc - pc) / pc
        if ret > 0.001:
            h1_dir[p] = 1
        elif ret < -0.001:
            h1_dir[p] = -1
        else:
            h1_dir[p] = 0
    if sum(1 for v in h1_dir.values() if v != 0) < 8: continue
    rets = []
    for p in ALL_PAIRS:
        h1d = h1_dir.get(p, 0)
        if h1d == 0: continue
        pi = i - 6
        if pi < 0: continue
        pv = records[pi].get(p); cc = r.get(p)
        if any(x is None or np.isnan(x) for x in (pv, cc)): continue
        if pv <= 0: continue
        m5_ret = (cc - pv) / pv
        if m5_ret * h1d < 0: continue
        rets.append((p, m5_ret))
    if len(rets) < 5: continue
    rets.sort(key=lambda x: x[1], reverse=True)
    for pair, _ in rets[:3]:
        fi = i + 12
        if fi >= n: continue
        eo = r.get(f'{pair}_open', r.get(pair))
        fc = records[fi].get(pair)
        if any(x is None or np.isnan(x) for x in (eo, fc)): continue
        ret = (fc - eo) / eo
        pnl7.append(ret); w7 += ret > 0; l7 += ret < 0
wr7 = w7/(w7+l7)*100 if w7+l7 else 0
avg7 = np.mean(pnl7)*100 if pnl7 else 0
results['7_h1_alignment'] = {'events': len(pnl7), 'WR': f'{wr7:.1f}%', 'avg_ret_pct': f'{avg7:+.4f}%'}
print(f'  Events={len(pnl7)} WR={wr7:.1f}% Avg={avg7:+.4f}%')

# ─── SUMMARY ──────────────────────────────────────────────
print('\n' + '='*60)
print('P0 SUMMARY')
print('='*60)
for k, v in sorted(results.items()):
    ev = v['events']; wr = float(v['WR'].rstrip('%')); ar = v['avg_ret_pct']
    tag = 'LIVE' if wr > 58 and ev > 30 else 'DEAD'
    print(f'  {k:30s} | ev={ev:>5d} WR={v["WR"]:>6s} avg={ar:>8s}  {tag}')

out = Path(__file__).parent / 'round5_p0_results.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nDone in {time.time()-t0:.0f}s')
