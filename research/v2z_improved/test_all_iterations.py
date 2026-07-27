"""
V2+z 7 Iteration Test Suite — relative comparison.
Fixes: enter at bar[i+1] open, stop-before-trail check order, spread modeling.
"""
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime
import sys

SLIPPAGE_PIPS = 1.0
COMMISSION_PER_LOT = 5.0
BASE_LOT = 0.75
MAX_HOLD_BARS = 54
Z_WINDOW = 50
ATR_PERIOD = 20

FOCUS_PAIRS = ["EURAUD", "GBPAUD", "AUDNZD", "EURNZD", "GBPCAD", "GBPNZD"]
FORWARD = (datetime(2026, 6, 8), datetime(2026, 7, 25))

def get_data(pair, start, end):
    r = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, start, end)
    if r is None or len(r) == 0: return None
    bars = pd.DataFrame(r)
    bars['time'] = pd.to_datetime(bars['time'], unit='s')
    return bars

def spread(pair):
    if 'JPY' in pair: return 3.0
    if pair == 'EURAUD': return 3.5
    if pair == 'GBPAUD': return 4.0
    if pair == 'EURNZD': return 4.5
    if pair == 'GBPNZD': return 5.0
    if pair == 'AUDNZD': return 3.0
    if pair == 'GBPCAD': return 4.0
    return 2.0

def run(bars, cfg):
    zt = cfg.get('z', 3.5)
    sa = cfg.get('stop_a', 3.0)
    ta = cfg.get('trig_a', 1.0)
    ga = cfg.get('gap_a', 0.05)
    va = cfg.get('vol_adj', False)
    rf = cfg.get('reg_filt', False)
    te = cfg.get('time_exit', False)
    eb = cfg.get('exit_bars', 30)
    pair = cfg.get('pair', 'EURUSD')
    no_trail = cfg.get('no_trail', False)

    sprd = spread(pair)  # spread in pips
    pip = 0.0001 if 'JPY' not in pair else 0.01
    pip_usd = {"EURAUD":6.70,"GBPAUD":6.10,"AUDNZD":5.60,"EURNZD":6.10,"GBPCAD":7.50,"GBPNZD":5.60}.get(pair, 10.0)

    trades = []
    pos = 0; entry_p = 0.0; best_p = 0.0; stop_p = 0.0; held = 0
    z_entry = 0.0

    atr = bars['high'].sub(bars['low']).rolling(ATR_PERIOD).mean()
    rets = bars['close'].diff()
    rmu = rets.rolling(Z_WINDOW).mean()
    rsd = rets.rolling(Z_WINDOW).std(ddof=1)
    zs = (rets - rmu) / rsd

    # Regime filter: ADX
    if rf:
        tr = pd.concat([bars['high']-bars['low'],
                        (bars['high']-bars['close'].shift()).abs(),
                        (bars['low']-bars['close'].shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        up = rets.clip(lower=0); dn = (-rets).clip(lower=0)
        pdm = up.rolling(14).mean() / atr14 * 100
        ndm = dn.rolling(14).mean() / atr14 * 100
        adx = (abs(pdm-ndm)/(pdm+ndm+1e-10)*100).rolling(14).mean()

    if va:
        atr_med = atr.rolling(100).median()

    start_i = max(Z_WINDOW, ATR_PERIOD) + 5
    n = len(bars)
    i = start_i

    while i < n - 1:
        hour = bars.iloc[i]['time'].hour
        if hour >= 7:
            i += 1; continue

        if pos != 0:
            held += 1
            if held > MAX_HOLD_BARS:
                ep = bars.iloc[i]['close']
                ep_adj = ep - sprd/2*pip if pos == 1 else ep + sprd/2*pip
                raw = (ep_adj - entry_p) * BASE_LOT * 100000 if pos == 1 else (entry_p - ep_adj) * BASE_LOT * 100000
                trades.append(dict(raw=raw,c=BASE_LOT*COMMISSION_PER_LOT,n=raw-BASE_LOT*COMMISSION_PER_LOT,r='exp',h=held,ze=z_entry))
                pos = 0; i += 1; continue

            atr_v = atr.iloc[i]
            if pd.notna(atr_v) and atr_v > 0:
                hi = bars.iloc[i]['high']; lo = bars.iloc[i]['low']
                if no_trail:
                    # No trailing: just check initial stop
                    if (pos == 1 and lo <= stop_p) or (pos == -1 and hi >= stop_p):
                        raw = (stop_p - entry_p) * BASE_LOT * 100000 if pos == 1 else (entry_p - stop_p) * BASE_LOT * 100000
                        trades.append(dict(raw=raw,c=BASE_LOT*COMMISSION_PER_LOT,n=raw-BASE_LOT*COMMISSION_PER_LOT,r='stp',h=held,ze=z_entry))
                        pos = 0; i += 1; continue
                else:
                    tg = ta * atr_v; gp = ga * atr_v
                    if pos == 1:
                        if lo <= stop_p:
                            raw = (stop_p - entry_p) * BASE_LOT * 100000
                            trades.append(dict(raw=raw,c=BASE_LOT*COMMISSION_PER_LOT,n=raw-BASE_LOT*COMMISSION_PER_LOT,r='stp',h=held,ze=z_entry))
                            pos = 0; i += 1; continue
                        if hi > best_p:
                            best_p = hi
                            if best_p - entry_p > tg:
                                ns = best_p - gp
                                if ns > stop_p: stop_p = ns
                    else:
                        if hi >= stop_p:
                            raw = (entry_p - stop_p) * BASE_LOT * 100000
                            trades.append(dict(raw=raw,c=BASE_LOT*COMMISSION_PER_LOT,n=raw-BASE_LOT*COMMISSION_PER_LOT,r='stp',h=held,ze=z_entry))
                            pos = 0; i += 1; continue
                        if lo < best_p:
                            best_p = lo
                            if entry_p - best_p > tg:
                                ns = best_p + gp
                                if ns < stop_p: stop_p = ns

                if te and held >= eb:
                    ep = bars.iloc[i]['close']
                    ep_adj = ep - sprd/2*pip if pos == 1 else ep + sprd/2*pip
                    raw = (ep_adj - entry_p) * BASE_LOT * 100000 if pos == 1 else (entry_p - ep_adj) * BASE_LOT * 100000
                    trades.append(dict(raw=raw,c=BASE_LOT*COMMISSION_PER_LOT,n=raw-BASE_LOT*COMMISSION_PER_LOT,r='tex',h=held,ze=z_entry))
                    pos = 0; i += 1; continue

            i += 1; continue

        z_v = zs.iloc[i]; a_v = atr.iloc[i]
        if pd.isna(z_v) or pd.isna(a_v) or a_v <= 0:
            i += 1; continue

        thresh = zt
        if va and atr_med.iloc[i] > 1e-10:
            ratio = a_v / atr_med.iloc[i]
            thresh = zt * max(0.3, ratio)

        if abs(z_v) < thresh:
            i += 1; continue

        if rf and not pd.isna(adx.iloc[i]) and adx.iloc[i] > 25:
            i += 1; continue

        direction = -1 if z_v > 0 else 1
        # Entry at next bar open, adjusted for spread
        next_open = bars.iloc[i + 1]['open']
        entry_p = next_open + sprd/2*pip if direction == 1 else next_open - sprd/2*pip
        entry_p += (SLIPPAGE_PIPS * pip / 10) * direction * -1

        s = sa * a_v
        stop_p = entry_p - s if direction == 1 else entry_p + s
        best_p = entry_p; held = 0
        z_entry = z_v; pos = direction
        i += 1

    return trades

def summary(trades):
    if not trades: return {'n':0,'net':0,'wr':0,'pf':0,'agw':0,'agl':0,'gross':0,'comm':0,'payoff':0}
    df = pd.DataFrame(trades)
    wins = df[df['n'] > 0]; losses = df[df['n'] < 0]
    n = len(df); net = df['n'].sum(); gross = df['raw'].sum(); comm = df['c'].sum()
    wr = len(wins)/n*100 if n else 0
    pf = abs(wins['raw'].sum()/losses['raw'].sum()) if len(losses) and losses['raw'].sum() != 0 else 99
    agw = wins['n'].mean() if len(wins) else 0
    agl = losses['n'].mean() if len(losses) else 0
    payoff = abs(agw/agl) if agl != 0 else 0
    return dict(n=n,net=net,wr=wr,pf=pf,gross=gross,comm=comm,payoff=payoff,agw=agw,agl=agl)

configs = {
    'baseline': {},
    'vol_adj': {'vol_adj': True},
    'reg_filt': {'reg_filt': True},
    'no_trail': {'no_trail': True},
    'time_exit30': {'time_exit': True, 'exit_bars': 30},
    'time_exit54': {'time_exit': True, 'exit_bars': 54},
    'z_5': {'z': 5.0},
    'stop_wider': {'stop_a': 5.0},
    'stop_wider_trail': {'stop_a': 5.0, 'trig_a': 2.0},
    'vol+reg': {'vol_adj': True, 'reg_filt': True},
    'vol+reg+zt4': {'vol_adj': True, 'reg_filt': True, 'z': 4.0},
    'no_trail_stop5': {'no_trail': True, 'stop_a': 5.0},
}

if __name__ == '__main__':
    if not mt5.initialize():
        print("MT5 init failed"); sys.exit(1)

    all_results = []
    for pair in FOCUS_PAIRS:
        print(f"\n=== {pair} ===")
        bars = get_data(pair, FORWARD[0], FORWARD[1])
        if bars is None or len(bars) < 5000:
            print(f"  SKIP: {len(bars) if bars is not None else 0} bars")
            continue
        print(f"  {len(bars):,} bars ({len(bars)/60/24:.1f}d)")

        for cname, cparams in configs.items():
            cparams['pair'] = pair
            trades = run(bars.copy(), cparams)
            s = summary(trades)
            all_results.append(dict(pair=pair, cfg=cname, **s))
            print(f"  {cname:15s}: n={s['n']:3d} net=${s['net']:>+7.0f} "
                  f"wr={s['wr']:5.1f}% pf={s['pf']:.2f} comm=${s['comm']:>4.0f} "
                  f"payoff={s['payoff']:.2f} agw=${s['agw']:>+.0f} agl=${s['agl']:>+.0f}")

    mt5.shutdown()

    # Summary table
    print("\n" + "=" * 120)
    print("NET PnL AFTER COMMISSION: Config × Pair")
    print("=" * 120)
    cfgs = list(configs.keys())
    hdr = f"{'Config':<18}" + "".join(f"{p:<10}" for p in FOCUS_PAIRS) + f"{'TOTAL':<10}"
    print(hdr)
    print("-" * 120)

    totals = {}
    for cname in cfgs:
        row = f"{cname:<18}"
        ct = 0
        for pair in FOCUS_PAIRS:
            v = 0
            for r in all_results: 
                if r['pair'] == pair and r['cfg'] == cname: v = r['net']
            row += f"${v:>+7.0f} "
            ct += v
        totals[cname] = ct
        marker = " ✓" if ct > 0 else " ✗"
        row += f"${ct:>+7.0f}{marker}"
        print(row)

    print("-" * 120)
    best = max(totals, key=totals.get)
    print(f"BEST: {best} = ${totals[best]:+.0f}")
    worst = min(totals, key=totals.get)
    print(f"WORST: {worst} = ${totals[worst]:+.0f}")
    base_net = totals['baseline']
    for cname in cfgs:
        if cname == 'baseline': continue
        delta = totals[cname] - base_net
        print(f"  {cname:15s}: Δ={delta:+.0f} vs baseline ({'+' if delta>0 else ''}{delta/base_net*100:.0f}%)")
    print("=" * 120)
