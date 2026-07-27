"""Test best combos: stop_wider × z_higher + more extreme values."""
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime
import sys

COMMISSION_PER_LOT = 5.0; BASE_LOT = 0.75
MAX_HOLD_BARS = 54; Z_WINDOW = 50; ATR_PERIOD = 20
SLIPPAGE_PIPS = 1.0

FOCUS_PAIRS = ["EURAUD","GBPAUD","AUDNZD","EURNZD","GBPCAD","GBPNZD"]

def spread(pair):
    return {"EURAUD":3.5,"GBPAUD":4.0,"AUDNZD":3.0,"EURNZD":4.5,"GBPCAD":4.0,"GBPNZD":5.0}.get(pair,2.0)

def run(bars, cfg):
    zt = cfg.get('z',3.5); sa = cfg.get('stop_a',3.0); ta = cfg.get('trig_a',1.0); ga = cfg.get('gap_a',0.05)
    pair = cfg.get('pair','EURUSD'); pip = 0.0001 if 'JPY' not in pair else 0.01

    trades = []; pos = 0; entry_p = 0.0; best_p = 0.0; stop_p = 0.0; held = 0; ze = 0.0
    atr = bars['high'].sub(bars['low']).rolling(ATR_PERIOD).mean()
    rets = bars['close'].diff()
    zs = (rets - rets.rolling(Z_WINDOW).mean()) / rets.rolling(Z_WINDOW).std(ddof=1)

    start_i = max(Z_WINDOW,ATR_PERIOD) + 5; n = len(bars); i = start_i
    while i < n - 1:
        hour = bars.iloc[i]['time'].hour
        if hour >= 7: i += 1; continue
        if pos != 0:
            held += 1
            if held > MAX_HOLD_BARS:
                ep = bars.iloc[i]['close']; ea = ep - spread(pair)/2*pip if pos==1 else ep + spread(pair)/2*pip
                raw = (ea-entry_p)*BASE_LOT*100000 if pos==1 else (entry_p-ea)*BASE_LOT*100000
                trades.append(dict(raw=raw,c=BASE_LOT*COMMISSION_PER_LOT,n=raw-BASE_LOT*COMMISSION_PER_LOT,r='exp',h=held))
                pos = 0; i += 1; continue
            atr_v = atr.iloc[i]
            if pd.notna(atr_v) and atr_v > 0:
                hi = bars.iloc[i]['high']; lo = bars.iloc[i]['low']
                tg = ta*atr_v; gp = ga*atr_v
                if pos == 1:
                    if lo <= stop_p:
                        raw = (stop_p-entry_p)*BASE_LOT*100000
                        trades.append(dict(raw=raw,c=BASE_LOT*COMMISSION_PER_LOT,n=raw-BASE_LOT*COMMISSION_PER_LOT,r='stp',h=held))
                        pos = 0; i += 1; continue
                    if hi > best_p:
                        best_p = hi
                        if best_p-entry_p > tg:
                            ns = best_p - gp
                            if ns > stop_p: stop_p = ns
                else:
                    if hi >= stop_p:
                        raw = (entry_p-stop_p)*BASE_LOT*100000
                        trades.append(dict(raw=raw,c=BASE_LOT*COMMISSION_PER_LOT,n=raw-BASE_LOT*COMMISSION_PER_LOT,r='stp',h=held))
                        pos = 0; i += 1; continue
                    if lo < best_p:
                        best_p = lo
                        if entry_p-best_p > tg:
                            ns = best_p + gp
                            if ns < stop_p: stop_p = ns
            i += 1; continue

        z_v = zs.iloc[i]; a_v = atr.iloc[i]
        if pd.isna(z_v) or pd.isna(a_v) or a_v <= 0: i += 1; continue
        if abs(z_v) < zt: i += 1; continue

        direction = -1 if z_v > 0 else 1
        noi = bars.iloc[i+1]['open']
        entry_p = noi + spread(pair)/2*pip if direction==1 else noi - spread(pair)/2*pip
        entry_p += (SLIPPAGE_PIPS*pip/10)*direction*-1
        s = sa*a_v
        stop_p = entry_p - s if direction==1 else entry_p + s
        best_p = entry_p; held = 0; pos = direction; i += 1

    return trades

def summary(trades):
    if not trades: return dict(n=0,net=0,wr=0,gross=0,comm=0,payoff=0)
    df = pd.DataFrame(trades)
    wins = df[df['n']>0]; losses = df[df['n']<0]
    n = len(df); net = df['n'].sum(); gross = df['raw'].sum(); comm = df['c'].sum()
    wr = len(wins)/n*100 if n else 0
    agw = wins['n'].mean() if len(wins) else 0; agl = losses['n'].mean() if len(losses) else 0
    payoff = abs(agw/agl) if agl!=0 else 0
    return dict(n=n,net=net,wr=wr,gross=gross,comm=comm,payoff=payoff)

configs = {
    'baseline': {'z':3.5,'stop_a':3.0,'trig_a':1.0,'gap_a':0.05},
    'stop_wider': {'z':3.5,'stop_a':5.0,'trig_a':1.0,'gap_a':0.05},
    'z_5': {'z':5.0,'stop_a':3.0,'trig_a':1.0,'gap_a':0.05},
    'stop6': {'z':3.5,'stop_a':6.0,'trig_a':1.0,'gap_a':0.05},
    'z_4_stop5': {'z':4.0,'stop_a':5.0,'trig_a':1.0,'gap_a':0.05},
    'z_5_stop5': {'z':5.0,'stop_a':5.0,'trig_a':1.0,'gap_a':0.05},
    'z_6_stop5': {'z':6.0,'stop_a':5.0,'trig_a':1.0,'gap_a':0.05},
    'z_5_stop6': {'z':5.0,'stop_a':6.0,'trig_a':1.0,'gap_a':0.05},
    'z_4_stop6': {'z':4.0,'stop_a':6.0,'trig_a':1.0,'gap_a':0.05},
    'stop8': {'z':3.5,'stop_a':8.0,'trig_a':1.0,'gap_a':0.05},
    'z_5_stop8': {'z':5.0,'stop_a':8.0,'trig_a':1.0,'gap_a':0.05},
}

if __name__ == '__main__':
    if not mt5.initialize(): print("MT5 init failed"); sys.exit(1)

    all_results = []
    for pair in FOCUS_PAIRS:
        print(f"\n=== {pair} ===")
        r = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, datetime(2026,6,8), datetime(2026,7,25))
        if r is None or len(r) < 5000: print(" SKIP"); continue
        bars = pd.DataFrame(r); bars['time'] = pd.to_datetime(bars['time'], unit='s')
        print(f" {len(bars):,} bars")

        for cname, cp in configs.items():
            cp['pair'] = pair
            trades = run(bars.copy(), cp)
            s = summary(trades)
            all_results.append(dict(pair=pair,cfg=cname,**s))
            print(f"  {cname:13s}: n={s['n']:3d} net=${s['net']:>+7.0f} wr={s['wr']:5.1f}% payoff={s['payoff']:.2f}")

    mt5.shutdown()

    print("\n" + "=" * 120)
    print("NET PnL AFTER COMMISSION: Config × Pair (Jun 8 - Jul 25)")
    print("=" * 120)
    cfgs = list(configs.keys())
    header = f"{'Config':<15}" + "".join(f"{p:<10}" for p in FOCUS_PAIRS) + "TOTAL     "
    print(header); print("-"*120)

    totals = {}
    for cname in cfgs:
        row = f"{cname:<15}"; ct = 0
        for pair in FOCUS_PAIRS:
            v = 0
            for r in all_results:
                if r['pair']==pair and r['cfg']==cname: v = r['net']
            row += f"${v:>+7.0f} "; ct += v
        totals[cname] = ct
        row += f"${ct:>+7.0f}{' ✓' if ct>0 else ' ✗'}"
        print(row)

    print("-"*120)
    best = max(totals, key=totals.get)
    print(f"BEST: {best} = ${totals[best]:+.0f}")

    print(f"\nImprovement over baseline:")
    base = totals['baseline']
    for cname in cfgs[1:]:
        d = totals[cname] - base
        pct = d/base*100 if base != 0 else float('inf')
        print(f"  {cname:13s}: ${totals[cname]:+>7.0f} (Δ={d:+>+7.0f}, {pct:+.0f}%)")
    print("="*120)
