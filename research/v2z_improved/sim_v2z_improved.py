"""
V2+z Improved Simulator — tests 7 iterations for commission survivability.
Fixes look-ahead bias: enters at bar[i+1] open (real EA behavior).
"""
import sys
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime

SLIPPAGE_PIPS = 1.0
COMMISSION_PER_LOT = 5.0
MAX_HOLD_BARS = 54
Z_WINDOW = 50
ATR_PERIOD = 20
BASE_LOT = 0.75

FOCUS_PAIRS = ["EURAUD", "GBPAUD", "AUDNZD", "EURNZD", "GBPCAD", "GBPNZD"]

def pip_value_usd(symbol):
    m = {"EURAUD": 6.70, "EURNZD": 6.10, "GBPAUD": 6.10,
         "GBPCAD": 7.50, "GBPNZD": 5.60, "AUDNZD": 5.60}
    return m.get(symbol, 10.0)

def run_sim(bars):
    return _run(bars, {})  # baseline

def run_sim_with(bars, params):
    return _run(bars, params)

def _run(bars, p):
    zt = p.get('z', 3.5)
    sa = p.get('stop_a', 3.0)
    ta = p.get('trig_a', 1.0)
    ga = p.get('gap_a', 0.05)
    vol_adj = p.get('vol_adj', False)
    reg_filt = p.get('reg_filt', False)
    time_exit = p.get('time_exit', False)
    exit_bars = p.get('exit_bars', 30)
    pair = p.get('pair', '')

    trades = []
    pos = 0; entry_p = 0.0; entry_i = 0
    best_p = 0.0; stop_p = 0.0; held = 0
    z_entry = 0.0

    bars = bars.reset_index(drop=True if 'time' not in bars.columns else False)
    n = len(bars)

    atr = bars['high'].sub(bars['low']).rolling(ATR_PERIOD).mean()
    rets = bars['close'].diff()
    rmu = rets.rolling(Z_WINDOW).mean()
    rsd = rets.rolling(Z_WINDOW).std(ddof=1)
    zs = (rets - rmu) / rsd

    # ADX for regime filter
    if reg_filt:
        tr = pd.concat([bars['high'] - bars['low'],
                        (bars['high'] - bars['close'].shift()).abs(),
                        (bars['low'] - bars['close'].shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        up = rets.clip(lower=0)
        dn = (-rets).clip(lower=0)
        pdm = up.rolling(14).mean() / atr14 * 100
        ndm = dn.rolling(14).mean() / atr14 * 100
        dx = abs(pdm - ndm) / (pdm + ndm + 1e-10) * 100
        adx = dx.rolling(14).mean()

    if vol_adj:
        atr_med = atr.rolling(100).median()

    pip_mul = 0.0001 if 'JPY' not in pair else 0.01
    slip = SLIPPAGE_PIPS * pip_mul / 10

    start_i = max(Z_WINDOW, ATR_PERIOD) + 5
    i = start_i
    while i < n - 1:
        hour = bars.iloc[i]['time'].hour if 'time' in bars.columns else i % 24
        if hour >= 7:
            i += 1; continue

        if pos != 0:
            held += 1
            atr_v = atr.iloc[i]
            if pd.notna(atr_v) and atr_v > 0:
                if time_exit:
                    if held >= exit_bars:
                        ep = bars.iloc[i]['close']
                        raw = (ep - entry_p) * BASE_LOT * 100000 if pos == 1 else (entry_p - ep) * BASE_LOT * 100000
                        trades.append(dict(dir='LONG' if pos==1 else 'SHORT', entry=entry_p, exit=ep,
                                           raw=raw, comm=BASE_LOT*COMMISSION_PER_LOT,
                                           net=raw-BASE_LOT*COMMISSION_PER_LOT, reason='exp', held=held, z=z_entry))
                        pos = 0; i += 1; continue
                else:
                    tg = ta * atr_v; gp = ga * atr_v
                    hi = bars.iloc[i]['high']; lo = bars.iloc[i]['low']
                    if pos == 1:
                        if hi > best_p:
                            best_p = hi
                            if best_p - entry_p > tg:
                                ns = best_p - gp
                                if ns > stop_p: stop_p = ns
                        if lo <= stop_p:
                            raw = (stop_p - entry_p) * BASE_LOT * 100000
                            trades.append(dict(dir='LONG', entry=entry_p, exit=stop_p, raw=raw,
                                               comm=BASE_LOT*COMMISSION_PER_LOT, net=raw-BASE_LOT*COMMISSION_PER_LOT,
                                               reason='stp', held=held, z=z_entry))
                            pos = 0; i += 1; continue
                    else:
                        if lo < best_p:
                            best_p = lo
                            if entry_p - best_p > tg:
                                ns = best_p + gp
                                if ns < stop_p: stop_p = ns
                        if hi >= stop_p:
                            raw = (entry_p - stop_p) * BASE_LOT * 100000
                            trades.append(dict(dir='SHORT', entry=entry_p, exit=stop_p, raw=raw,
                                               comm=BASE_LOT*COMMISSION_PER_LOT, net=raw-BASE_LOT*COMMISSION_PER_LOT,
                                               reason='stp', held=held, z=z_entry))
                            pos = 0; i += 1; continue

            if held >= MAX_HOLD_BARS:
                ep = bars.iloc[i]['close']
                raw = (ep - entry_p) * BASE_LOT * 100000 if pos == 1 else (entry_p - ep) * BASE_LOT * 100000
                trades.append(dict(dir='LONG' if pos==1 else 'SHORT', entry=entry_p, exit=ep, raw=raw,
                                   comm=BASE_LOT*COMMISSION_PER_LOT, net=raw-BASE_LOT*COMMISSION_PER_LOT,
                                   reason='exp', held=held, z=z_entry))
                pos = 0
            i += 1; continue

        z_v = zs.iloc[i]
        a_v = atr.iloc[i]
        if pd.isna(z_v) or pd.isna(a_v) or a_v <= 0:
            i += 1; continue

        thresh = zt
        if vol_adj and atr_med.iloc[i] > 1e-10:
            ratio = a_v / atr_med.iloc[i]
            thresh = zt * max(0.3, ratio)

        if abs(z_v) < thresh:
            i += 1; continue

        if reg_filt and not pd.isna(adx.iloc[i]) and adx.iloc[i] > 25:
            i += 1; continue

        direction = -1 if z_v > 0 else 1
        next_open = bars.iloc[i + 1]['open']
        entry_p = next_open + (slip * direction * -1)
        s = sa * a_v
        stop_p = entry_p - s if direction == 1 else entry_p + s
        best_p = entry_p; held = 0
        z_entry = z_v; pos = direction; entry_i = i + 1
        i += 1

    return trades

def summarize(trades, label, pair):
    if not trades:
        return {'pair': pair, 'label': label, 'n': 0, 'net': 0, 'wr': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'gross': 0, 'comm': 0}
    df = pd.DataFrame(trades)
    wins = df[df['net'] > 0]; losses = df[df['net'] < 0]
    n = len(df); net = df['net'].sum(); gross = df['raw'].sum(); comm = df['comm'].sum()
    wr = len(wins)/n*100 if n else 0
    pf = abs(wins['raw'].sum() / losses['raw'].sum()) if len(losses) and losses['raw'].sum() != 0 else float('inf')
    return {'pair': pair, 'label': label, 'n': n, 'net': net, 'wr': wr, 'pf': pf,
            'avg_win': wins['net'].mean() if len(wins) else 0,
            'avg_loss': losses['net'].mean() if len(losses) else 0,
            'gross': gross, 'comm': comm}

def get_data(pair, start_dt, end_dt):
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, start_dt, end_dt)
    if rates is None or len(rates) == 0: return None
    bars = pd.DataFrame(rates)
    bars['time'] = pd.to_datetime(bars['time'], unit='s')
    bars['symbol'] = pair
    return bars

# === TEST CONFIGURATIONS ===
CONFIGS = {
    'baseline': {},
    'vol_adj': {'vol_adj': True},
    'reg_filt': {'reg_filt': True},
    'time_exit': {'time_exit': True, 'exit_bars': 30},
    'z_higher': {'z': 5.0},
    'z_lower': {'z': 2.5},
    'stop_wider': {'stop_a': 5.0},
    'stop_tighter': {'stop_a': 2.0},
    'trail_tighter': {'trig_a': 0.5, 'gap_a': 0.02},
    'trail_looser': {'trig_a': 2.0, 'gap_a': 0.1},
    'vol_adj+reg': {'vol_adj': True, 'reg_filt': True},
    'vol_adj+z5': {'vol_adj': True, 'z': 5.0},
    'all_filters': {'vol_adj': True, 'reg_filt': True, 'z': 4.5},
}

if __name__ == '__main__':
    start_dt = datetime(2026, 1, 1)
    end_dt   = datetime(2026, 7, 25)

    print("=" * 100)
    print(f"V2+z Improved: Testing {len(CONFIGS)} configurations on {len(FOCUS_PAIRS)} pairs")
    print(f"Period: {start_dt.date()} to {end_dt.date()}")
    print("=" * 100)

    if not mt5.initialize():
        print("MT5 init failed!"); sys.exit(1)

    all_results = []
    for pair in FOCUS_PAIRS:
        print(f"\n--- Downloading {pair} ---")
        bars = get_data(pair, start_dt, end_dt)
        if bars is None or len(bars) < 5000:
            print(f"  SKIP: insufficient data ({len(bars) if bars is not None else 0} bars)")
            continue
        print(f"  {len(bars)} bars ({len(bars)/60/24:.1f} days)")

        # Run baseline first to validate
        base_trades = run_sim(bars.copy())
        base_s = summarize(base_trades, 'baseline', pair)
        print(f"  baseline: n={base_s['n']:4d} net=${base_s['net']:+.0f} wr={base_s['wr']:5.1f}% "
              f"pf={base_s['pf']:.2f} gross=${base_s['gross']:+.0f} comm=${base_s['comm']:.0f}")
        all_results.append(base_s)

        # Run each config
        for cname, cparams in CONFIGS.items():
            if cname == 'baseline': continue
            cparams['pair'] = pair
            trades = run_sim_with(bars.copy(), cparams)
            s = summarize(trades, cname, pair)
            delta = s['net']
            print(f"  {cname:15s}: n={s['n']:4d} net=${s['net']:+.0f} wr={s['wr']:5.1f}% "
                  f"pf={s['pf']:.2f} gross=${s['gross']:+.0f} comm=${s['comm']:.0f} "
                  f"Δ=${delta-base_s['net']:+.0f}")
            all_results.append(s)

    mt5.shutdown()

    # Summary table
    print("\n" + "=" * 120)
    print("SUMMARY: Net PnL by Pair × Config (after commission)")
    print("=" * 120)
    header = f"{'Config':<18}" + "".join(f"{p:<12}" for p in FOCUS_PAIRS) + f"{'TOTAL':<12}"
    print(header)
    print("-" * 120)

    totals = {}
    for cname in ['baseline'] + list(CONFIGS.keys())[1:]:
        row = f"{cname:<18}"
        row_total = 0
        for pair in FOCUS_PAIRS:
            val = 0
            for r in all_results:
                if r['pair'] == pair and r['label'] == cname:
                    val = r['net']
            row += f"${val:>+8.0f}  "
            row_total += val
        row_total_name = cname
        row += f"${row_total:>+8.0f}"
        if row_total > 0: row += " ✓"
        print(row)
        totals[cname] = row_total

    print("-" * 120)
    # Best config
    best = max(totals, key=totals.get)
    print(f"Best: {best} = ${totals[best]:+.0f}")
    print("=" * 120)
