"""Dollar PnL for a $25k funded account, including realistic slippage estimates."""
import numpy as np, pandas as pd, time
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'

# USDJPY rate for JPY pair conversion ~157 (Oct-Dec 2025 average)
USDJPY = 157.0

PIP_VALUES = {  # per standard lot (100k units)
    'EURUSD': 10.0,         # $10/pip
    'EURJPY': 1000/USDJPY,  # ~$6.37/pip
    'GBPJPY': 1000/USDJPY,  # ~$6.37/pip
}

SPREAD_COST = {'EURUSD': 0.15, 'EURJPY': 50, 'GBPJPY': 60}
# Actual median spreads from tick data
REAL_SPREAD = {'EURUSD': 0.04, 'EURJPY': 40, 'GBPJPY': 30}  # MP

def load(pair):
    dfs = []
    for y,m in [(2025,10),(2025,11),(2025,12)]:
        fn = TICK_DIR / f'{pair}_Raw_Spread_{y}_{m:02d}.zip'
        d = pd.read_csv(fn, compression='zip',
            names=['E','S','Ts','B','A'], skiprows=1, header=None,
            dtype={'Ts':str,'B':np.float64,'A':np.float64})
        d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
            format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        dfs.append(d.dropna(subset=['Ts']))
    df = pd.concat(dfs, ignore_index=True).sort_values('Ts').reset_index(drop=True)
    df['MP'] = ((df['B']+df['A'])/2) * 10000
    return df.set_index('Ts')

def run(pair):
    t = load(pair)
    b = t['MP'].resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    ret = b['close'].diff()
    z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
    atr = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)
    valid = z.notna() & atr.notna()
    idxs = np.where(valid)[0]
    closes = b['close'].values; highs = b['high'].values; lows = b['low'].values
    z_vals = z.values; atr_vals = atr.values

    pnls = []; entry_prices = []; exit_prices = []
    stop_a, trig_a, gap_a = 0.15, 0.20, 0.10
    for pos in idxs:
        if pos + 2 >= len(b): continue
        direction = -1 if z_vals[pos] > 0 else 1
        entry = closes[pos]; atr_v = atr_vals[pos]
        if np.isnan(atr_v) or atr_v < 1e-10: continue
        s = stop_a * atr_v; tg = trig_a * atr_v; gp = gap_a * atr_v
        best = entry; exited = False
        for j in range(1, 55):
            bp = pos + j
            if bp >= len(b): break
            if direction == 1:
                best = max(best, highs[bp])
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if lows[bp] <= sl:
                    pnls.append((sl - entry) * direction)
                    entry_prices.append(entry); exit_prices.append(sl)
                    exited=True; break
            else:
                best = min(best, lows[bp])
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if highs[bp] >= sl:
                    pnls.append((sl - entry) * direction)
                    entry_prices.append(entry); exit_prices.append(sl)
                    exited=True; break
        if not exited:
            exit_px = closes[min(pos+54, len(b)-1)]
            pnls.append((exit_px - entry) * direction)
            entry_prices.append(entry); exit_prices.append(exit_px)

    return np.array(pnls), entry_prices, exit_prices

print("=" * 75)
print("REALISTIC P&L PROJECTION — $25k Funded Account")
print("=" * 75)

total_daily = 0
for pair in ['EURUSD', 'EURJPY', 'GBPJPY']:
    pnls, entries, exits = run(pair)
    wr = np.mean(pnls > 0)
    avg_mp = np.mean(pnls)
    net_mp = avg_mp - SPREAD_COST[pair]
    real_net_mp = avg_mp - REAL_SPREAD[pair]

    # Per standard lot in dollars
    pip_val = PIP_VALUES[pair]
    # Convert MP to pips: for EURUSD 1MP=1pip, for JPY 100MP=1pip
    if pair == 'EURUSD':
        mp_per_pip = 1.0
    else:
        mp_per_pip = 100.0

    net_pips = net_mp / mp_per_pip
    real_net_pips = real_net_mp / mp_per_pip

    usd_per_lot = net_pips * pip_val
    real_usd_per_lot = real_net_pips * pip_val

    trades_per_day = len(pnls) / 78  # ~78 trading days in 3 months

    print(f"\n--- {pair} ---")
    print(f"  WR: {wr:.1%}  avg gross MP: {avg_mp:+.2f}")
    print(f"  Assumed cost: {SPREAD_COST[pair]} MP  net: {net_mp:+.2f} MP  ({net_pips:+.3f} pips)")
    print(f"  Real cost:    {REAL_SPREAD[pair]} MP  net: {real_net_mp:+.2f} MP  ({real_net_pips:+.3f} pips)")
    print(f"  Trades/day: {trades_per_day:.0f}")

    print(f"  --- Per standard lot (100k) ---")
    print(f"  $/trade (assumed cost): ${usd_per_lot:.2f}")
    print(f"  $/trade  (real cost):   ${real_usd_per_lot:.2f}")

    for lots in [0.01, 0.03, 0.05, 0.10]:
        daily = trades_per_day * usd_per_lot * lots
        real_daily = trades_per_day * real_usd_per_lot * lots
        monthly = daily * 22
        real_monthly = real_daily * 22
        max_positions = lots * 3  # 3 pairs × 0.1 lot
        total_risk = lots * abs(avg_mp - 2*SPREAD_COST[pair]) * pip_val / mp_per_pip * 10  # 10 consecutive losses
        print(f"    {lots:>4.0f} lot: ${daily:>6.0f}/d (${monthly:>6.0f}/mo)  "
              f"real: ${real_daily:>6.0f}/d (${real_monthly:>6.0f}/mo)  "
              f"10-loss risk: ${total_risk:.0f}")

# Combined scenario
print(f"\n{'='*75}")
print("COMBINED (all 3 pairs, 0.03 lot per trade)")
print(f"{'='*75}")
total_daily = 0
total_real_daily = 0
total_monthly = 0
total_real_monthly = 0
total_trades = 0
max_exposure = 0

for pair in ['EURUSD', 'EURJPY', 'GBPJPY']:
    pnls, _, _ = run(pair)
    avg_mp = np.mean(pnls)
    wr = np.mean(pnls > 0)
    net_mp = avg_mp - SPREAD_COST[pair]
    real_net_mp = avg_mp - REAL_SPREAD[pair]

    pip_val = PIP_VALUES[pair]
    mp_per_pip = 1.0 if pair == 'EURUSD' else 100.0

    net_pips = net_mp / mp_per_pip
    real_net_pips = real_net_mp / mp_per_pip

    usd_per_lot = net_pips * pip_val
    real_usd_per_lot = real_net_pips * pip_val

    trades_day = len(pnls) / 78
    total_trades += trades_day

    lot = 0.03
    daily = trades_day * usd_per_lot * lot
    real_daily = trades_day * real_usd_per_lot * lot
    total_daily += daily
    total_real_daily += real_daily
    total_monthly += daily * 22
    total_real_monthly += real_daily * 22

print(f"  Total trades/day: {total_trades:.0f} ({total_trades/60:.1f}/min)")
print(f"  Daily PnL (assumed cost ${SPREAD_COST}): ${total_daily:.0f}")
print(f"  Daily PnL (real cost ${REAL_SPREAD}):   ${total_real_daily:.0f}")
print(f"  Monthly PnL (assumed): ${total_monthly:.0f}")
print(f"  Monthly PnL (real):    ${total_real_monthly:.0f}")
print(f"  Return on $25k (assumed): {total_monthly/25000*100:.0f}%/mo")
print(f"  Return on $25k (real):    {total_real_monthly/25000*100:.0f}%/mo")

# Slippage sensitivity
print(f"\n{'='*75}")
print("SLIPPAGE SENSITIVITY — 0.03 lot per trade")
print(f"{'='*75}")
for slip_pips in [0, 0.1, 0.2, 0.5]:
    slip_cost_usd = {}
    total_slip = 0
    for pair in ['EURUSD', 'EURJPY', 'GBPJPY']:
        pnls, _, _ = run(pair)
        avg_mp = np.mean(pnls)
        pip_val = PIP_VALUES[pair]
        mp_per_pip = 1.0 if pair == 'EURUSD' else 100.0
        net_pips = (avg_mp - SPREAD_COST[pair]) / mp_per_pip

        # Slippage on 20% of trades (trailing stop fills)
        slip = slip_pips * pip_val * 0.03 * 0.20  # per trade
        slip_per_trade = slip_pips * pip_val * 0.03  # per trade if applied to all
        net_after_slip = net_pips - slip_pips * 0.20  # slippage on 20%

        trades_day = len(pnls) / 78
        daily_no_slip = trades_day * net_pips * pip_val * 0.03
        daily_with_slip = trades_day * net_after_slip * pip_val * 0.03
        monthly_with_slip = daily_with_slip * 22
        total_slip += monthly_with_slip

    print(f"  {slip_pips:.1f}p slip on 20%: ${total_slip:>6.0f}/mo")

# News gap risk
print(f"\n{'='*75}")
print("RISK ANALYSIS")
print(f"{'='*75}")
print(f"  Max concurrent positions (3 pairs × 0.03 lot): ~0.09 lot total")
print(f"  Max notional: ~$11,700 (at 1.30 EURUSD)")
print(f"  Leverage: ~0.47x on $25k")
print(f"  Daily loss on 20 consecutive losses (EURUSD 0.03 lot): ${0.19*0.03*10:.2f}")
print(f"  News gap risk (e.g. NFP 20 pip jump): ${20*10*0.03:.0f} on EURUSD 0.03 lot")
