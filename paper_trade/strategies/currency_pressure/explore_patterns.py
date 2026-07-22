"""Brute-force pattern exploration on M1 data.
Tests multiple distinct strategy ideas for WR > 70% and high TPD."""
import MetaTrader5 as mt5
import numpy as np
from time import time
from collections import defaultdict, deque
import sys

ALL_CURRENCIES = ['USD','EUR','JPY','GBP','AUD','NZD','CAD','CHF']
ALL_PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD","USDCHF",
    "EURJPY","GBPJPY","EURGBP","EURAUD","EURCHF","EURCAD","EURNZD",
    "GBPAUD","GBPCAD","GBPCHF","GBPNZD","AUDJPY","AUDCAD","AUDCHF","AUDNZD",
    "NZDJPY","NZDCAD","NZDCHF","CADJPY","CADCHF","CHFJPY"]
PAIR_SPREAD_PIPS = {"AUDUSD":1.5,"EURUSD":1.5,"GBPUSD":2.0,"NZDUSD":2.0,
    "USDCAD":2.0,"USDCHF":2.0,"USDJPY":1.8,"EURJPY":2.5,"GBPJPY":4.0,
    "EURGBP":2.0,"EURAUD":2.5,"EURCHF":2.0,"EURCAD":2.0,"EURNZD":2.5,
    "GBPAUD":2.5,"GBPCAD":2.5,"GBPCHF":3.0,"GBPNZD":3.0,"AUDJPY":2.5,
    "AUDCAD":2.5,"AUDCHF":2.5,"AUDNZD":2.5,"NZDJPY":3.0,"NZDCAD":2.5,
    "NZDCHF":2.5,"CADJPY":2.5,"CADCHF":2.5,"CHFJPY":3.0}

PIP_VAL = 5.0  # $5 per pip for 0.5 lot
PIP_SZ = np.array([0.01 if "JPY" in p else 0.0001 for p in ALL_PAIRS])

t0 = time()
if not mt5.initialize():
    print("MT5 init failed", file=sys.stderr); sys.exit(1)
for p in ALL_PAIRS: mt5.symbol_select(p, True)

print("Loading M1 data...", file=sys.stderr)
PC = {}
for p in ALL_PAIRS:
    r = mt5.copy_rates_from_pos(p, mt5.TIMEFRAME_M1, 0, 60000)
    if r is not None: PC[p] = np.array([float(x[4]) for x in r])
n = min(len(v) for v in PC.values())
P = np.zeros((n, len(ALL_PAIRS)))
for pj, p in enumerate(ALL_PAIRS):
    if p in PC: P[:n, pj] = PC[p][:n]
print(f"Loaded {n} bars", file=sys.stderr)

lr = np.diff(np.log(P + 1e-12), axis=0)
n = len(lr)
rets = lr  # log returns
dirs = np.sign(rets)

# ============================================================
# STRATEGY 1: Consecutive Candle Exhaustion
# ============================================================
print(f"\n{'='*80}", file=sys.stderr)
print(f"  STRATEGY 1: CONSECUTIVE CANDLE EXHAUSTION", file=sys.stderr)
print(f"  Idea: After N same-direction M1 bars, counter-trade next bar", file=sys.stderr)
print(f"{'='*80}", file=sys.stderr)

for n_consec in [3, 4, 5, 6, 7]:
    total = 0; wins = 0; pnl = []
    for pj in [ALL_PAIRS.index(p) for p in ['EURUSD','GBPUSD','AUDUSD','NZDUSD','USDCAD','USDCHF']]:
        pair = ALL_PAIRS[pj]
        sp = PAIR_SPREAD_PIPS[pair] * PIP_SZ[pj]
        for i in range(n_consec + 1, n - 5):
            seg = dirs[max(0,i-n_consec):i, pj]
            if np.all(seg >= 0) or np.all(seg <= 0):
                # same direction N bars in a row
                expected_dir = -np.sign(dirs[i-1, pj])
                actual_dir = dirs[i, pj]
                if actual_dir != 0:
                    total += 1
                    is_win = (actual_dir == expected_dir)
                    if is_win: wins += 1
                    if is_win:
                        pnl.append(abs(rets[i, pj]) / PIP_SZ[pj] * PIP_VAL - sp)
                    else:
                        pnl.append(-(abs(rets[i, pj]) / PIP_SZ[pj] * PIP_VAL) - sp)
    wr = wins/total*100 if total > 0 else 0
    avg_pnl = np.mean(pnl) if pnl else 0
    tpd = total / (n / 1440) if n > 1440 else 0
    print(f"  Consec={n_consec}: WR={wr:.1f}%, TPD={tpd:.1f}, trades={total}, avg=${avg_pnl:.2f}", file=sys.stderr)

# ============================================================
# STRATEGY 2: Correlation Divergence (EURUSD vs GBPUSD)
# ============================================================
print(f"\n{'='*80}", file=sys.stderr)
print(f"  STRATEGY 2: CORRELATION DIVERGENCE", file=sys.stderr)
print(f"  Idea: When EURUSD vs GBPUSD exceed divergence threshold, trade convergence", file=sys.stderr)
print(f"{'='*80}", file=sys.stderr)

pj_eu = ALL_PAIRS.index('EURUSD')
pj_gu = ALL_PAIRS.index('GBPUSD')

divergences = [0.0003, 0.0005, 0.0008, 0.0010, 0.0015, 0.0020]
for hold in [1, 2, 3, 5, 10]:
    best_wr = 0
    for div in divergences:
        total = 0; wins = 0; pnl = []
        for i in range(hold, n - 5):
            eu_ret = rets[i, pj_eu]
            gu_ret = rets[i, pj_gu]
            gap = abs(eu_ret - gu_ret)
            if gap > div:
                total += 1
                # Bet: the underperformer outperforms in next `hold` bars
                winner = 'eu' if eu_ret > gu_ret else 'gu'
                future_eu = np.sum(rets[i+1:i+1+hold, pj_eu])
                future_gu = np.sum(rets[i+1:i+1+hold, pj_gu])
                future_winner = 'eu' if future_eu > future_gu else 'gu'
                is_win = winner == future_winner
                if is_win: wins += 1
        wr = wins/total*100 if total > 0 else 0
        tpd = total / (n / 1440) if n > 1440 else 0
        if wr > best_wr: best_wr = wr
        if total >= 50:
            print(f"  Hold={hold:2d}m, div={div:.4f}: WR={wr:.1f}%, TPD={tpd:.1f}, trades={total}", file=sys.stderr)
    if best_wr > 60:
        print(f"  ** Hold={hold}: Best WR={best_wr:.1f}%", file=sys.stderr)

# ============================================================
# STRATEGY 3: Session Transition Volatility
# ============================================================
print(f"\n{'='*80}", file=sys.stderr)
print(f"  STRATEGY 3: SESSION TRANSITION MEAN REVERSION", file=sys.stderr)
print(f"  Idea: At session boundaries, sharp moves revert", file=sys.stderr)
print(f"{'='*80}", file=sys.stderr)

# Session boundaries: 0, 8, 13, 22 UTC (Tokyo, London, NY, Sydney)
boundaries = [0, 8, 13, 22]
for lookback in [10, 15, 20, 30, 60]:
    for hold in [5, 10, 15, 30]:
        total = 0; wins = 0
        for pj in [ALL_PAIRS.index(p) for p in ['EURUSD','GBPUSD','AUDUSD','USDJPY']]:
            for boundary in boundaries:
                # We need bar timestamps - skip since we only have close prices
                pass
        if total > 0:
            wr = wins/total*100
            print(f"  Lookback={lookback}m, hold={hold}m: WR={wr:.1f}%, trades={total}", file=sys.stderr)
        else:
            pass  # need timestamps

# ============================================================
# STRATEGY 4: Large Bar Retracement
# ============================================================
print(f"\n{'='*80}", file=sys.stderr)
print(f"  STRATEGY 4: LARGE BAR RETRACEMENT", file=sys.stderr)
print(f"  Idea: After a bar > 2 SD, next 1-3 bars retrace", file=sys.stderr)
print(f"{'='*80}", file=sys.stderr)

for lookback in [20, 50, 100, 200]:
    for z_thresh in [2.0, 2.5, 3.0]:
        for hold in [1, 2, 3]:
            total = 0; wins = 0; pnl = []
            for pj in range(len(ALL_PAIRS)):
                pair = ALL_PAIRS[pj]
                sp = PAIR_SPREAD_PIPS[pair] * PIP_SZ[pj]
                for i in range(lookback, n - hold):
                    window = rets[i-lookback:i, pj]
                    mean = np.mean(window)
                    std = np.std(window)
                    if std < 1e-12: continue
                    z = (rets[i-1, pj] - mean) / std
                    if abs(z) > z_thresh:
                        total += 1
                        expected_dir = -np.sign(rets[i-1, pj])
                        future_dir = np.sign(np.sum(rets[i:i+hold, pj]))
                        if future_dir == expected_dir:
                            wins += 1
                        ret = np.sum(rets[i:i+hold, pj])
                        pnl_val = ret / PIP_SZ[pj] * PIP_VAL
                        if abs(pnl_val) < sp:
                            pnl.append(-sp)
                        elif pnl_val > 0:
                            pnl.append(pnl_val - sp)
                        else:
                            pnl.append(pnl_val - sp)
            wr = wins/total*100 if total > 0 else 0
            avg_pnl = np.mean(pnl) if pnl else 0
            tpd = total / (n / 1440) if n > 1440 else 0
            if total >= 20:
                print(f"  lb={lookback}, Z>{z_thresh:.0f}, hold={hold}: WR={wr:.1f}%, TPD={tpd:.1f}, trades={total}, avg=${avg_pnl:.2f}", file=sys.stderr)

# ============================================================
# STRATEGY 5: Volatility Contraction + Expansion
# ============================================================
print(f"\n{'='*80}", file=sys.stderr)
print(f"  STRATEGY 5: VOLATILITY CONTRACTION + EXPANSION", file=sys.stderr)
print(f"  Idea: When vol drops to X% of recent avg, next bar expands directionally", file=sys.stderr)
print(f"{'='*80}", file=sys.stderr)

for vol_window in [10, 20, 30]:
    for ratio_thresh in [0.3, 0.5, 0.7]:
        total = 0; wins = 0; pnl = []
        for pj in range(len(ALL_PAIRS)):
            pair = ALL_PAIRS[pj]
            sp = PAIR_SPREAD_PIPS[pair] * PIP_SZ[pj]
            for i in range(vol_window+10, n - 3):
                short_vol = np.std(rets[i-vol_window:i, pj])
                long_vol = np.std(rets[i-200:i, pj]) if i >= 200 else short_vol
                if long_vol < 1e-12: continue
                ratio = short_vol / long_vol
                if ratio < ratio_thresh:
                    total += 1
                    # Next 2 bars: which direction?
                    dir1 = np.sign(rets[i, pj])
                    dir2 = np.sign(rets[i+1, pj]) if i+1 < n else 0
                    dir3 = np.sign(rets[i+2, pj]) if i+2 < n else 0
                    # Is there a directional sequence in next 3 bars?
                    seq = [d for d in [dir1, dir2, dir3] if d != 0]
                    if len(seq) >= 2 and np.all(np.array(seq) > 0):
                        wins += 1
                    elif len(seq) >= 2 and np.all(np.array(seq) < 0):
                        wins += 1
        wr = wins/total*100 if total > 0 else 0
        tpd = total / (n / 1440) if n > 1440 else 0
        if total >= 20:
            print(f"  VolW={vol_window}, ratio<{ratio_thresh}: WR={wr:.1f}%, TPD={tpd:.1f}, trades={total}", file=sys.stderr)

# ============================================================
# STRATEGY 6: Multi-pair consensus exhaustion
# ============================================================
print(f"\n{'='*80}", file=sys.stderr)
print(f"  STRATEGY 6: MULTI-PAIR CONSENSUS EXHAUSTION", file=sys.stderr)
print(f"  Idea: When X% of pairs move same direction for N bars, counter-trade", file=sys.stderr)
print(f"{'='*80}", file=sys.stderr)

for n_bars in [3, 5]:
    for pct in [0.6, 0.7, 0.8]:
        total = 0; wins = 0; pnl = []
        for i in range(n_bars + 1, n - 3):
            # Check last N bars: what % moved in one direction?
            dirs_hist = dirs[i-n_bars:i, :]
            up_count = np.sum(dirs_hist > 0, axis=1)
            dn_count = np.sum(dirs_hist < 0, axis=1)
            up_pct = up_count / len(ALL_PAIRS)
            dn_pct = dn_count / len(ALL_PAIRS)
            
            consensus_up = np.any(up_pct > pct)
            consensus_dn = np.any(dn_pct > pct)
            
            if consensus_up or consensus_dn:
                expected_dir = -1 if consensus_up else 1
                # Next 2 bars - does consensus reverse?
                actual = np.sign(dirs[i, :].sum())
                if actual != 0:
                    total += 1
                    if actual == expected_dir:
                        wins += 1
                        pnl.append(20.0)
                    else:
                        pnl.append(-20.0)
        wr = wins/total*100 if total > 0 else 0
        tpd = total / (n / 1440) if n > 1440 else 0
        if total >= 20:
            print(f"  {n_bars}bars>{pct*100:.0f}%: WR={wr:.1f}%, TPD={tpd:.1f}, trades={total}", file=sys.stderr)

# ============================================================
# STRATEGY 7: Opening Range Breakdown (hourly)
# ============================================================
print(f"\n{'='*80}", file=sys.stderr)
print(f"  STRATEGY 7: HOURLY RANGE BREAK", file=sys.stderr)
print(f"  Idea: First X min of hour = range. Breakout beyond range fades back in.", file=sys.stderr)
# Need hour timestamps - skip for now since we don't have them

# ============================================================
# STRATEGY 8: Tick Volume Climax
# ============================================================
print(f"\n{'='*80}", file=sys.stderr)
print(f"  STRATEGY 8: POSITION CLIMAX (using returns, not volume)", file=sys.stderr)
print(f"  Idea: Rate of change deceleration = exhaustion", file=sys.stderr)
print(f"{'='*80}", file=sys.stderr)

for pj in [ALL_PAIRS.index(p) for p in ['EURUSD','GBPUSD','AUDUSD']]:
    pair = ALL_PAIRS[pj]
    sp = PAIR_SPREAD_PIPS[pair] * PIP_SZ[pj]
    for accel_window in [3, 5]:
        total = 0; wins = 0; pnl_list = []
        for i in range(accel_window + 2, n - 3):
            rets_window = rets[i-accel_window:i, pj]
            d1 = rets_window[-1] - rets_window[-2]
            d2 = rets_window[-2] - rets_window[-3]
            # Acceleration: first move up, then deceleration
            if rets_window[-3] > 0 and rets_window[-2] > 0 and rets_window[-1] < rets_window[-2]:
                total += 1
                future = np.sum(rets[i:i+2, pj])
                is_win = future < 0
                if is_win: wins += 1
                pnl_val = abs(future) / PIP_SZ[pj] * PIP_VAL - sp
                pnl_list.append(pnl_val)
        wr = wins/total*100 if total > 0 else 0
        avg_p = np.mean(pnl_list) if pnl_list else 0
        tpd = total / (n / 1440)
        if total >= 20:
            print(f"  {pair} accel_w={accel_window}: WR={wr:.1f}%, TPD={tpd:.1f}, trades={total}, avg=${avg_p:.2f}", file=sys.stderr)

# ============================================================
# STRATEGY 9: Support/Resistance proximity bounce
# ============================================================
print(f"\n{'='*80}", file=sys.stderr)
print(f"  STRATEGY 9: RECENT HIGH/LOW BOUNCE", file=sys.stderr)
print(f"  Idea: Price near recent N-bar extreme -> trade bounce", file=sys.stderr)
print(f"{'='*80}", file=sys.stderr)

for lookback in [20, 50, 100]:
    for pj in [ALL_PAIRS.index(p) for p in ['EURUSD','GBPUSD','AUDUSD']]:
        pair = ALL_PAIRS[pj]
        sp = PAIR_SPREAD_PIPS[pair] * PIP_SZ[pj]
        total = 0; wins = 0; pnl_list = []
        for i in range(lookback, n - 5):
            window = P[i-lookback:i, pj]
            high = np.max(window)
            low = np.min(window)
            curr = P[i, pj]
            near_high = (high - curr) / high < 0.0005
            near_low = (curr - low) / low < 0.0005
            if near_high:
                total += 1
                future = np.sum(rets[i+1:i+4, pj])
                is_win = future < 0
                if is_win: wins += 1
                pnl_val = abs(future) / PIP_SZ[pj] * PIP_VAL - sp
                pnl_list.append(pnl_val)
            elif near_low:
                total += 1
                future = np.sum(rets[i+1:i+4, pj])
                is_win = future > 0
                if is_win: wins += 1
                pnl_val = future / PIP_SZ[pj] * PIP_VAL - sp
                pnl_list.append(pnl_val)
        wr = wins/total*100 if total > 0 else 0
        avg_p = np.mean(pnl_list) if pnl_list else 0
        tpd = total / (n / 1440)
        if total >= 20:
            print(f"  {pair} lookback={lookback}: WR={wr:.1f}%, TPD={tpd:.1f}, trades={total}, avg=${avg_p:.2f}", file=sys.stderr)

# ============================================================
# STRATEGY 10: Pair rotation / relative strength
# ============================================================
print(f"\n{'='*80}", file=sys.stderr)
print(f"  STRATEGY 10: RELATIVE STRENGTH", file=sys.stderr)
print(f"  Idea: When pair A outperforms pair B by >X for N bars, rotate trade to underperformer", file=sys.stderr)
print(f"{'='*80}", file=sys.stderr)

# Compare EURUSD vs AUDUSD 
pj_eu = ALL_PAIRS.index('EURUSD')
pj_au = ALL_PAIRS.index('AUDUSD')
for lookback in [10, 20, 30]:
    total = 0; wins = 0
    for hold in [3, 5, 10]:
        for i in range(lookback, n - hold):
            eu_sum = np.sum(rets[i-lookback:i, pj_eu])
            au_sum = np.sum(rets[i-lookback:i, pj_au])
            gap = eu_sum - au_sum
            if gap > 0.005:  # EUR outperformed AUD
                total += 1
                # Expect AUD to outperform EUR next
                future_eu = np.sum(rets[i:i+hold, pj_eu])
                future_au = np.sum(rets[i:i+hold, pj_au])
                is_win = future_au > future_eu
                if is_win: wins += 1
        if total > 0:
            wr = wins/total*100
            tpd = total / (n / 1440)
            if total >= 20:
                print(f"  EURvsAUD lb={lookback} hold={hold}: WR={wr:.1f}%, TPD={tpd:.1f}, trades={total}", file=sys.stderr)

mt5.shutdown()
print(f"\nTotal time: {time()-t0:.1f}s", file=sys.stderr)
