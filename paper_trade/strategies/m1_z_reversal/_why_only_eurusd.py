"""Deep-dive: why does impulse fade only work on EURUSD?"""
import sys, time, numpy as np
from collections import deque
from pathlib import Path
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")

TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\fundednext_ticks")
PIP = 0.0001
PIP_USD = 10.0

def load(pair):
    d = np.load(str(TICK_DIR / (pair + ".npy")))
    ts = np.array([t[0] for t in d], dtype=np.int64)
    bid = np.array([t[1] for t in d], dtype=np.float64)
    ask = np.array([t[2] for t in d], dtype=np.float64)
    return ts, bid, ask

def analyze_impulses(ts, mid, pip, spread_price, comm_price, window_s=20, detect_pips=5, hold_s=30):
    """Detect impulses and measure what happens after each one."""
    n = len(mid)
    min_q = deque(); max_q = deque(); ws_idx = 0
    thresh = detect_pips * pip
    cost = spread_price + comm_price
    
    impulses = []  # (direction, extreme_idx, entry_price, hold_exit_price, 10s_price, 30s_price, 60s_price, stop_hit)
    
    for i in range(n):
        v = float(mid[i])
        while min_q and min_q[-1][0] >= v: min_q.pop()
        while max_q and max_q[-1][0] <= v: max_q.pop()
        min_q.append((v, i)); max_q.append((v, i))
        while ts[i] - ts[ws_idx] > window_s:
            if min_q and min_q[0][1] == ws_idx: min_q.popleft()
            if max_q and max_q[0][1] == ws_idx: max_q.popleft()
            ws_idx += 1
        if i > ws_idx:
            wp = mid[ws_idx]
            hp = float(max_q[0][0] - wp)
            lp = float(wp - min_q[0][0])
            span = ts[i] - ts[ws_idx]
            if span <= window_s and (hp >= thresh or lp >= thresh):
                if impulses and impulses[-1]["ws_idx"] >= ws_idx:
                    continue
                direction = 1 if hp >= lp else -1
                ext_idx = max_q[0][1] if direction == 1 else min_q[0][1]
                
                # Now measure: what happens at 0s (entry), 5s, 10s, 15s, 30s, 60s after extreme?
                entry_idx = ext_idx + 1
                if entry_idx >= n: continue
                
                entry_price = ask[entry_idx] if direction == -1 else bid[entry_idx]  # fade: go opposite
                ed = -direction  # fade direction
                
                # Measure mid price at various intervals
                intervals = {}
                for label, delay in [("t+0", 0), ("t+2s", 2), ("t+5s", 5), ("t+10s", 10), ("t+15s", 15), ("t+30s", 30), ("t+60s", 60)]:
                    ts_target = ts[entry_idx] + delay
                    j = int(np.searchsorted(ts, ts_target, side="right"))
                    if j < n:
                        intervals[label] = mid[j]
                    else:
                        intervals[label] = mid[-1]
                
                # Entry and exit prices for trade simulation
                ep = entry_price
                he = int(np.searchsorted(ts, ts[entry_idx] + hold_s, side="right"))
                if he >= n: continue
                
                stop_pips = 10 * pip
                stop_price = ep - stop_pips if ed == 1 else ep + stop_pips
                stop_hit = False
                if stop_pips > 0:
                    for j in range(entry_idx + 1, he):
                        if ed == 1 and bid[j] <= stop_price:
                            stop_hit = True; break
                        if ed == -1 and ask[j] >= stop_price:
                            stop_hit = True; break
                
                xp = stop_price if stop_hit else (bid[he] if ed == 1 else ask[he])
                pnl = (xp - ep) * ed - cost
                
                # Also measure raw mid movement (no spread cost) at various holds
                hold_pnls = {}
                for label, delay in [("+2s", 2), ("+5s", 5), ("+10s", 10), ("+15s", 15), ("+30s", 30), ("+60s", 60)]:
                    ts_target = ts[entry_idx] + delay
                    j = int(np.searchsorted(ts, ts_target, side="right"))
                    if j < n:
                        hold_pnls[label] = (mid[j] - ep) * ed
                    else:
                        hold_pnls[label] = 0.0
                
                impulses.append({
                    "ws_idx": ws_idx,
                    "direction": direction,
                    "ed": ed,
                    "entry_idx": entry_idx,
                    "entry_price": entry_price,
                    "exit_price": xp,
                    "pnl": pnl,
                    "pnl_pips": pnl / pip,
                    "stop_hit": stop_hit,
                    "intervals": intervals,
                    "hold_pnls": hold_pnls,
                })
    return impulses

t0 = time.time()

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD"]
RESULTS = {}

for pair in PAIRS:
    spread_pips = 0.9 if pair == "AUDUSD" else 0.8
    pip_usd = 7.8 if pair == "USDCAD" else 10.0
    spread_price = spread_pips * PIP
    comm_price = 3.0 / pip_usd * PIP
    cost_pips = (spread_price + comm_price) / PIP
    
    ts, bid, ask = load(pair)
    mid = (bid + ask) / 2.0
    print(f"\n{'='*60}")
    print(f"{pair}: cost={cost_pips:.2f}p spread={spread_pips}p pip_usd=${pip_usd}")
    
    # Tick statistics
    price_changes = np.sum(np.diff(mid) != 0)
    total_ticks = len(mid)
    unique_prices = np.unique(mid).size
    avg_tick_move = np.mean(np.abs(np.diff(mid[mid != np.roll(mid, 1)])))
    
    print(f"  Ticks: {total_ticks:,}  Price changes: {price_changes:,} ({price_changes/total_ticks*100:.1f}%)")
    print(f"  Unique prices: {unique_prices:,}  Avg tick move: {avg_tick_move/PIP:.2f}p")
    
    # Tick density (avg ticks per second during active hours)
    time_span = ts[-1] - ts[0]
    ticks_per_sec = total_ticks / time_span
    print(f"  Timespan: {time_span/86400:.1f}d  Ticks/sec: {ticks_per_sec:.2f}")
    
    # Detect impulses and measure fade
    impulses = analyze_impulses(ts, mid, PIP, spread_price, comm_price)
    print(f"  Impulses (5p/20s): {len(impulses)}")
    
    if len(impulses) == 0:
        RESULTS[pair] = None
        continue
    
    # Trade results at 30s hold, 10p stop, FUNDEDNEXT cost
    pnls_30s = np.array([imp["pnl"] for imp in impulses])
    pnls_10s = np.array([imp["hold_pnls"].get("+10s", 0) for imp in impulses])
    pnls_15s = np.array([imp["hold_pnls"].get("+15s", 0) for imp in impulses])
    pnls_30s_raw = np.array([imp["hold_pnls"].get("+30s", 0) for imp in impulses])
    
    pips_30s = pnls_30s / PIP
    pips_10s = pnls_10s / PIP
    pips_15s = pnls_15s / PIP
    pips_30s_raw = pnls_30s_raw / PIP  # raw mid pnl, no cost
    
    wr_30s = (pips_30s > 0).mean() * 100
    wr_10s = (pips_10s > 0).mean() * 100
    wr_30s_raw = (pips_30s_raw > 0).mean() * 100
    
    print(f"  === Trade Simulation (30s hold, 10p stop, with cost) ===")
    print(f"  WR: {wr_30s:.1f}%  Avg: {pips_30s.mean():+.2f}p  Gross: {pips_30s.sum():+.0f}p")
    print(f"  Median: {np.median(pips_30s):+.2f}p  P25/P75: {np.percentile(pips_30s,25):+.2f}/{np.percentile(pips_30s,75):+.2f}p")
    
    wins = pips_30s[pips_30s > 0]
    losses = pips_30s[pips_30s <= 0]
    print(f"  Avg win: {wins.mean():+.2f}p  Avg loss: {losses.mean():+.2f}p")
    print(f"  Win% of total PnL: {wins.sum()/pips_30s.sum()*100:.0f}%" if pips_30s.sum() > 0 else "  All negative")
    
    # What WR would we need to break even?
    break_even_wr = -losses.mean() / (wins.mean() - losses.mean()) * 100 if len(losses) > 0 and len(wins) > 0 else 0
    print(f"  Break-even WR: {break_even_wr:.1f}%")
    
    # Check without cost (raw mid price)
    print(f"  === Raw mid (no cost, 30s hold) ===")
    print(f"  WR: {wr_30s_raw:.1f}%  Avg: {pips_30s_raw.mean():+.2f}p")
    print(f"  Median: {np.median(pips_30s_raw):+.2f}p  P25/P75: {np.percentile(pips_30s_raw,25):+.2f}/{np.percentile(pips_30s_raw,75):+.2f}p")
    
    # What if we had 0 spread?
    print(f"  === With only spread (no commission), 10p stop ===")
    pnls_spread_only = pips_30s_raw - spread_price / PIP
    wr_spread = (pnls_spread_only > 0).mean() * 100
    print(f"  WR: {wr_spread:.1f}%  Avg: {pnls_spread_only.mean():+.2f}p")
    
    # Check directional asymmetry
    long_pnls = [imp["pnl_pips"] for imp in impulses if imp["ed"] == 1]
    short_pnls = [imp["pnl_pips"] for imp in impulses if imp["ed"] == -1]
    if long_pnls:
        long_arr = np.array(long_pnls)
        print(f"  === Directional ===")
        print(f"  LONGs:  {len(long_arr)}t WR={(long_arr>0).mean()*100:.1f}% avg={long_arr.mean():+.2f}p")
    if short_pnls:
        short_arr = np.array(short_pnls)
        print(f"  SHORTs: {len(short_arr)}t WR={(short_arr>0).mean()*100:.1f}% avg={short_arr.mean():+.2f}p")
    
    # Also check: what's the average raw impulse move (before fade)?
    impulse_sizes = []
    for imp in impulses:
        entry_mid = mid[imp["entry_idx"]]
        extreme_mid = mid[imp["entry_idx"] - 1]  # the extreme tick
        impulse_size = abs(entry_mid - extreme_mid) / PIP
        impulse_sizes.append(impulse_size)
    impulse_arr = np.array(impulse_sizes)
    print(f"  === Impulse characteristics ===")
    print(f"  Avg impulse size: {impulse_arr.mean():.2f}p  Median: {np.median(impulse_arr):.2f}p")
    print(f"  P10/P90: {np.percentile(impulse_arr,10):.1f}/{np.percentile(impulse_arr,90):.1f}p")
    
    # How often does the price continue vs reverse in 2s, 5s, 10s?
    cont_2s = sum(1 for imp in impulses if imp["hold_pnls"].get("+2s", 0) * imp["ed"] > 0)  # price continues in same direction
    cont_5s = sum(1 for imp in impulses if imp["hold_pnls"].get("+5s", 0) * imp["ed"] > 0)
    cont_10s = sum(1 for imp in impulses if imp["hold_pnls"].get("+10s", 0) * imp["ed"] > 0)
    n_imp = len(impulses)
    print(f"  % continue (vs fade) at +2s: {cont_2s/n_imp*100:.0f}%")
    print(f"  % continue (vs fade) at +5s: {cont_5s/n_imp*100:.0f}%")
    print(f"  % continue (vs fade) at +10s: {cont_10s/n_imp*100:.0f}%")
    
    RESULTS[pair] = {
        "wr_30s": wr_30s, "avg_30s": pips_30s.mean(), "gross_30s": pips_30s.sum(),
        "wr_30s_raw": wr_30s_raw, "avg_30s_raw": pips_30s_raw.mean(),
        "wr_spread_only": wr_spread, "avg_spread_only": pnls_spread_only.mean(),
        "avg_win": wins.mean() if len(wins) else 0,
        "avg_loss": losses.mean() if len(losses) else 0,
        "impulse_size_avg": impulse_arr.mean(),
        "impulse_size_median": np.median(impulse_arr),
        "continue_2s": cont_2s/n_imp*100,
        "continue_5s": cont_5s/n_imp*100,
        "continue_10s": cont_10s/n_imp*100,
        "long_wr": (np.array(long_pnls)>0).mean()*100 if long_pnls else 0,
        "short_wr": (np.array(short_pnls)>0).mean()*100 if short_pnls else 0,
        "long_avg": np.array(long_pnls).mean() if long_pnls else 0,
        "short_avg": np.array(short_pnls).mean() if short_pnls else 0,
        "n_trades": len(impulses),
        "ticks_per_sec": ticks_per_sec,
    }

print(f"\n{'='*60}")
print("SUMMARY: WHY ONLY EURUSD?")
print(f"{'='*60}")
print(f"  {'Pair':<10s} {'n':>5s} {'WR%':>5s} {'Avg(p)':>8s} {'RawAvg':>7s} {'OnlySprd':>9s} {'ImpAvg':>7s} {'Cont2s':>6s} {'Cont5s':>6s} {'LongWR':>7s} {'ShortWR':>8s}")
print(f"  {'-'*78}")
for pair in PAIRS:
    r = RESULTS[pair]
    if r:
        print(f"  {pair:<10s} {r['n_trades']:>5d} {r['wr_30s']:>5.1f}% {r['avg_30s']:>+8.2f}p {r['avg_30s_raw']:>+7.2f}p {r['avg_spread_only']:>+8.2f}p {r['impulse_size_avg']:>6.1f}p {r['continue_2s']:>5.0f}% {r['continue_5s']:>5.0f}% {r['long_wr']:>6.0f}% {r['short_wr']:>7.0f}%")

print(f"\nTotal: {time.time()-t0:.1f}s")
