"""Find what signals a bad week for the impulse fade strategy.

Compares W28 (losing week) vs profitable weeks on FundedNext data.
Looks for detectable signals: tick rate, spread, volatility, etc.
"""
import sys, numpy as np
from collections import deque, defaultdict
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\fundednext_ticks")
PIP = 0.0001
PIP_USD = 10.0
COST = 0.8 * PIP + 3.0 / PIP_USD * PIP

def load(pair):
    d = np.load(str(TICK_DIR / (pair + ".npy")))
    ts = np.array([t[0] for t in d], dtype=np.int64)
    bid = np.array([t[1] for t in d], dtype=np.float64)
    ask = np.array([t[2] for t in d], dtype=np.float64)
    return ts, bid, ask

def detect_with_info(ts, mid):
    """Detect impulses and return both events and rolling stats."""
    n = len(mid)
    min_q, max_q = deque(), deque()
    ws_idx = 0
    evs = []  # (ws_idx, ext_idx, ext_dir, entry_time)
    stats_by_hour = defaultdict(lambda: {"ticks": 0, "impulses": 0, "hp_sum": 0, "lp_sum": 0, "hp_cnt": 0, "lp_cnt": 0})
    
    for i in range(n):
        v = float(mid[i])
        while min_q and min_q[-1][0] >= v: min_q.pop()
        while max_q and max_q[-1][0] <= v: max_q.pop()
        min_q.append((v, i)); max_q.append((v, i))
        while ts[i] - ts[ws_idx] > 20:
            if min_q and min_q[0][1] == ws_idx: min_q.popleft()
            if max_q and max_q[0][1] == ws_idx: max_q.popleft()
            ws_idx += 1
        
        hour = datetime.fromtimestamp(ts[i], tz=timezone.utc).hour
        stats_by_hour[hour]["ticks"] += 1
        
        if i > ws_idx:
            wp = mid[ws_idx]
            hp = float(max_q[0][0] - wp)
            lp = float(wp - min_q[0][0])
            span = ts[i] - ts[ws_idx]
            
            stats_by_hour[hour]["hp_sum"] += hp
            stats_by_hour[hour]["hp_cnt"] += 1 if hp > 0 else 0
            stats_by_hour[hour]["lp_sum"] += lp
            stats_by_hour[hour]["lp_cnt"] += 1 if lp > 0 else 0
            
            if span <= 20 and (hp >= 5*PIP or lp >= 5*PIP):
                if evs and evs[-1][0] >= ws_idx: continue
                ext_idx = max_q[0][1] if hp >= lp else min_q[0][1]
                d = 1 if hp >= lp else -1
                evs.append((ws_idx, ext_idx, d, ts[i]))
                stats_by_hour[hour]["impulses"] += 1
    
    # Convert hourly stats to averaged
    avg_stats = {}
    for hr, s in stats_by_hour.items():
        avg_stats[hr] = {
            "ticks": s["ticks"],
            "impulses": s["impulses"],
            "avg_hp": s["hp_sum"] / max(s["hp_cnt"], 1),
            "avg_lp": s["lp_sum"] / max(s["lp_cnt"], 1),
            "impulse_rate": s["impulses"] / max(s["ticks"], 1) * 1000,  # per 1000 ticks
        }
    
    return evs, avg_stats, stats_by_hour

def sim_one(ev_list, ts, bid, ask):
    """Simulate one trade at a time, returning PnL + entry_time."""
    results = []
    for ws_i, ext_i, ext_dir, entry_t in ev_list:
        ed = -ext_dir
        ei2 = ext_i + 1
        if ei2 >= len(bid) - 1: continue
        ep = ask[ei2] if ed == 1 else bid[ei2]
        et = ts[ei2]
        he = int(np.searchsorted(ts, et + 30, side="right"))
        if he >= len(bid): continue
        sp = ep - 10*PIP if ed == 1 else ep + 10*PIP
        hit = False
        for j in range(ei2 + 1, he):
            if ed == 1 and bid[j] <= sp: hit = True; break
            if ed == -1 and ask[j] >= sp: hit = True; break
        xp = sp if hit else (bid[he] if ed == 1 else ask[he])
        pnl = (xp - ep) * ed - COST
        results.append((entry_t, pnl / PIP))  # (timestamp, pnl_in_pips)
    return results

print("Loading EURUSD...")
ts, bid, ask = load("EURUSD")
mid = (bid + ask) / 2.0

print("Detecting with hourly stats...")
evs, hourly_stats, raw_hourly = detect_with_info(ts, mid)
print(f"  {len(evs)} events")

print("Simulating trades...")
trades = sim_one(evs, ts, bid, ask)  # (timestamp, pnl_pips)

# Group by ISO week
from collections import OrderedDict
weeks = OrderedDict()
for t, pnl in trades:
    d = datetime.fromtimestamp(t, tz=timezone.utc).date()
    iso = d.isocalendar()
    wk = f"{iso[0]}-W{iso[1]:02d}"
    weeks.setdefault(wk, []).append((t, pnl))

# Get daily tick counts
from collections import defaultdict
daily_ticks = defaultdict(int)
for i, t in enumerate(ts):
    daily_ticks[datetime.fromtimestamp(t, tz=timezone.utc).date()] += 1

print(f"\n{'='*80}")
print("WEEKLY ANALYSIS")
print(f"{'='*80}")
for wk in sorted(weeks.keys()):
    arr = [x[1] for x in weeks[wk]]
    arr_np = np.array(arr)
    wr = (arr_np > 0).mean() * 100
    avg = arr_np.mean()
    gross = arr_np.sum()
    n = len(arr)
    
    # Daily stats for this week
    wk_dates = sorted(set(datetime.fromtimestamp(x[0], tz=timezone.utc).date() for x in weeks[wk]))
    dr = f"{wk_dates[0].strftime('%m/%d')}-{wk_dates[-1].strftime('%m/%d')}" if wk_dates else "?"
    days_str = ", ".join(d.strftime('%a %m/%d') for d in wk_dates)
    
    # Tick counts per day in this week
    tick_rates = [daily_ticks.get(d, 0) for d in wk_dates if d in daily_ticks]
    avg_ticks_day = np.mean(tick_rates) if tick_rates else 0
    
    print(f"\n{wk} ({dr}): {n:>3d}t WR={wr:.1f}% avg={avg:+.2f}p gross={gross:+.0f}p")
    print(f"  Days: {days_str}")
    print(f"  Avg ticks/day: {avg_ticks_day:.0f}")
    
    # Hourly impulse rate
    wk_hours = defaultdict(int)
    wk_ticks = defaultdict(int)
    for t, pnl in weeks[wk]:
        hr = datetime.fromtimestamp(t, tz=timezone.utc).hour
        wk_hours[hr] += 1
    for d in wk_dates:
        for hr in range(24):
            day_start = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
            hr_start = day_start + hr * 3600
            hr_end = hr_start + 3600
            tick_count = np.sum((ts >= hr_start) & (ts < hr_end))
            wk_ticks[hr] += tick_count
    
    # Show hourly impulse rates
    print(f"  Hour | Impulses | Ticks | Rate/1K")
    all_hours = sorted(set(wk_hours.keys()) | set(wk_ticks.keys()))
    for hr in all_hours:
        imp = wk_hours.get(hr, 0)
        tck = wk_ticks.get(hr, 0)
        rate = imp / max(tck, 1) * 1000
        print(f"  {hr:>4d}: {imp:>9d} {tck:>7,d} {rate:>7.2f}")

# ═══ KEY FINDER: Compare losing vs winning periods ═══
print(f"\n{'='*80}")
print("LOSING VS WINNING PERIOD COMPARISON")
print(f"{'='*80}")

# Split trades by time of day
hourly_pnl = defaultdict(list)
for t, pnl in trades:
    hr = datetime.fromtimestamp(t, tz=timezone.utc).hour
    hourly_pnl[hr].append(pnl)

print("\nHourly performance (FundedNext, all weeks):")
print(f"  Hour | n | WR | Avg(p)")
for hr in sorted(hourly_pnl.keys()):
    arr = np.array(hourly_pnl[hr])
    wr = (arr > 0).mean() * 100
    avg = arr.mean()
    n = len(arr)
    print(f"  {hr:>4d}: {n:>4d} {wr:>5.1f}% {avg:>+7.2f}p")

# Check if losing week has same tick density but different outcome
print(f"\n{'='*80}")
print(f"TICK DENSITY BY WEEK (avg ticks per second during active hours)")
print(f"{'='*80}")

for wk in sorted(weeks.keys()):
    wk_trades = weeks[wk]
    if not wk_trades: continue
    first_t = min(x[0] for x in wk_trades)
    last_t = max(x[0] for x in wk_trades)
    span = last_t - first_t
    tick_count = np.sum((ts >= first_t - 3600) & (ts <= last_t + 3600))  # buffer
    tps = tick_count / max(span, 1)
    n = len(wk_trades)
    arr = np.array([x[1] for x in wk_trades])
    avg = arr.mean()
    print(f"  {wk}: {tps:.2f} ticks/s  ({tick_count:,} ticks in {span/3600:.0f}h)  n={n} avg={avg:+.2f}p")

# ═══ WHAT ABOUT ROLLING CONSECUTIVE LOSSES? ═══
print(f"\n{'='*80}")
print("CONSECUTIVE LOSS ANALYSIS - Can streaks predict bad periods?")
print(f"{'='*80}")

# Split into sequential chunks of 20 trades
pnl_arr = np.array([x[1] for x in trades])
chunk_size = 20
for i in range(0, len(pnl_arr), chunk_size):
    chunk = pnl_arr[i:i+chunk_size]
    if len(chunk) < chunk_size: continue
    wr = (chunk > 0).mean() * 100
    avg = chunk.mean()
    gross = chunk.sum()
    start_t = trades[i][0]
    start_d = datetime.fromtimestamp(start_t, tz=timezone.utc).strftime('%m/%d %H:%M')
    print(f"  Trades {i:>4d}-{i+chunk_size-1:<4d} ({start_d}): WR={wr:>5.1f}% avg={avg:>+.2f}p gross={gross:>+.0f}p")
