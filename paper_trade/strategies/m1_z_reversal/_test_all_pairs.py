"""Test impulse fade on EURUSD, EURJPY, GBPJPY with FundedNext costs."""
import sys, time, numpy as np, pandas as pd
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")

# FundedNext Server 3 verified spreads
FN_SPREADS = {"EURUSD": 8, "EURJPY": 14, "GBPJPY": 14, "GBPUSD": 8, "USDJPY": 9}
PIP_VALUES = {"EURUSD": 0.0001, "EURJPY": 0.01, "GBPJPY": 0.01}
PIP_USD = {"EURUSD": 10.0, "EURJPY": 6.45, "GBPJPY": 6.2}  # at current rates

def load(pair, months):
    dfs = []
    for y, m in months:
        p = TICK_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts":str,"B":np.float64,"A":np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
                                 format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    t["ts_s"] = t["Ts"].astype(np.int64) // 10**9
    return t

def detect(pair, ticks, pip):
    """Detect impulse events: 5-pip move in 20-sec window."""
    mid = ((ticks["B"].values + ticks["A"].values) / 2.0).astype(np.float64)
    ts = ticks["ts_s"].values.astype(np.int64); n = len(mid)
    min_q = deque(); max_q = deque(); ws_idx = 0; evs = []
    thresh = 5 * pip
    for i in range(n):
        v = float(mid[i])
        while min_q and min_q[-1][0] >= v: min_q.pop()
        while max_q and max_q[-1][0] <= v: max_q.pop()
        min_q.append((v, i)); max_q.append((v, i))
        while ts[i] - ts[ws_idx] > 20:
            if min_q and min_q[0][1] == ws_idx: min_q.popleft()
            if max_q and max_q[0][1] == ws_idx: max_q.popleft()
            ws_idx += 1
        if i > ws_idx:
            wp = mid[ws_idx]; hp = float(max_q[0][0] - wp); lp = float(wp - min_q[0][0])
            if ts[i] - ts[ws_idx] <= 20 and (hp >= thresh or lp >= thresh):
                if evs and evs[-1][0] >= ws_idx: continue
                d = 1 if hp >= lp else -1
                ext_idx = max_q[0][1] if d == 1 else min_q[0][1]
                evs.append((ws_idx, ext_idx, d))
    return evs

def sim(ev_list, ticks, pip, cost, hold_s, stop_pips, direction="both"):
    bid = ticks["B"].values.astype(np.float64)
    ask = ticks["A"].values.astype(np.float64)
    ts = ticks["ts_s"].values.astype(np.int64); n = len(ts)
    pnls = []
    for ws_i, ext_i, ext_dir in ev_list:
        ed = -ext_dir
        if direction == "short" and ed == 1: continue
        if direction == "long" and ed == -1: continue
        ei2 = ext_i + 1
        if ei2 >= n - 1: continue
        ep = ask[ei2] if ed == 1 else bid[ei2]
        et = ts[ei2]
        he = int(np.searchsorted(ts, et + hold_s, side="right"))
        if he >= n: continue
        sp = ep - stop_pips * pip if ed == 1 else ep + stop_pips * pip
        hit = False
        if stop_pips > 0:
            for j in range(ei2 + 1, he):
                if (ed == 1 and bid[j] <= sp) or (ed == -1 and ask[j] >= sp):
                    hit = True; break
        xp = sp if hit else (bid[he] if ed == 1 else ask[he])
        pnl = (xp - ep) * ed - cost
        pnls.append(pnl)
    return np.array(pnls, dtype=np.float64)

def calc_fn_cost(pair, pip, fn_spread_points, comm_usd=3.0):
    """Calculate equivalent cost for FundedNext at 1.0 lot."""
    spread_price = fn_spread_points * pip / 10  # 10 points = 1 pip
    # Commission in price units: $3 / (contract * pip_value_usd * pip)
    pip_val_usd = PIP_USD[pair]
    comm_price = comm_usd / (100000 * pip_val_usd / pip)  # comm as fraction of price
    # Wait, simplify: comm_price = comm_usd * pip / pip_val_usd / contract
    comm_price = comm_usd / (100000 / pip) * pip / pip_val_usd
    # Actually: at 1 lot, 1 pip = pip_val_usd dollars. 
    # comm $3 = 3/pip_val_usd pips = 3/pip_val_usd * pip (in price)
    comm_price = comm_usd / pip_val_usd * pip
    return spread_price + comm_price

# Run all pairs
pairs_config = {
    "EURUSD": {
        "pip": 0.0001, "detect_pips": 5, "window_s": 20, "hold_s": 30,
        "stop_pips": 5, "direction": "both", "months": [(2025,10),(2025,11),(2025,12)],
        "fn_points": 8, "exness_cost": 0.00003,
    },
    "EURJPY": {
        "pip": 0.01, "detect_pips": 10, "window_s": 20, "hold_s": 30,
        "stop_pips": 7, "direction": "short", "months": [(2025,10),(2025,11),(2025,12)],
        "fn_points": 14, "exness_cost": 0.006,
    },
    "GBPJPY": {
        "pip": 0.01, "detect_pips": 10, "window_s": 20, "hold_s": 30,
        "stop_pips": 7, "direction": "both", "months": [(2025,10),(2025,11),(2025,12)],
        "fn_points": 14, "exness_cost": 0.006,
    },
}

DAYS = 65
t0 = time.time()

for pair, cfg in pairs_config.items():
    pip = cfg["pip"]
    detect_pips = cfg.get("detect_pips", 5)
    window_s = cfg.get("window_s", 20)
    hold_s = cfg.get("hold_s", 30)
    stop_pips = cfg.get("stop_pips", 5)
    direction = cfg.get("direction", "both")
    fn_points = cfg["fn_points"]
    exness_cost = cfg["exness_cost"]
    pip_usd = PIP_USD[pair]

    # Exness cost in price units
    cost_ex = exness_cost

    # FundedNext cost in price units
    # FN spread = fn_points / 10 pips
    fn_spread_price = fn_points * pip / 10
    # FN commission: $3/lot per trade
    fn_comm_price = 3.0 / pip_usd * pip
    cost_fn = fn_spread_price + fn_comm_price

    print(f"\n{'='*60}")
    print(f"{pair}: Exness cost={cost_ex:.6f}  FN cost={cost_fn:.6f} "
          f"(spread={fn_spread_price:.6f} comm={fn_comm_price:.6f})")
    print(f"  Detect: {detect_pips}p/{window_s}s  Hold: {hold_s}s  Stop: {stop_pips}p  Dir: {direction}")

    ticks = load(pair, cfg["months"])
    print(f"  {len(ticks):,} ticks loaded")
    
    evs = detect(pair, ticks, pip)
    print(f"  {len(evs)} events detected")

    for label, cost in [("Exness", cost_ex), ("FundedNext", cost_fn)]:
        pnls = sim(evs, ticks, pip, cost, hold_s, stop_pips, direction)
        if len(pnls) == 0:
            print(f"  {label}: NO TRADES")
            continue
        p = pnls / pip  # in pips
        n = len(p)
        wr = (p > 0).mean() * 100
        avg = p.mean()
        gross = p.sum()
        wins = p[p > 0]; losses = p[p <= 0]
        aw = wins.mean() if len(wins) else 0
        al = losses.mean() if len(losses) else 0
        worst_3 = sum(sorted(p)[:3])

        print(f"  {label}: {n}t ({n/DAYS:.1f}/d) WR={wr:.1f}% avg={avg:+.2f}p "
              f"gross={gross:+.0f}p W={aw:+.2f}p L={al:+.2f}p")
        print(f"    Worst 3 trades: {worst_3:.2f}p  MaxCL: calc'd")

        # What pass rate at 1.0 and 1.5 lots?
        daily_pnl_per_lot = gross / DAYS * pip_usd  # $/day at 1 lot
        print(f"    Daily PnL at 1.0 lot: ${daily_pnl_per_lot:.0f}/day")
        
        # Quick Monte Carlo
        TRADES_PER_DAY = int(n / DAYS)
        for lot in [0.75, 1.0, 1.5, 2.0, 2.5]:
            N = 10000
            n_pass = 0
            for _ in range(N):
                trades = np.random.choice(p, size=TRADES_PER_DAY * 5, replace=True)
                day_chunks = trades.reshape(5, TRADES_PER_DAY)
                day_pnls_usd = day_chunks.sum(axis=1) * lot * pip_usd
                if day_pnls_usd.sum() >= 2000 and day_pnls_usd.min() > -1250:
                    n_pass += 1
            pass_pct = n_pass / N * 100
            blow = (np.random.choice(p, size=(N, TRADES_PER_DAY * 5), replace=True)
                    .reshape(N, 5, TRADES_PER_DAY).sum(axis=2).min(axis=1) * lot * pip_usd < -1250).mean() * 100
            print(f"    {lot:.1f} lot: {pass_pct:.1f}% pass  blow={blow:.1f}%")

print(f"\nTotal time: {time.time()-t0:.1f}s")
