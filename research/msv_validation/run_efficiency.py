"""Session-specific optimization and the REAL efficiency frontier.
Testing: what's the max achievable WR at different trade frequencies?
"""

import sys, os, numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque

project_root = str(Path(__file__).resolve().parents[2])
os.chdir(project_root)
cd_root = os.path.join(project_root, "currency_decomposition")
sys.path.insert(0, cd_root)

import MetaTrader5 as mt5
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

from config.settings import BASE_CURRENCY_MAP
ALL_PAIRS = list(BASE_CURRENCY_MAP.keys())[:15]

def load_data():
    end = datetime.now()
    start = end - timedelta(days=120)
    all_data = {}
    for pair in ALL_PAIRS:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    return all_data

def backtest(all_data, top_n=3, lookback=3, hold=3, max_pos=3,
             sessions="all", vol_filter_asia=True, min_move_bp=0,
             bias_asia="long", bias_ny="short", exclude_pairs=None,
             hold_ny=None):
    N = min(len(v) for v in all_data.values())
    atr_window = deque(maxlen=288)
    positions = {}
    trades = []
    
    hold_ny = hold_ny or hold
    
    for idx in range(lookback, N - max(hold, hold_ny)):
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        
        if sessions == "asia" and hour >= 7: continue
        if sessions == "ny" and not (16 <= hour < 24): continue
        if sessions == "asia_ny" and (7 <= hour < 16): continue
        
        atr = 0.0
        for p in ALL_PAIRS:
            hi = float(all_data[p][idx]["high"])
            lo = float(all_data[p][idx]["low"])
            pc = float(all_data[p][idx - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            atr += tr / float(all_data[p][idx]["close"])
        atr /= len(ALL_PAIRS)
        atr_window.append(atr)
        
        if vol_filter_asia and hour < 7:
            if len(atr_window) >= 30:
                thresh = sorted(atr_window)[2 * len(atr_window) // 3]
                if atr <= thresh: continue
        
        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        if len(positions) >= max_pos: continue
        
        pair_moves = []
        for p in ALL_PAIRS:
            if p in positions: continue
            if exclude_pairs and p in exclude_pairs: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            if min_move_bp > 0 and abs(ret * 10000) < min_move_bp: continue
            pair_moves.append((p, abs(ret), ret))
        
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        
        h = hold if hour < 16 else hold_ny
        
        for p, mag, ret in pair_moves[:top_n]:
            if p in positions: continue
            if len(positions) >= max_pos: break
            
            if hour < 7:  # Asia
                if bias_asia == "long" and ret > 0: continue
                if bias_asia == "long_strong" and ret > -0.0003: continue
                direction = 1
            elif hour >= 16:  # NY
                if bias_ny == "short" and ret < 0: continue
                if bias_ny == "short_strong" and ret < 0.0003: continue
                direction = -1
            else:  # London
                direction = 1 if ret < 0 else -1
                if abs(ret) < 0.0005: continue
            
            if idx + 1 + h >= N: continue
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + h]["close"])
            pnl = direction * (exit_ / entry - 1) if entry > 0 else 0
            won = pnl > 0
            trades.append({"pnl": pnl * 10000, "won": won, "hour": hour, "pair": p})
            positions[p] = idx + h
    
    return trades

def stats(trades):
    if not trades: return {"n":0,"wr":0,"mean_bp":0,"mean_usd":0,"t":0,"n_day":0}
    pnls = [t["pnl"] for t in trades]
    mu, s = float(np.mean(pnls)), float(np.std(pnls))
    wr = sum(1 for t in trades if t["won"]) / len(trades) * 100
    t = mu / (s / np.sqrt(len(trades))) if s > 0 else 0
    return {"n":len(trades),"wr":wr,"mean_bp":mu,"mean_usd":mu*10,"t_stat":t}

def main():
    all_data = load_data()
    
    print(f"\n{'='*70}")
    print("SESSION-SPECIFIC OPTIMIZATION")
    print("=" * 70)
    
    # ── ASIA SWEEP ──
    print(f"\n  ASIA SESSION SWEEP:")
    print(f"  {'Config':>30s} {'n':>5s} {'WR':>6s} {'Mean':>8s} {'$/trade':>8s} {'t':>7s}")
    print(f"  {'-'*67}")
    
    asia_configs = [
        ("Asia 15min hold T3", dict(sessions="asia", hold=3, top_n=3)),
        ("Asia 15min hold T2", dict(sessions="asia", hold=3, top_n=2)),
        ("Asia 15min hold T1", dict(sessions="asia", hold=3, top_n=1)),
        ("Asia 10min T3", dict(sessions="asia", hold=2, top_n=3)),
        ("Asia 10min T2", dict(sessions="asia", hold=2, top_n=2)),
        ("Asia 10min T1", dict(sessions="asia", hold=2, top_n=1)),
        ("Asia 5min T3", dict(sessions="asia", hold=1, top_n=3)),
        ("Asia 5min T2", dict(sessions="asia", hold=1, top_n=2)),
        ("Asia 5min T1", dict(sessions="asia", hold=1, top_n=1)),
        ("Asia T3 strong decl", dict(sessions="asia", hold=3, top_n=3, bias_asia="long_strong")),
        ("Asia T2 strong decl", dict(sessions="asia", hold=3, top_n=2, bias_asia="long_strong")),
        ("Asia T1 0.3bp min", dict(sessions="asia", hold=3, top_n=1, min_move_bp=0.3)),
        ("Asia T2 0.5bp min", dict(sessions="asia", hold=3, top_n=2, min_move_bp=0.5)),
    ]
    
    for label, params in asia_configs:
        trades = backtest(all_data, **params)
        s = stats(trades)
        print(f"  {label:>30s}: {s['n']:5d}  {s['wr']:5.1f}%  {s['mean_bp']:>+6.2f}bp  ${s['mean_usd']:>+5.1f}  {s['t_stat']:>+6.2f}")
    
    # ── NY SESSION SWEEP ──
    print(f"\n  NY SESSION SWEEP:")
    for label, params in [
        ("NY 15min T3", dict(sessions="ny", hold=3, top_n=3)),
        ("NY 15min T2", dict(sessions="ny", hold=3, top_n=2)),
        ("NY 15min T1", dict(sessions="ny", hold=3, top_n=1)),
        ("NY 10min T3", dict(sessions="ny", hold_ny=2, hold=2, top_n=3)),
        ("NY 10min T2", dict(sessions="ny", hold_ny=2, hold=2, top_n=2)),
        ("NY 10min T1", dict(sessions="ny", hold_ny=2, hold=2, top_n=1)),
        ("NY 5min T3", dict(sessions="ny", hold_ny=1, hold=1, top_n=3)),
        ("NY 5min T2", dict(sessions="ny", hold_ny=1, hold=1, top_n=2)),
        ("NY 5min T1", dict(sessions="ny", hold_ny=1, hold=1, top_n=1)),
        ("NY T3 strong rise", dict(sessions="ny", hold=3, top_n=3, bias_ny="short_strong")),
        ("NY T2 strong rise", dict(sessions="ny", hold=3, top_n=2, bias_ny="short_strong")),
        ("NY T1 0.3bp min", dict(sessions="ny", hold=3, top_n=1, min_move_bp=0.3)),
    ]:
        trades = backtest(all_data, **params)
        s = stats(trades)
        print(f"  {label:>30s}: {s['n']:5d}  {s['wr']:5.1f}%  {s['mean_bp']:>+6.2f}bp  ${s['mean_usd']:>+5.1f}  {s['t_stat']:>+6.2f}")
    
    # ── EFFICIENCY FRONTIER ──
    print(f"\n{'='*70}")
    print("EFFICIENCY FRONTIER — every config plotted")
    print("=" * 70)
    print(f"\n  {'Config':>35s} {'n/day':>6s} {'WR':>6s} {'$/trade':>8s}")
    print(f"  {'-'*57}")
    
    # Collect all configs
    all_configs = []
    
    # Asia only — varying strictness
    for top_n in [3, 2, 1]:
        for hold_b in [1, 2, 3]:
            for min_move in [0, 0.3, 0.5]:
                for bias in ["long", "long_strong"]:
                    all_configs.append((
                        f"Asia T{top_n} H{hold_b}" + (f" MM{min_move}" if min_move else "") + (" strong" if bias=="long_strong" else ""),
                        dict(sessions="asia", top_n=top_n, hold=hold_b, min_move_bp=min_move, bias_asia=bias)))
    
    # NY only
    for top_n in [3, 2, 1]:
        for hold_b in [1, 2, 3]:
            for min_move in [0, 0.3]:
                for bias in ["short", "short_strong"]:
                    all_configs.append((
                        f"NY T{top_n} H{hold_b}" + (f" MM{min_move}" if min_move else "") + (" strong" if bias=="short_strong" else ""),
                        dict(sessions="ny", top_n=top_n, hold=hold_b, hold_ny=hold_b, min_move_bp=min_move, bias_ny=bias)))
    
    # Asia+NY combined
    for top_n in [3, 2]:
        for hold_asia in [2, 3]:
            for hold_ny in [1, 2]:
                all_configs.append((
                    f"A+T{top_n}H{hold_asia}_NY+T{top_n}H{hold_ny}",
                    dict(sessions="asia_ny", top_n=top_n, hold=hold_asia, hold_ny=hold_ny)))
    
    # Run and sort by WR
    frontier = []
    for label, params in all_configs:
        trades = backtest(all_data, **params)
        s = stats(trades)
        if s["n"] >= 50:
            frontier.append((s["wr"], s["n"]/85, label, s["mean_usd"]))
    
    frontier.sort(key=lambda x: -x[0])
    
    for wr, nday, label, usd in frontier[:30]:
        print(f"  {label:>35s}:  {nday:5.0f}  {wr:5.1f}%  ${usd:>+7.1f}")
    
    # ── BEST ASIA ONLY WALK-FORWARD ──
    print(f"\n{'='*70}")
    print("WALK-FORWARD: Best Asia config (T2, H=3, strong decl)")
    print("=" * 70)
    
    N = min(len(v) for v in all_data.values())
    for wf_idx, (sp, ep) in enumerate([(0, 0.5), (0.25, 0.75), (0.5, 1.0)]):
        sub = {}
        si, ei = int(N * sp), int(N * ep)
        for p in all_data:
            sub[p] = all_data[p][si:ei]
        trades = backtest(sub, sessions="asia", top_n=2, hold=3, bias_asia="long", vol_filter_asia=True)
        s = stats(trades)
        print(f"  WF{wf_idx+1} ({sp:.0%}-{ep:.0%}):  n={s['n']:4d}  wr={s['wr']:5.1f}%  mean={s['mean_bp']:>+5.2f}bp  t={s['t_stat']:>+5.2f}")
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
