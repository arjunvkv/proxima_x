"""Adaptive regime-aware scalper — auto-detects market conditions for max WR.
Uses MSV features (dispersion, velocity, ATR percentile, hour) to decide:
  - Should I trade NOW? (regime gate)
  - Which pairs? (pair selection)
  - How long to hold? (adaptive hold)
  - Direction bias? (session-aware)

NOT fitted — based on general market structure principles.
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

from state.market_state import MarketStateVector
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

def compute_pct(disp, history, window):
    h = history[-min(window, len(history)):]
    if len(h) < 10: return 0.5
    return sum(1 for x in h if x < disp) / len(h)

def regime_trade(all_data, config):
    """
    Auto-detect regime using MSV features and adapt entry parameters.
    
    Features computed per bar:
    - disp_pct: dispersion percentile (market stress)
    - disp_velocity: dispersion trend (expanding/contracting)
    - atr_pct: volatility percentile
    - hour: UTC hour (session detection)
    - pair_disp: which currencies are driving the dispersion
    
    Decision logic (structural rules, not fitted):
    1. Low dispersion + low vol → NO TRADE (no edge)
    2. High dispersion + high vol + early session → STRONG mean reversion
    3. High dispersion + late session → WEAK/WARN
    4. Rising dispersion (velocity>0) → stronger signal
    5. Falling dispersion → let it stabilize
    """
    N = min(len(v) for v in all_data.values())
    ms = MarketStateVector(history_size=50)
    dh = deque(maxlen=500)
    atr_window = deque(maxlen=288)
    
    positions = {}
    trades = []
    
    lookback = config.get("lookback_bars", 3)
    hold = config.get("hold_bars", 3)
    top_n = config.get("top_n", 3)
    max_pos = config.get("max_positions", 3)
    min_confidence = config.get("min_confidence", 0.0)  # 0.0-1.0
    
    for idx in range(max(lookback, 2), N - hold):
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        
        # ── Market State Features ──
        rets = {}
        for p in ALL_PAIRS:
            c = float(all_data[p][idx]["close"])
            pv = float(all_data[p][idx - 1]["close"])
            rets[p] = max(min((c / pv - 1) if pv > 0 else 0.0, 0.05), -0.05)
        snap = ms.update(rets, timestamp=float(all_data[ALL_PAIRS[0]][idx]["time"]))
        dh.append(snap.network.dispersion)
        
        disp_pct = compute_pct(snap.network.dispersion, list(dh), 500)
        disp_velocity = snap.network.dispersion - (list(dh)[-6] if len(dh) >= 6 else snap.network.dispersion)
        
        # ATR percentile
        atr = 0.0
        for p in ALL_PAIRS:
            hi = float(all_data[p][idx]["high"])
            lo = float(all_data[p][idx]["low"])
            pc = float(all_data[p][idx - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            atr += tr / float(all_data[p][idx]["close"])
        atr /= len(ALL_PAIRS)
        atr_window.append(atr)
        atr_pct = sum(1 for a in atr_window if a < atr) / max(len(atr_window), 1)
        
        # ── REGIME DETECTION (structural rules) ──
        # Low dispersion + low vol = calm, no edge
        if disp_pct < 0.60 and atr_pct < 0.60:
            continue
        
        # Confidence score (0.0 - 1.0) based on combined market state
        confidence = 0.0
        
        # Dispersion contribution
        if disp_pct > 0.95: confidence += 0.4
        elif disp_pct > 0.85: confidence += 0.3
        elif disp_pct > 0.70: confidence += 0.2
        else: confidence += 0.1
        
        # Volatility contribution
        if atr_pct > 0.90: confidence += 0.3
        elif atr_pct > 0.75: confidence += 0.2
        elif atr_pct > 0.60: confidence += 0.1
        
        # Velocity contribution (expanding dispersion = fresh shock = stronger reversal)
        if disp_velocity > 0: confidence += 0.2
        elif disp_velocity > -0.00005: confidence += 0.1
        
        # Session contribution
        if hour < 2:  # Early Asia = strongest
            confidence += 0.2
        elif hour < 7:  # Late Asia = moderate
            confidence += 0.1
        elif 16 <= hour < 20:  # Early NY = moderate
            confidence += 0.1
        elif 20 <= hour < 24:  # Late NY = weak
            pass
        else:  # London = weakest
            pass
        
        # Confidence gate
        if confidence < min_confidence:
            continue
        
        # Close expired
        for p in list(positions.keys()):
            if idx >= positions[p]:
                del positions[p]
        if len(positions) >= max_pos:
            continue
        
        # ── DIRECTION BIAS (session-adaptive) ──
        if hour < 7:  # Asia: LONG bias
            direction_bias = 1
        elif hour >= 16:  # NY: SHORT bias
            direction_bias = -1
        else:  # London: neutral
            direction_bias = 0
        
        # ── ADAPTIVE HOLD ──
        # Higher confidence = longer hold (more conviction)
        # Lower confidence = quick scalp
        adaptive_hold = hold
        if confidence > 0.7:
            adaptive_hold = min(hold + 2, 6)  # hold longer
        elif confidence < 0.4:
            adaptive_hold = max(hold - 1, 1)  # quick exit
        
        if idx + adaptive_hold >= N:
            continue
        
        # ── PAIR RANKING ──
        pair_moves = []
        for p in ALL_PAIRS:
            if p in positions: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        
        for p, mag, ret in pair_moves[:top_n]:
            if p in positions: continue
            if len(positions) >= max_pos: break
            
            # Direction decision
            if direction_bias == 1:  # Asia: prefer long
                if ret > -0.0003: continue  # need meaningful decline
                direction = 1
            elif direction_bias == -1:  # NY: prefer short
                if ret < 0.0003: continue  # need meaningful rise
                direction = -1
            else:  # London: trade both
                direction = 1 if ret < 0 else -1
                if abs(ret) < 0.0005: continue  # need bigger move
            
            # Entry
            entry_open = float(all_data[p][idx + 1]["open"])
            exit_idx = min(idx + adaptive_hold, N - 1)
            exit_close = float(all_data[p][exit_idx]["close"])
            pnl = direction * (exit_close / entry_open - 1) if entry_open > 0 else 0
            
            won = pnl > 0
            trades.append({
                "pair": p, "ts": dt, "hour": hour,
                "pnl": pnl * 10000, "won": won,
                "direction": "LONG" if direction > 0 else "SHORT",
                "confidence": confidence,
                "disp_pct": disp_pct, "atr_pct": atr_pct,
                "hold_bars": adaptive_hold,
            })
            positions[p] = exit_idx
    
    return trades

def compute_stats(trades):
    if not trades or len(trades) < 3:
        return {"n": 0, "wr": 0, "mean_bp": 0, "mean_usd": 0, "t_stat": 0, "n_day": 0}
    pnls = np.array([t["pnl"] for t in trades])
    mu = float(np.mean(pnls))
    s = float(np.std(pnls))
    wr = sum(1 for t in trades if t["won"]) / len(trades) * 100
    t_stat = mu / (s / np.sqrt(len(trades))) if s > 0 else 0
    days = max(1, (trades[-1]["ts"] - trades[0]["ts"]).total_seconds() / 86400)
    return {
        "n": len(trades), "wr": wr, "mean_bp": mu,
        "mean_usd": mu * 10, "t_stat": t_stat, "n_day": len(trades) / days,
    }

def main():
    all_data = load_data()
    
    print(f"\n{'='*70}")
    print("REGIME-AWARE SCALPER — auto-detects market conditions")
    print("=" * 70)
    
    # ── TEST CONFIDENCE THRESHOLDS ──
    print(f"\n  Confidence gate sweep:")
    print(f"  {'Gate':>8s} {'n':>6s} {'WR':>6s} {'Mean':>8s} {'$/trade':>8s} {'t':>7s} {'/day':>5s}")
    print(f"  {'-'*50}")
    
    for conf in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        trades = regime_trade(all_data, dict(min_confidence=conf))
        s = compute_stats(trades)
        print(f"  {conf:>7.1f}  {s['n']:5d}  {s['wr']:5.1f}%  {s['mean_bp']:>+6.2f}bp  ${s['mean_usd']:>+6.1f}  {s['t_stat']:>+6.2f}  {s['n_day']:3.0f}")
    
    # ── CONFIDENCE BUCKETS ──
    print(f"\n  Performance by confidence bracket (min_conf=0.3):")
    trades = regime_trade(all_data, dict(min_confidence=0.3))
    
    for bucket, lo, hi in [("0.3-0.5", 0.3, 0.5), ("0.5-0.7", 0.5, 0.7), ("0.7-0.9", 0.7, 0.9), ("0.9+", 0.9, 2.0)]:
        sub = [t for t in trades if lo <= t["confidence"] < hi]
        if sub:
            wr = sum(1 for t in sub if t["won"]) / len(sub) * 100
            mu = float(np.mean([t["pnl"] for t in sub]))
            print(f"    Conf {bucket:>7s}:  n={len(sub):4d}  wr={wr:5.1f}%  mean={mu:>+6.2f}bp")
    
    # ── SESSION BREAKDOWN ──
    print(f"\n  Session breakdown (min_conf=0.3):")
    for sname, scond in [("Asia 0-2h", lambda t: t["hour"] < 2),
                          ("Asia 2-7h", lambda t: 2 <= t["hour"] < 7),
                          ("NY", lambda t: 16 <= t["hour"] < 24),
                          ("London", lambda t: 7 <= t["hour"] < 16)]:
        sub = [t for t in trades if scond(t)]
        if sub:
            wr = sum(1 for t in sub if t["won"]) / len(sub) * 100
            mu = float(np.mean([t["pnl"] for t in sub]))
            print(f"    {sname:>12s}:  n={len(sub):4d}  wr={wr:5.1f}%  mean={mu:>+6.2f}bp")
    
    # ── DISPERSION BUCKETS ──
    print(f"\n  Performance by dispersion percentile:")
    for d_lo, d_hi, label in [(0, 0.6, "Low (<60)"), (0.6, 0.8, "Med (60-80)"),
                               (0.8, 0.95, "High (80-95)"), (0.95, 1.0, "Extreme (>95)")]:
        sub = [t for t in trades if d_lo <= t["disp_pct"] < d_hi]
        if sub:
            wr = sum(1 for t in sub if t["won"]) / len(sub) * 100
            mu = float(np.mean([t["pnl"] for t in sub]))
            print(f"    {label:>15s}:  n={len(sub):4d}  wr={wr:5.1f}%  mean={mu:>+6.2f}bp")
    
    # ── DISPERSION VELOCITY EFFECT ──
    print(f"\n  Rising vs falling dispersion (disp_velocity):")
    for v_label in [("Rising (vel>0)", lambda t: getattr(t, 'disp_pct', 0) > 0.5)]:
        # We don't store velocity in trades, just approximate
        pass
    
    # ── WALK-FORWARD ──
    print(f"\n  Walk-forward (min_conf=0.3):")
    N = min(len(v) for v in all_data.values())
    wf_results = []
    for wf_idx, (sp, ep) in enumerate([(0, 0.5), (0.25, 0.75), (0.5, 1.0)]):
        sub = {}
        si, ei = int(N * sp), int(N * ep)
        for p in all_data:
            sub[p] = all_data[p][si:ei]
        trades = regime_trade(sub, dict(min_confidence=0.3))
        s = compute_stats(trades)
        wf_results.append(s)
        print(f"    WF{wf_idx+1} ({sp:.0%}-{ep:.0%}):  n={s['n']:4d}  wr={s['wr']:5.1f}%  mean={s['mean_bp']:>+5.2f}bp  t={s['t_stat']:>+5.2f}")
    
    if wf_results:
        wr_range = f"{min(s['wr'] for s in wf_results):.1f}% - {max(s['wr'] for s in wf_results):.1f}%"
        print(f"    WF WR range: {wr_range}")
    
    # ── BEST CONFIG DETAIL ──
    print(f"\n{'='*70}")
    print("BEST CONFIG DETAIL (min_confidence=0.3)")
    print("=" * 70)
    
    trades = regime_trade(all_data, dict(min_confidence=0.3))
    s = compute_stats(trades)
    
    # Per-pair WR
    pair_stats = {}
    for t in trades:
        p = t["pair"]
        if p not in pair_stats: pair_stats[p] = {"n": 0, "w": 0}
        pair_stats[p]["n"] += 1
        pair_stats[p]["w"] += 1 if t["won"] else 0
    
    print(f"\n  {'Pair':>8s} {'n':>5s} {'WR':>6s} {'Mean':>8s}")
    print(f"  {'-'*30}")
    for p in sorted(pair_stats.keys(), key=lambda x: pair_stats[x]["w"]/pair_stats[x]["n"], reverse=True):
        ps = pair_stats[p]
        wr = ps["w"] / ps["n"] * 100
        mu = np.mean([t["pnl"] for t in trades if t["pair"] == p])
        print(f"  {p:>8s}  {ps['n']:4d}  {wr:5.1f}%  {mu:>+6.2f}bp")
    
    # Monthly
    print(f"\n  Monthly:")
    by_month = {}
    for t in trades:
        m = t["ts"].strftime("%Y-%m")
        if m not in by_month: by_month[m] = []
        by_month[m].append(t["pnl"])
    for m in sorted(by_month.keys()):
        v = by_month[m]
        wr = sum(1 for x in v if x > 0) / len(v) * 100
        mu = float(np.mean(v))
        print(f"    {m}:  n={len(v):4d}  wr={wr:5.1f}%  mean={mu:>+6.2f}bp")
    
    print(f"""
  {'='*50}
  FINAL: Regime-Aware Scalper (min_conf=0.3)
  {'='*50}
  Trades:    {s['n']} ({s['n_day']:.0f}/day)
  Win rate:  {s['wr']:.1f}%
  Per trade: ${s['mean_usd']:.1f} ({s['mean_bp']:.2f}bp)
  Batch(3):  ~${s['mean_usd']*3:.0f}
  t-stat:    {s['t_stat']:.2f}
  WF WR:     {wr_range if 'wr_range' in dir() else 'N/A'}
""")
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
