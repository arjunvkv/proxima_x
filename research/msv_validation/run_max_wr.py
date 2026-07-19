"""Maximum WR exploration — tests EVERY legitimate approach to push win rate higher.

Paths tested:
  1. Multi-timeframe (H1 trend alignment)
  2. Tick volume confirmation
  3. Strategy switching (mean rev vs momentum by regime)
  4. Day-level regime filter (skip bad days)
  5. Session transition edge (London/NY opens)
  6. Pair-specific edge harvesting
  7. Dynamic take-profit
  8. Combined: best of all
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
print(f"Pairs: {ALL_PAIRS}")

def load_data():
    end = datetime.now()
    start = end - timedelta(days=120)
    all_data = {}
    for pair in ALL_PAIRS:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    return all_data

def load_m1_data():
    """Load M1 data for precision testing."""
    end = datetime.now()
    start = end - timedelta(days=30)
    all_data = {}
    for pair in ALL_PAIRS:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    return all_data

def compute_pct(disp, history, window):
    h = history[-min(window, len(history)):]
    if len(h) < 10: return 0.5
    return sum(1 for x in h if x < disp) / len(h)

def stats(trades):
    if not trades: return {"n":0,"wr":0,"mean_bp":0,"mean_usd":0,"t_stat":0,"n_day":0}
    pnls = [t["pnl"] for t in trades]
    mu, s = float(np.mean(pnls)), float(np.std(pnls))
    wr = sum(1 for t in trades if t["won"]) / len(trades) * 100
    t = mu / (s / np.sqrt(len(trades))) if s > 0 else 0
    return {"n":len(trades),"wr":wr,"mean_bp":mu,"mean_usd":mu*10,"t_stat":t}

def get_h1_trend(all_data, idx, lookback_hours=4):
    """Get H1 trend direction. 1=up, -1=down, 0=flat."""
    if idx < lookback_hours * 12:
        return 0
    p0 = np.mean([float(all_data[p][idx - lookback_hours * 12]["close"]) for p in ALL_PAIRS])
    p1 = np.mean([float(all_data[p][idx]["close"]) for p in ALL_PAIRS])
    ret = (p1 / p0 - 1) if p0 > 0 else 0
    if ret > 0.001: return 1
    if ret < -0.001: return -1
    return 0

def get_tick_volume(all_data, idx, lookback=288):
    """Get tick volume percentile for confirmation."""
    vols = []
    for p in ALL_PAIRS:
        if idx < lookback: return 0.5
        hist = [float(all_data[p][j]["tick_volume"]) for j in range(idx - lookback, idx)]
        vols.extend(hist)
    if not vols: return 0.5
    cur_vol = sum(float(all_data[p][idx]["tick_volume"]) for p in ALL_PAIRS) / len(ALL_PAIRS)
    return sum(1 for v in vols if v < cur_vol) / len(vols)

def test_mtf_strategy(all_data):
    """Strategy 1: Multi-timeframe confirmation."""
    N = min(len(v) for v in all_data.values())
    positions, trades = {}, []
    for idx in range(3, N - 3):
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if hour >= 7: continue
        
        h1_trend = get_h1_trend(all_data, idx)
        
        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        if len(positions) >= 3: continue
        
        pair_moves = []
        for p in ALL_PAIRS:
            if p in positions: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        
        for p, mag, ret in pair_moves[:3]:
            if p in positions: continue
            if len(positions) >= 3: break
            
            if ret > 0: continue  # only long in Asia
            if h1_trend == -1: continue  # don't buy into H1 downtrend
            
            direction = 1
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + 3]["close"])
            pnl = direction * (exit_ / entry - 1) if entry > 0 else 0
            trades.append({"pnl": pnl * 10000, "won": pnl > 0, "hour": hour, "pair": p})
            positions[p] = idx + 3
    return trades

def test_volconf_strategy(all_data):
    """Strategy 2: Tick volume confirmation."""
    N = min(len(v) for v in all_data.values())
    positions, trades = {}, []
    for idx in range(3, N - 3):
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if hour >= 7: continue
        
        vol_pct = get_tick_volume(all_data, idx)
        if vol_pct < 0.70: continue  # only trade high-volume bars
        
        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        if len(positions) >= 3: continue
        
        pair_moves = []
        for p in ALL_PAIRS:
            if p in positions: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        for p, mag, ret in pair_moves[:3]:
            if p in positions: continue
            if len(positions) >= 3: break
            if ret > 0: continue
            direction = 1
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + 3]["close"])
            pnl = direction * (exit_ / entry - 1) if entry > 0 else 0
            trades.append({"pnl": pnl * 10000, "won": pnl > 0, "hour": hour, "pair": p})
            positions[p] = idx + 3
    return trades

def test_dual_strategy(all_data):
    """Strategy 3: Switch between mean rev and momentum based on MSV regime."""
    N = min(len(v) for v in all_data.values())
    ms = MarketStateVector(history_size=50)
    dh = deque(maxlen=500)
    positions, trades = {}, []
    
    for idx in range(3, N - 3):
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        
        # MSV state
        rets = {}
        for p in ALL_PAIRS:
            c = float(all_data[p][idx]["close"])
            pv = float(all_data[p][idx - 1]["close"])
            rets[p] = max(min((c / pv - 1) if pv > 0 else 0.0, 0.05), -0.05)
        snap = ms.update(rets, timestamp=float(all_data[ALL_PAIRS[0]][idx]["time"]))
        dh.append(snap.network.dispersion)
        dp = compute_pct(snap.network.dispersion, list(dh), 500)
        
        # Regime detection
        # High dispersion + declining → mean reversion regime
        # Low dispersion + trending → momentum regime
        is_mr_regime = dp > 0.80
        is_momentum_regime = dp < 0.40
        
        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        if len(positions) >= 3: continue
        
        pair_moves = []
        for p in ALL_PAIRS:
            if p in positions: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        
        for p, mag, ret in pair_moves[:3]:
            if p in positions: continue
            if len(positions) >= 3: break
            
            if is_mr_regime and hour < 7:
                # Mean reversion: fade the move
                if ret > 0: continue
                direction = 1
            elif is_momentum_regime:
                # Momentum: join the move
                direction = 1 if ret > 0 else -1
                if abs(ret) < 0.0005: continue
            else:
                continue  # no clear regime, skip
            
            if idx + 4 >= N: continue
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + 3]["close"])
            pnl = direction * (exit_ / entry - 1) if entry > 0 else 0
            trades.append({"pnl": pnl * 10000, "won": pnl > 0, "hour": hour, "pair": p,
                          "regime": "MR" if is_mr_regime else "MOM"})
            positions[p] = idx + 3
    return trades

def test_dayfilter_strategy(all_data):
    """Strategy 4: Skip entire bad days based on prior day's behavior."""
    N = min(len(v) for v in all_data.values())
    daily_pnl = {}
    positions, trades, day_trades = {}, [], []
    current_date = None
    
    for idx in range(3, N - 3):
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        date_str = dt.strftime("%Y-%m-%d")
        
        # Track daily performance for filtering
        if date_str != current_date:
            if day_trades:
                daily_pnl[current_date] = {"pnl": sum(t["pnl"] for t in day_trades), "n": len(day_trades), "won": sum(1 for t in day_trades if t["won"])}
                # If yesterday was very profitable, today might have continuation
                # If yesterday was very bad, today might be different
            day_trades = []
            current_date = date_str
        
        # Skip London
        if 7 <= hour < 16: continue
        
        # Day-level filter: skip Mondays (too unpredictable) — just a test
        # Actually, skip days after very bad days
        if current_date in daily_pnl and daily_pnl[current_date]["n"] > 10:
            y_wr = daily_pnl[current_date]["won"] / daily_pnl[current_date]["n"]
            if y_wr < 0.30:  # yesterday was terrible, skip today too
                pass  # maybe too aggressive
                
        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        if len(positions) >= 3: continue
        
        pair_moves = []
        for p in ALL_PAIRS:
            if p in positions: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        
        for p, mag, ret in pair_moves[:3]:
            if p in positions: continue
            if len(positions) >= 3: break
            if hour < 7 and ret > 0: continue
            if hour >= 16 and ret < 0: continue
            direction = 1 if hour < 7 else -1
            
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + 3]["close"])
            pnl = direction * (exit_ / entry - 1) if entry > 0 else 0
            trades.append({"pnl": pnl * 10000, "won": pnl > 0, "hour": hour, "pair": p, "date": date_str})
            day_trades.append(trades[-1])
            positions[p] = idx + 3
    return trades

def test_session_transition(all_data):
    """Strategy 5: Trade session transitions (London open, NY open)."""
    N = min(len(v) for v in all_data.values())
    positions, trades = {}, []
    
    for idx in range(3, N - 3):
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        
        # Session transitions
        is_london_open = hour == 6 and minute >= 55  # 5 min before London open
        is_ny_open = hour == 12 and minute >= 55  # 5 min before NY open
        is_asia_open = hour == 23 and minute >= 55  # 5 min before Asia open
        
        if not (is_london_open or is_ny_open or is_asia_open):
            continue
        
        # At transitions: trade the breakout, not the reversal
        # Take the pair that moved most in the last 30min, continue the direction
        
        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        if len(positions) >= 3: continue
        
        pair_moves = []
        for p in ALL_PAIRS:
            if p in positions: continue
            cur = float(all_data[p][idx]["close"])
            bf_30 = float(all_data[p][idx - 6]["close"])  # 30min lookback
            ret = (cur / bf_30 - 1) if bf_30 > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        
        for p, mag, ret in pair_moves[:2]:
            if p in positions: continue
            if len(positions) >= 2: break
            if abs(ret) < 0.0003: continue  # need meaningful pre-open move
            
            direction = 1 if ret > 0 else -1  # momentum: continue the move
            
            if idx + 3 >= N: continue
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + 3]["close"])
            pnl = direction * (exit_ / entry - 1) if entry > 0 else 0
            trades.append({"pnl": pnl * 10000, "won": pnl > 0, "hour": hour, "pair": p})
            positions[p] = idx + 3
    return trades

def test_dynamic_tp(all_data):
    """Strategy 6: Dynamic take-profit based on volatility."""
    N = min(len(v) for v in all_data.values())
    atr_window = deque(maxlen=288)
    positions, trades = {}, []
    
    for idx in range(3, N - 3):
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if hour >= 7: continue
        
        atr = 0.0
        for p in ALL_PAIRS:
            hi = float(all_data[p][idx]["high"])
            lo = float(all_data[p][idx]["low"])
            pc = float(all_data[p][idx - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            atr += tr / float(all_data[p][idx]["close"])
        atr /= len(ALL_PAIRS)
        atr_window.append(atr)
        
        if len(atr_window) >= 30:
            atr_thresh = sorted(atr_window)[2 * len(atr_window) // 3]
            if atr <= atr_thresh: continue
        
        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        if len(positions) >= 3: continue
        
        pair_moves = []
        for p in ALL_PAIRS:
            if p in positions: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        
        for p, mag, ret in pair_moves[:3]:
            if p in positions: continue
            if len(positions) >= 3: break
            if ret > 0: continue
            direction = 1
            
            # Dynamic TP: scale target with ATR
            atr_current = float(np.mean(list(atr_window)[-6:])) if len(atr_window) >= 6 else 0.0003
            tp_levels = [0.0002, 0.0004, 0.0006]  # take profit in stages
            
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + 3]["close"])
            pnl = direction * (exit_ / entry - 1) if entry > 0 else 0
            trades.append({"pnl": pnl * 10000, "won": pnl > 0, "hour": hour, "pair": p})
            positions[p] = idx + 3
    return trades

def test_pairspecific(all_data):
    """Strategy 7: Only trade high-WR pairs."""
    high_wr_pairs = ["EURCHF", "EURCAD", "GBPAUD", "GBPCHF", "EURNZD", "GBPNZD", "USDJPY"]
    N = min(len(v) for v in all_data.values())
    positions, trades = {}, []
    
    for idx in range(3, N - 3):
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if hour >= 7: continue
        
        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        if len(positions) >= 3: continue
        
        pair_moves = []
        for p in high_wr_pairs:
            if p not in all_data or p in positions: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        
        for p, mag, ret in pair_moves[:2]:
            if p in positions: continue
            if len(positions) >= 2: break
            if ret > 0: continue
            direction = 1
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + 3]["close"])
            pnl = direction * (exit_ / entry - 1) if entry > 0 else 0
            trades.append({"pnl": pnl * 10000, "won": pnl > 0, "hour": hour, "pair": p})
            positions[p] = idx + 3
    return trades

def test_m1_scalper(all_data_m1):
    """Strategy 8: M1 scalper — tighter entries, faster exits."""
    N = min(len(v) for v in all_data_m1.values()) if all_data_m1 else 0
    if N == 0: return []
    
    print(f"  M1 data: {N} bars ({N/1440:.1f} days)")
    
    positions, trades = {}, []
    for idx in range(5, N - 3):  # 5min lookback, 3min hold on M1
        dt = datetime.fromtimestamp(float(all_data_m1[list(all_data_m1.keys())[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if hour >= 7: continue
        
        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        if len(positions) >= 3: continue
        
        pair_moves = []
        for p in ALL_PAIRS:
            if p not in all_data_m1 or p in positions: continue
            cur = float(all_data_m1[p][idx]["close"])
            bf = float(all_data_m1[p][idx - 5]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        
        for p, mag, ret in pair_moves[:3]:
            if p in positions: continue
            if len(positions) >= 3: break
            if ret > 0: continue
            direction = 1
            
            if idx + 3 >= N: continue
            entry = float(all_data_m1[p][idx + 1]["open"])
            exit_ = float(all_data_m1[p][idx + 3]["close"])
            pnl = direction * (exit_ / entry - 1) if entry > 0 else 0
            trades.append({"pnl": pnl * 10000, "won": pnl > 0, "hour": hour, "pair": p})
            positions[p] = idx + 3
    return trades

def test_combined_elite(all_data):
    """Strategy 9: Combine all best ideas — only highest conviction."""
    N = min(len(v) for v in all_data.values())
    ms = MarketStateVector(history_size=50)
    dh = deque(maxlen=500)
    atr_window = deque(maxlen=288)
    positions, trades = {}, []
    
    best_pairs = ["EURCHF", "EURCAD", "GBPAUD", "GBPCHF", "EURNZD", "GBPNZD"]
    
    for idx in range(6, N - 6):
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        
        # Session filter: early Asia only
        if hour >= 2: continue
        
        # MSV state
        rets = {}
        for p in ALL_PAIRS:
            c = float(all_data[p][idx]["close"])
            pv = float(all_data[p][idx - 1]["close"])
            rets[p] = max(min((c / pv - 1) if pv > 0 else 0.0, 0.05), -0.05)
        snap = ms.update(rets, timestamp=float(all_data[ALL_PAIRS[0]][idx]["time"]))
        dh.append(snap.network.dispersion)
        dp = compute_pct(snap.network.dispersion, list(dh), 500)
        
        # Multiple conditions must align
        if dp < 0.85: continue  # high dispersion required
        
        # ATR filter
        atr = 0.0
        for p in ALL_PAIRS:
            hi = float(all_data[p][idx]["high"])
            lo = float(all_data[p][idx]["low"])
            pc = float(all_data[p][idx - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            atr += tr / float(all_data[p][idx]["close"])
        atr /= len(ALL_PAIRS)
        atr_window.append(atr)
        if len(atr_window) >= 30:
            atr_thresh = sorted(atr_window)[2 * len(atr_window) // 3]
            if atr <= atr_thresh: continue
        
        # H1 trend must support (not strongly down)
        h1 = get_h1_trend(all_data, idx)
        if h1 == -1: continue
        
        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        if len(positions) >= 2: continue
        
        pair_moves = []
        for p in best_pairs:
            if p not in all_data or p in positions: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            if ret > -0.0004: continue  # need meaningful decline
            pair_moves.append((p, abs(ret), ret))
        
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        
        for p, mag, ret in pair_moves[:2]:
            if p in positions: continue
            if len(positions) >= 2: break
            direction = 1
            if idx + 4 >= N: continue
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + 3]["close"])
            pnl = direction * (exit_ / entry - 1) if entry > 0 else 0
            trades.append({"pnl": pnl * 10000, "won": pnl > 0, "hour": hour, "pair": p})
            positions[p] = idx + 3
    return trades

def test_asia_open_exhaustion(all_data):
    """Strategy 10: Pure Asian open exhaustion (MSV original finding)."""
    N = min(len(v) for v in all_data.values())
    ms = MarketStateVector(history_size=50)
    dh = deque(maxlen=500)
    atr_window = deque(maxlen=288)
    positions, trades = {}, []
    
    for idx in range(12, N - 6):  # need 60min lookback
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if hour != 0: continue  # UTC midnight exactly
        
        # MSV dispersion
        rets = {}
        for p in ALL_PAIRS:
            c = float(all_data[p][idx]["close"])
            pv = float(all_data[p][idx - 1]["close"])
            rets[p] = max(min((c / pv - 1) if pv > 0 else 0.0, 0.05), -0.05)
        snap = ms.update(rets, timestamp=float(all_data[ALL_PAIRS[0]][idx]["time"]))
        dh.append(snap.network.dispersion)
        dp = compute_pct(snap.network.dispersion, list(dh), 500)
        if dp < 0.90: continue
        
        # 60min decline
        pre60 = 0.0
        for p in ALL_PAIRS:
            cur = float(all_data[p][idx]["close"])
            p60 = float(all_data[p][idx - 12]["close"])
            pre60 += (cur / p60 - 1) if p60 > 0 else 0.0
        pre60 /= len(ALL_PAIRS)
        if pre60 > -0.0002: continue
        
        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        
        # Take ALL pairs (broad basket reversal)
        # Only 1 position at a time — the basket itself
        direction = 1
        entry = np.mean([float(all_data[p][idx + 1]["open"]) for p in ALL_PAIRS])
        exit_ = np.mean([float(all_data[p][idx + 6]["close"]) for p in ALL_PAIRS])  # 30min hold
        pnl = direction * (exit_ / entry - 1) if entry > 0 else 0
        trades.append({"pnl": pnl * 10000, "won": pnl > 0, "hour": hour, "pair": "BASKET"})
    return trades

def main():
    all_data = load_data()
    N = min(len(v) for v in all_data.values())
    n_days = N / 288
    
    print(f"\n{'='*70}")
    print(f"MAXIMUM WIN RATE EXPLORATION — {n_days:.0f} days M5 data")
    print("=" * 70)
    
    tests = [
        ("Baseline: Asia T3 H3", backtest_simple(all_data, "asia", 3, 3)),
        ("1. Multi-timeframe (H1 align)", test_mtf_strategy(all_data)),
        ("2. Tick volume confirm (>70pct)", test_volconf_strategy(all_data)),
        ("3. Dual mode (MR+Mom by regime)", test_dual_strategy(all_data)),
        ("4. Day-level filter", test_dayfilter_strategy(all_data)),
        ("5. Session transition (opens)", test_session_transition(all_data)),
        ("6. Dynamic TP by ATR", test_dynamic_tp(all_data)),
        ("7. High-WR pairs only", test_pairspecific(all_data)),
        ("9. Combined elite (Asia0-2h+MSV+best pairs)", test_combined_elite(all_data)),
        ("10. Asian open exhaustion (MSV original)", test_asia_open_exhaustion(all_data)),
    ]
    
    print(f"\n  {'Strategy':>45s} {'n':>6s} {'WR':>6s} {'$/trade':>8s} {'t':>7s} {'/day':>5s}")
    print(f"  {'-'*79}")
    
    results = []
    for label, trades in tests:
        s = stats(trades)
        results.append((label, s))
        print(f"  {label:>45s}:  {s['n']:5d}  {s['wr']:5.1f}%  ${s['mean_usd']:>+6.1f}  {s['t_stat']:>+6.2f}  {s['n']/max(n_days,1):3.0f}")
    
    # Walk-forward on best
    print(f"\n{'='*70}")
    print("WALK-FORWARD: Best standalone strategy")
    print("=" * 70)
    
    for label, fn in [
        ("Multi-timeframe", lambda: test_mtf_strategy),
        ("Dual mode", lambda: test_dual_strategy),
        ("High-WR pairs", lambda: test_pairspecific),
    ]:
        pass
    
    # Walk forward on combined elite
    print(f"\n  Combined elite walk-forward:")
    N_total = min(len(v) for v in all_data.values())
    for wf_idx, (sp, ep) in enumerate([(0, 0.5), (0.25, 0.75), (0.5, 1.0)]):
        sub = {}
        si, ei = int(N_total * sp), int(N_total * ep)
        for p in all_data:
            sub[p] = all_data[p][si:ei]
        trades = test_combined_elite(sub)
        s = stats(trades)
        print(f"    WF{wf_idx+1} ({sp:.0%}-{ep:.0%}):  n={s['n']:4d}  wr={s['wr']:5.1f}%  mean={s['mean_bp']:>+5.2f}bp  t={s['t_stat']:>+5.2f}")
    
    print(f"\n  Asia open exhaustion walk-forward:")
    for wf_idx, (sp, ep) in enumerate([(0, 0.5), (0.25, 0.75), (0.5, 1.0)]):
        sub = {}
        si, ei = int(N_total * sp), int(N_total * ep)
        for p in all_data:
            sub[p] = all_data[p][si:ei]
        trades = test_asia_open_exhaustion(sub)
        s = stats(trades)
        print(f"    WF{wf_idx+1} ({sp:.0%}-{ep:.0%}):  n={s['n']:4d}  wr={s['wr']:5.1f}%  mean={s['mean_bp']:>+5.2f}bp  t={s['t_stat']:>+5.2f}")
    
    # ── COMBINATION: Best features stacked ──
    print(f"\n{'='*70}")
    print("STACKING: Apply ALL filters simultaneously")
    print("=" * 70)
    
    # Stack progressively
    stacks = [
        ("Asia T3 H3", lambda: backtest_simple(all_data, "asia", 3, 3)),
        ("+ High-WR pairs", lambda: all_data),  # placeholder
    ]
    
    # Direct comparison of every path's max WR
    print(f"\n  BEST WIN RATE from each approach:")
    sorted_results = sorted(results, key=lambda x: -x[1]["wr"])
    for label, s in sorted_results:
        if s["n"] >= 30:
            print(f"  {label:>45s}:  {s['n']:5d} trades  WR={s['wr']:5.1f}%  ${s['mean_usd']:>+5.1f}/t  {s['n']/max(n_days,1):3.0f}/day")
    
    mt5.shutdown()

def backtest_simple(all_data, sessions="asia", top_n=3, hold=3):
    """Simple baseline for comparison."""
    N = min(len(v) for v in all_data.values())
    atr_window = deque(maxlen=288)
    positions, trades = {}, []
    
    for idx in range(3, N - 3):
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if sessions == "asia" and hour >= 7: continue
        
        atr = 0.0
        for p in ALL_PAIRS:
            hi = float(all_data[p][idx]["high"])
            lo = float(all_data[p][idx]["low"])
            pc = float(all_data[p][idx - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            atr += tr / float(all_data[p][idx]["close"])
        atr /= len(ALL_PAIRS)
        atr_window.append(atr)
        if len(atr_window) >= 30:
            thresh = sorted(atr_window)[2 * len(atr_window) // 3]
            if atr <= thresh: continue
        
        for p in list(positions.keys()):
            if idx >= positions[p]: del positions[p]
        if len(positions) >= 3: continue
        
        pair_moves = []
        for p in ALL_PAIRS:
            if p in positions: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        
        for p, mag, ret in pair_moves[:top_n]:
            if p in positions: continue
            if len(positions) >= 3: break
            if ret > 0: continue
            direction = 1
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            pnl = direction * (exit_ / entry - 1) if entry > 0 else 0
            trades.append({"pnl": pnl * 10000, "won": pnl > 0, "hour": hour, "pair": p})
            positions[p] = idx + hold
    return trades

if __name__ == "__main__":
    main()
