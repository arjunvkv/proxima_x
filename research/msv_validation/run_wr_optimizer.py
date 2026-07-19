"""Win rate optimization sweep — find the efficiency frontier.
Tests multiple filter combinations to show achievable WR at different frequencies.
Goal: maximize WR without data mining.
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

def backtest_config(all_data, config):
    """
    config dict:
      sessions: 'all','asia','ny','asia_ny'
      top_n: 1-5
      vol_filter: True/False — only trade high-vol Asia
      msv_filter: True/False — require dispersion>0.80 for entries
      lookback_bars: bars for measuring move (3=15min, 6=30min)
      hold_bars: bars to hold
      min_move_bp: minimum 15min move to trigger
      pair_whitelist: None or list of pairs
      entry_confirm: True — wait 1 extra bar to confirm reversal before entering
      exit_atr_mult: None or float — dynamic TP based on ATR
    """
    N = min(len(v) for v in all_data.values())
    atr_window = deque(maxlen=288)
    positions = {}
    trades = []

    # MSV for dispersion tracking
    ms = MarketStateVector(history_size=50) if config.get("msv_filter") else None
    dh = deque(maxlen=500) if config.get("msv_filter") else None

    sessions = config.get("sessions", "all")
    top_n = config.get("top_n", 3)
    vol_filter = config.get("vol_filter", False)
    msv_filter = config.get("msv_filter", False)
    lookback = config.get("lookback_bars", 3)
    hold = config.get("hold_bars", 3)
    min_move = config.get("min_move_bp", 0) / 10000  # convert bp to decimal
    pair_whitelist = config.get("pair_whitelist", None)
    entry_confirm = config.get("entry_confirm", False)

    for idx in range(lookback, N - hold - (1 if entry_confirm else 0)):
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour

        # Session filter
        if sessions == "asia" and hour >= 7: continue
        if sessions == "ny" and not (16 <= hour < 24): continue
        if sessions == "asia_ny" and (7 <= hour < 16): continue

        # ATR
        atr = 0.0
        for p in ALL_PAIRS:
            hi = float(all_data[p][idx]["high"])
            lo = float(all_data[p][idx]["low"])
            pc = float(all_data[p][idx - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            atr += tr / float(all_data[p][idx]["close"])
        atr /= len(ALL_PAIRS)
        atr_window.append(atr)

        # Vol filter for Asia
        if vol_filter and hour < 7:
            if len(atr_window) >= 30:
                thresh = sorted(atr_window)[2 * len(atr_window) // 3]
                if atr <= thresh:
                    continue

        # MSV dispersion filter
        if msv_filter and ms is not None:
            rets = {}
            for p in ALL_PAIRS:
                c = float(all_data[p][idx]["close"])
                pv = float(all_data[p][idx - 1]["close"])
                rets[p] = max(min((c / pv - 1) if pv > 0 else 0.0, 0.05), -0.05)
            snap = ms.update(rets, timestamp=float(all_data[ALL_PAIRS[0]][idx]["time"]))
            dh.append(snap.network.dispersion)
            dp = compute_pct(snap.network.dispersion, list(dh), 500)
            if dp < 0.80:  # require elevated dispersion
                continue

        # Close expired
        for p in list(positions.keys()):
            if idx >= positions[p]:
                del positions[p]

        if len(positions) >= config.get("max_positions", 3):
            continue

        # Rank pairs
        pair_moves = []
        for p in ALL_PAIRS:
            if p in positions: continue
            if pair_whitelist and p not in pair_whitelist: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            if min_move > 0 and abs(ret) < min_move: continue
            pair_moves.append((p, abs(ret), ret))

        pair_moves.sort(key=lambda x: x[1], reverse=True)

        for p, mag, ret in pair_moves[:top_n]:
            if p in positions: continue
            if len(positions) >= config.get("max_positions", 3): break

            # Session-adaptive direction
            if hour < 7:  # Asia
                if ret > 0: continue  # only long in Asia
                direction = 1
            elif 16 <= hour < 24:  # NY
                if ret < 0: continue  # only short in NY
                direction = -1
            elif 7 <= hour < 16:  # London (if not skipped)
                direction = 1 if ret < 0 else -1
            else:
                direction = 1 if ret < 0 else -1

            # Entry confirmation: wait 1 bar to see if move continues
            entry_idx = idx + 1
            if entry_confirm:
                # Check if the pair reversed in the next bar
                next_ret = float(all_data[p][idx + 1]["close"]) / float(all_data[p][idx]["close"]) - 1
                if direction > 0 and next_ret < 0: continue  # still falling
                if direction < 0 and next_ret > 0: continue  # still rising
                entry_idx = idx + 2  # enter one bar later

            entry_open = float(all_data[p][entry_idx]["open"])
            exit_close = float(all_data[p][entry_idx + hold - 1]["close"])
            pnl = direction * (exit_close / entry_open - 1) if entry_open > 0 else 0

            won = pnl > 0
            trades.append({
                "pair": p, "ts": dt, "hour": hour,
                "pnl": pnl * 10000, "won": won,
                "direction": "LONG" if direction > 0 else "SHORT",
            })
            positions[p] = entry_idx + hold - 1

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
    print("WIN RATE EFFICIENCY FRONTIER")
    print("=" * 70)
    print(f"\nTesting configurations to find max achievable WR at different frequencies")
    print(f"Goal: show the trade-off between win rate and trades/day")

    configs = []

    # 1. Baseline: current best
    configs.append(("Baseline adaptive + vol", dict(sessions="asia_ny", top_n=3, vol_filter=True)))

    # 2. Tighter pair selection (only top 1 or top 2)
    for n in [1, 2]:
        configs.append((f"Top {n} only", dict(sessions="asia_ny", top_n=n, vol_filter=True)))

    # 3. Higher min move threshold
    for bp in [0.3, 0.5, 1.0, 2.0]:
        configs.append((f"Min move {bp}bp", dict(sessions="asia_ny", top_n=3, vol_filter=True, min_move_bp=bp)))

    # 4. Longer lookback (30min instead of 15min)
    configs.append(("30min lookback", dict(sessions="asia_ny", top_n=3, vol_filter=True, lookback_bars=6)))

    # 5. Add MSV dispersion filter
    configs.append(("+ MSV disp>80", dict(sessions="asia_ny", top_n=3, vol_filter=True, msv_filter=True)))

    # 6. Entry confirmation (wait for reversal bar)
    configs.append(("+ Entry confirm", dict(sessions="asia_ny", top_n=3, vol_filter=True, entry_confirm=True)))

    # 7. Asia only (skip NY)
    configs.append(("Asia only", dict(sessions="asia", top_n=3, vol_filter=True)))
    configs.append(("Asia only + confirm", dict(sessions="asia", top_n=3, vol_filter=True, entry_confirm=True)))

    # 8. Best pairs only
    best_pairs = ["EURCHF", "GBPAUD", "EURCAD", "GBPCHF", "EURNZD", "GBPNZD"]
    configs.append(("Top 6 pairs only", dict(sessions="asia_ny", top_n=3, vol_filter=True, pair_whitelist=best_pairs)))

    # 9. Longer hold (30min instead of 15min) — more time for reversal
    configs.append(("30min hold", dict(sessions="asia_ny", top_n=3, vol_filter=True, hold_bars=6)))

    # 10. Combined
    configs.append(("Asia + confirm + top6", dict(sessions="asia", top_n=2, vol_filter=True, entry_confirm=True, pair_whitelist=best_pairs)))

    results = []
    for label, params in configs:
        trades = backtest_config(all_data, params)
        s = compute_stats(trades)
        results.append((label, s))
        print(f"  {label:>30s}:  n={s['n']:5d}  wr={s['wr']:5.1f}%  "
              f"mean={s['mean_bp']:>+5.2f}bp  ${s['mean_usd']:>+5.1f}/trade  "
              f"t={s['t_stat']:>+5.2f}  {s['n_day']:4.0f}/day")

    # ── COMBINATION SCAN ──
    print(f"\n{'='*70}")
    print("COMBINATION SCAN — systematically combine filters")
    print("=" * 70)

    combo_configs = {
        "Asia+K6+conf+msv": dict(sessions="asia", top_n=2, vol_filter=True, entry_confirm=True, pair_whitelist=best_pairs, msv_filter=True),
        "Asia+K6+conf+30m": dict(sessions="asia", top_n=2, vol_filter=True, entry_confirm=True, pair_whitelist=best_pairs, hold_bars=6),
        "Asia+conf+0.3bp": dict(sessions="asia", top_n=3, vol_filter=True, entry_confirm=True, min_move_bp=0.3),
        "Asia+K6+0.5bp": dict(sessions="asia", top_n=2, vol_filter=True, pair_whitelist=best_pairs, min_move_bp=0.5),
        "NY+conf+0.3bp": dict(sessions="ny", top_n=2, vol_filter=False, entry_confirm=True, min_move_bp=0.3),
    }

    for label, params in combo_configs.items():
        trades = backtest_config(all_data, params)
        s = compute_stats(trades)
        results.append((label, s))
        print(f"  {label:>30s}:  n={s['n']:5d}  wr={s['wr']:5.1f}%  "
              f"mean={s['mean_bp']:>+5.2f}bp  ${s['mean_usd']:>+5.1f}/trade  "
              f"t={s['t_stat']:>+5.2f}  {s['n_day']:4.0f}/day")

    # ── WALK-FORWARD ON BEST NON-OVERFIT CONFIG ──
    print(f"\n{'='*70}")
    print("WALK-FORWARD: Asia + entry confirm (simple, least overfit)")
    print("=" * 70)

    N = min(len(v) for v in all_data.values())
    wf_config = dict(sessions="asia", top_n=3, vol_filter=True, entry_confirm=True)
    wf_results = []
    for wf_idx, (sp, ep) in enumerate([(0, 0.5), (0.25, 0.75), (0.5, 1.0)]):
        sub = {}
        si, ei = int(N * sp), int(N * ep)
        for p in all_data:
            sub[p] = all_data[p][si:ei]
        trades = backtest_config(sub, wf_config)
        s = compute_stats(trades)
        wf_results.append(s)
        print(f"  WF{wf_idx+1} ({sp:.0%}-{ep:.0%}):  n={s['n']:4d}  wr={s['wr']:5.1f}%  mean={s['mean_bp']:>+5.2f}bp  t={s['t_stat']:>+5.2f}")

    if wf_results:
        wr_range = f"{min(s['wr'] for s in wf_results):.1f}% - {max(s['wr'] for s in wf_results):.1f}%"
        print(f"  WF WR range: {wr_range}")

    # ── ASIA-ONLY DEEP DIVE ──
    print(f"\n{'='*70}")
    print("ASIA-ONLY DEEP DIVE (highest WR session)")
    print("=" * 70)

    for hold_mins, hold_b in [("5m (1bar)", 1), ("10m (2bar)", 2), ("15m (3bar)", 3), ("20m (4bar)", 4)]:
        for confirm in [False, True]:
            cfg = dict(sessions="asia", top_n=3, vol_filter=True, hold_bars=hold_b, entry_confirm=confirm)
            label = f"Hold {hold_mins}" + ("+confirm" if confirm else "")
            trades = backtest_config(all_data, cfg)
            s = compute_stats(trades)
            print(f"  {label:>25s}:  n={s['n']:5d}  wr={s['wr']:5.1f}%  "
                  f"mean={s['mean_bp']:>+5.2f}bp  ${s['mean_usd']:>+5.1f}/trade  {s['n_day']:4.0f}/day")

    # ── RECOMMENDED CONFIG ──
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print("=" * 70)

    # Best Asia config with entry confirm
    best_cfg = dict(sessions="asia", top_n=3, vol_filter=True, entry_confirm=True, hold_bars=3)
    trades = backtest_config(all_data, best_cfg)
    s = compute_stats(trades)

    # Win rate by pair
    pair_wr = {}
    for t in trades:
        p = t["pair"]
        if p not in pair_wr:
            pair_wr[p] = {"n": 0, "w": 0}
        pair_wr[p]["n"] += 1
        pair_wr[p]["w"] += 1 if t["won"] else 0

    worst_pairs = sorted(pair_wr.keys(), key=lambda p: pair_wr[p]["w"]/pair_wr[p]["n"])[:3]

    print(f"""
  Recommended: Asia-only + high-vol filter + entry confirmation

  Trades:  {s['n']} ({s['n_day']:.0f}/day)
  Win rate: {s['wr']:.1f}%
  Per trade: ${s['mean_usd']:.1f} ({s['mean_bp']:.2f}bp)
  With 3 concurrent: ~${s['mean_usd']*3:.0f} per entry batch
  t-stat: {s['t_stat']:.2f}

  Worst 3 pairs: {[p for p in worst_pairs]}
  (consider excluding these for higher WR)

  Walk-forward WR range: {wr_range if 'wr_range' in dir() else 'N/A'}
""")

    mt5.shutdown()

if __name__ == "__main__":
    main()
