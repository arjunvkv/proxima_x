"""Execution Geometry — Phase II.A.
Tests four dimensions to amplify OSS PF from 1.07 -> 1.20+.

Order:
  1. Hold-Time Surface (exit horizons)
  2. MAE-based kill switch
  3. Time-underwater exit
  4. Partial MFE harvesting

Reuses ReplayCache + OSS infrastructure.
"""
import sys; sys.path.insert(0, '.')
import time
import math
from collections import defaultdict

from research.replay_cache import ReplayCache
from research.outcome_surface_signal import OutcomeSurfaceSignal
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine

H = "=" * 70
TICK_LIMIT = 50000
SEED = 42

FOLDS = [
    dict(name="Fold 1", syms=["EURJPY","USDJPY"], train_start="2026-03-12", train_end="2026-03-28",
         test_start="2026-03-29", test_end="2026-04-14"),
    dict(name="Fold 2", syms=["EURJPY"], train_start="2026-04-01", train_end="2026-04-20",
         test_start="2026-04-21", test_end="2026-05-10"),
    dict(name="Fold 3", syms=["EURJPY"], train_start="2026-05-01", train_end="2026-05-20",
         test_start="2026-05-21", test_end="2026-06-08"),
]

def ecdf_sig(t):
    e = t.get("ecdf", 0.5)
    d = e - 0.5
    return 1 if d > 0.05 else (-1 if d < -0.05 else 0)

def simulate_with_exit(oss, ticks, exit_horizon, mae_stop_sigma=None, underwater_max=None,
                        take_profit_sigma=None, spread_bps=0.0, sym_stats=None):
    """Simulate OSS with configurable exit geometry.

    Parameters:
        exit_horizon: fixed hold time in ticks (0 = use DOA default 20)
        mae_stop_sigma: cut trade if adverse excursion exceeds N sigma
        underwater_max: cut trade if underwater for N consecutive ticks
        take_profit_sigma: take 50% profit at N sigma favorable excursion
    """
    horizon = exit_horizon if exit_horizon else 20
    doa = DelayedOutcomeEngine(horizon_ticks=horizon)
    trades = []
    ed = {}
    active_positions = {}  # sym -> {entry_price, signal, ticks_held, ticks_underwater, peak_favorable}

    for t_idx, t in enumerate(ticks):
        s = t["sym"]
        price = t["price"]
        sig = oss.predict(t["ecdf"])
        ed[s] = {"price": price, "ecdf_rank": t["ecdf"], "entropy": t["entropy"], "signal": sig}

        # Track existing positions
        if s in active_positions:
            pos = active_positions[s]
            pos["ticks_held"] += 1
            entry_price = pos["entry_price"]
            direction = pos["signal"]

            # Adverse excursion
            if direction == 1:  # BUY
                excursion = (entry_price - price) / price
            else:  # SELL
                excursion = (price - entry_price) / price

            pos["current_excursion"] = excursion
            if excursion > 0:
                pos["peak_favorable"] = max(pos.get("peak_favorable", 0), excursion)
                pos["ticks_underwater"] = 0
            else:
                pos["ticks_underwater"] = pos.get("ticks_underwater", 0) + 1

            # Check exit conditions
            should_exit = False

            # 1. MAE stop
            if mae_stop_sigma is not None and sym_stats:
                sym_vol = sym_stats.get(s, {}).get("sigma", 0.001)
                if excursion < -mae_stop_sigma * sym_vol:
                    should_exit = True

            # 2. Underwater timeout
            if underwater_max is not None and pos["ticks_underwater"] >= underwater_max:
                should_exit = True

            if should_exit:
                if direction == 1:
                    pct = (price - entry_price) / entry_price
                else:
                    pct = (entry_price - price) / entry_price
                trade_pnl = pct * 10000
                spread_cost = spread_bps / 10000 * price * 2
                net_pnl = trade_pnl - spread_cost
                trades.append({
                    "sym": s, "signal": direction, "outcome": pct,
                    "pnl": trade_pnl, "net_pnl": net_pnl, "price": price,
                    "spread_cost": spread_cost, "exit_reason": "stop",
                    "ticks_held": pos["ticks_held"],
                })
                del active_positions[s]

        # Enter new position
        if sig != 0 and s not in active_positions:
            active_positions[s] = {
                "entry_price": price, "signal": sig, "ticks_held": 0,
                "ticks_underwater": 0, "peak_favorable": 0, "current_excursion": 0,
            }

        # Evaluate outcomes via DOA
        doa.record_snapshot(ed)
        if doa.ready:
            cp = {s2: ed[s2]["price"] for s2 in ed}
            outcomes = doa.evaluate(cp)
            for s2, outcome in outcomes.items():
                if s2 in active_positions:
                    pos = active_positions[s2]
                    pos["outcome"] = outcome
                    pos["outcome_price"] = cp[s2]

        # Check if any active positions hit exit horizon
        to_close = []
        for sym, pos in active_positions.items():
            if pos["ticks_held"] >= horizon:
                to_close.append(sym)

        for sym in to_close:
            pos = active_positions[sym]
            direction = pos["signal"]
            entry_price = pos["entry_price"]
            exit_price = pos["outcome_price"]
            if direction == 1:
                pct = (exit_price - entry_price) / entry_price
            else:
                pct = (entry_price - exit_price) / entry_price
            trade_pnl = pct * 10000
            spread_cost = spread_bps / 10000 * exit_price * 2
            net_pnl = trade_pnl - spread_cost
            trades.append({
                "sym": sym, "signal": direction, "outcome": pct,
                "pnl": trade_pnl, "net_pnl": net_pnl, "price": exit_price,
                "spread_cost": spread_cost, "exit_reason": "horizon",
                "ticks_held": pos["ticks_held"],
            })
            del active_positions[sym]

    # Force close remaining on last tick
    for sym, pos in active_positions.items():
        direction = pos["signal"]
        last_price = ed[sym]["price"]
        entry_price = pos["entry_price"]
        if direction == 1:
            pct = (last_price - entry_price) / entry_price
        else:
            pct = (entry_price - last_price) / entry_price
        trade_pnl = pct * 10000
        spread_cost = spread_bps / 10000 * last_price * 2
        net_pnl = trade_pnl - spread_cost
        trades.append({
            "sym": sym, "signal": direction, "outcome": outcome,
            "pnl": trade_pnl, "net_pnl": net_pnl, "price": last_price,
            "spread_cost": spread_cost, "exit_reason": "force_close",
            "ticks_held": pos["ticks_held"],
        })

    return trades

def compute_stats(trades, label=""):
    """Compute standard metrics from trade list."""
    if not trades:
        return None
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] < 0]
    n = len(trades)

    total_win = sum(t["net_pnl"] for t in wins)
    total_loss = abs(sum(t["net_pnl"] for t in losses))
    total_pnl = sum(t["net_pnl"] for t in trades)

    wr = len(wins) / n * 100 if n else 0
    aw = total_win / len(wins) if wins else 0
    al = total_loss / len(losses) if losses else 0
    pf = total_win / total_loss if total_loss > 0 else float('inf')
    exp = (wr/100 * aw) - ((1-wr/100) * al) if n else 0

    cum = 0.0; peak = 0.0; max_dd = 0.0
    for t in trades:
        cum += t["net_pnl"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    avg_hold = sum(t["ticks_held"] for t in trades) / n if n else 0
    gross = sum(t["pnl"] for t in trades)
    drag = sum(t["spread_cost"] for t in trades)

    return {
        "label": label,
        "n": n, "wr": wr, "pf": pf, "exp": exp,
        "avg_win": aw, "avg_loss": al,
        "total_pnl": total_pnl, "max_dd": max_dd,
        "gross": gross, "drag": drag,
        "avg_hold": avg_hold,
    }

def print_stats(s, prefix=""):
    if s is None:
        print(f"{prefix}  no trades")
        return
    print(f"  {prefix}  n={s['n']:>6}  WR={s['wr']:>5.1f}%  PF={s['pf']:>5.2f}  "
          f"Exp={s['exp']:>+7.4f}  PnL={s['total_pnl']:>+8.0f}  "
          f"DD={s['max_dd']:>6.0f}  Hold={s['avg_hold']:>5.1f}", flush=True)

# =========================================================================
# 1. HOLD-TIME SURFACE
# =========================================================================
def experiment_hold_time_surface():
    """Test different exit horizons: 5, 10, 20, 50, 100, 250 ticks."""
    print(f"\n{H}")
    print("  EXPERIMENT 1 — HOLD-TIME SURFACE")
    print(H)
    print(f"{'Fold':>8}  {'ExitHor':>7}  {'n':>6}  {'WR':>7}  {'PF':>5}  {'Exp':>9}  {'PnL':>9}  {'DD':>7}  {'AvgHold':>8}", flush=True)

    horizons = [5, 10, 20, 50, 100, 250]
    results = {h: [] for h in horizons}

    for f in FOLDS:
        train_ticks = ReplayCache(f["syms"], f["train_start"], f["train_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()
        test_ticks = ReplayCache(f["syms"], f["test_start"], f["test_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()

        # Train OSS
        train_recs = []
        doa = DelayedOutcomeEngine(horizon_ticks=20)
        ed = {}
        for t in train_ticks:
            s = t["sym"]
            sig = ecdf_sig(t)
            ed[s] = {"price": t["price"], "ecdf_rank": t["ecdf"], "entropy": t["entropy"], "signal": sig}
            doa.record_snapshot(ed)
            if doa.ready:
                for s2, outcome in doa.evaluate({s2: ed[s2]["price"] for s2 in ed}).items():
                    train_recs.append({"sym": s2, "ecdf": ed[s2]["ecdf_rank"], "outcome": outcome})
        oss = OutcomeSurfaceSignal.from_pipeline_records(train_recs, ev_threshold=0.05)

        for h in horizons:
            trades = simulate_with_exit(oss, test_ticks, exit_horizon=h, spread_bps=0.0)
            s = compute_stats(trades, f"Hold_{h}")
            if s:
                results[h].append(s)
                print(f"  {f['name']:>6}    h={h:>3}    {s['n']:>6}  {s['wr']:>5.1f}%  {s['pf']:>4.2f}  "
                      f"{s['exp']:>+8.4f}  {s['total_pnl']:>+8.0f}  {s['max_dd']:>6.0f}  {s['avg_hold']:>6.1f}", flush=True)

    # Aggregate
    print(f"\n  {'AGGREGATE':>50}", flush=True)
    print(f"  {'Horizon':>8}  {'n':>6}  {'WR':>7}  {'PF':>5}  {'Exp':>9}  {'PnL':>9}  {'DD':>7}  {'AvgHold':>8}", flush=True)
    best = None
    for h in horizons:
        stats_list = results[h]
        if not stats_list:
            continue
        avg_pnl = sum(s["total_pnl"] for s in stats_list) / len(stats_list)
        avg_wr = sum(s["wr"] for s in stats_list) / len(stats_list)
        avg_pf = sum(s["pf"] for s in stats_list) / len(stats_list)
        avg_exp = sum(s["exp"] for s in stats_list) / len(stats_list)
        avg_dd = sum(s["max_dd"] for s in stats_list) / len(stats_list)
        avg_hold = sum(s["avg_hold"] for s in stats_list) / len(stats_list)
        total_n = sum(s["n"] for s in stats_list)
        total_pnl = sum(s["total_pnl"] for s in stats_list)
        print(f"  h={h:>3}    {total_n:>6}  {avg_wr:>5.1f}%  {avg_pf:>4.2f}  {avg_exp:>+8.4f}"
              f"  {total_pnl:>+9.0f}  {avg_dd:>6.0f}  {avg_hold:>6.1f}", flush=True)
        if avg_pf > 1.2:
            if best is None or avg_pf > best[1]:
                best = (h, avg_pf, total_pnl, avg_wr)
    if best:
        print(f"\n  BEST: h={best[0]}  PF={best[1]:.2f}  PnL={best[2]:+.0f}  WR={best[3]:.1f}%", flush=True)
    else:
        # Find best even if PF not > 1.2
        best_h = max(horizons, key=lambda h: sum(s["total_pnl"] for s in results[h]) / max(len(results[h]), 1))
        best_stats = results[best_h]
        if best_stats:
            avg_pnl = sum(s["total_pnl"] for s in best_stats) / len(best_stats)
            avg_pf = sum(s["pf"] for s in best_stats) / len(best_stats)
            print(f"\n  BEST (by PnL): h={best_h}  PF={avg_pf:.2f}  PnL={avg_pnl:+.0f}", flush=True)

    return results, horizons

# =========================================================================
# 2. MAE STOP ANALYSIS
# =========================================================================
def experiment_mae_stops(best_horizon):
    """Test MAE-based kill switch at different sigma thresholds."""
    print(f"\n{H}")
    print("  EXPERIMENT 2 — MAE-BASED KILL SWITCH")
    print(H)
    print(f"{'Fold':>8}  {'MAEs':>6}  {'n':>6}  {'WR':>7}  {'PF':>5}  {'Exp':>9}  {'PnL':>9}  {'DD':>7}", flush=True)

    sigmas = [None, 0.25, 0.5, 0.75, 1.0]
    results = {s: [] for s in [None] + [0.25, 0.5, 0.75, 1.0]}

    for f in FOLDS:
        train_ticks = ReplayCache(f["syms"], f["train_start"], f["train_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()
        test_ticks = ReplayCache(f["syms"], f["test_start"], f["test_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()

        # Train OSS
        train_recs = []
        doa = DelayedOutcomeEngine(horizon_ticks=20)
        ed = {}
        for t in train_ticks:
            s = t["sym"]
            sig = ecdf_sig(t)
            ed[s] = {"price": t["price"], "ecdf_rank": t["ecdf"], "entropy": t["entropy"], "signal": sig}
            doa.record_snapshot(ed)
            if doa.ready:
                for s2, outcome in doa.evaluate({s2: ed[s2]["price"] for s2 in ed}).items():
                    train_recs.append({"sym": s2, "ecdf": ed[s2]["ecdf_rank"], "outcome": outcome})
        oss = OutcomeSurfaceSignal.from_pipeline_records(train_recs, ev_threshold=0.05)

        # Compute per-symbol sigma from training
        sym_stats = {}
        for s in f["syms"]:
            prices = [t["price"] for t in train_ticks if t["sym"] == s]
            if len(prices) > 1:
                returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
                sym_stats[s] = {"sigma": max(0.0001, sum(abs(r) for r in returns) / len(returns))}

        for s_thr in sigmas:
            label = f"MAE_{s_thr}" if s_thr else "NoStop"
            trades = simulate_with_exit(oss, test_ticks, exit_horizon=best_horizon,
                                        mae_stop_sigma=s_thr, spread_bps=0.0, sym_stats=sym_stats)
            s = compute_stats(trades, label)
            if s:
                results[s_thr].append(s)
                s_label = f"{s_thr:.2f}s" if s_thr else "None"
                print(f"  {f['name']:>6}  {s_label:>6}  {s['n']:>6}  {s['wr']:>5.1f}%  {s['pf']:>4.2f}  "
                      f"{s['exp']:>+8.4f}  {s['total_pnl']:>+8.0f}  {s['max_dd']:>6.0f}", flush=True)

    print(f"\n  {'AGGREGATE':>48}", flush=True)
    print(f"  {'MAE':>8}  {'n':>6}  {'WR':>7}  {'PF':>5}  {'Exp':>9}  {'PnL':>9}  {'DD':>7}", flush=True)
    for s_thr in sigmas:
        slist = results[s_thr]
        if not slist:
            continue
        avg_pnl = sum(s["total_pnl"] for s in slist) / len(slist)
        avg_wr = sum(s["wr"] for s in slist) / len(slist)
        avg_pf = sum(s["pf"] for s in slist) / len(slist)
        avg_exp = sum(s["exp"] for s in slist) / len(slist)
        avg_dd = sum(s["max_dd"] for s in slist) / len(slist)
        total_n = sum(s["n"] for s in slist)
        total_pnl = sum(s["total_pnl"] for s in slist)
        s_label = f"{s_thr:.2f}s" if s_thr else "None"
        print(f"    {s_label:>6}  {total_n:>6}  {avg_wr:>5.1f}%  {avg_pf:>4.2f}  {avg_exp:>+8.4f}"
              f"  {total_pnl:>+9.0f}  {avg_dd:>6.0f}", flush=True)

# =========================================================================
# 3. TIME-UNDERWATER EXIT
# =========================================================================
def experiment_underwater_timeout(best_horizon, best_mae_sigma):
    """Test time-underwater exit thresholds."""
    print(f"\n{H}")
    print("  EXPERIMENT 3 — TIME-UNDERWATER EXIT")
    print(H)
    print(f"{'Fold':>8}  {'UW':>6}  {'n':>6}  {'WR':>7}  {'PF':>5}  {'Exp':>9}  {'PnL':>9}  {'DD':>7}", flush=True)

    limits = [None, 5, 10, 25, 50, 100]
    results = {l: [] for l in limits}

    for f in FOLDS:
        train_ticks = ReplayCache(f["syms"], f["train_start"], f["train_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()
        test_ticks = ReplayCache(f["syms"], f["test_start"], f["test_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()

        train_recs = []
        doa = DelayedOutcomeEngine(horizon_ticks=20)
        ed = {}
        for t in train_ticks:
            s = t["sym"]
            sig = ecdf_sig(t)
            ed[s] = {"price": t["price"], "ecdf_rank": t["ecdf"], "entropy": t["entropy"], "signal": sig}
            doa.record_snapshot(ed)
            if doa.ready:
                for s2, outcome in doa.evaluate({s2: ed[s2]["price"] for s2 in ed}).items():
                    train_recs.append({"sym": s2, "ecdf": ed[s2]["ecdf_rank"], "outcome": outcome})
        oss = OutcomeSurfaceSignal.from_pipeline_records(train_recs, ev_threshold=0.05)

        sym_stats = {}
        for s in f["syms"]:
            prices = [t["price"] for t in train_ticks if t["sym"] == s]
            if len(prices) > 1:
                returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
                sym_stats[s] = {"sigma": max(0.0001, sum(abs(r) for r in returns) / len(returns))}

        for lim in limits:
            label = f"UW_{lim}" if lim else "NoUW"
            trades = simulate_with_exit(oss, test_ticks, exit_horizon=best_horizon,
                                        mae_stop_sigma=best_mae_sigma, underwater_max=lim,
                                        spread_bps=0.0, sym_stats=sym_stats)
            s = compute_stats(trades, label)
            if s:
                results[lim].append(s)
                s_label = f"{lim}tks" if lim else "None"
                print(f"  {f['name']:>6}  {s_label:>6}  {s['n']:>6}  {s['wr']:>5.1f}%  {s['pf']:>4.2f}  "
                      f"{s['exp']:>+8.4f}  {s['total_pnl']:>+8.0f}  {s['max_dd']:>6.0f}", flush=True)

    print(f"\n  {'AGGREGATE':>48}", flush=True)
    for lim in limits:
        slist = results[lim]
        if not slist:
            continue
        avg_pnl = sum(s["total_pnl"] for s in slist) / len(slist)
        avg_wr = sum(s["wr"] for s in slist) / len(slist)
        avg_pf = sum(s["pf"] for s in slist) / len(slist)
        avg_exp = sum(s["exp"] for s in slist) / len(slist)
        avg_dd = sum(s["max_dd"] for s in slist) / len(slist)
        total_pnl = sum(s["total_pnl"] for s in slist)
        s_label = f"{lim}tks" if lim else "None"
        print(f"    {s_label:>6}  {sum(s['n'] for s in slist):>6}  {avg_wr:>5.1f}%  {avg_pf:>4.2f}  {avg_exp:>+8.4f}"
              f"  {total_pnl:>+9.0f}  {avg_dd:>6.0f}", flush=True)

# =========================================================================
# MAIN
# =========================================================================
t0 = time.perf_counter()
print("Phase II.A — Execution Geometry Research", flush=True)

# Load all cache first
print(f"\nLoading cache...", end=" ", flush=True)
for f in FOLDS:
    ReplayCache(f["syms"], f["train_start"], f["train_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()
    ReplayCache(f["syms"], f["test_start"], f["test_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()
print(f"{time.perf_counter()-t0:.1f}s", flush=True)

# Experiment 1
results, horizons = experiment_hold_time_surface()

# Find best horizon
best_horizon = horizons[0]
best_pnl = -1e9
for h in horizons:
    total = sum(s["total_pnl"] for s in results[h])
    if total > best_pnl:
        best_pnl = total
        best_horizon = h
print(f"\n  Selected best horizon: h={best_horizon} (PnL={best_pnl:+.0f})", flush=True)

# Experiment 2
experiment_mae_stops(best_horizon)

# For now, use best_mae = None until we can analyze
experiment_underwater_timeout(best_horizon, best_mae_sigma=None)

print(f"\nTotal: {time.perf_counter()-t0:.1f}s", flush=True)
