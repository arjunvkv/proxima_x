"""Spread Sensitivity — real price-based PnL via post-hoc analysis.
Uses the proven approach: DOA simulation + price path recording + MAE + spread cost.
All PnL in price-difference units.

PnL = direction * (exit_price - entry_price)
Spread cost (round trip) = spread_bps / 10000 * (entry_price + exit_price)
"""
import sys; sys.path.insert(0, '.')
import time
from collections import defaultdict

from research.replay_cache import ReplayCache
from research.outcome_surface_signal import OutcomeSurfaceSignal
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine

TICK_LIMIT = 50000; SEED = 42; H = "=" * 70

FOLDS = [
    dict(name="Fold 1", syms=["EURJPY","USDJPY"], train_start="2026-03-12", train_end="2026-03-28",
         test_start="2026-03-29", test_end="2026-04-14"),
    dict(name="Fold 2", syms=["EURJPY"], train_start="2026-04-01", train_end="2026-04-20",
         test_start="2026-04-21", test_end="2026-05-10"),
    dict(name="Fold 3", syms=["EURJPY"], train_start="2026-05-01", train_end="2026-05-20",
         test_start="2026-05-21", test_end="2026-06-08"),
]

MAE_SIGMA = 0.15
HORIZON = 5
SPREADS = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5]

def ecdf_sig(t):
    d = t.get("ecdf", 0.5) - 0.5
    return 1 if d > 0.05 else (-1 if d < -0.05 else 0)

def run_simulation(ticks, oss):
    """Run DOA h=5 simulation, recording price path in each trade.

    Returns list of trades: {pnl_doa, signal, sym, price_path}
    price_path is [entry_price, price_t+1, ..., exit_price] (len=horizon+1)
    """
    horizon = HORIZON
    doa = DelayedOutcomeEngine(horizon_ticks=horizon)
    ed = {}
    trades = []
    price_buffer = defaultdict(lambda: [])

    for t in ticks:
        s = t["sym"]
        price = t["price"]
        sig_at_entry = oss.predict(t["ecdf"])
        ed[s] = {"price": price, "ecdf_rank": t["ecdf"], "entropy": t["entropy"], "signal": sig_at_entry}
        price_buffer[s].append(price)

        doa.record_snapshot(ed)

        if doa.ready:
            cp = {s2: ed[s2]["price"] for s2 in ed}
            outcomes = doa.evaluate(cp)
            for s2, outcome in outcomes.items():
                sig = ed[s2]["signal"]
                if sig != 0:
                    trades.append({
                        "pnl_doa": sig * outcome,
                        "signal": sig,
                        "sym": s2,
                        "price_path": price_buffer[s2][-(horizon+1):]
                        if len(price_buffer[s2]) >= horizon + 1 else None,
                    })

    return trades

def apply_mae_spread(base_trades, mae_sigma, spread_bps, sym_stats):
    """Apply MAE stop and spread cost to base trades.

    Computes real price-based PnL from price paths stored in each trade.
    PnL = direction * (exit_price - entry_price)
    Spread cost = spread_bps/10000 * (entry_price + exit_price)

    Returns list of trade dicts with pnl_net.
    """
    result = []
    for t in base_trades:
        path = t.get("price_path")

        if path is None or len(path) < 2:
            # No price path available — use DOA PnL as rough estimate
            t2 = dict(t, pnl_raw=t["pnl_doa"], spread_cost=0.0, pnl_net=t["pnl_doa"],
                      stopped=False, ticks_to_stop=HORIZON, entry_price=0, exit_price=0)
            result.append(t2)
            continue

        entry_price = path[0]
        direction = t["signal"]
        sym_vol = sym_stats.get(t["sym"], {}).get("sigma", 0.001)
        threshold = (mae_sigma * sym_vol) if mae_sigma else 0

        stop_idx = HORIZON
        for i in range(1, len(path)):
            p = path[i]
            if direction == 1:
                excursion = (entry_price - p) / entry_price if entry_price != 0 else 0
            else:
                excursion = (p - entry_price) / entry_price if entry_price != 0 else 0
            if excursion < -threshold:
                stop_idx = i
                break

        exit_price = path[stop_idx]
        stopped = stop_idx < HORIZON

        # Real PnL in price units
        if direction == 1:
            pnl_raw = exit_price - entry_price
        else:
            pnl_raw = entry_price - exit_price

        # Spread cost (round trip, same price units)
        spread_cost = spread_bps / 10000 * (entry_price + exit_price)

        result.append({
            "pnl_raw": pnl_raw,
            "spread_cost": spread_cost,
            "pnl_net": pnl_raw - spread_cost,
            "signal": direction,
            "sym": t["sym"],
            "stopped": stopped,
            "ticks_to_stop": stop_idx,
            "entry_price": entry_price,
            "exit_price": exit_price,
        })

    return result

def compute_stats(trades, label=""):
    n = len(trades)
    if n == 0:
        return None
    wins = [t for t in trades if t["pnl_net"] > 0]
    losses = [t for t in trades if t["pnl_net"] < 0]
    wr = len(wins) / n * 100 if n else 0
    tw = sum(t["pnl_net"] for t in wins)
    tl = abs(sum(t["pnl_net"] for t in losses))
    aw = tw / len(wins) if wins else 0
    al = tl / len(losses) if losses else 0
    pf = tw / tl if tl > 0 else float('inf')
    exp = (wr/100 * aw) - ((1-wr/100) * al) if n else 0
    total_pnl = sum(t["pnl_net"] for t in trades)

    cum = 0.0; peak = 0.0; max_dd = 0.0
    for t in trades:
        cum += t["pnl_net"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    gross = sum(t["pnl_raw"] for t in trades)
    drag = sum(t["spread_cost"] for t in trades)
    drag_pct = drag / abs(gross) * 100 if gross != 0 else 0
    stopped = sum(1 for t in trades if t.get("stopped"))
    stop_pct = stopped / n * 100 if n else 0

    return {"n": n, "wr": wr, "pf": pf, "exp": exp, "total_pnl": total_pnl,
            "max_dd": max_dd, "gross": gross, "drag": drag, "drag_pct": drag_pct,
            "stopped": stopped, "stop_pct": stop_pct, "avg_win": aw, "avg_loss": al}


t0 = time.perf_counter()
print(f"{H}")
print(f"  SPREAD SENSITIVITY (real price-based PnL)")
print(f"  OSS + h={HORIZON} + MAE {MAE_SIGMA}")
print(f"{H}")

all_stats = {sp: [] for sp in SPREADS}

for f in FOLDS:
    print(f"\n  --- {f['name']} ---")
    train_ticks = ReplayCache(f["syms"], f["train_start"], f["train_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()
    test_ticks = ReplayCache(f["syms"], f["test_start"], f["test_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()

    # Train OSS
    train_recs = []
    doa = DelayedOutcomeEngine(horizon_ticks=20)
    ed = {}
    for t in train_ticks:
        s = t["sym"]; sig = ecdf_sig(t)
        ed[s] = {"price": t["price"], "ecdf_rank": t["ecdf"], "entropy": t["entropy"], "signal": sig}
        doa.record_snapshot(ed)
        if doa.ready:
            for s2, outcome in doa.evaluate({s2: ed[s2]["price"] for s2 in ed}).items():
                train_recs.append({"sym": s2, "ecdf": ed[s2]["ecdf_rank"], "outcome": outcome})
    oss = OutcomeSurfaceSignal.from_pipeline_records(train_recs, ev_threshold=0.05)

    # Per-symbol sigma
    sym_stats = {}
    for s in f["syms"]:
        prices = [t["price"] for t in train_ticks if t["sym"] == s]
        if len(prices) > 1:
            returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
            sym_stats[s] = {"sigma": max(0.0001, sum(abs(r) for r in returns) / len(returns))}

    # Run simulation
    base_trades = run_simulation(test_ticks, oss)

    print(f"  Base trades: {len(base_trades)}  With paths: {sum(1 for t in base_trades if t.get('price_path') is not None)}")

    # Header
    print(f"  {'Spr':>5}  {'n':>7}  {'WR':>6}  {'PF':>6}  {'Exp':>9}  {'NetPnL':>10}  "
          f"{'Gross':>10}  {'Drag%':>7}  {'StopRt':>7}", flush=True)

    for sp in SPREADS:
        trades = apply_mae_spread(base_trades, MAE_SIGMA, sp, sym_stats)
        s = compute_stats(trades)
        if s:
            all_stats[sp].append(s)
            print(f"  {sp:>4.1f}  {s['n']:>7}  {s['wr']:>5.1f}%  {s['pf']:>5.2f}  {s['exp']:>+8.4f}  "
                  f"{s['total_pnl']:>+9.4f}  {s['gross']:>+9.4f}  {s['drag_pct']:>5.1f}%  "
                  f"{s['stop_pct']:>5.1f}%", flush=True)

    # Baseline: no MAE, zero spread
    base_adj = apply_mae_spread(base_trades, None, 0.0, sym_stats)
    bs = compute_stats(base_adj)
    if bs:
        print(f"  Base  {bs['n']:>7}  {bs['wr']:>5.1f}%  {bs['pf']:>5.2f}  {bs['exp']:>+8.4f}  "
              f"{bs['total_pnl']:>+9.4f}", flush=True)

# Aggregate
print(f"\n{'='*70}")
print(f"  AGGREGATE SPREAD SENSITIVITY")
print(f"{'='*70}")
print(f"  {'Spr':>5}  {'n':>7}  {'WR':>6}  {'PF':>6}  {'Exp':>9}  {'NetPnL':>10}  "
      f"{'Gross':>10}  {'Drag%':>7}", flush=True)

for sp in SPREADS:
    slist = all_stats[sp]
    if not slist:
        continue
    avg_wr = sum(s["wr"] for s in slist) / len(slist)
    avg_pf = sum(s["pf"] for s in slist) / len(slist)
    avg_exp = sum(s["exp"] for s in slist) / len(slist)
    avg_dd = sum(s["max_dd"] for s in slist) / len(slist)
    total_pnl = sum(s["total_pnl"] for s in slist)
    total_n = sum(s["n"] for s in slist)
    gross = sum(s["gross"] for s in slist)
    drag = sum(s["drag"] for s in slist)
    drag_pct = drag / abs(gross) * 100 if gross != 0 else 0
    paper = "  PAPER" if avg_pf >= 1.30 else ""
    live = "  LIVE" if avg_pf >= 1.15 else ""
    print(f"  {sp:>4.1f}  {total_n:>7}  {avg_wr:>5.1f}%  {avg_pf:>5.2f}  {avg_exp:>+8.4f}  "
          f"{total_pnl:>+9.4f}  {gross:>+9.4f}  {drag_pct:>5.1f}%{paper}{live}", flush=True)

print(f"\n{time.perf_counter()-t0:.1f}s")
