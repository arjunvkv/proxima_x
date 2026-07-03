"""State Transition Engine — 3-fold validation.
Compares: OSS, OSS+SAL, OSS+SAL+TCL, STE (standalone), STE+OSS+SAL.
"""
import sys; sys.path.insert(0, '.')
import time
from collections import defaultdict

from research.replay_cache import ReplayCache
from research.outcome_surface_signal import OutcomeSurfaceSignal
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from signals.sal_mapper import SignalAggregationLayer
from signals.tcl_mapper import TemporalCompressionLayer
from signals.state_transition import RollingTransitionTracker

TICK_LIMIT = 50000; SEED = 42; H = "=" * 70

FOLDS = [
    dict(name="Fold 1", syms=["EURJPY","USDJPY"], train_start="2026-03-12", train_end="2026-03-28",
         test_start="2026-03-29", test_end="2026-04-14"),
    dict(name="Fold 2", syms=["EURJPY"], train_start="2026-04-01", train_end="2026-04-20",
         test_start="2026-04-21", test_end="2026-05-10"),
    dict(name="Fold 3", syms=["EURJPY"], train_start="2026-05-01", train_end="2026-05-20",
         test_start="2026-05-21", test_end="2026-06-08"),
]

def ecdf_sig(t):
    d = t.get("ecdf", 0.5) - 0.5
    return 1 if d > 0.05 else (-1 if d < -0.05 else 0)

def run_simulation(ticks, signal_fn, sym_stats, horizon=5):
    doa = DelayedOutcomeEngine(horizon_ticks=horizon)
    ed = {}; trades = []; price_buffer = defaultdict(lambda: [])
    for t in ticks:
        s = t["sym"]; price = t["price"]
        trade_sig = signal_fn(s, price, t["ecdf"])
        price_buffer[s].append(price)
        ed[s] = {"price": price, "ecdf_rank": t["ecdf"], "entropy": t["entropy"], "signal": trade_sig}
        doa.record_snapshot(ed)
        if doa.ready:
            cp = {s2: ed[s2]["price"] for s2 in ed}
            outcomes = doa.evaluate(cp)
            for s2, outcome in outcomes.items():
                sig_at_entry = ed[s2]["signal"]
                if sig_at_entry != 0:
                    buf = price_buffer[s2]
                    entry_price = buf[-(horizon+1)] if len(buf) >= horizon+1 else cp[s2]
                    exit_price = cp[s2]; direction = sig_at_entry
                    sym_vol = sym_stats.get(s2, {}).get("sigma", 0.001)
                    threshold = 0.15 * sym_vol; stop_idx = horizon
                    if len(buf) >= horizon + 1:
                        path = buf[-(horizon+1):]
                        for i in range(1, len(path)):
                            p = path[i]
                            if direction == 1: excursion = (entry_price - p) / entry_price if entry_price != 0 else 0
                            else: excursion = (p - entry_price) / entry_price if entry_price != 0 else 0
                            if excursion < -threshold: stop_idx = i; break
                        exit_price = path[stop_idx]
                    pnl_raw = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
                    trades.append({"pnl_net": pnl_raw})
    return trades

def stats(trades):
    if not trades: return None
    n = len(trades)
    wins = [t for t in trades if t["pnl_net"] > 0]
    losses = [t for t in trades if t["pnl_net"] < 0]
    wr = len(wins)/n*100
    tw = sum(t["pnl_net"] for t in wins)
    tl = abs(sum(t["pnl_net"] for t in losses))
    pf = tw/tl if tl > 0 else float('inf')
    aw = tw/len(wins) if wins else 0
    al = tl/len(losses) if losses else 0
    total = sum(t["pnl_net"] for t in trades)
    exp = (wr/100*aw)-((1-wr/100)*al)
    return {"n": n, "wr": wr, "pf": pf, "exp": exp, "total_pnl": total, "avg_win": aw, "avg_loss": al}

t0 = time.perf_counter()
print(f"{H}\n  STATE TRANSITION ENGINE — 3-FOLD VALIDATION\n{H}")

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

    sym_stats = {}
    for s in f["syms"]:
        prices = [t["price"] for t in train_ticks if t["sym"] == s]
        if len(prices) > 1:
            returns = [(prices[i+1]-prices[i])/prices[i] for i in range(len(prices)-1)]
            sym_stats[s] = {"sigma": max(0.0001, sum(abs(r) for r in returns)/len(returns))}

    print(f"  {'System':>15}  {'n':>7}  {'WR':>5}  {'PF':>5}  {'Exp':>8}  {'PnL':>9}  {'AvgW':>7}  {'AvgL':>7}  {'BAR':>5}")

    results = []

    # 1. OSS only
    def oss_fn(s, p, ecdf): return oss.predict(ecdf)
    t1 = run_simulation(test_ticks, oss_fn, sym_stats)
    s1 = stats(t1)
    if s1:
        bar = s1["avg_win"] / max(0.0001, s1["avg_loss"])
        print(f"  {'OSS':>15}  {s1['n']:>7}  {s1['wr']:>4.1f}%  {s1['pf']:>4.2f}  {s1['exp']:>+7.4f}  "
              f"{s1['total_pnl']:>+8.4f}  {s1['avg_win']:>+6.4f}  {s1['avg_loss']:>+6.4f}  {bar:>4.2f}")
        results.append(("OSS", s1))

    # 2. OSS + SAL
    sal = SignalAggregationLayer(sym_stats=sym_stats, entry_threshold=0.65,
                                  accum_window=20, accum_decay=0.85, consensus_min=0.60)
    sal_scores = {}
    def sal_fn(s, p, ecdf):
        sal_scores[s] = sal.accumulator.score()
        return sal.update(s, oss.predict(ecdf), 1.0, p, all_scores=sal_scores)
    t2 = run_simulation(test_ticks, sal_fn, sym_stats)
    s2 = stats(t2)
    if s2:
        bar = s2["avg_win"] / max(0.0001, s2["avg_loss"])
        print(f"  {'OSS+SAL':>15}  {s2['n']:>7}  {s2['wr']:>4.1f}%  {s2['pf']:>4.2f}  {s2['exp']:>+7.4f}  "
              f"{s2['total_pnl']:>+8.4f}  {s2['avg_win']:>+6.4f}  {s2['avg_loss']:>+6.4f}  {bar:>4.2f}")
        results.append(("OSS+SAL", s2))

    # 3. OSS + SAL + TCL
    sal2 = SignalAggregationLayer(sym_stats=sym_stats, entry_threshold=0.65,
                                   accum_window=20, accum_decay=0.85, consensus_min=0.60)
    tcl = TemporalCompressionLayer(compression_window=8, min_same_dir=2, min_compression=0.40,
                                    min_density=0.40, vol_expansion=1.15, sal_threshold=0.50)
    sal2_scores = {}
    def tcl_fn(s, p, ecdf):
        sal2_scores[s] = sal2.accumulator.score()
        sig = sal2.update(s, oss.predict(ecdf), 1.0, p, all_scores=sal2_scores)
        return tcl.update(sig, 1.0, p)
    t3 = run_simulation(test_ticks, tcl_fn, sym_stats)
    s3 = stats(t3)
    if s3:
        bar = s3["avg_win"] / max(0.0001, s3["avg_loss"])
        print(f"  {'OSS+SAL+TCL':>15}  {s3['n']:>7}  {s3['wr']:>4.1f}%  {s3['pf']:>4.2f}  {s3['exp']:>+7.4f}  "
              f"{s3['total_pnl']:>+8.4f}  {s3['avg_win']:>+6.4f}  {s3['avg_loss']:>+6.4f}  {bar:>4.2f}")
        results.append(("OSS+SAL+TCL", s3))

    # 4. STE standalone (per-symbol instances)
    ste_instances = {}
    def ste_fn(s, p, ecdf):
        if s not in ste_instances:
            ste_instances[s] = RollingTransitionTracker(window=100, entropy_threshold=0.60,
                                                         min_direction=0.15, min_amplitude=0.02)
        return ste_instances[s].update(ecdf, p)
    t4 = run_simulation(test_ticks, ste_fn, sym_stats)
    s4 = stats(t4)
    if s4:
        bar = s4["avg_win"] / max(0.0001, s4["avg_loss"])
        print(f"  {'STE':>15}  {s4['n']:>7}  {s4['wr']:>4.1f}%  {s4['pf']:>4.2f}  {s4['exp']:>+7.4f}  "
              f"{s4['total_pnl']:>+8.4f}  {s4['avg_win']:>+6.4f}  {s4['avg_loss']:>+6.4f}  {bar:>4.2f}")
        results.append(("STE", s4))

    # 5. STE + OSS combined (vote, per-symbol instances)
    ste2_instances = {}
    def steoss_fn(s, p, ecdf):
        if s not in ste2_instances:
            ste2_instances[s] = RollingTransitionTracker(window=100, entropy_threshold=0.60,
                                                          min_direction=0.15, min_amplitude=0.02)
        ste_sig = ste2_instances[s].update(ecdf, p)
        oss_sig = oss.predict(ecdf)
        if ste_sig == oss_sig:
            return ste_sig
        return 0
    t5 = run_simulation(test_ticks, steoss_fn, sym_stats)
    s5 = stats(t5)
    if s5:
        bar = s5["avg_win"] / max(0.0001, s5["avg_loss"])
        print(f"  {'STE+OSS':>15}  {s5['n']:>7}  {s5['wr']:>4.1f}%  {s5['pf']:>4.2f}  {s5['exp']:>+7.4f}  "
              f"{s5['total_pnl']:>+8.4f}  {s5['avg_win']:>+6.4f}  {s5['avg_loss']:>+6.4f}  {bar:>4.2f}")
        results.append(("STE+OSS", s5))

print(f"\n{time.perf_counter()-t0:.1f}s")
