"""SAL 3-fold test v2 — SAL feeds directly into DOA snapshot.
No separate position tracking. SAL signal replaces raw OSS in DOA pipeline.
"""
import sys; sys.path.insert(0, '.')
import time
from collections import defaultdict

from research.replay_cache import ReplayCache
from research.outcome_surface_signal import OutcomeSurfaceSignal
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from signals.sal_mapper import SignalAggregationLayer

TICK_LIMIT = 50000; SEED = 42; H = "=" * 70

FOLDS = [
    dict(name="Fold 1", syms=["EURJPY","USDJPY"], train_start="2026-03-12", train_end="2026-03-28",
         test_start="2026-03-29", test_end="2026-04-14"),
    dict(name="Fold 2", syms=["EURJPY"], train_start="2026-04-01", train_end="2026-04-20",
         test_start="2026-04-21", test_end="2026-05-10"),
    dict(name="Fold 3", syms=["EURJPY"], train_start="2026-05-01", train_end="2026-05-20",
         test_start="2026-05-21", test_end="2026-06-08"),
]

SAL_PARAMS = dict(entry_threshold=0.65, accum_window=20, accum_decay=0.85,
                  consensus_min=0.60, vol_window=20, vol_min_z=0.15, vol_max_z=2.0)

def ecdf_sig(t):
    d = t.get("ecdf", 0.5) - 0.5
    return 1 if d > 0.05 else (-1 if d < -0.05 else 0)

def run_simulation(ticks, oss, signal_fn, sym_stats):
    """Generic simulation: signal_fn(sym, price, oss_signal) -> trade_signal.
    Uses DOA h=5 for exit timing, real price-based PnL from buffer.
    """
    horizon = 5; doa = DelayedOutcomeEngine(horizon_ticks=horizon)
    ed = {}; trades = []; price_buffer = defaultdict(lambda: [])
    sal_interface = signal_fn.__closure__[0].cell_contents if hasattr(signal_fn, '__closure__') and signal_fn.__closure__ else None

    for t in ticks:
        s = t["sym"]; price = t["price"]
        raw_oss = oss.predict(t["ecdf"])
        trade_sig = signal_fn(s, price, raw_oss)
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
                    entry_price = buf[-(horizon+1)] if len(buf) >= horizon+1 else (cp[s2] - 0.001)
                    exit_price = cp[s2]
                    direction = sig_at_entry

                    # MAE check using intra-horizon path
                    sym_vol = sym_stats.get(s2, {}).get("sigma", 0.001)
                    mae = 0.15
                    threshold = mae * sym_vol
                    stop_idx = horizon
                    if len(buf) >= horizon + 1:
                        path = buf[-(horizon+1):]
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

                    if direction == 1:
                        pnl_raw = exit_price - entry_price
                    else:
                        pnl_raw = entry_price - exit_price
                    trades.append({"pnl_raw": pnl_raw, "pnl_net": pnl_raw, "signal": direction, "sym": s2,
                                   "entry_price": entry_price, "exit_price": exit_price})
    return trades

def stats(trades):
    if not trades: return None
    n = len(trades)
    wins = [t for t in trades if t["pnl_net"] > 0]
    losses = [t for t in trades if t["pnl_net"] < 0]
    wr = len(wins) / n * 100 if n else 0
    tw = sum(t["pnl_net"] for t in wins)
    tl = abs(sum(t["pnl_net"] for t in losses))
    pf = tw / tl if tl > 0 else float('inf')
    aw = tw / len(wins) if wins else 0
    al = tl / len(losses) if losses else 0
    exp = (wr/100 * aw) - ((1-wr/100) * al) if n else 0
    return {"n": n, "wr": wr, "pf": pf, "exp": exp, "total_pnl": sum(t["pnl_net"] for t in trades),
            "avg_win": aw, "avg_loss": al}

t0 = time.perf_counter()
print(f"{H}\n  SIGNAL AGGREGATION LAYER v2 — DOA-integrated\n{H}")

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
            returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
            sym_stats[s] = {"sigma": max(0.0001, sum(abs(r) for r in returns) / len(returns))}

    # Baseline: raw OSS signal
    def raw_signal(sym, price, oss_sig):
        return oss_sig

    base_trades = run_simulation(test_ticks, oss, raw_signal, sym_stats)
    bs = stats(base_trades)

    # SAL: aggregated signal
    thr_grid = [0.20, 0.30, 0.40, 0.50, 0.65]
    print(f"  {'System':>10}  {'n':>7}  {'WR':>6}  {'PF':>6}  {'Exp':>9}  {'PnL':>10}  {'AvgW':>8}  {'AvgL':>8}")
    if bs:
        print(f"  {'Baseline':>10}  {bs['n']:>7}  {bs['wr']:>5.1f}%  {bs['pf']:>5.2f}  {bs['exp']:>+8.4f}  "
              f"{bs['total_pnl']:>+9.4f}  {bs['avg_win']:>+7.4f}  {bs['avg_loss']:>+7.4f}")

    for thr in thr_grid:
        sal = SignalAggregationLayer(sym_stats=sym_stats, **dict(SAL_PARAMS, entry_threshold=thr))
        sal_scores = {}

        def make_sal_fn(sal_instance=sal, scores_dict=sal_scores):
            def fn(sym, price, oss_sig):
                scores_dict[sym] = sal_instance.accumulator.score()
                return sal_instance.update(sym, oss_sig, 1.0, price, all_scores=scores_dict)
            return fn

        sal_trades = run_simulation(test_ticks, oss, make_sal_fn(), sym_stats)
        ss = stats(sal_trades)
        if ss:
            print(f"  SAL-{thr:.2f}     {ss['n']:>7}  {ss['wr']:>5.1f}%  {ss['pf']:>5.2f}  {ss['exp']:>+8.4f}  "
                  f"{ss['total_pnl']:>+9.4f}  {ss['avg_win']:>+7.4f}  {ss['avg_loss']:>+7.4f}")

print(f"\n{time.perf_counter()-t0:.1f}s")
