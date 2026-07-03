"""TCL 3-fold validation — clean instance management.
Compares OSS-only, OSS+SAL, OSS+SAL+TCL with fresh instances per test.
"""
import sys; sys.path.insert(0, '.')
import time
from collections import defaultdict

from research.replay_cache import ReplayCache
from research.outcome_surface_signal import OutcomeSurfaceSignal
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from signals.sal_mapper import SignalAggregationLayer
from signals.tcl_mapper import TemporalCompressionLayer

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

def run_simulation(ticks, oss, signal_fn, sym_stats):
    horizon = 5; doa = DelayedOutcomeEngine(horizon_ticks=horizon)
    ed = {}; trades = []; price_buffer = defaultdict(lambda: [])

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
                    entry_price = buf[-(horizon+1)] if len(buf) >= horizon+1 else cp[s2]
                    exit_price = cp[s2]
                    direction = sig_at_entry
                    sym_vol = sym_stats.get(s2, {}).get("sigma", 0.001)
                    threshold = 0.15 * sym_vol
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
                                stop_idx = i; break
                        exit_price = path[stop_idx]
                    if direction == 1:
                        pnl_raw = exit_price - entry_price
                    else:
                        pnl_raw = entry_price - exit_price
                    trades.append({"pnl_raw": pnl_raw, "pnl_net": pnl_raw, "signal": direction, "sym": s2})
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
    exp = (wr/100*aw)-((1-wr/100)*al) if n else 0
    return {"n": n, "wr": wr, "pf": pf, "exp": exp, "total_pnl": sum(t["pnl_net"] for t in trades),
            "avg_win": aw, "avg_loss": al}

def make_oss_fn():
    def fn(sym, price, oss_sig):
        return oss_sig
    return fn

def make_sal_fn(sym_stats):
    sal = SignalAggregationLayer(sym_stats=sym_stats, entry_threshold=0.65,
                                  accum_window=20, accum_decay=0.85, consensus_min=0.60)
    scores = {}
    def fn(sym, price, oss_sig):
        scores[sym] = sal.accumulator.score()
        return sal.update(sym, oss_sig, 1.0, price, all_scores=scores)
    return fn

def make_tcl_fn(sym_stats):
    sal = SignalAggregationLayer(sym_stats=sym_stats, entry_threshold=0.65,
                                  accum_window=20, accum_decay=0.85, consensus_min=0.60)
    tcl = TemporalCompressionLayer(sal_threshold=0.65)
    scores = {}
    def fn(sym, price, oss_sig):
        scores[sym] = sal.accumulator.score()
        sal_sig = sal.update(sym, oss_sig, 1.0, price, all_scores=scores)
        return tcl.update(sal_sig, 1.0, price)
    return fn

t0 = time.perf_counter()
print(f"{H}\n  TCL 3-FOLD VALIDATION\n{H}")

for f in FOLDS:
    print(f"\n  --- {f['name']} ---")
    train_ticks = ReplayCache(f["syms"], f["train_start"], f["train_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()
    test_ticks = ReplayCache(f["syms"], f["test_start"], f["test_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()

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

    print(f"  {'System':>15}  {'n':>7}  {'WR':>6}  {'PF':>6}  {'Exp':>9}  {'PnL':>10}  {'AvgW':>8}  {'AvgL':>8}")

    for name, fn_maker in [("OSS", make_oss_fn), ("OSS+SAL", make_sal_fn), ("OSS+SAL+TCL", make_tcl_fn)]:
        fn = fn_maker() if name == "OSS" else fn_maker(sym_stats)
        trades = run_simulation(test_ticks, oss, fn, sym_stats)
        s = stats(trades)
        if s:
            bar = s["avg_win"] / max(0.0001, s["avg_loss"])
            print(f"  {name:>15}  {s['n']:>7}  {s['wr']:>5.1f}%  {s['pf']:>5.2f}  {s['exp']:>+8.4f}  "
                  f"{s['total_pnl']:>+9.4f}  {s['avg_win']:>+7.4f}  {s['avg_loss']:>+7.4f}  {bar:>5.2f}", flush=True)

print(f"\n{time.perf_counter()-t0:.1f}s")
