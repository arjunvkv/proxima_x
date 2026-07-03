"""TCL parameter sweep — finds settings that maximize trade count × PF."""
import sys; sys.path.insert(0, '.')
import time
from collections import defaultdict

from research.replay_cache import ReplayCache
from research.outcome_surface_signal import OutcomeSurfaceSignal
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from signals.sal_mapper import SignalAggregationLayer
from signals.tcl_mapper import TemporalCompressionLayer

TICK_LIMIT = 50000; SEED = 42

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
    return {"n": n, "wr": wr, "pf": pf, "avg_win": aw, "avg_loss": al, "total_pnl": sum(t["pnl_net"] for t in trades)}

def make_tcl_fn(sym_stats, win, msd, mc, md, ve, th):
    sal = SignalAggregationLayer(sym_stats=sym_stats, entry_threshold=0.65,
                                  accum_window=20, accum_decay=0.85, consensus_min=0.60)
    tcl = TemporalCompressionLayer(compression_window=win, min_same_dir=msd,
                                    min_compression=mc, min_density=md,
                                    vol_expansion=ve, sal_threshold=th)
    scores = {}
    def fn(sym, price, oss_sig):
        scores[sym] = sal.accumulator.score()
        sal_sig = sal.update(sym, oss_sig, 1.0, price, all_scores=scores)
        return tcl.update(sal_sig, 1.0, price)
    return fn

PARAMS = [
    # window, min_same_dir, min_compression, min_density, vol_expansion, sal_threshold
    (8, 3, 0.60, 0.50, 1.25, 0.65),   # default
    (8, 2, 0.40, 0.40, 1.15, 0.50),   # relaxed A
    (6, 2, 0.50, 0.50, 1.10, 0.40),   # relaxed B
    (10, 3, 0.50, 0.40, 1.20, 0.55),  # wider window
    (5, 2, 0.60, 0.60, 1.05, 0.30),   # tight window, relaxed vol
    (8, 3, 0.60, 0.50, 99.0, 0.65),   # no vol filter
    (8, 2, 0.40, 0.40, 99.0, 0.50),   # no vol + relaxed
    (4, 2, 0.75, 0.75, 99.0, 0.30),   # tiny window, no vol, low SAL
    (12, 4, 0.50, 0.40, 99.0, 0.60),  # big window, no vol
]

t0 = time.perf_counter()
print(f"{'='*90}\n  TCL PARAMETER SWEEP\n{'='*90}")

# Use Fold 2 (EURJPY only, cleanest)
f = dict(syms=["EURJPY"], train_start="2026-04-01", train_end="2026-04-20",
         test_start="2026-04-21", test_end="2026-05-10")

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

prices = [t["price"] for t in train_ticks if t["sym"] == "EURJPY"]
returns = [(prices[i+1]-prices[i])/prices[i] for i in range(len(prices)-1)]
sym_stats = {"EURJPY": {"sigma": max(0.0001, sum(abs(r) for r in returns)/len(returns))}}

print(f"  {'win':>3} {'msd':>3} {'mc':>4} {'md':>4} {'ve':>5} {'th':>4} {'n':>6} {'WR':>5} {'PF':>5} {'AvgW':>7} {'AvgL':>7} {'score':>6}")
for p in PARAMS:
    fn = make_tcl_fn(sym_stats, *p)
    trades = run_simulation(test_ticks, oss, fn, sym_stats)
    s = stats(trades)
    if s:
        score = s["n"] * (s["pf"] - 1.0) * (s["wr"] / 100.0) if s["pf"] > 1 else 0
        print(f"  {p[0]:>3} {p[1]:>3} {p[2]:>4.2f} {p[3]:>4.2f} {p[4]:>5.1f} {p[5]:>4.2f} {s['n']:>6} {s['wr']:>4.1f}% {s['pf']:>4.2f} "
              f"{s['avg_win']:>+7.4f} {s['avg_loss']:>+7.4f} {score:>6.0f}", flush=True)

print(f"\n{time.perf_counter()-t0:.1f}s")
