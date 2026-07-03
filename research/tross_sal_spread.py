"""TrOSS + SAL + spread — fixed closure binding."""
import sys; sys.path.insert(0, '.')
import time
from collections import defaultdict

from research.replay_cache import ReplayCache
from research.outcome_surface_signal import OutcomeSurfaceSignal
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from signals.sal_mapper import SignalAggregationLayer

TICK_LIMIT = 50000; SEED = 42; H = "=" * 70
HOLD_TIMES = [5, 10, 20, 50]

def ecdf_sig(t):
    d = t.get("ecdf", 0.5) - 0.5
    return 1 if d > 0.05 else (-1 if d < -0.05 else 0)

def run_simulation(ticks, signal_fn, sym_stats, horizon, spread_bps=0.5):
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
                    buf = price_buffer[s2]; entry_price = buf[-(horizon+1)] if len(buf) >= horizon+1 else cp[s2]
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
                    spread_cost = entry_price * (spread_bps / 10000) + exit_price * (spread_bps / 10000)
                    trades.append({"pnl_net": pnl_raw - spread_cost, "pnl_raw": pnl_raw, "spread": spread_cost})
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
    exp = (wr/100*aw)-((1-wr/100)*al)
    raw_wins = [t for t in trades if t["pnl_raw"] > 0]
    raw_losses = [t for t in trades if t["pnl_raw"] < 0]
    raw_pf = sum(t["pnl_raw"] for t in raw_wins) / max(0.0001, abs(sum(t["pnl_raw"] for t in raw_losses)))
    return {"n": n, "wr": wr, "pf": pf, "exp": exp, "total_pnl": sum(t["pnl_net"] for t in trades),
            "raw_total": sum(t["pnl_raw"] for t in trades), "raw_pf": raw_pf,
            "avg_win": aw, "avg_loss": al,
            "avg_spread": sum(t["spread"] for t in trades)/n if n else 0}

t0 = time.perf_counter()
print(f"{H}\n  TrOSS + SAL + SPREAD (Fold 2)\n{H}")

f = dict(syms=["EURJPY"], train_start="2026-04-01", train_end="2026-04-20",
         test_start="2026-04-21", test_end="2026-05-10")

train_ticks = ReplayCache(f["syms"], f["train_start"], f["train_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()
test_ticks = ReplayCache(f["syms"], f["test_start"], f["test_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()

train_recs = []; doa = DelayedOutcomeEngine(horizon_ticks=20); ed = {}
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

SYSTEMS = []

# 1. OSS baseline
SYSTEMS.append(("OSS", None, None))
# 2. TrOSS cross1
SYSTEMS.append(("TrOSS cross1", "cross1", None))
# 3. TrOSS cross2
SYSTEMS.append(("TrOSS cross2", "cross2", None))
# 4. TrOSS+SAL cross1
SYSTEMS.append(("TrOSS+SAL cross1", "cross1", True))
# 5. TrOSS+SAL cross2
SYSTEMS.append(("TrOSS+SAL cross2", "cross2", True))

print(f"  {'System':>20}  {'h':>3}  {'n':>6}  {'WR':>5}  {'PF':>5}  {'PFraw':>5}  {'Exp':>8}  {'PnL':>9}  {'AvgSprd':>8}")
print(f"{'-'*80}")

for name, mode, use_sal in SYSTEMS:
    for h in HOLD_TIMES:
        # Create fresh instances per iteration
        prev_bucket = {}

        if mode is None:
            def fn(s, p, ecdf, _prev=prev_bucket):
                return oss.predict(ecdf)
        elif not use_sal:
            def fn(s, p, ecdf, _mode=mode, _prev=prev_bucket):
                bucket = min(int(ecdf * 10), 9)
                prev_b = _prev.get(s)
                _prev[s] = bucket
                if prev_b is None: return 0
                diff = bucket - prev_b
                if _mode == "cross1": trigger = abs(diff) > 1
                elif _mode == "cross2": trigger = abs(diff) > 2
                else: trigger = False
                return oss.predict(ecdf) if trigger else 0
        else:
            sal = SignalAggregationLayer(sym_stats=sym_stats, entry_threshold=0.50,
                                          accum_window=20, accum_decay=0.85, consensus_min=0.60)
            scores = {}
            def fn(s, p, ecdf, _mode=mode, _prev=prev_bucket, _sal=sal, _scores=scores):
                bucket = min(int(ecdf * 10), 9)
                prev_b = _prev.get(s)
                _prev[s] = bucket
                if prev_b is None: return 0
                diff = bucket - prev_b
                if _mode == "cross1": trigger = abs(diff) > 1
                elif _mode == "cross2": trigger = abs(diff) > 2
                else: trigger = False
                if trigger:
                    _scores[s] = _sal.accumulator.score()
                    return _sal.update(s, oss.predict(ecdf), 1.0, p, all_scores=_scores)
                return 0

        trades = run_simulation(test_ticks, fn, sym_stats, h)
        s = stats(trades)
        if s:
            flag = " ⭐" if s["pf"] > 1.0 and s["wr"] > 65 and s["n"] > 50 else ""
            print(f"  {name:>20}  {h:>3}  {s['n']:>6}  {s['wr']:>4.1f}%  {s['pf']:>4.2f}  "
                  f"{s['raw_pf']:>4.2f}  {s['exp']:>+7.4f}  {s['total_pnl']:>+8.4f}  "
                  f"{s['avg_spread']:>+8.5f}{flag}", flush=True)

print(f"\n{time.perf_counter()-t0:.1f}s")
