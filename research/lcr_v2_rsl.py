"""Layer Contribution v2 — OSS + RSL: regime-specific bucket tables.

Tests whether regime-conditioned OSS thresholds outperform global OSS.
Regime detected from entropy (RTD-style).
"""
import sys; sys.path.insert(0, '.')
import time
from collections import defaultdict

from research.replay_cache import ReplayCache
from research.outcome_surface_signal import OutcomeSurfaceSignal
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from validation.wfv_engine import WalkForwardValidator, StatisticalEdgeTest

TICK_LIMIT = 50000; SEED = 42

FOLDS = [
    dict(name="1", syms=["EURJPY","USDJPY"], train_start="2026-03-12", train_end="2026-03-28",
         test_start="2026-03-29", test_end="2026-04-14"),
    dict(name="2", syms=["EURJPY"], train_start="2026-04-01", train_end="2026-04-20",
         test_start="2026-04-21", test_end="2026-05-10"),
    dict(name="3", syms=["EURJPY"], train_start="2026-05-01", train_end="2026-05-20",
         test_start="2026-05-21", test_end="2026-06-08"),
]

def ecdf_sig(t):
    e = t.get("ecdf", 0.5)
    d = e - 0.5
    return 1 if d > 0.05 else (-1 if d < -0.05 else 0)

def detect_regime(entropy, chaotic=0.80, structured=0.65):
    """Regime detection based on real entropy distribution (mean=0.84)."""
    if entropy > chaotic:
        return "CHAOTIC"
    elif entropy < structured:
        return "STRUCTURED"
    return "TRANSITION"

def simulate_oss_rsl(ticks, oss_global, oss_regime, regime_thresholds, spread_bps=0.0):
    """Simulate OSS+RSL: use regime-specific OSS thresholds."""
    doa = DelayedOutcomeEngine(horizon_ticks=20)
    trades = []; ed = {}
    for t in ticks:
        s = t["sym"]
        ent = t["entropy"]
        regime = detect_regime(ent)
        thr = regime_thresholds.get(regime, 0.05)
        oss = oss_regime.get(regime, oss_global)
        if oss is None:
            oss = oss_global
        oss._ev_threshold = thr
        sig = oss.predict(t["ecdf"])
        ed[s] = {"price": t["price"], "ecdf_rank": t["ecdf"], "entropy": ent, "signal": sig}
        doa.record_snapshot(ed)
        if doa.ready:
            cp = {s2: ed[s2]["price"] for s2 in ed}
            outcomes = doa.evaluate(cp)
            for s2, outcome in outcomes.items():
                sig = ed[s2]["signal"]
                if sig != 0:
                    price = cp[s2]
                    trade_pnl = sig * outcome
                    spread_cost = spread_bps / 10000 * price * 2
                    trades.append({"pnl": trade_pnl, "net_pnl": trade_pnl - spread_cost,
                                   "regime": detect_regime(ed[s2]["entropy"])})
    return trades

t0 = time.perf_counter()
print("OSS vs OSS+RSL comparison (zero spread)", flush=True)
print(f"{'Fold':>5} {'Config':>18} {'Trades':>8} {'WR':>7} {'PF':>7} {'Exp':>9} {'PnL':>9}", flush=True)

for f in FOLDS:
    train_ticks = ReplayCache(f["syms"], f["train_start"], f["train_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()
    test_ticks = ReplayCache(f["syms"], f["test_start"], f["test_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()

    # Train global OSS
    doa = DelayedOutcomeEngine(horizon_ticks=20); ed = {}; recs = []
    for t in train_ticks:
        s = t["sym"]; sig = ecdf_sig(t)
        ed[s] = {"price": t["price"], "ecdf_rank": t["ecdf"], "entropy": t["entropy"], "signal": sig}
        doa.record_snapshot(ed)
        if doa.ready:
            for s2, outcome in doa.evaluate({s2: ed[s2]["price"] for s2 in ed}).items():
                recs.append({"sym": s2, "ecdf": ed[s2]["ecdf_rank"], "outcome": outcome, "entropy": ed[s2]["entropy"]})
    oss_global = OutcomeSurfaceSignal.from_pipeline_records(recs, ev_threshold=0.05)

    # Train regime-specific OSS
    regime_recs = {"CHAOTIC": [], "STRUCTURED": [], "TRANSITION": []}
    for r in recs:
        reg = detect_regime(r["entropy"])
        regime_recs[reg].append(r)

    oss_regime = {}
    for reg, rrecs in regime_recs.items():
        if len(rrecs) > 100:
            oss_regime[reg] = OutcomeSurfaceSignal.from_pipeline_records(rrecs, ev_threshold=0.05)
        else:
            oss_regime[reg] = None

    # Regime-specific thresholds
    regime_thresholds = {
        "CHAOTIC": 0.03,   # low threshold in chaos (expect less edge)
        "STRUCTURED": 0.10, # high threshold in structure (fewer, better signals)
        "TRANSITION": 0.05, # default
    }

    # Simulate both
    for cfg_name, tfunc in [
        ("OSS_global", lambda t, oss=oss_global: oss.predict(t["ecdf"])),
        ("OSS+RSL", lambda t, oss_g=oss_global, oss_r=oss_regime, thr=regime_thresholds: (
            oss_r.get(detect_regime(t["entropy"]), oss_g).predict(t["ecdf"])
            if False else None  # placeholder - we'll use the simulate function
        )),
    ]:
        pass

    # Use the proper simulation
    global_trades = simulate_oss_rsl(test_ticks, oss_global, {}, {"CHAOTIC":0.05,"STRUCTURED":0.05,"TRANSITION":0.05})
    rsl_trades = simulate_oss_rsl(test_ticks, oss_global, oss_regime, regime_thresholds)

    def stats(trades, label):
        wins = [t for t in trades if t["net_pnl"] > 0]
        losses = [t for t in trades if t["net_pnl"] < 0]
        n = len(trades)
        if n == 0: return
        wr = len(wins) / n * 100
        tw = sum(t["net_pnl"] for t in wins)
        tl = abs(sum(t["net_pnl"] for t in losses))
        tp = sum(t["net_pnl"] for t in trades)
        pf = tw / tl if tl > 0 else float('inf')
        aw = tw / len(wins) if wins else 0
        al = tl / len(losses) if losses else 0
        exp = (wr/100 * aw) - ((1-wr/100) * al)
        print(f"  {f['name']:>3}  {label:>18}  {n:>8}  {wr:>6.1f}%  {pf:>6.2f}  {exp:>+8.4f}  {tp:>+8.0f}", flush=True)

    stats(global_trades, "OSS_global")
    stats(rsl_trades, "OSS+RSL")

    # Per-regime breakdown for RSL
    regime_stats = defaultdict(lambda: {"trades": [], "pnl": 0})
    for t in rsl_trades:
        regime_stats[t["regime"]]["trades"].append(t)

    for reg in ["CHAOTIC", "STRUCTURED", "TRANSITION"]:
        if reg in regime_stats and regime_stats[reg]["trades"]:
            rs = regime_stats[reg]["trades"]
            wins = [t for t in rs if t["net_pnl"] > 0]
            losses = [t for t in rs if t["net_pnl"] < 0]
            n = len(rs)
            wr = len(wins) / n * 100 if n else 0
            tp = sum(t["net_pnl"] for t in rs)
            print(f"      {reg:>12} trades={n} wr={wr:.1f}% pnl={tp:+.0f}", flush=True)

print(f"\nTotal: {time.perf_counter()-t0:.1f}s", flush=True)
