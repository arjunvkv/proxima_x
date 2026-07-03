"""OSS-only simulation — no V3/V4 layers, pure OSS → H20 → Execution."""
import sys; sys.path.insert(0, '.')
import time
import math
from collections import defaultdict

from research.replay_cache import ReplayCache
from research.outcome_surface_signal import OutcomeSurfaceSignal
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from validation.wfv_engine import WalkForwardValidator, StatisticalEdgeTest

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
    e = t.get("ecdf", 0.5) if isinstance(t, dict) else 0.5
    d = e - 0.5
    return 1 if d > 0.05 else (-1 if d < -0.05 else 0)


def simulate(oss, ticks, spread_bps=0.5):
    """Simulate OSS trades on cached ticks with full metrics."""
    doa = DelayedOutcomeEngine(horizon_ticks=20)
    trades = []
    ed = {}
    balance = 0.0

    for t in ticks:
        s = t["sym"]
        ed[s] = {"price": t["price"], "ecdf_rank": t["ecdf"], "entropy": t["entropy"],
                  "signal": oss.predict(t["ecdf"])}
        doa.record_snapshot(ed)

        if doa.ready:
            cp = {s2: ed[s2]["price"] for s2 in ed}
            outcomes = doa.evaluate(cp)
            for s2, outcome in outcomes.items():
                sig_at_entry = ed[s2]["signal"]
                price = cp[s2]
                if sig_at_entry != 0:
                    trade_pnl = sig_at_entry * outcome  # -1, 0, or +1 in outcome units
                    spread_cost = spread_bps / 10000 * price * 2
                    net_pnl = trade_pnl - spread_cost
                    balance += net_pnl
                    trades.append({
                        "sym": s2,
                        "signal": sig_at_entry,
                        "outcome": outcome,
                        "pnl": trade_pnl,
                        "net_pnl": net_pnl,
                        "price": price,
                        "spread_cost": spread_cost,
                    })

    if not trades:
        return {"n_trades": 0}

    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] < 0]
    n = len(trades)

    total_win = sum(t["net_pnl"] for t in wins)
    total_loss = abs(sum(t["net_pnl"] for t in losses))
    total_pnl = sum(t["net_pnl"] for t in trades)

    win_rate = len(wins) / n * 100 if n else 0
    avg_win = total_win / len(wins) if wins else 0
    avg_loss = total_loss / len(losses) if losses else 0
    profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
    expectancy = (win_rate/100 * avg_win) - ((1-win_rate/100) * avg_loss) if n else 0

    # Drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cum += t["net_pnl"]
        peak = max(peak, cum)
        dd = peak - cum
        max_dd = max(max_dd, dd)

    gross_alpha = sum(t["pnl"] for t in trades)
    net_alpha = total_pnl
    spread_drag = sum(t["spread_cost"] for t in trades)
    spread_drag_pct = spread_drag / abs(gross_alpha) * 100 if gross_alpha != 0 else 0

    return {
        "n_trades": n,
        "win_rate": round(win_rate, 2),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 4),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown": round(max_dd, 2),
        "gross_alpha": round(gross_alpha, 2),
        "net_alpha": round(net_alpha, 2),
        "spread_drag": round(spread_drag, 2),
        "spread_drag_pct": round(spread_drag_pct, 2),
    }


t0 = time.perf_counter()

print(f"{H}", flush=True)
print("  OSS-ONLY SIMULATION (ECDF -> OSS -> DOA -> Metrics)", flush=True)
print(H, flush=True)

all_metrics = []

for f in FOLDS:
    name = f["name"]
    print(f"\n  --- {name} ---", flush=True)

    train_ticks = ReplayCache(f["syms"], f["train_start"], f["train_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()
    test_ticks = ReplayCache(f["syms"], f["test_start"], f["test_end"], tick_limit=TICK_LIMIT, seed=SEED).compute()

    # Train OSS on train window using ECDF-only signal for outcomes
    train_recs = []
    doa = DelayedOutcomeEngine(horizon_ticks=20)
    ed = {}
    for t in train_ticks:
        s = t["sym"]
        sig = ecdf_sig(t)
        ed[s] = {"price": t["price"], "ecdf_rank": t["ecdf"], "entropy": t["entropy"], "signal": sig}
        doa.record_snapshot(ed)
        if doa.ready:
            cp = {s2: ed[s2]["price"] for s2 in ed}
            outcomes = doa.evaluate(cp)
            for s2, outcome in outcomes.items():
                train_recs.append({
                    "sym": s2, "ecdf": ed[s2]["ecdf_rank"],
                    "outcome": outcome, "signal": ed[s2]["signal"],
                })

    oss = OutcomeSurfaceSignal.from_pipeline_records(train_recs, ev_threshold=0.05)
    print(f"  OSS trained on {len(train_recs)} records, {oss.bucket_count()} buckets, density={oss.signal_density():.2f}", flush=True)

    # Simulate on test window
    metrics = simulate(oss, test_ticks, spread_bps=0.5)
    all_metrics.append(metrics)

    print(f"  Trades: {metrics['n_trades']}  WinRate: {metrics['win_rate']}%  PF: {metrics['profit_factor']}", flush=True)
    print(f"  Total PnL: {metrics['total_pnl']:+.2f}  MaxDD: {metrics['max_drawdown']:.2f}  Expectancy: {metrics['expectancy']:.4f}", flush=True)
    print(f"  GrossAlpha: {metrics['gross_alpha']:+.2f}  NetAlpha: {metrics['net_alpha']:+.2f}  SpreadDrag: {metrics['spread_drag_pct']}%", flush=True)

# ───── Aggregate ─────
print(f"\n{H}", flush=True)
print("  AGGREGATE OSS SIMULATION", flush=True)
print(H, flush=True)

avg_metrics = {}
for k in all_metrics[0].keys():
    vals = [m[k] for m in all_metrics]
    avg_metrics[k] = sum(vals) / len(vals) if isinstance(vals[0], (int, float)) else vals[0]

print(f"  Total trades across 3 folds: {sum(m['n_trades'] for m in all_metrics)}", flush=True)
print(f"  Avg win rate: {avg_metrics['win_rate']:.1f}%", flush=True)
print(f"  Avg profit factor: {avg_metrics['profit_factor']:.2f}", flush=True)
print(f"  Avg expectancy: {avg_metrics['expectancy']:.4f}", flush=True)
print(f"  Total net PnL: {sum(m['total_pnl'] for m in all_metrics):+.2f}", flush=True)
print(f"  Avg max drawdown: {avg_metrics['max_drawdown']:.2f}", flush=True)
print(f"  Avg spread drag: {avg_metrics['spread_drag_pct']:.1f}%", flush=True)

# Acceptance criteria
print(f"\n  ACCEPTANCE:", flush=True)
if avg_metrics["profit_factor"] > 1.2:
    print(f"  PASS: PF > 1.2 ({avg_metrics['profit_factor']:.2f})", flush=True)
else:
    print(f"  FAIL: PF > 1.2 ({avg_metrics['profit_factor']:.2f})", flush=True)
if avg_metrics["expectancy"] > 0:
    print(f"  PASS: Expectancy > 0 ({avg_metrics['expectancy']:.4f})", flush=True)
else:
    print(f"  FAIL: Expectancy > 0 ({avg_metrics['expectancy']:.4f})", flush=True)
if avg_metrics["win_rate"] > 54:
    print(f"  PASS: Win rate > 54% ({avg_metrics['win_rate']}%)", flush=True)
else:
    print(f"  FAIL: Win rate > 54% ({avg_metrics['win_rate']}%)", flush=True)

print(f"\nTotal: {time.perf_counter()-t0:.1f}s", flush=True)
