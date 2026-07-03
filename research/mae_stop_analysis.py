"""MAE-stop intra-horizon — correct approach.
Uses DOA simulation as base, then applies MAE as a "what-if" post-processing
on the price path during each trade's horizon.
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

MAE_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.35, 0.50]

def ecdf_sig(t):
    d = t.get("ecdf", 0.5) - 0.5
    return 1 if d > 0.05 else (-1 if d < -0.05 else 0)

def simulate_with_mae_posthoc(oss, ticks, sym_stats):
    """Run correct DOA simulation + record intra-horizon price paths for MAE analysis.

    Returns:
        base_trades: list of {pnl, signal, entry_price, entry_idx}
        price_paths: dict mapping (sym, entry_idx) -> [price at t, t+1, ..., t+h]
    """
    horizon = 5
    doa = DelayedOutcomeEngine(horizon_ticks=horizon)
    ed = {}
    trades = []
    price_paths = {}
    tick_counter = defaultdict(int)  # per-symbol tick counter

    # For each symbol, track the last horizon+1 prices
    price_buffer = defaultdict(lambda: [])

    for t_idx, t in enumerate(ticks):
        s = t["sym"]
        price = t["price"]
        sig = oss.predict(t["ecdf"])
        ed[s] = {"price": price, "ecdf_rank": t["ecdf"], "entropy": t["entropy"], "signal": sig}
        price_buffer[s].append(price)
        tick_counter[s] += 1

        doa.record_snapshot(ed)

        if doa.ready:
            cp = {s2: ed[s2]["price"] for s2 in ed}
            outcomes = doa.evaluate(cp)
            for s2, outcome in outcomes.items():
                sig_at_entry = ed[s2]["signal"]
                if sig_at_entry != 0:
                    trade_pnl = sig_at_entry * outcome
                    entry_idx = max(0, tick_counter[s2] - horizon)
                    trades.append({
                        "pnl": trade_pnl, "signal": sig_at_entry,
                        "sym": s2, "entry_idx": entry_idx,
                    })

                    # Extract price path for this trade
                    buf = price_buffer[s2]
                    path = buf[-(horizon+1):]
                    price_paths[(s2, entry_idx)] = path

    return trades, price_paths

def apply_mae(trades, price_paths, mae_sigma, sym_stats):
    """Given base trades and price paths, compute MAE-adjusted PnL."""
    new_trades = []
    for t in trades:
        key = (t["sym"], t["entry_idx"])
        path = price_paths.get(key)
        if path is None or len(path) < 2:
            new_trades.append(dict(**t, stopped=False, ticks_to_stop=0, full_pnl=t["pnl"]))
            continue

        entry_price = path[0]
        sig = t["signal"]
        sym_vol = sym_stats.get(t["sym"], {}).get("sigma", 0.001)
        threshold = mae_sigma * sym_vol

        stop_triggered = False
        stop_pnl = 0
        ticks_to_stop = 0

        for i, price in enumerate(path[1:], 1):
            if sig == 1:
                excursion = (entry_price - price) / price
            else:
                excursion = (price - entry_price) / price

            if excursion < -threshold:
                stop_triggered = True
                ticks_to_stop = i
                if sig == 1:
                    stop_pct = (price - entry_price) / entry_price
                else:
                    stop_pct = (entry_price - price) / entry_price
                stop_pnl = stop_pct * 10000
                break

        if stop_triggered:
            full_pnl = t["pnl"]
            new_trades.append(dict(
                pnl=stop_pnl, signal=sig, sym=t["sym"],
                stopped=True, ticks_to_stop=ticks_to_stop,
                full_pnl=full_pnl, exit_reason="mae",
            ))
        else:
            new_trades.append(dict(**t, stopped=False, ticks_to_stop=5, full_pnl=t["pnl"], exit_reason="horizon"))

    return new_trades


t0 = time.perf_counter()
print(f"{H}", flush=True)
print("  MAE-STOP INTRA-HORIZON (correct post-hoc analysis)", flush=True)
print(H, flush=True)

for f in FOLDS:
    print(f"\n  --- Fold {f['name']} ---", flush=True)
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

    # Per-symbol sigma from training
    sym_stats = {}
    for s in f["syms"]:
        prices = [t["price"] for t in train_ticks if t["sym"] == s]
        if len(prices) > 1:
            returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
            sym_stats[s] = {"sigma": max(0.0001, sum(abs(r) for r in returns) / len(returns))}

    # Run simulation with price path recording
    base_trades, price_paths = simulate_with_mae_posthoc(oss, test_ticks, sym_stats)

    print(f"  Base trades: {len(base_trades)}  Price paths: {len(price_paths)}", flush=True)

    # Apply MAE at different thresholds
    results = {}
    for m in MAE_THRESHOLDS:
        new_trades = apply_mae(base_trades, price_paths, m, sym_stats)
        results[m] = new_trades

    # Also compute baseline (no MAE)
    results[None] = base_trades

    # Print header
    print(f"\n  {'MAE':>6}  {'n':>7}  {'WR':>6}  {'PF':>6}  {'PnL':>9}  "
          f"{'Stopped':>8}  {'StopRt':>7}  {'Salvage':>8}  {'AvgTF':>6}", flush=True)

    for m in [None] + MAE_THRESHOLDS:
        trades = results[m]
        n = len(trades)
        if n == 0:
            continue

        stopped = [t for t in trades if t.get("stopped")]
        ns = len(stopped)
        nh = n - ns
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] < 0]
        wr = len(wins) / n * 100 if n else 0
        tw = sum(t["pnl"] for t in wins)
        tl = abs(sum(t["pnl"] for t in losses))
        pf = tw / tl if tl > 0 else float('inf')
        total_pnl = sum(t["pnl"] for t in trades)
        stop_hit_rate = ns / n * 100 if n else 0

        # Salvage: stopped trades where full_pnl > current pnl (would've recovered)
        if ns:
            salvaged = sum(1 for t in stopped if t.get("full_pnl") is not None and t["full_pnl"] > t["pnl"])
            salvage_ratio = salvaged / ns * 100
            avg_tf = sum(t["ticks_to_stop"] for t in stopped) / ns
        else:
            salvage_ratio = 0.0
            avg_tf = 0.0

        label = f"{m:.2f}" if m else "None"
        print(f"  {label:>6}  {n:>7}  {wr:>5.1f}%  {pf:>5.2f}  {total_pnl:>+8.0f}  "
              f"{ns:>8}  {stop_hit_rate:>6.1f}%  {salvage_ratio:>7.1f}%  {avg_tf:>5.1f}", flush=True)

    # Best PF per fold
    best_m = max([m for m in [None] + MAE_THRESHOLDS], key=lambda m: (
        sum(t["pnl"] for t in results[m] if t["pnl"] > 0) /
        max(0.001, abs(sum(t["pnl"] for t in results[m] if t["pnl"] < 0)))
    ) if len([t for t in results[m] if t["pnl"] > 0]) > 0 else 0)
    print(f"  Best: MAE={best_m}", flush=True)

print(f"\nTotal: {time.perf_counter()-t0:.1f}s", flush=True)
