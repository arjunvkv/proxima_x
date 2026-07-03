"""Cross-Scale State Projection — 3-fold validation.
Tests whether microstructure state (ECDF bucket, transition type) 
predicts mesostructure price excursion at M1/M5/M15 horizons.
"""
import sys; sys.path.insert(0, '.')
import time
from collections import defaultdict

from research.cross_scale_cache import CrossScaleCache, StateProjectionEngine

TICK_LIMIT = 50000; SEED = 42; H = "=" * 70

FOLDS = [
    dict(name="Fold 1", syms=["EURJPY"], train_start="2026-03-12", train_end="2026-03-28",
         test_start="2026-03-29", test_end="2026-04-14"),
    dict(name="Fold 2", syms=["EURJPY"], train_start="2026-04-01", train_end="2026-04-20",
         test_start="2026-04-21", test_end="2026-05-10"),
    dict(name="Fold 3", syms=["EURJPY"], train_start="2026-05-01", train_end="2026-05-20",
         test_start="2026-05-21", test_end="2026-06-08"),
]

HORIZONS = ["fwd_1m", "fwd_5m", "fwd_15m"]
HORIZON_LABELS = {"fwd_1m": "M1", "fwd_5m": "M5", "fwd_15m": "M15"}

t0 = time.perf_counter()
print(f"{H}\n  PROGRAM v3 — CROSS-SCALE STATE PROJECTION\n{H}")

for f in FOLDS:
    print(f"\n  --- {f['name']} ---")
    train_ticks = CrossScaleCache(f["syms"], f["train_start"], f["train_end"],
                                   tick_limit=TICK_LIMIT, seed=SEED).compute()
    test_ticks = CrossScaleCache(f["syms"], f["test_start"], f["test_end"],
                                  tick_limit=TICK_LIMIT, seed=SEED).compute()

    print(f"  Train: {len(train_ticks)} ticks, Test: {len(test_ticks)} ticks")
    print(f"  Time range test: {test_ticks[0]['time_sec']} - {test_ticks[-1]['time_sec']}")
    span_hours = (test_ticks[-1]['time_sec'] - test_ticks[0]['time_sec']) / 3600
    print(f"  Test span: {span_hours:.1f} hours = {span_hours/24:.1f} days")

    # Build training surface
    train_engine = StateProjectionEngine()
    train_prev_ecdf = {}
    train_avg_abs = 0.0
    for t in train_ticks:
        s = t["sym"]
        pe = train_prev_ecdf.get(s)
        for h in HORIZONS:
            train_engine.record(t["ecdf"], pe, t["entropy"], t.get(h, 0.0))
        train_avg_abs += abs(t.get("fwd_1m", 0))
        train_prev_ecdf[s] = t["ecdf"]
    train_avg_abs /= max(len(train_ticks), 1)

    print(f"\n  Training Surface Summary (min_n=10):")
    rows = train_engine.surface_summary(min_samples=10)
    print(f"  Total states: {len(rows)}")

    # Top states by mean forward amplitude
    print(f"\n  Top-5 states by M1 mean_abs (n>=10):")
    for h in HORIZONS:
        h_rows = [r for r in rows if abs(r["mean_fwd"]) > 0]
        h_rows.sort(key=lambda r: r["mean_abs"], reverse=True)
        print(f"  --- {HORIZON_LABELS[h]} ---")
        for r in h_rows[:5]:
            print(f"    bkt={r['bucket']} tt={r['trans_type']} ed={r['entropy_dec']}  "
                  f"n={r['n']}  WR={r['wr']:.1f}%  PF={r['pf']:.2f}  "
                  f"mean={r['mean_fwd']:+.6f}  |mean|={r['mean_abs']:.6f}")

    # AER
    tick_avg_abs_1m = sum(abs(t.get("fwd_1m", 0)) for t in test_ticks) / max(len(test_ticks), 1)
    aer = train_engine.amplitude_expansion_ratio(tick_avg_abs_1m)
    print(f"\n  Tick avg abs M1 (test): {tick_avg_abs_1m:.6f}")
    print(f"  Amplitude Expansion Ratio (AER): {aer:.2f}x  {'PASS' if aer >= 4.0 else 'FAIL'} (target: 4.0x)")

    # Test: use top training states to trade on test data
    print(f"\n  --- Test: Trade top training states on test ---")
    for h in HORIZONS:
        top_states = train_engine.top_states(min_n=10, min_wr=55, n=5)
        if not top_states:
            print(f"  {HORIZON_LABELS[h]}: no qualified states")
            continue

        test_engine = StateProjectionEngine()
        prev_ecdf = {}
        trades = []
        start_tick = None
        in_trade = False
        entry_price = 0.0
        entry_state = None

        for t in test_ticks:
            s = t["sym"]; pe = prev_ecdf.get(s)

            # Check if current state is in top_states
            ecdf_bucket = min(int(t["ecdf"] * 10), 9)
            if pe is None:
                trans_type = 0
            else:
                prev_bucket = min(int(pe * 10), 9)
                diff = abs(ecdf_bucket - prev_bucket)
                if diff > 2: trans_type = 3
                elif diff > 1: trans_type = 2
                elif diff >= 1: trans_type = 1
                else: trans_type = 0
            entropy_dec = min(int(t["entropy"] * 10), 9)

            matching = [r for r in top_states if r["bucket"] == ecdf_bucket
                        and r["trans_type"] == trans_type and r["entropy_dec"] == entropy_dec]

            fwd_return = t.get(h, 0.0)
            test_engine.record(t["ecdf"], pe, t["entropy"], fwd_return)
            prev_ecdf[s] = t["ecdf"]

            if matching:
                # State matches — check forward return
                if not in_trade:
                    in_trade = True
                    entry_price = t["price"]
                    entry_state = matching[0]
                else:
                    # Exit: compute PnL
                    pnl = t["price"] - entry_price
                    direction = 1 if matching[0]["mean_fwd"] > 0 else -1
                    trades.append({"pnl_net": pnl * direction, "state_mean": matching[0]["mean_fwd"],
                                   "state_wr": matching[0]["wr"], "horizon": HORIZON_LABELS[h]})
                    in_trade = False
            elif in_trade:
                # Exit if state no longer matches
                pnl = t["price"] - entry_price
                direction = 1 if entry_state["mean_fwd"] > 0 else -1
                trades.append({"pnl_net": pnl * direction, "state_mean": entry_state["mean_fwd"],
                               "state_wr": entry_state["wr"], "horizon": HORIZON_LABELS[h]})
                in_trade = False

        if trades:
            n = len(trades)
            wins = [tr for tr in trades if tr["pnl_net"] > 0]
            losses = [tr for tr in trades if tr["pnl_net"] < 0]
            wr = len(wins) / n * 100
            tw = sum(tr["pnl_net"] for tr in wins)
            tl = abs(sum(tr["pnl_net"] for tr in losses))
            pf = tw / tl if tl > 0 else float('inf')
            aw = tw / len(wins) if wins else 0
            al = tl / len(losses) if losses else 0
            avg_abs_fwd = sum(abs(tr["pnl_net"]) for tr in trades) / n if n else 0
            aer_actual = avg_abs_fwd / max(tick_avg_abs_1m, 0.0001)
            pf_flag = " ⭐" if pf > 1.4 else ""
            print(f"  {HORIZON_LABELS[h]}: n={n:>5}  WR={wr:>4.1f}%  PF={pf:>4.2f}{pf_flag}  "
                  f"AvgPnL={sum(tr['pnl_net'] for tr in trades)/n:+.6f}  "
                  f"AvgAbs={avg_abs_fwd:.6f}  AER={aer_actual:.1f}x")
        else:
            print(f"  {HORIZON_LABELS[h]}: 0 trades")

    # Cross-scale vs tick-level comparison
    print(f"\n  --- Comparison: tick-level (TrOSS cross2 h=5) ---")
    print(f"  Reference: Fold 2, cross2 h=5: n=73, WR=71.2%, PF=2.79, AvgWin=0.0113")
    print(f"  Reference: tick avg abs fwd = {tick_avg_abs_1m:.6f}")

elapsed = time.perf_counter() - t0
print(f"\n{elapsed:.1f}s")
