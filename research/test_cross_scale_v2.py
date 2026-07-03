"""Cross-Scale State Projection v2 — fixed hold-to-horizon logic."""
import sys; sys.path.insert(0, '.')
import time
from collections import defaultdict

from research.cross_scale_cache import CrossScaleCache, StateProjectionEngine

TICK_LIMIT = 50000; SEED = 42; H = "=" * 70

FOLDS = [
    dict(name="Fold 2", syms=["EURJPY"], train_start="2026-04-01", train_end="2026-04-20",
         test_start="2026-04-21", test_end="2026-05-10"),
    dict(name="Fold 3", syms=["EURJPY"], train_start="2026-05-01", train_end="2026-05-20",
         test_start="2026-05-21", test_end="2026-06-08"),
]

HORIZONS = ["fwd_1m", "fwd_5m", "fwd_15m"]
HLABEL = {"fwd_1m": "M1", "fwd_5m": "M5", "fwd_15m": "M15"}
HORIZON_SEC = {"fwd_1m": 60, "fwd_5m": 300, "fwd_15m": 900}

t0 = time.perf_counter()
print(f"{H}\n  PROGRAM v3 — CROSS-SCALE STATE PROJECTION v2\n{H}")

for f in FOLDS:
    print(f"\n  --- {f['name']} ---")
    train_ticks = CrossScaleCache(f["syms"], f["train_start"], f["train_end"],
                                   tick_limit=TICK_LIMIT, seed=SEED).compute()
    test_ticks = CrossScaleCache(f["syms"], f["test_start"], f["test_end"],
                                  tick_limit=TICK_LIMIT, seed=SEED).compute()

    print(f"  Train: {len(train_ticks)} ticks, Test: {len(test_ticks)} ticks")
    span_h = (test_ticks[-1]['time_sec'] - test_ticks[0]['time_sec']) / 3600
    print(f"  Test span: {span_h:.1f}h")

    # Build per-horizon training surfaces
    train_engines = {h: StateProjectionEngine() for h in HORIZONS}
    train_prev_ecdf = {}
    for t in train_ticks:
        s = t["sym"]; pe = train_prev_ecdf.get(s)
        for h in HORIZONS:
            train_engines[h].record(t["ecdf"], pe, t["entropy"], t.get(h, 0.0))
        train_prev_ecdf[s] = t["ecdf"]

    # Top states per horizon
    print(f"\n  Top-3 states (mean_abs, n>=20) per horizon:")
    for h in HORIZONS:
        top = train_engines[h].top_states(min_n=20, min_wr=55, n=3)
        if top:
            r = top[0]
            print(f"  {HLABEL[h]}: bkt={r['bucket']} tt={r['trans_type']} ed={r['entropy_dec']}  "
                  f"n={r['n']}  WR={r['wr']:.1f}%  PF={r['pf']:.2f}  mean={r['mean_fwd']:+.5f}  "
                  f"abs={r['mean_abs']:.5f}")

    # Tick-level baseline (avg absolute fwd_return for this horizon)
    tick_avg_abs = {}
    for h in HORIZONS:
        vals = [abs(t.get(h, 0)) for t in test_ticks if t.get(h, 0) != 0]
        tick_avg_abs[h] = sum(vals) / len(vals) if vals else 0

    # AER from training surface
    for h in HORIZONS:
        aer = train_engines[h].amplitude_expansion_ratio(tick_avg_abs.get(h, 0.001), min_n=20)
        pf = "PASS" if aer >= 4.0 else "FAIL"
        print(f"  {HLABEL[h]} AER: {aer:.2f}x [{pf}] (tick_avg_abs={tick_avg_abs.get(h, 0):.5f})")

    # --- TEST: Use top training states, hold to time horizon ---
    print(f"\n  --- Test: state-triggered trades, hold to horizon ---")
    print(f"  {'Horizon':>8}  {'n':>5}  {'WR':>5}  {'PF':>5}  {'Exp':>8}  {'PnL':>9}  {'AvgAbs':>7}  {'AER':>5}  {'SprAdj':>6}")

    for h in HORIZONS:
        delta_sec = HORIZON_SEC[h]
        top_states = train_engines[h].top_states(min_n=20, min_wr=55, n=5)
        if not top_states:
            print(f"  {HLABEL[h]}: no qualified states")
            continue

        # Convert to set for fast lookup
        top_set = set((r["bucket"], r["trans_type"], r["entropy_dec"]) for r in top_states)
        top_dir = {}
        for r in top_states:
            top_dir[(r["bucket"], r["trans_type"], r["entropy_dec"])] = 1 if r["mean_fwd"] > 0 else -1

        # Walk through test ticks
        active_trades = {}  # sym -> {entry_tick, entry_time, entry_price, direction, state_key}
        trades = []

        for i in range(len(test_ticks)):
            t = test_ticks[i]
            s = t["sym"]
            cur_time = t["time_sec"]
            cur_price = t["price"]

            # Check for exits
            if s in active_trades:
                at = active_trades[s]
                if cur_time >= at["entry_time"] + delta_sec:
                    # Time horizon reached — exit
                    entry_price = at["entry_price"]
                    direction = at["direction"]
                    pnl = (cur_price - entry_price) * direction
                    trades.append({"pnl_net": pnl, "pnl_abs": abs(cur_price - entry_price),
                                   "horizon": HLABEL[h], "state_key": at["state_key"],
                                   "entry_time": at["entry_time"], "exit_time": cur_time})
                    del active_trades[s]

            # Check for entries (skip if already in trade)
            if s in active_trades:
                continue

            # Get state
            prev_ecdf = test_ticks[i-1]["ecdf"] if i > 0 else None
            ecdf_bucket = min(int(t["ecdf"] * 10), 9)
            if prev_ecdf is None:
                trans_type = 0
            else:
                prev_bucket = min(int(prev_ecdf * 10), 9)
                diff = abs(ecdf_bucket - prev_bucket)
                if diff > 2: trans_type = 3
                elif diff > 1: trans_type = 2
                elif diff >= 1: trans_type = 1
                else: trans_type = 0
            entropy_dec = min(int(t["entropy"] * 10), 9)
            state_key = (ecdf_bucket, trans_type, entropy_dec)

            if state_key in top_set:
                direction = top_dir[state_key]
                active_trades[s] = {
                    "entry_time": cur_time,
                    "entry_price": cur_price,
                    "direction": direction,
                    "state_key": state_key,
                }

        # Flush remaining trades at end
        for s, at in active_trades.items():
            last_tick = test_ticks[-1]
            entry_price = at["entry_price"]
            direction = at["direction"]
            pnl = (last_tick["price"] - entry_price) * direction
            trades.append({"pnl_net": pnl, "pnl_abs": abs(last_tick["price"] - entry_price),
                           "horizon": HLABEL[h], "state_key": at["state_key"],
                           "entry_time": at["entry_time"], "exit_time": last_tick["time_sec"]})

        # Compute stats
        if trades:
            n_trades = len([tr for tr in trades if tr["pnl_net"] != 0])
            wins = [tr for tr in trades if tr["pnl_net"] > 0]
            losses = [tr for tr in trades if tr["pnl_net"] < 0]
            wr = len(wins) / n_trades * 100 if n_trades else 0
            tw = sum(tr["pnl_net"] for tr in wins)
            tl = abs(sum(tr["pnl_net"] for tr in losses))
            pf = tw / tl if tl > 0 else float('inf')
            exp = (sum(tr["pnl_net"] for tr in trades)) / n_trades if n_trades else 0
            total_pnl = sum(tr["pnl_net"] for tr in trades)
            avg_abs = sum(tr["pnl_abs"] for tr in trades) / n_trades if n_trades else 0
            aer_actual = avg_abs / max(tick_avg_abs.get(h, 0.001), 0.0001)

            # Spread-adjusted: 0.5bps round trip on EURJPY ~160 → 0.016
            spread_cost = 0.016
            adj_exp = exp - spread_cost
            adj_pnl = sum(tr["pnl_net"] for tr in trades) - spread_cost * n_trades
            adj_wins = [tr for tr in trades if tr["pnl_net"] - spread_cost > 0]
            adj_wr = len(adj_wins) / n_trades * 100 if n_trades else 0
            spr_flag = " ⭐" if pf > 1.4 and adj_exp > 0 else ""
            print(f"  {HLABEL[h]:>8}  {n_trades:>5}  {wr:>4.1f}%  {pf:>4.2f}{spr_flag}  "
                  f"{exp:>+7.4f}  {total_pnl:>+8.4f}  {avg_abs:>6.4f}  {aer_actual:>4.1f}x  "
                  f"{adj_exp:>+5.3f}", flush=True)
        else:
            print(f"  {HLABEL[h]}: 0 trades")

elapsed = time.perf_counter() - t0
print(f"\n{elapsed:.1f}s")
