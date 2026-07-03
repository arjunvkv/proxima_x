"""Sweep B: Friction Collapse Point — sweep spread with zero other friction."""
import sys; sys.path.insert(0, ".")
import json, os, time
from collections import defaultdict
from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from features.ecdf_transform import PerSymbolECDF
from signals.outcome_surface_signal import OutcomeSurfaceSignal
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from research.cep.counterfactual_execution import CounterfactualExecutionEngine
from research.cep.execution_profile import ExecutionProfile
from research.cep.metrics import CEPMetrics


SPREAD_SWEEP_BPS = [0.000, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020]


def run_sweep_b(n_ticks=80000):
    import math

    print("=" * 60)
    print("SWEEP B: FRICTION COLLAPSE POINT")
    print(f"Zero latency, 100% fill, 0 slippage. Sweep spread 0-0.020 bps")
    print("=" * 60)

    # 1. Build replay + walk ticks
    print("\n1. Loading April data...")
    t0 = time.time()
    config = ReplayConfig(
        symbols=["EURJPY"], start="2026-04-01", end="2026-04-20",
        speed=500000, burst=True, latency=False, slippage=False, seed=42,
    )
    env = build_replay_environment(config)
    patch_clock(env.clock)
    feed = env.replay_feed
    print(f"   Feed: {len(feed._merged)} merged ticks ({time.time()-t0:.2f}s)")

    # 2. Walk + train OSS
    print("2. Walking ticks + training OSS...")
    ecdf = PerSymbolECDF(window_size=2000)
    oss_recs, ed, doa = [], {}, DelayedOutcomeEngine(horizon_ticks=20)
    price_lookup = {}
    signals_oss = []
    last_minute, tick_count = -1, 0
    price_bufs = defaultdict(list)

    while tick_count < n_ticks:
        tick = feed.next()
        if tick is None:
            break
        tick_count += 1
        sym = tick.get("symbol", "EURJPY")
        price = float(tick.get("ask", tick.get("bid", tick.get("price", 0))))
        ts = int(tick.get("time_sec", 0))
        ecdf_rank = ecdf.update(sym, price)

        price_bufs[sym].append(price)
        if len(price_bufs[sym]) > 50:
            price_bufs[sym] = price_bufs[sym][-50:]

        if ts and ts not in price_lookup:
            price_lookup[ts] = price

        if tick_count < 50000:
            d = ecdf_rank - 0.5
            sig = 1 if d > 0.05 else (-1 if d < -0.05 else 0)
            ed[sym] = {"price": price, "ecdf_rank": ecdf_rank, "signal": sig}
            doa.record_snapshot(ed)
            if doa.ready:
                for s2, outcome in doa.evaluate({s2: ed[s2]["price"] for s2 in ed}).items():
                    oss_recs.append({"sym": s2, "ecdf": ed[s2]["ecdf_rank"], "outcome": outcome})

        if tick_count % 20000 == 0:
            print(f"   ... {tick_count} ticks ({time.time()-t0:.2f}s)")

    print(f"   {tick_count} ticks, {len(price_lookup)} seconds, {len(oss_recs)} OSS records")

    oss = OutcomeSurfaceSignal.from_pipeline_records(oss_recs) if oss_recs else OutcomeSurfaceSignal()
    print(f"   OSS: {oss.bucket_count()} buckets, density={oss.signal_density():.2f}")

    # 3. Re-walk for signals
    print("3. Generating signals...")
    if hasattr(feed, "seek"):
        feed.seek(0)
    else:
        print("   No seek, re-creating feed...")
        env2 = build_replay_environment(config)
        patch_clock(env2.clock)
        feed = env2.replay_feed

    ecdf2 = PerSymbolECDF(window_size=2000)
    last_minute, tick_count = -1, 0

    while tick_count < n_ticks:
        tick = feed.next()
        if tick is None:
            break
        tick_count += 1
        sym = tick.get("symbol", "EURJPY")
        price = float(tick.get("ask", tick.get("bid", tick.get("price", 0))))
        ts = int(tick.get("time_sec", 0))
        ecdf_rank = ecdf2.update(sym, price)

        minute_key = ts // 60 if ts else 0
        if minute_key > last_minute and tick_count >= 5000:
            last_minute = minute_key
            oss_sig = oss.predict(ecdf_rank)
            if oss_sig != 0:
                signals_oss.append({
                    "ts": ts, "price": price, "direction": oss_sig,
                    "source": "OSS", "sym": sym,
                })

    print(f"   {len(signals_oss)} OSS signals")

    # 4. Sweep spread
    print("4. Running spread sweep...")
    metrics_engine = CEPMetrics()
    results = []

    for sp in SPREAD_SWEEP_BPS:
        profile = ExecutionProfile(
            name=f"spread_{sp:.3f}",
            spread_bps=sp,
            latency_ms_mean=0,
            latency_ms_std=0,
            fill_probability=1.0,
            reject_probability=0.0,
            slippage_bps_mean=0.0,
            slippage_bps_std=0.0,
            queue_priority=1.0,
        )
        engine = CounterfactualExecutionEngine(profile, seed=42)
        trades = engine.run_signals(signals_oss, price_lookup, "OSS")
        m = metrics_engine.compute(trades)
        row = {
            "spread_bps": sp,
            "round_trip_cost": round(185 * sp / 10000 * 2, 8) if sp > 0 else 0,
            "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "expectancy": m["expectancy"],
            "avg_win": m["avg_win"],
            "avg_loss": m["avg_loss"],
            "sharpe": m["sharpe"],
            "total_pnl": m["total_pnl"],
        }
        results.append(row)
        print(f"   {sp:.3f} bps: PF={row['profit_factor']:.4f}, WR={row['win_rate']:.4f}, "
              f"n={row['n_trades']}, AvgWin={row['avg_win']:.6f}")

    # 5. Interpolate FCP
    print("\n5. Friction Collapse Point interpolation...")
    fcp = None
    for i in range(len(results) - 1):
        r0, r1 = results[i], results[i+1]
        if r0["profit_factor"] >= 1.0 >= r1["profit_factor"]:
            # Linear interpolation
            x0, x1 = r0["spread_bps"], r1["spread_bps"]
            y0, y1 = r0["profit_factor"], r1["profit_factor"]
            if y0 != y1:
                fcp = x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
                break

    if fcp is not None:
        print(f"   FCP = {fcp:.4f} bps (spread where PF = 1.0)")
    else:
        if results[0]["profit_factor"] < 1.0:
            print(f"   PF < 1.0 even at zero spread. FCP = 0.000 (edge is sub-friction at limit)")
            fcp = 0.0
        else:
            print(f"   PF >= 1.0 at max sweep. FCP > {SPREAD_SWEEP_BPS[-1]} bps")
            fcp = SPREAD_SWEEP_BPS[-1] * 2  # estimate

    # CME minimum executable friction
    cme_min_friction = 185 * 0.02 / 10000 * 2
    print(f"   CME minimum round-trip friction: {cme_min_friction:.6f} price units")
    print(f"   Maximum viable round-trip friction: {185 * fcp / 10000 * 2:.6f} price units (if FCP > 0)")

    # Save
    output = {
        "sweep": "B",
        "spread_sweep_bps": SPREAD_SWEEP_BPS,
        "friction_collapse_point_bps": round(fcp, 6) if fcp else 0,
        "cme_min_friction_price": cme_min_friction,
        "results": results,
        "verdict": "NON-DEPLOYABLE" if (fcp is not None and fcp < 0.02) else "BORDERLINE",
        "avg_win": sum(r["avg_win"] for r in results if r["avg_win"] > 0) / max(sum(1 for r in results if r["avg_win"] > 0), 1),
        "n_signals": len(signals_oss),
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/cep_sweep_b_fcp.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to reports/cep_sweep_b_fcp.json")

    # Summary table
    print(f"\n{'Spread(bps)':<12} {'RTCost':<10} {'n':<6} {'PF':<8} {'WR':<8} {'AvgWin':<10} {'AvgLoss':<10} {'Sharpe':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['spread_bps']:<12.3f} {r['round_trip_cost']:<10.6f} {r['n_trades']:<6} "
              f"{r['profit_factor']:<8.4f} {r['win_rate']:<8.4f} {r['avg_win']:<10.6f} "
              f"{r['avg_loss']:<10.6f} {r['sharpe']:<8.4f}")

    return output, results


if __name__ == "__main__":
    t0 = time.time()
    output, results = run_sweep_b()
    print(f"\nTotal time: {time.time() - t0:.2f}s")
