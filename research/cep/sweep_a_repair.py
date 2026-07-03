"""Sweep A Repair v2 — uses build_replay_environment for April data."""
import sys; sys.path.insert(0, ".")
import json, os, time
from collections import defaultdict
from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from features.ecdf_transform import PerSymbolECDF
from signals.outcome_surface_signal import OutcomeSurfaceSignal
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from research.cep.execution_profile import load_profiles
from research.cep.counterfactual_execution import CounterfactualExecutionEngine
from research.cep.metrics import CEPMetrics


class SweepARepairV2:
    def __init__(self, n_ticks: int = 80000):
        self._n_ticks = n_ticks
        self._profiles = load_profiles()
        self._metrics = CEPMetrics()

    def run(self):
        print("=" * 60)
        print(f"SWEEP A REPAIR V2: April cache, {self._n_ticks} ticks")
        print("=" * 60)

        print("\n1. Building replay environment (April 1-20)...")
        t0 = time.time()
        config = ReplayConfig(
            symbols=["EURJPY"],
            start="2026-04-01", end="2026-04-20",
            speed=500000, burst=True,
            latency=False, slippage=False, seed=42,
        )
        env = build_replay_environment(config)
        patch_clock(env.clock)
        feed = env.replay_feed
        print(f"   Feed: {feed._total_loaded} loaded, {len(feed._merged)} merged ({time.time()-t0:.2f}s)")

        # Walk ticks, build stream, train OSS, generate signals
        print("\n2. Walking ticks + training OSS + generating signals...")
        t0 = time.time()
        ecdf = PerSymbolECDF(window_size=2000)
        price_lookup = {}
        oss_recs, ed, doa = [], {}, DelayedOutcomeEngine(horizon_ticks=20)

        signals_oss = []
        signals_tross = []
        last_minute = -1
        tick_count = 0
        price_bufs = defaultdict(list)

        while tick_count < self._n_ticks:
            tick = feed.next()
            if tick is None:
                break
            tick_count += 1

            sym = tick.get("symbol", "EURJPY")
            price = float(tick.get("ask", tick.get("bid", tick.get("price", 0))))
            ts = int(tick.get("time_sec", 0))

            # ECDF update
            ecdf_rank = ecdf.update(sym, price)

            # Entropy
            price_bufs[sym].append(price)
            if len(price_bufs[sym]) > 50:
                price_bufs[sym] = price_bufs[sym][-50:]
            entropy = self._entropy(price_bufs[sym])

            # Price lookup per second
            if ts and ts not in price_lookup:
                price_lookup[ts] = price

            # DOA training records (first N ticks)
            if tick_count < 50000:
                d = ecdf_rank - 0.5
                sig = 1 if d > 0.05 else (-1 if d < -0.05 else 0)
                ed[sym] = {"price": price, "ecdf_rank": ecdf_rank, "signal": sig}
                doa.record_snapshot(ed)
                if doa.ready:
                    for s2, outcome in doa.evaluate({s2: ed[s2]["price"] for s2 in ed}).items():
                        oss_recs.append({"sym": s2, "ecdf": ed[s2]["ecdf_rank"], "outcome": outcome})

            # Minute-boundary signal (after warmup)
            minute_key = ts // 60 if ts else 0
            if minute_key > last_minute and tick_count >= 5000:
                last_minute = minute_key
                oss_sig = self._predict_oss_later(ecdf_rank)  # placeholder, will use trained OSS

            if tick_count % 10000 == 0:
                print(f"   ... {tick_count} ticks processed ({time.time()-t0:.2f}s)")

        print(f"   {tick_count} ticks processed ({time.time()-t0:.2f}s)")
        print(f"   {len(price_lookup)} unique seconds")
        print(f"   {len(oss_recs)} OSS training records")

        # Train OSS
        print("\n3. Training OSS...")
        oss = OutcomeSurfaceSignal()
        if oss_recs:
            oss = OutcomeSurfaceSignal.from_pipeline_records(oss_recs)
            print(f"   {oss.bucket_count()} buckets, density={oss.signal_density():.2f}")

        # Re-walk to generate signals with trained OSS
        print("\n4. Generating signals with trained OSS...")
        feed.seek(0)
        last_minute = -1
        tick_count = 0
        ecdf2 = PerSymbolECDF(window_size=2000)

        while tick_count < self._n_ticks:
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
                signals_tross.append({
                    "ts": ts, "price": price,
                    "direction": oss_sig if oss_sig != 0 else 0,
                    "source": "TrOSS_cross1", "sym": sym,
                })

        print(f"   OSS: {len(signals_oss)} signals")
        print(f"   TrOSS_cross1: {len(signals_tross)} signals")

        # Sweep profiles
        print("\n5. Sweeping execution profiles...")
        all_results = []
        for signal_name, sigs in [("OSS", signals_oss), ("TrOSS_cross1", signals_tross)]:
            print(f"\n--- {signal_name} ---")
            if not sigs:
                print("   No signals, skipping")
                continue
            for pname, profile in self._profiles.items():
                engine = CounterfactualExecutionEngine(profile, seed=42)
                trades = engine.run_signals(sigs, price_lookup, signal_name)
                metrics = self._metrics.compute(trades)
                row = {
                    "profile": pname, "signal": signal_name,
                    "n_trades": metrics["n_trades"],
                    "win_rate": metrics["win_rate"],
                    "profit_factor": metrics["profit_factor"],
                    "expectancy": metrics["expectancy"],
                    "sharpe": metrics["sharpe"],
                    "monetization_ratio": metrics["monetization_ratio"],
                    "total_pnl": metrics["total_pnl"],
                    "avg_win": metrics["avg_win"],
                    "avg_loss": metrics["avg_loss"],
                }
                all_results.append(row)
                print(f"   {pname:<15} n={row['n_trades']:<5} PF={row['profit_factor']:<8.3f} "
                      f"WR={row['win_rate']:<8.3f} MR={row['monetization_ratio']:<8.3f}")

        return all_results

    def _predict_oss_later(self, ecdf_rank):
        return 0

    def _entropy(self, prices):
        if len(prices) < 3:
            return 0.5
        n = len(prices)
        diffs = [prices[i] - prices[i-1] for i in range(1, n)]
        pos = sum(1 for d in diffs if d > 0)
        neg = sum(1 for d in diffs if d < 0)
        total = pos + neg
        if total == 0:
            return 0.5
        pp = pos / total
        pn = neg / total
        if pp <= 0 or pn <= 0:
            return 0.0
        return - (pp * math.log2(pp) + pn * math.log2(pn))

    def print_table(self, results):
        print(f"\n{'Profile':<15} {'Signal':<15} {'n':<6} {'PF':<8} {'WR':<8} {'MR':<8} {'Sharpe':<8} {'AvgWin':<10}")
        print("-" * 90)
        for r in sorted(results, key=lambda x: x["profit_factor"], reverse=True):
            print(f"{r['profile']:<15} {r['signal']:<15} {r['n_trades']:<6} "
                  f"{r['profit_factor']:<8.3f} {r['win_rate']:<8.3f} "
                  f"{r['monetization_ratio']:<8.3f} {r['sharpe']:<8.3f} "
                  f"{r['avg_win']:<10.6f}")

    def save(self, results, path="reports/cep_sweep_a_repaired.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    import math
    t0 = time.time()
    runner = SweepARepairV2(n_ticks=80000)
    results = runner.run()
    runner.print_table(results)
    runner.save(results)
    print(f"\nTotal time: {time.time() - t0:.2f}s")
