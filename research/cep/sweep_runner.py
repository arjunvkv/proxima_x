"""CEP Sweep Runner — orchestrates counterfactual sweeps."""
import sys; sys.path.insert(0, ".")
import json, os, time
from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from signals.outcome_surface_signal import OutcomeSurfaceSignal
from research.replay_cache import ReplayCache
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from research.cep.execution_profile import load_profiles
from research.cep.counterfactual_execution import CounterfactualExecutionEngine
from research.cep.metrics import CEPMetrics


class CEPSweepRunner:
    def __init__(self, config: ReplayConfig = None):
        self._config = config or ReplayConfig(
            symbols=["EURJPY", "USDJPY"],
            start="2026-06-22", end="2026-06-22",
            speed=500000, burst=True,
            latency=False, slippage=False, seed=42,
        )
        self._profiles = load_profiles()
        self._metrics = CEPMetrics()
        self._signals: dict[str, list[dict]] = {}
        self._price_lookup: dict[int, float] = {}

    def load_and_generate_signals(self):
        """Build replay env, train OSS, walk ticks, generate signals."""
        print("Building replay environment...")
        env = build_replay_environment(self._config)
        patch_clock(env.clock)
        feed = env.replay_feed

        print(f"Feed loaded: {feed._total_loaded} ticks, merged: {len(feed._merged)}")

        print("Training OSS from cache...")
        oss = OutcomeSurfaceSignal()
        try:
            cache = ReplayCache(["EURJPY"], "2026-04-01", "2026-04-20",
                                tick_limit=50000, seed=42)
            ticks = cache.compute()
            recs, ed, doa = [], {}, DelayedOutcomeEngine(horizon_ticks=20)
            for t in ticks:
                if len(recs) >= 5000:
                    break
                sym = t["sym"]
                d = t.get("ecdf", 0.5) - 0.5
                sig = 1 if d > 0.05 else (-1 if d < -0.05 else 0)
                ed[sym] = {"price": t["price"], "ecdf_rank": t["ecdf"], "signal": sig}
                doa.record_snapshot(ed)
                if doa.ready:
                    for s2, outcome in doa.evaluate({s2: ed[s2]["price"] for s2 in ed}).items():
                        recs.append({"sym": s2, "ecdf": ed[s2]["ecdf_rank"], "outcome": outcome})
            if recs:
                oss = OutcomeSurfaceSignal.from_pipeline_records(recs)
                print(f"OSS trained: {oss.bucket_count()} buckets, density={oss.signal_density():.2f}")
        except Exception as e:
            print(f"OSS train error: {e}")

        print("Walking replay ticks and generating signals...")
        signals_all = []
        last_minute = -1

        while True:
            tick = feed.next()
            if tick is None:
                break
            ts = int(tick.get("time_sec", 0))
            price = float(tick.get("ask", tick.get("bid", 0)))
            sym = tick.get("symbol", "")

            # Store price lookup per second (first tick per second)
            if ts not in self._price_lookup:
                self._price_lookup[ts] = price

            # Minute-boundary signal cycle
            minute_key = ts // 60
            if minute_key > last_minute:
                last_minute = minute_key
                ecdf = 0.5  # placeholder
                oss_sig = oss.predict(ecdf)

                # OSS signal
                if oss_sig != 0:
                    signals_all.append({
                        "ts": ts, "price": price, "direction": oss_sig,
                        "source": "OSS", "sym": sym,
                    })

                # TrOSS cross1
                signals_all.append({
                    "ts": ts, "price": price, "direction": oss_sig if oss_sig != 0 else 0,
                    "source": "TrOSS_cross1", "sym": sym,
                })
                signals_all.append({
                    "ts": ts, "price": price, "direction": oss_sig if oss_sig != 0 else 0,
                    "source": "TrOSS_cross2", "sym": sym,
                })

        # Group by signal source
        for sig in signals_all:
            src = sig["source"]
            self._signals.setdefault(src, []).append(sig)

        for src, sigs in self._signals.items():
            print(f"  {src}: {len(sigs)} signals")

        return self._signals, self._price_lookup

    def sweep_profiles(self, signal_name: str = "OSS") -> list[dict]:
        """Sweep all execution profiles for a given signal."""
        sigs = self._signals.get(signal_name, [])
        if not sigs:
            print(f"No signals for {signal_name}")
            return []

        results = []
        for pname, profile in self._profiles.items():
            engine = CounterfactualExecutionEngine(profile)
            trades = engine.run_signals(sigs, self._price_lookup, signal_name)
            metrics = self._metrics.compute(trades)
            results.append({
                "profile": pname,
                "signal": signal_name,
                "n_trades": metrics["n_trades"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "expectancy": metrics["expectancy"],
                "sharpe": metrics["sharpe"],
                "monetization_ratio": metrics["monetization_ratio"],
                "total_pnl": metrics["total_pnl"],
                "avg_win": metrics["avg_win"],
                "avg_loss": metrics["avg_loss"],
            })
            print(f"  {pname}: n={metrics['n_trades']}, PF={metrics['profit_factor']:.3f}, "
                  f"WR={metrics['win_rate']:.3f}, MR={metrics['monetization_ratio']:.3f}")

        return results

    def run_sweep_a(self):
        """Sweep A: all profiles × all signals."""
        print("\n" + "=" * 60)
        print("SWEEP A: EXECUTION PROFILE PORTABILITY")
        print("=" * 60)
        all_results = []
        for sig_name in ["OSS", "TrOSS_cross1", "TrOSS_cross2"]:
            print(f"\n--- Signal: {sig_name} ---")
            results = self.sweep_profiles(sig_name)
            all_results.extend(results)

        # Find best by PF
        best = max(all_results, key=lambda r: r["profit_factor"]) if all_results else None
        if best:
            print(f"\nBest combination: {best['signal']} + {best['profile']}")
            print(f"  PF={best['profit_factor']:.3f}, WR={best['win_rate']:.3f}, "
                  f"MR={best['monetization_ratio']:.3f}, n={best['n_trades']}")

        return all_results, best

    def save_results(self, results: list[dict], path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {path}")

    def print_summary_table(self, results: list[dict]):
        print(f"\n{'Profile':<15} {'Signal':<15} {'n':<6} {'PF':<8} {'WR':<8} {'MR':<8} {'Sharpe':<8} {'PnL':<10}")
        print("-" * 80)
        for r in sorted(results, key=lambda x: x["profit_factor"], reverse=True):
            print(f"{r['profile']:<15} {r['signal']:<15} {r['n_trades']:<6} "
                  f"{r['profit_factor']:<8.3f} {r['win_rate']:<8.3f} "
                  f"{r['monetization_ratio']:<8.3f} {r['sharpe']:<8.3f} "
                  f"{r['total_pnl']:<10.6f}")


if __name__ == "__main__":
    t0 = time.time()
    runner = CEPSweepRunner()
    runner.load_and_generate_signals()
    results, best = runner.run_sweep_a()
    runner.print_summary_table(results)
    runner.save_results(results, "reports/cep_sweep_a.json")
    print(f"\nTotal time: {time.time() - t0:.2f}s")
