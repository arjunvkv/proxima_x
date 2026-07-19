"""Run all Engine 2 backtests in sequence. Collect written evidence."""
import time
import sys
import importlib
import MetaTrader5 as mt5

MODULES = [
    'run_range_ratio',
    'run_regime',
    'run_lead_lag',
    'run_spread_compression',
    'run_tick_vol_divergence',
    'run_triangle_coherence',
    'run_entry_gate',
]

def main():
    total_start = time.time()

    for mod_name in MODULES:
        print(f"\n\n{'#' * 70}")
        print(f"# RUNNING: {mod_name}")
        print(f"{'#' * 70}")
        # Force fresh MT5 state by shutting down before each run
        try:
            mt5.shutdown()
        except:
            pass
        time.sleep(0.5)

        if mod_name in sys.modules:
            del sys.modules[mod_name]
        mod = importlib.import_module(mod_name)
        mod.run()
        print(f"\nCompleted in {time.time() - total_start:.0f}s")

    print(f"\n\n{'=' * 70}")
    print("ALL ENGINE 2 BACKTESTS COMPLETE")
    print(f"Total time: {time.time() - total_start:.0f}s")
    print(f"{'=' * 70}")

if __name__ == '__main__':
    main()
