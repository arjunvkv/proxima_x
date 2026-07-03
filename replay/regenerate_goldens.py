"""Regenerate golden snapshots for all parity test profiles."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from replay.parity import ParityLedger, GOLDEN_DIR
import json


def _run_single(symbols, n_ticks=100):
    config = ReplayConfig(
        symbols=symbols,
        start="2026-03-12",
        end="2026-05-12",
        speed=500000,
        burst=True,
        latency=False,
        slippage=False,
        seed=42,
    )
    env = build_replay_environment(config)
    patch_clock(env.clock)
    ledger = ParityLedger(symbol="_".join(symbols), seed=42)
    for sym in symbols:
        for _ in range(n_ticks):
            tick = env.tick_source.get_tick(sym)
            if tick:
                ledger.add_tick(tick)
    ledger.finalize({"cursor": len(symbols) * n_ticks})
    return ledger


def _save_golden(ledger, name):
    path = os.path.join(GOLDEN_DIR, f"{name}.json")
    payload = {**ledger.build(), **ledger._build_meta()}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved golden: {path}")
    return payload


if __name__ == "__main__":
    print("Regenerating single-symbol golden (EURJPY, seed=42)...")
    led1 = _run_single(["EURJPY"], n_ticks=100)
    g1 = _save_golden(led1, "golden_eurjpy_seed42")
    print(f"  H_ticks={g1['H_ticks'][:16]}... tick_count={g1['tick_count']}")
    print(f"  git_sha={g1['_meta']['git_sha']}")

    print("\nRegenerating dual-symbol golden (EURJPY+USDJPY, seed=42)...")
    led2 = _run_single(["EURJPY", "USDJPY"], n_ticks=200)
    g2 = _save_golden(led2, "golden_eurjpy_usdjpy_seed42")
    print(f"  H_ticks={g2['H_ticks'][:16]}... tick_count={g2['tick_count']}")
    print(f"  git_sha={g2['_meta']['git_sha']}")

    print("\nDone. Golden snapshots regenerated.")
