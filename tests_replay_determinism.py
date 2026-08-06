"""Replay determinism regression test — the G3 guard.

G3 incident (2026-08-06): verify_walkforward_gate.load_ticks() used
polars .unique(subset=["timestamp_ns"]) WITHOUT maintain_order=True.
Row order shuffled across threads, so same code + same tape produced
different trade counts on every run (observed: 35/43/57/60). The
strategy was fine; the measurement pipeline was unstable.

Fix: sort -> unique(keep-first, maintain_order=True) -> sort, making the
canonical tick stream byte-reproducible. This test enforces the contract
at the reusable layer (not just main()).

Run:  python3 tests_replay_determinism.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scripts.verify_walkforward_gate as g

_results = []
def check(name: str, cond: bool, detail: str = ""):
    _results.append((name, bool(cond)))


def main() -> int:
    print("=" * 60)
    print("TEST 1: load_ticks twice -> identical stream (hash + count)")
    print("=" * 60)
    a = g.load_ticks(g.TRAIN_START, g.TRAIN_END)
    b = g.load_ticks(g.TRAIN_START, g.TRAIN_END)
    check("train count stable", len(a) == len(b))
    ha, hb = g.tick_hash(a), g.tick_hash(b)
    check("train order-stable hash", ha == hb)
    check("assert_deterministic self-consistent",
          g.assert_deterministic(a, b, "train") == ha)
    print(f"  train ticks={len(a)} hash={ha[:16]}")
    for n, c in _results: print(f"  {'PASS' if c else 'FAIL'} {n}")

    print("=" * 60)
    print("TEST 2: val window also deterministic")
    print("=" * 60)
    va = g.load_ticks(g.VAL_START, g.VAL_END)
    vb = g.load_ticks(g.VAL_START, g.VAL_END)
    check("val count stable", len(va) == len(vb))
    check("val hash stable", g.tick_hash(va) == g.tick_hash(vb))
    print(f"  val ticks={len(va)}")
    for name, c in _results[3:]:
        print(f"  {'PASS' if c else 'FAIL'} {name}")

    print("=" * 60)
    print("TEST 3: guard actually raises on mismatch (G3 protection)")
    print("=" * 60)
    # deterministic negative: reordered stream must trigger the guard.
    swapped = [dict(a[1]), dict(a[0])] if len(a) >= 2 else []
    try:
        g.assert_deterministic(a, swapped, "neg")
        caught = False
    except AssertionError:
        caught = True
    check("order-sensitivity raises", caught and g.tick_hash(swapped) != g.tick_hash(a))
    print(f"  trigger-on-reorder: {'PASS' if caught else 'FAIL'}")

    print("=" * 60)
    fails = [n for n, c in _results if not c]
    print(f"RESULT: {len(_results) - len(fails)} passed, {len(fails)} failed")
    for name in fails:
        print(f"  FAILED: {name}")
    print("=" * 60)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())