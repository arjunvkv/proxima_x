"""verify_nova_parity.py — P1 gate: NOVA == legacy engine on the same tape.

Same specs, same costs, same universe. Asserts per-trade byte equality of the
raw path (entry_ts/exit_ts/side/reason/pnl_pts) AND the USD path (gross,
commission, spread, net). Any drift is a fill-semantics bug, not 'close enough'.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from proxima_ops.backtest.spec import StrategySpec  # noqa: E402
from proxima_ops.backtest import engine as legacy  # noqa: E402
from proxima_ops.backtest.feed import load_bars_cached  # noqa: E402
from proxima_ops.nova import feed as nfeed  # noqa: E402
from proxima_ops.nova import engine as nova  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts", "_absorb"))
from costmaps_r3 import corrected_maps  # noqa: E402

FX18 = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "EURJPY", "GBPJPY",
        "AUDJPY", "EURAUD", "EURNZD", "GBPAUD", "GBPNZD", "GBPCAD",
        "USDCAD", "NZDUSD", "AUDNZD", "EURGBP", "EURCHF", "USDCHF"]

BIG = [1e9, 1e9]

CONFIGS = [
    ("tokyo_hold", dict(rule="session_exhaustion", sessions=[0], lookback=1440,
                        top_n=8, hold_bars=24, sl_tp=BIG)),
    ("cascade_hold", dict(rule="session_exhaustion", sessions=[2, 3, 4],
                          lookback=1440, top_n=8, hold_bars=24, sl_tp=BIG)),
    ("london_hold", dict(rule="session_exhaustion", sessions=[7, 8, 9],
                         lookback=1440, top_n=5, hold_bars=12, sl_tp=BIG)),
    ("usfade_hold", dict(rule="session_momentum", sessions=list(range(14, 20)),
                         lookback=50, top_n=5, hold_bars=24, sl_tp=BIG)),
    ("tokyo_sltp", dict(rule="session_exhaustion", sessions=[0], lookback=1440,
                        top_n=8, hold_bars=12, sl_tp=[0.20, 0.20])),
    ("sr_usfade", dict(rule="session_reversion", sessions=list(range(14, 20)),
                       lookback=1440, top_n=8, hold_bars=24, sl_tp=BIG,
                       side="both")),
    ("bmf_full", dict(rule="big_move_fade", sessions=None, lookback=48,
                      top_n=3, hold_bars=12, sl_tp=BIG, side="both")),
]


def make_spec(cfg: dict) -> StrategySpec:
    d = {
        "name": cfg["rule"],
        "feed": {"kind": "bar", "timeframe": "M5"},
        "universe": FX18,
        "signal": {"rule": cfg["rule"], "lookback": cfg["lookback"],
                   "top_n": cfg["top_n"], "fill_bar": 1,
                   "side": cfg.get("side", "long"),
                   "pick": "n_worst" if cfg["rule"] == "session_exhaustion" else "n_best"},
        "exit": {"mode": "sl_tp_hold", "hold_bars": cfg["hold_bars"],
                 "stop_first": True,
                 "jpy_sl_tp": cfg["sl_tp"], "non_jpy_sl_tp": cfg["sl_tp"]},
    }
    if cfg["sessions"] is not None:
        d["signal"]["sessions"] = cfg["sessions"]
    return StrategySpec.from_dict(d)


def main():
    TICK, SPR = corrected_maps()
    spread_map = {s: SPR[s] for s in FX18}
    bars_legacy = {s: load_bars_cached(s) for s in FX18}
    bars_nova = {s: nfeed.bars_list_to_arrays(bars_legacy[s]) for s in FX18}
    t0 = time.time()
    total_mismatch = 0
    for name, cfg in CONFIGS:
        spec = make_spec(cfg)
        a = legacy.run_strategy(bars_legacy, spec, tick_value_map=TICK,
                                volume=0.15, commission_per_lot=3.0,
                                spread_pips_map=spread_map)
        b = nova.run_strategy(bars_nova, spec, tick_value_map=TICK,
                              volume=0.15, commission_per_lot=3.0,
                              spread_pips_map=spread_map)
        keys = ("symbol", "side", "entry_ts", "exit_ts", "reason",
                "pnl_pts", "gross_usd", "commission", "spread", "net")
        mism = 0
        if len(a) != len(b):
            mism = abs(len(a) - len(b))
        else:
            for x, y in zip(a, b):
                for k in keys:
                    if x.get(k) != y.get(k):
                        mism += 1
                        if mism <= 3:
                            print(f"  [{name}] {k}: legacy={x.get(k)} nova={y.get(k)} sym={x.get('symbol')}")
        total_mismatch += mism
        verdict = "PASS" if mism == 0 else "FAIL"
        print(f"{name:12s} {verdict}  n={len(a):5d}  mismatches={mism}")
    dt = time.time() - t0
    print(f"parity runtime: {dt:.2f}s for {len(CONFIGS)} configs "
          f"({dt / max(len(CONFIGS), 1):.2f}s/config)")
    if total_mismatch:
        print("NOVA PARITY: FAIL")
        sys.exit(1)
    print("NOVA PARITY: PASS — byte-identical trade sequences and USD nets")


if __name__ == "__main__":
    main()
