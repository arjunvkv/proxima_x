"""scripts/_absorb/run_study.py — Hypothesis A/B study over the 7 primary FX majors.

Run: unset PYTHONPATH && ./.venv/Scripts/python.exe scripts/_absorb/run_study.py
Output: scripts/_absorb/results/study_<W>_<T>.json + printed verdict table.
"""
from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from proxima_ops.backtest.feed import build_bars_map  # read-only import
import study  # sibling module (script dir is on sys.path[0])

UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY", "AUDUSD", "USDCAD"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

WS = [24, 36, 48]   # absorption window (bars) — 2h / 3h / 4h
TS = [4, 6]         # transition window (bars) — 20 / 30 min
D = 240             # regime window (bars) = 20h trailing


def fmt(v):
    if v is None:
        return "  n/a   "
    return f"{v:8.4f}"


def main() -> None:
    bars_map = build_bars_map(UNIVERSE)
    nbars = {s: len(b) for s, b in bars_map.items()}
    print(f"tape: {len(UNIVERSE)} symbols, bars {nbars}")
    for W in WS:
        for T in TS:
            path = os.path.join(OUT_DIR, f"study_{W}_{T}.json")
            print(f"\n=== W={W} T={T} (purple shuffle running...) ===")
            res = study.run(bars_map, UNIVERSE, W, T, D, purples=True, out_path=path)
            pg = res["pooled_gated"]
            pr = res["pooled_raw"]
            pn = res["random_null"]
            print(f"{'H':>3} | {'gated n':>8} {'gated mean':>12} {'gated t':>9} | "
                  f"{'raw mean':>10} {'raw t':>8} | {'null mean':>10} {'null sd':>9} {'z':>7}")
            for H in [6, 12, 24, 48]:
                g, r_, n_ = pg[str(H)], pr[str(H)], pn[str(H)]
                z = (g["mean"] - n_["mean"]) / n_["sd"] if n_["sd"] else None
                print(f"{H:>3} | {g['n']:>8} {fmt(g['mean'])} {fmt(g['t'])} | "
                      f"{fmt(r_['mean'])} {fmt(r_['t'])} | "
                      f"{fmt(n_['mean'])} {fmt(n_['sd'])} {fmt(z)}")
            # per-symbol headline (H=12)
            print("per-symbol pooled contribution mean @ H=12 (continuation if >0):")
            for s in res["per_symbol"]:
                print(f"  {s['symbol']:>7} n={s['n_signal']:>5}  "
                      f"gated={fmt(s['gated']['12']['mean'])}  "
                      f"raw={fmt(s['raw']['12']['mean'])}")
            # pooled blocks across symbols
            blocks = {k: {str(h): [] for h in [6, 12, 24, 48]}
                      for k in study.BLOCKS}
            for s in res["per_symbol"]:
                for k, v in s["by_block"].items():
                    for h in [6, 12, 24, 48]:
                        if v[str(h)]["mean"] is not None:
                            blocks[k][str(h)].append(v[str(h)]["mean"])
            print("  blocks (gated mean @H12): " +
                  ", ".join(f"{k}={fmt(sum(v['12']) / len(v['12']) if v['12'] else None)}"
                            for k, v in blocks.items()))


if __name__ == "__main__":
    main()