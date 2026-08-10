"""scripts/_absorb/tick_study.py — the decisive tick-level A/B study.

Loads 1-min aggregates built from genuine trade ticks (tick_pull.py), runs the
SAME absorption->impact-transition battery at native resolution:

  T1 (price-only): measures.AbsorbSignals on minute bars — identical definition
                   to the M5 study, resolution changed.
  T2 (flow-aware): TickAbsorbSignals using tick-rule signed flow — Kyle-style
                   lambda = |price move| / |net flow|; direction = net flow.

Both: forward close-vs-open x direction at H = 30/60/120/240 min (~0.5-4h),
hour-boundary signals, same-day guard, random-position +/-1 null.

Also runs the window-matched M5 comparison (last 60 days = tick window) for an
apples-to-apples resolution test of the SAME calendar.

Run: unset PYTHONPATH && ./.venv/Scripts/python.exe scripts/_absorb/tick_study.py
"""
from __future__ import annotations
import json, os

import numpy as np
import polars as pl

from measures import AbsorbSignals
from study import raw_indices as m5_raw
from tick_measures import TickAbsorbSignals, HORIZONS_FWD, BLOCKS

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS = os.path.join(HERE, "results", "ticks")
UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY", "AUDUSD", "USDCAD"]
CONFIGS = [(24, 4), (24, 6), (36, 4), (36, 6), (48, 4), (48, 6)]
H = HORIZONS_FWD


def load_minutes(sym: str) -> tuple[dict, np.ndarray]:
    """Return (bars dict in engine format, signed-flow array) from the cache."""
    df = pl.read_parquet(os.path.join(TICKS, f"{sym}.pqt"))
    bars = {
        "ts": df["ts"].to_numpy().astype("int64"),
        "open": df["open"].to_numpy().astype("float64"),
        "high": df["high"].to_numpy().astype("float64"),
        "low": df["low"].to_numpy().astype("float64"),
        "close": df["close"].to_numpy().astype("float64"),
    }
    flow = df["imb"].to_numpy().astype("float64")
    return bars, flow


def study_symbol(sig, sym: str, tag: str) -> dict:
    """Forward-return study for one state object (M5 or minute, price or flow)."""
    contrib = {hh: [] for hh in H}          # gated
    raw = {hh: [] for hh in H}
    blocks = {k: {str(hh): [] for hh in H} for k in BLOCKS}
    for i in sig.signal_indices():
        d = float(sig.dir_tr[i])
        hr = int(sig.a["ts"][i]) // 3600 % 24
        blk = next((k for k, (a, b) in BLOCKS.items() if a <= hr < b), "Late")
        for hh in H:
            r, flag = sig.forward_return(int(i), hh)
            if flag:
                contrib[hh].append(r * d)
                blocks[blk][str(hh)].append(r * d)
    for i in raw_indices_for(sig):
        d = float(sig.dir_tr[i])
        for hh in H:
            r, flag = sig.forward_return(int(i), hh)
            if flag:
                raw[hh].append(r * d)
    out = {"symbol": sym, "tag": tag, "n_signal": len(contrib[H[0]]),
           "gated": {str(hh): summarize(contrib[hh]) for hh in H},
           "raw": {str(hh): summarize(raw[hh]) for hh in H},
           "by_block": {k: {str(hh): summarize(v.get(str(hh), [])) for hh in H}
                        for k, v in blocks.items()}}
    return out


def raw_indices_for(sig):
    """Guarded hour-boundary bars via the state object's own arrays."""
    return m5_raw(sig) if isinstance(sig, AbsorbSignals) else _raw_tick(sig)


def _raw_tick(sig: TickAbsorbSignals) -> np.ndarray:
    out = []
    ts = sig.a["ts"]
    day = ts // 86400
    for i in range(max(300, 1), sig.n):
        if ts[i] % 3600 != 0:
            continue
        if not (day[i - 1] == day[i - 290] == day[i - 5]):
            continue
        if sig.a["close"][i - 1] == sig.a["open"][i - 4]:
            continue
        out.append(i)
    return np.array(out, dtype=np.int64)


def summarize(arr) -> dict:
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": None, "t": None}
    sd = float(arr.std(ddof=1))
    m = float(arr.mean())
    t = m / (sd / np.sqrt(n)) if sd > 0 else 0.0
    return {"n": n, "mean": round(m, 6), "t": round(t, 3)}


def random_null(pools: dict, n_sig: int, n_iter: int = 500, seed: int = 7):
    rng = np.random.default_rng(seed)
    guard = {hh: np.array(pools[hh]) for hh in H}
    ng = {hh: len(v) for hh, v in guard.items()}   # per-horizon lengths can differ
    out = {}
    for hh in H:
        a = guard[hh]
        if ng[hh] == 0:
            out[str(hh)] = {"mean": None, "sd": None}
            continue
        means = np.empty(n_iter)
        for k in range(n_iter):
            idx = rng.integers(0, ng[hh], size=n_sig)
            sgn = rng.choice([-1.0, 1.0], size=n_sig)
            means[k] = float((a[idx] * sgn).mean())
        out[str(hh)] = {"mean": round(float(means.mean()), 6),
                        "sd": round(float(means.std(ddof=1)), 6)}
    return out, ng


def run_variant(bars_map: dict, flow_map: dict, syms: list, W: int, T: int,
                variant: str) -> dict:
    sigs, fwd = {}, {hh: [] for hh in H}
    n_sig = 0
    for s in syms:
        if variant == "T1":
            sig = AbsorbSignals(bars_map[s], W=W, T=T, D=240)
        else:
            sig = TickAbsorbSignals(bars_map[s], flow_map[s], W=W, T=T, D=240,
                                    use_flow=True)
        sigs[s] = sig
        n_sig += len(sig.signal_indices())
        for i in raw_indices_for(sig):
            for hh in H:
                r, flag = sig.forward_return(int(i), hh)
                if flag:
                    fwd[hh].append(r)
    null, ng = random_null(fwd, n_sig)
    per = [study_symbol(sigs[s], s, variant) for s in syms]
    pooled = {hh: [] for hh in H}
    for s_ in per:
        for hh in H:
            if s_["gated"][str(hh)]["mean"] is not None:
                pooled[hh].append(s_["gated"][str(hh)]["mean"])
    return {"variant": variant, "W": W, "T": T,
            "per_symbol": per, "null": null, "n_signal": n_sig,
            "n_guarded": ng,
            "pooled_mean": {str(hh): round(float(np.mean(pooled[hh])), 6)
                            if pooled[hh] else None for hh in H}}


def fmt(x):
    return "--" if x is None else f"{x:+.4f}"


def print_variant(res: dict) -> None:
    print(f"--- {res['variant']} W={res['W']} T={res['T']} "
          f"(signals={res['n_signal']}, guarded={res['n_guarded']})")
    print(f"{'H':>4} | {'gated mean':>11} | {'raw mean':>10} | "
          f"{'null mean':>10} {'null sd':>9} {'z':>7}")
    for hh in H:
        gm = np.mean([s["gated"][str(hh)]["mean"] for s in res["per_symbol"]
                      if s["gated"][str(hh)]["mean"] is not None])
        rm = np.mean([s["raw"][str(hh)]["mean"] for s in res["per_symbol"]
                      if s["raw"][str(hh)]["mean"] is not None])
        nm, nsd = res["null"][str(hh)]["mean"], res["null"][str(hh)]["sd"]
        z = (gm - nm) / nsd if nsd else None
        print(f"{hh:>4} | {fmt(gm)!s:>11} | {fmt(rm)!s:>10} | "
              f"{nm:+.4f} {nsd:.4f} {fmt(z)}")
    print("  per-symbol gated mean @ H=60:")
    for s in res["per_symbol"]:
        g = s["gated"]["60"]["mean"]
        print(f"    {s['symbol']:>7} n={s['n_signal']:>5} gated={fmt(g)}")


def window_m5(bars_map: dict, syms: list, from_ts: int) -> dict:
    """Window-matched M5 study (last 60 days = tick window)."""
    print(f"--- M5 window-matched study (ts >= {from_ts})")
    out = {}
    for W, T in CONFIGS:
        gated, raw = {hh: [] for hh in [6, 12, 24, 48]}, {hh: [] for hh in [6, 12, 24, 48]}
        for s in syms:
            bars = bars_map[s]
            # build_bars_map returns list-of-dicts; convert view to dict-of-arrays
            if isinstance(bars, list) and bars and isinstance(bars[0], dict):
                bars = {k: np.asarray([b[k] for b in bars])
                        for k in ("ts", "open", "high", "low", "close")}
            keep = np.asarray(bars["ts"]) >= from_ts
            sl = {k: np.asarray(v)[keep] for k, v in bars.items()}
            sig = AbsorbSignals(sl, W=W, T=T, D=240)
            for i in sig.signal_indices():
                d = float(sig.dir_tr[i])
                for hh in [6, 12, 24, 48]:
                    r, f_ = sig.forward_return(int(i), hh)
                    if f_:
                        gated[hh].append(r * d)
            for i in m5_raw(sig):
                d = float(sig.dir_tr[i])
                for hh in [6, 12, 24, 48]:
                    r, f_ = sig.forward_return(int(i), hh)
                    if f_:
                        raw[hh].append(r * d)
        out[f"{W}x{T}"] = {
            "gated": {str(hh): summarize(gated[hh]) for hh in [6, 12, 24, 48]},
            "raw": {str(hh): summarize(raw[hh]) for hh in [6, 12, 24, 48]}}
        print(f"  W={W} T={T}: gated " + " ".join(
            f"H{hh}={fmt(out[f'{W}x{T}']['gated'][str(hh)]['mean'])}"
            for hh in [6, 12, 24, 48]))
    return out


def main() -> None:
    bars_map, flow_map = {}, {}
    with open(os.path.join(TICKS, "_window.json")) as f:
        win = json.load(f)
    for s in UNIVERSE:
        p = os.path.join(TICKS, f"{s}.pqt")
        if not os.path.exists(p):
            print(f"missing cache {p}")
            continue
        bars, flow = load_minutes(s)
        bars_map[s] = bars
        flow_map[s] = flow
        print(f"{s}: {len(bars['ts']):,} minutes, "
              f"net flow {flow.sum():+.0f}, |flow| {np.abs(flow).sum():,.0f}")
    print()
    results = {"window": win}
    for W, T in CONFIGS:
        for variant in ["T1", "T2"]:
            res = run_variant(bars_map, flow_map, list(bars_map), W, T, variant)
            print_variant(res)
            results[f"{variant}_{W}x{T}"] = res
            print()
    # window-matched M5 (todo #10)
    import sys as _sys
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if ROOT not in _sys.path:
        _sys.path.insert(0, ROOT)
    from proxima_ops.backtest.feed import build_bars_map
    m5map = build_bars_map(["EURUSD", "GBPUSD", "USDJPY", "EURJPY",
                            "GBPJPY", "AUDUSD", "USDCAD"])
    from_ts = win["from_s"]
    results["m5_window"] = window_m5(m5map, list(m5map), from_ts)
    out_path = os.path.join(HERE, "results", "tick_study.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()