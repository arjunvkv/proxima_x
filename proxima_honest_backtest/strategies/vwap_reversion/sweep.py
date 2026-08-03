#!/usr/bin/env python3
"""VWAP Reversion — P0 empirical analysis, P1 parameter sweep.

Usage:
    python strategies/vwap_reversion/sweep.py
"""
import sys, time, json, math
from pathlib import Path
from collections import defaultdict, deque
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.vwap_reversion.strategy import (
    VWAPReversionStrategy, _pip_value, ALL_PAIRS,
)
from proxima_honest_backtest.strategies.multi_pair_engine import MultiPairBacktestEngine
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from data.providers.mt5_provider import MT5Provider

BROKERS = ["exness","ftmo","fundednext","fusionmarkets","dukascopy"]
MONTHS = [(2026, m) for m in range(1, 8)]


def load_and_align(tf="m5"):
    provider = MT5Provider()
    raw = {}
    for p in ALL_PAIRS:
        frames = [provider.load_rates(p, y, m, tf) for y, m in MONTHS]
        frames = [f for f in frames if not f.empty]
        if frames:
            d = pd.concat(frames, ignore_index=True)
            d.sort_values("time", inplace=True)
            d.reset_index(drop=True, inplace=True)
            raw[p] = d
    pieces = []
    for pair, df in raw.items():
        sub = df.set_index("time")[["close","open","high","low","tick_volume","spread"]]
        sub.columns = [pair, f"{pair}_open", f"{pair}_high", f"{pair}_low", f"{pair}_volume", f"{pair}_spread"]
        pieces.append(sub)
    aligned = pd.concat(pieces, axis=1, sort=True)
    aligned.sort_index(inplace=True)
    aligned.ffill(inplace=True); aligned.bfill(inplace=True)
    aligned.reset_index(inplace=True); aligned.rename(columns={"index": "time"}, inplace=True)
    return raw, aligned.to_dict("records")


def run_p0(pre_align):
    """Phase 0: VWAP deviation analysis with CORRECT z-score (deviation std, not price std)."""
    print("=" * 70)
    print("PHASE 0: VWAP Deviation Reversion (correct z-score)")
    print("=" * 70)

    pair_data = defaultdict(lambda: {
        "events": 0, "wins": 0, "total_ret": 0.0,
        "fwd_rets": [], "z_sizes": [],
    })
    _cur_date = None
    _vctx: dict = {}  # pair -> {cum_tp, cum_vol, deviations_deque}
    _debug = set()

    for row_idx, row in enumerate(pre_align):
        ts = row["time"]
        hour = ts.hour if hasattr(ts, "hour") else 0
        today = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)

        if _cur_date != today:
            _cur_date = today
            _vctx = {}

        for pair in ALL_PAIRS:
            close = row.get(pair)
            high = row.get(f"{pair}_high")
            low = row.get(f"{pair}_low")
            volume = row.get(f"{pair}_volume", 0)
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (close, high, low)):
                continue
            vol = max(float(volume), 0.0)
            tp = (high + low + close) / 3.0
            ctx = _vctx.setdefault(pair, {"cum_tp": 0.0, "cum_vol": 0.0, "devs": deque(maxlen=50)})
            ctx["cum_tp"] += tp * vol
            ctx["cum_vol"] += vol

            if ctx["cum_vol"] <= 0:
                continue
            vwap = ctx["cum_tp"] / ctx["cum_vol"]
            dev = close - vwap
            ctx["devs"].append(dev)

            if len(ctx["devs"]) < 24:
                continue
            arr = list(ctx["devs"])[-20:]
            mean_d = sum(arr) / len(arr)
            var_d = sum((d - mean_d) ** 2 for d in arr) / len(arr)
            std = math.sqrt(var_d) if var_d > 1e-24 else 0.0
            if std <= 0:
                continue

            z = dev / std
            pip = _pip_value(pair)
            dev_pips = abs(dev) / pip
            if dev_pips < 3:
                continue

            if abs(z) > 2.0:
                pair_data[pair]["events"] += 1

                # Forward return over hold_bars=10 (50 min)
                future_idx = row_idx + 10
                if future_idx < len(pre_align):
                    fwd_row = pre_align[future_idx]
                    fwd_close = fwd_row.get(pair)
                    if fwd_close is not None and not np.isnan(fwd_close):
                        entry_open = row.get(f"{pair}_open", close)
                        if z > 0:
                            fwd_ret = (entry_open - fwd_close) / entry_open
                        else:
                            fwd_ret = (fwd_close - entry_open) / entry_open

                        pair_data[pair]["total_ret"] += fwd_ret
                        pair_data[pair]["fwd_rets"].append(fwd_ret)
                        pair_data[pair]["z_sizes"].append(abs(z))
                        if fwd_ret > 0:
                            pair_data[pair]["wins"] += 1

                        if len(_debug) < 30:
                            _debug.add(f"{today}_{pair}")
                            direction = "SHORT" if z > 0 else "LONG"
                            print(f"  {today} {pair:>6s} {direction:>5s} z={z:>+5.2f} dev_pips={dev_pips:>4.1f} fwd_ret={fwd_ret:>+7.4f}")

    print(f"\n{'─' * 70}")
    print(f"{'Pair':>6s}  {'Events':>7s}  {'Wins':>5s}  {'WR':>5s}  {'AvgRet%':>8s}  {'AvgZ':>5s}")
    print(f"{'─' * 70}")
    results = []
    for pair in ALL_PAIRS:
        d = pair_data[pair]
        if d["events"] < 10:
            continue
        wr = d["wins"] / d["events"]
        avg_ret = d["total_ret"] / d["events"] * 100
        avg_z = float(np.mean(d["z_sizes"])) if d["z_sizes"] else 0
        results.append((pair, d["events"], d["wins"], wr, avg_ret, avg_z))
        print(f"{pair:>6s}  {d['events']:>7d}  {d['wins']:>5d}  {wr:>4.1%}  {avg_ret:>+7.3f}%  {avg_z:>4.2f}σ")

    results.sort(key=lambda x: x[3], reverse=True)
    print(f"\nTop by WR (min 100 events):")
    for pair, n, w, wr, ret, az in results:
        if n >= 100:
            print(f"  {pair:>6s}  n={n:>4d}  WR={wr:>4.1%}  ret={ret:>+6.2f}%  z={az:.2f}σ")

    return pair_data


def run_cfg(data, pre_align, strategy, broker):
    e = MultiPairBacktestEngine(strategy, ExecutionSimulator(broker))
    return e.run(data, pre_aligned=pre_align)


def main():
    t0 = time.time()
    print("Loading & aligning M5 data...")
    raw, pre_align = load_and_align()
    print(f"  {len(raw)} pairs, {len(pre_align):,} aligned bars ({time.time()-t0:.1f}s)")

    pair_data = run_p0(pre_align)

    # Build subsets
    high_wr = []
    positive = []
    for pair in ALL_PAIRS:
        d = pair_data[pair]
        if d["events"] < 50:
            continue
        wr = d["wins"] / d["events"]
        avg_ret = d["total_ret"] / d["events"]
        if wr >= 0.55 and avg_ret > 0:
            high_wr.append(pair)
        if avg_ret > 0:
            positive.append(pair)

    subsets = {"all": ALL_PAIRS}
    if high_wr:
        subsets["high_wr"] = high_wr
    if positive:
        subsets["positive"] = positive

    print(f"\nP0 complete. Subsets:")
    for name, plist in subsets.items():
        print(f"  {name:10s}: {len(plist)} pairs")

    t1 = time.time()

    # Phase 1
    print(f"\n{'=' * 70}")
    print(f"PHASE 1: Parameter sweep on Exness")
    print(f"{'=' * 70}")

    SIGMAS = [2.0, 2.5, 3.0]
    TOP_N_VALS = [3, 5]
    HOLDS = [5, 10, 15]

    phase1 = []
    total_cfgs = len(subsets) * len(SIGMAS) * len(TOP_N_VALS) * len(HOLDS)
    cfg_idx = 0

    for sname, spairs in subsets.items():
        for sig in SIGMAS:
            for tn in TOP_N_VALS:
                for hb in HOLDS:
                    cfg_idx += 1
                    params = {
                        "trade_pairs": spairs,
                        "sigma_entry": sig,
                        "sigma_exit": 0.5,
                        "hold_bars": hb,
                        "top_n": tn,
                        "max_positions": tn,
                    }
                    strat = VWAPReversionStrategy(params)
                    r = run_cfg(raw, pre_align, strat, "exness")
                    elapsed = time.time() - t1
                    t1 = time.time()

                    entry = {
                        "subset": sname, "n_pairs": len(spairs),
                        "sigma": sig, "top_n": tn, "hold": hb,
                        "trades": r.n_trades, "net_pnl": r.net_pnl,
                        "wr": r.win_rate, "pf": r.profit_factor,
                    }
                    phase1.append(entry)
                    bar = f"  [{cfg_idx}/{total_cfgs}] {sname:10s} σ={sig} tn={tn} h={hb:>2d} | "
                    bar += f"T={r.n_trades:>4d} Net=${r.net_pnl:>+8.2f} WR={r.win_rate*100:>5.1f}% PF={r.profit_factor:>5.2f}"
                    print(bar)
                    if r.n_trades > 0:
                        print(f"         -> best_win={r.avg_win:.2f} avg_loss={r.avg_loss:.2f} dd={r.max_drawdown_pct:.2f}%")

    phase1.sort(key=lambda x: x["net_pnl"], reverse=True)
    print(f"\nTop 5 Exness:")
    for i, r in enumerate(phase1[:5]):
        print(f"  #{i+1} {r['subset']:10s} σ={r['sigma']} tn={r['top_n']} h={r['hold']:>2d} "
              f"T={r['trades']:>4d} Net=${r['net_pnl']:>+8.2f} WR={r['wr']*100:>5.1f}% PF={r['pf']:>5.2f}")

    out = Path(__file__).parent / "sweep_results.json"
    with open(out, "w") as f:
        json.dump({"p0": {pair: dict(d) for pair, d in pair_data.items()},
                    "phase1": phase1, "total_sec": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nSaved to {out}")
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
