#!/usr/bin/env python3
"""London H7 — P0 empirical analysis, P1 parameter sweep, P2 multi-broker validation.

Usage:
    python strategies/london_h7/sweep.py
"""
import sys, time, json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.london_h7.strategy import (
    LondonH7Strategy, _pip_value, ALL_PAIRS, _pair_group,
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
    """Empirical analysis of London Open sweep-and-reversal."""
    print("=" * 70)
    print("PHASE 0: London Open Sweep Analysis")
    print("=" * 70)

    pair_stats = defaultdict(lambda: {
        "sw_events": 0, "judas": 0, "wins": 0, "losses": 0,
        "total_ret": 0.0, "sweep_pips": [], "fwd_rets": [],
    })
    asian_ranges = {}
    sw_candidates = {}
    close_0705 = {}
    printed = set()
    debug_rows = 0
    _cur_date = None

    for row_idx, row in enumerate(pre_align):
        ts = row["time"]
        hour = ts.hour if hasattr(ts, "hour") else 0
        minute = ts.minute if hasattr(ts, "minute") else 0
        today = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)

        if _cur_date != today:
            _cur_date = today
            asian_ranges = {}
            sw_candidates = {}
            close_0705 = {}

        # Track Asian range
        if hour < 7:
            for pair in ALL_PAIRS:
                val = row.get(pair)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                high = row.get(f"{pair}_high", val)
                low = row.get(f"{pair}_low", val)
                ar = asian_ranges.setdefault(pair, {"high": -1e9, "low": 1e9, "count": 0})
                ar["high"] = max(ar["high"], high)
                ar["low"] = min(ar["low"], low)
                ar["count"] += 1

        # At 07:00: detect sweep
        if hour == 7 and minute == 0:
            for pair in ALL_PAIRS:
                ar = asian_ranges.get(pair)
                if not ar:
                    continue
                pv = _pip_value(pair)
                range_pips = (ar["high"] - ar["low"]) / pv
                if range_pips < 8 or range_pips > 50:
                    continue

                high_val = row.get(f"{pair}_high")
                low_val = row.get(f"{pair}_low")
                if high_val is None or low_val is None or np.isnan(high_val) or np.isnan(low_val):
                    continue

                high_sweep = high_val - ar["high"]
                low_sweep = ar["low"] - low_val
                max_sweep = max(high_sweep, low_sweep)
                sweep_pips = max_sweep / pv
                if sweep_pips < 3 or sweep_pips > 30:
                    continue

                sw_candidates[pair] = {
                    "high_sweep": high_sweep, "low_sweep": low_sweep,
                    "sweep_pips": sweep_pips,
                    "is_high_sweep": high_sweep > low_sweep,
                    "asian_high": ar["high"], "asian_low": ar["low"],
                    "range_pips": range_pips, "today": today, "row_idx": row_idx,
                }

        # At 07:05: store close
        if hour == 7 and minute == 5:
            for pair in ALL_PAIRS:
                close_val = row.get(pair)
                if close_val is not None and not np.isnan(close_val):
                    close_0705[pair] = close_val

        # At 07:10: check Judas Swing + forward return
        if hour == 7 and minute == 10:
            for pair, sw in list(sw_candidates.items()):
                if sw["today"] != today:
                    continue
                pair_stats[pair]["sw_events"] += 1

                c5 = close_0705.get(pair)
                if c5 is None:
                    continue

                if not (sw["asian_low"] <= c5 <= sw["asian_high"]):
                    continue

                pv = _pip_value(pair)
                entry_price = row.get(f"{pair}_open", row.get(pair))
                if entry_price is None or np.isnan(entry_price):
                    continue

                future_idx = row_idx + 12
                if future_idx >= len(pre_align):
                    continue
                fwd_row = pre_align[future_idx]
                fwd_close = fwd_row.get(pair)
                if fwd_close is None or np.isnan(fwd_close):
                    continue

                if sw["is_high_sweep"]:
                    fwd_ret = (entry_price - fwd_close) / entry_price
                else:
                    fwd_ret = (fwd_close - entry_price) / entry_price

                pair_stats[pair]["judas"] += 1
                pair_stats[pair]["wins"] += 1 if fwd_ret > 0 else 0
                pair_stats[pair]["losses"] += 1 if fwd_ret <= 0 else 0
                pair_stats[pair]["total_ret"] += fwd_ret
                pair_stats[pair]["sweep_pips"].append(sw["sweep_pips"])
                pair_stats[pair]["fwd_rets"].append(fwd_ret)

                if debug_rows < 60:
                    debug_rows += 1
                    direction = "SHORT" if sw["is_high_sweep"] else "LONG"
                    print(f"  {today} {pair:>6s} {direction:>5s} "
                          f"sweep={sw['sweep_pips']:>4.1f}p range={sw['range_pips']:>4.1f}p "
                          f"fwd_ret={fwd_ret:>+7.4f}")

    # Summary
    print(f"\n{'─' * 70}")
    print(f"{'Pair':>6s}  {'Swp':>4s}  {'Judas':>5s}  {'WR':>5s}  {'AvgRet%':>8s}  {'AvgSwp':>7s}  {'PF':>5s}")
    print(f"{'─' * 70}")
    results = []
    for pair in ALL_PAIRS:
        s = pair_stats[pair]
        if s["judas"] < 2:
            print(f"  {pair:>6s}  {s['sw_events']:>4d}  {s['judas']:>5d}  {'N/A':>5s}  {'N/A':>8s}  {'N/A':>7s}  {'N/A':>5s}")
            continue
        wr = s["wins"] / s["judas"]
        avg_ret = s["total_ret"] / s["judas"] * 100
        avg_sw = float(np.mean(s["sweep_pips"])) or 0
        gw = sum(r for r in s["fwd_rets"] if r > 0) or 0.0001
        gl = abs(sum(r for r in s["fwd_rets"] if r <= 0)) or 0.0001
        pf = gw / gl
        results.append((pair, s["judas"], s["wins"], wr, avg_ret, avg_sw, pf, s["sw_events"]))
        print(f"  {pair:>6s}  {s['sw_events']:>4d}  {s['judas']:>5d}  {wr:>4.1%}  {avg_ret:>+7.3f}%  {avg_sw:>6.1f}p  {pf:>4.2f}")

    results.sort(key=lambda x: x[3], reverse=True)
    print(f"\nSorted by WR (min 5 Judas):")
    for pair, n, wins, wr, ret, asw, pf, swp in results:
        if wr >= 0.50 and n >= 5:
            print(f"  {pair:>6s}  judas={n:>3d}/{swp:>3d}  WR={wr:>4.1%}  ret={ret:>+6.2f}%  PF={pf:.2f}")

    return pair_stats


def run_cfg(data, pre_align, strategy, broker):
    e = MultiPairBacktestEngine(strategy, ExecutionSimulator(broker))
    return e.run(data, pre_aligned=pre_align)


def main():
    t0 = time.time()
    print("Loading & aligning M5 data...")
    raw, pre_align = load_and_align()
    print(f"  {len(raw)} pairs, {len(pre_align):,} aligned bars ({time.time()-t0:.1f}s)")

    pair_stats = run_p0(pre_align)

    # Build subsets
    high_wr = []
    positive = []
    for pair in ALL_PAIRS:
        s = pair_stats[pair]
        if s["judas"] < 5:
            continue
        wr = s["wins"] / s["judas"]
        avg_ret = s["total_ret"] / s["judas"]
        if wr >= 0.55 and avg_ret > 0:
            high_wr.append(pair)
        if avg_ret > 0:
            positive.append(pair)

    subsets = {}
    if high_wr:
        subsets["high_wr"] = high_wr
    if positive:
        subsets["positive"] = positive
    subsets["all"] = ALL_PAIRS

    print(f"\nP0 complete. Subsets:")
    for name, plist in subsets.items():
        print(f"  {name:10s}: {len(plist)} pairs: {', '.join(plist[:6])}{'...' if len(plist) > 6 else ''}")

    t1 = time.time()

    # Phase 1: Sweep on Exness
    print(f"\n{'=' * 70}")
    print(f"PHASE 1: Parameter sweep on Exness")
    print(f"{'=' * 70}")

    SWEEP_MINS = [3, 5, 8]
    TOP_N_VALS = [3, 5]
    HOLDS = [6, 9, 12, 15, 18]

    phase1 = []
    total_configs = len(subsets) * len(SWEEP_MINS) * len(TOP_N_VALS) * len(HOLDS)
    cfg_idx = 0

    for sname, spairs in subsets.items():
        for smin in SWEEP_MINS:
            for tn in TOP_N_VALS:
                for hb in HOLDS:
                    cfg_idx += 1
                    params = {
                        "trade_pairs": spairs,
                        "sweep_min_pips": smin,
                        "sweep_max_pips": 30,
                        "top_n": tn,
                        "hold_bars_major": hb,
                        "hold_bars_jpy": hb,
                        "hold_bars_volatile": hb,
                    }
                    strat = LondonH7Strategy(params)
                    r = run_cfg(raw, pre_align, strat, "exness")
                    elapsed = time.time() - t1
                    t1 = time.time()

                    entry = {
                        "subset": sname, "n_pairs": len(spairs),
                        "sweep_min": smin, "top_n": tn, "hold": hb,
                        "trades": r.n_trades, "net_pnl": r.net_pnl,
                        "wr": r.win_rate, "pf": r.profit_factor,
                        "sharpe": r.sharpe, "dd": r.max_drawdown_pct,
                    }
                    phase1.append(entry)
                    bar = f"  [{cfg_idx}/{total_configs}] {sname:10s} smin={smin} tn={tn} h={hb:>2d} | "
                    bar += f"T={r.n_trades:>3d} Net=${r.net_pnl:>+8.2f} WR={r.win_rate*100:>5.1f}% PF={r.profit_factor:>5.2f}"
                    print(bar)

    phase1.sort(key=lambda x: x["net_pnl"], reverse=True)
    print(f"\nTop 5 Exness:")
    for i, r in enumerate(phase1[:5]):
        print(f"  #{i+1} {r['subset']:10s} smin={r['sweep_min']} tn={r['top_n']} h={r['hold']:>2d} "
              f"T={r['trades']:>3d} Net=${r['net_pnl']:>+8.2f} WR={r['wr']*100:>5.1f}% PF={r['pf']:>5.2f}")

    # Bottom 5
    print(f"\nBottom 5:")
    for i, r in enumerate(phase1[-5:]):
        print(f"  #{len(phase1)-4+i} {r['subset']:10s} smin={r['sweep_min']} tn={r['top_n']} h={r['hold']:>2d} "
              f"T={r['trades']:>3d} Net=${r['net_pnl']:>+8.2f} WR={r['wr']*100:>5.1f}% PF={r['pf']:>5.2f}")

    # Save
    out = Path(__file__).parent / "sweep_results.json"
    p0_out = {}
    for pair, s in pair_stats.items():
        p0_out[pair] = {
            "sw_events": s["sw_events"], "judas": s["judas"],
            "wins": s["wins"],
            "wr": s["wins"] / s["judas"] if s["judas"] > 0 else 0,
            "avg_fwd_ret_pct": float(np.mean(s["fwd_rets"]) * 100) if s["fwd_rets"] else 0,
            "avg_sweep_pips": float(np.mean(s["sweep_pips"])) if s["sweep_pips"] else 0,
        }
    with open(out, "w") as f:
        json.dump({"p0": p0_out, "phase1": phase1, "total_sec": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nSaved to {out}")

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"LONDON H7 — COMPLETE ({total:.0f}s)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
