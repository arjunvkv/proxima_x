"""Verify v2z_bar produces identical results to hfdf_m1 backtest on Dukascopy parquet data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from paper_trade.strategies.v2z_bar.strategy import PerPairState, BarStopManager

PAIR = "GBPNZD"
Z_THRESH = 2.5
STOP_A, TRIG_A, GAP_A, MAX_BARS = 0.15, 0.20, 0.10, 54


def hfdf_m1(b):
    """Reference backtest (exact logic from _verify_alignment.py:hfdf_m1)."""
    c = b["close"].values
    h = b["high"].values
    l = b["low"].values
    n = len(b)
    ret = np.diff(c)
    z_arr = np.full(n, np.nan)
    for i in range(51, n):
        rw = ret[i - 51:i - 1]
        mu = rw.mean()
        sig = rw.std(ddof=1)
        z_arr[i] = (ret[i - 1] - mu) / sig if sig > 1e-10 else 0
    atr_arr = np.full(n, np.nan)
    for i in range(21, n):
        atr_arr[i] = np.mean(b["range"].values[i - 20:i])
    valid = np.where((~np.isnan(z_arr)) & (~np.isnan(atr_arr)) & (np.abs(z_arr) >= Z_THRESH))[0]
    out = []
    for pos in valid:
        if pos + 2 >= n:
            continue
        direction = -1 if z_arr[pos] > 0 else 1
        entry = c[pos]
        s = STOP_A * atr_arr[pos]
        tg = TRIG_A * atr_arr[pos]
        gp = GAP_A * atr_arr[pos]
        best = entry
        exited = False
        for j in range(1, min(MAX_BARS + 1, n - pos)):
            bp = pos + j
            if direction == 1:
                if h[bp] > best: best = h[bp]
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if l[bp] <= sl:
                    out.append({"dir": direction, "entry": entry, "exit": sl,
                                "entry_bar": pos, "exit_bar": bp,
                                "entry_time": int(b.index[pos].timestamp()),
                                "exit_time": int(b.index[bp].timestamp()),
                                "raw_pnl": (sl - entry) * direction,
                                "z": z_arr[pos], "atr": atr_arr[pos], "exit_reason": "stop"})
                    exited = True; break
            else:
                if l[bp] < best: best = l[bp]
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if h[bp] >= sl:
                    out.append({"dir": direction, "entry": entry, "exit": sl,
                                "entry_bar": pos, "exit_bar": bp,
                                "entry_time": int(b.index[pos].timestamp()),
                                "exit_time": int(b.index[bp].timestamp()),
                                "raw_pnl": (sl - entry) * direction,
                                "z": z_arr[pos], "atr": atr_arr[pos], "exit_reason": "stop"})
                    exited = True; break
        if not exited:
            eb = min(pos + MAX_BARS, n - 1)
            out.append({"dir": direction, "entry": entry, "exit": c[eb],
                        "entry_bar": pos, "exit_bar": eb,
                        "entry_time": int(b.index[pos].timestamp()),
                        "exit_time": int(b.index[eb].timestamp()),
                        "raw_pnl": (c[eb] - entry) * direction,
                        "z": z_arr[pos], "atr": atr_arr[pos], "exit_reason": "expiry"})
    return out


def v2z_replay(b):
    """Replay v2z_bar PerPairState + BarStopManager on parquet bars."""
    ps = PerPairState("TEST", {"z_window": 50, "atr_window": 20, "z_thresh": Z_THRESH})
    mgr = BarStopManager({"stop_a": STOP_A, "trig_a": TRIG_A, "gap_a": GAP_A, "max_hold_min": MAX_BARS})

    seed_n = 51  # gives z_buf=50 entries, matches backtest start at bar 51
    n = len(b)
    for i in range(seed_n):
        bar = {"open": float(b.iloc[i]["open"]), "high": float(b.iloc[i]["high"]),
               "low": float(b.iloc[i]["low"]), "close": float(b.iloc[i]["close"]),
               "time": int(b.index[i].timestamp())}
        ps.seed_bar(bar)

    trades = []
    pair = "TEST"
    _signal_info = {}
    _entry_bars = {}  # pair -> dataframe index of entry bar

    for i in range(seed_n, n):
        bar = {"open": float(b.iloc[i]["open"]), "high": float(b.iloc[i]["high"]),
               "low": float(b.iloc[i]["low"]), "close": float(b.iloc[i]["close"]),
               "time": int(b.index[i].timestamp())}
        bar_data = {pair: bar}
        bt = int(bar["time"])

        # Stop checks first (may close position before new signal)
        closed = mgr.check_stops(bar_data, bt)
        for ct in closed:
            si = _signal_info.pop(pair, {"z": 0, "atr": 0})
            entry_idx = _entry_bars.pop(pair, 0)
            trades.append({"dir": ct["direction"], "entry": ct["entry"], "exit": ct["exit"],
                           "entry_bar": entry_idx, "exit_bar": i,
                           "entry_time": ct["entry_time"], "exit_time": bt,
                           "raw_pnl": (ct["exit"] - ct["entry"]) * ct["direction"],
                           "z": si["z"], "atr": si["atr"], "exit_reason": ct["exit_reason"]})

        # Signal generation
        signal = ps.on_bar(bar)
        if signal is not None and mgr.pair_count(pair) == 0:
            _signal_info[pair] = {"z": signal["z_score"], "atr": signal["atr"]}
            _entry_bars[pair] = i
            mgr.add(pair, signal["direction"], signal["entry_price"],
                    signal["atr"], entry_time=int(signal["bar_time"]))

    return trades


def main():
    print(f"{'=' * 70}")
    print(f"V2+z BAR VERIFICATION: v2z_bar vs hfdf_m1")
    print(f"Pair: {PAIR}  |  Z>{Z_THRESH}  |  Stop: {STOP_A}/{TRIG_A}/{GAP_A} ATR  |  Max hold: {MAX_BARS}")
    print(f"{'=' * 70}")

    print(f"\n[1] Loading parquet data...")
    df = pd.read_parquet(f"research/phase_dislocation/dukascopy_data/{PAIR.lower()}.parquet")
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index)
    df["range"] = df["high"] - df["low"]
    print(f"  {len(df)} M1 bars, {df.index[0]} to {df.index[-1]}")

    print(f"\n[2] Running hfdf_m1 (reference backtest)...")
    t_bar = hfdf_m1(df)
    pnls_bar = [t["raw_pnl"] for t in t_bar if t["raw_pnl"] is not None]
    n_bar = len(pnls_bar)
    wr_bar = sum(1 for p in pnls_bar if p > 0) / n_bar if n_bar > 0 else 0
    net_bar = sum(pnls_bar)
    print(f"  Trades: {n_bar}  WR: {wr_bar:.1%}  Net: {net_bar:.6f} ({net_bar * 10000:.0f} MP)")

    print(f"\n[3] Running v2z_bar replay...")
    t_v2z = v2z_replay(df)
    pnls_v2z = [t["raw_pnl"] for t in t_v2z if t["raw_pnl"] is not None]
    n_v2z = len(pnls_v2z)
    wr_v2z = sum(1 for p in pnls_v2z if p > 0) / n_v2z if n_v2z > 0 else 0
    net_v2z = sum(pnls_v2z)
    print(f"  Trades: {n_v2z}  WR: {wr_v2z:.1%}  Net: {net_v2z:.6f} ({net_v2z * 10000:.0f} MP)")

    print(f"\n[4] Trade-by-trade comparison...")
    bar_by_entry = {t["entry_bar"]: t for t in t_bar}
    v2z_by_entry = {t["entry_bar"]: t for t in t_v2z}
    common = []
    bar_only = []
    v2z_only = []
    for eb in sorted(set(list(bar_by_entry.keys()) + list(v2z_by_entry.keys()))):
        tb = bar_by_entry.get(eb)
        tv = v2z_by_entry.get(eb)
        if tb and tv:
            common.append((tb, tv))
        elif tb:
            bar_only.append(tb)
        else:
            v2z_only.append(tv)

    print(f"  Common: {len(common)} trades")
    print(f"  hfdf_m1 only: {len(bar_only)} trades")
    print(f"  v2z_bar only: {len(v2z_only)} trades")

    if common:
        entry_diffs = []
        exit_diffs = []
        pnl_diffs = []
        dir_match = 0
        reason_match = 0
        for tb, tv in common:
            entry_diffs.append(abs(tb["entry"] - tv["entry"]))
            exit_diffs.append(abs(tb["exit"] - tv["exit"]))
            pnl_diffs.append(abs(tb["raw_pnl"] - tv["raw_pnl"]))
            if tb["dir"] == tv["dir"]: dir_match += 1
            if tb["exit_reason"] == tv["exit_reason"]: reason_match += 1
        print(f"  Entry diff:  mean={np.mean(entry_diffs):.8f}  median={np.median(entry_diffs):.8f}")
        print(f"  Exit diff:   mean={np.mean(exit_diffs):.8f}  median={np.median(exit_diffs):.8f}")
        print(f"  PnL diff:    mean={np.mean(pnl_diffs):.8f}  median={np.median(pnl_diffs):.8f}")
        print(f"  Direction agreement: {dir_match}/{len(common)} = {dir_match / len(common):.0%}")
        print(f"  Exit reason agreement: {reason_match}/{len(common)} = {reason_match / len(common):.0%}")
        max_ed = max(entry_diffs)
        max_xd = max(exit_diffs)
        print(f"  Max entry diff: {max_ed:.8f} ({max_ed * 10000:.2f} MP)")
        print(f"  Max exit diff:  {max_xd:.8f} ({max_xd * 10000:.2f} MP)")

    if bar_only:
        pnls_bo = [t["raw_pnl"] for t in bar_only]
        print(f"\n  hfdf_m1-only trades ({len(bar_only)}): WR={sum(1 for p in pnls_bo if p>0)/len(pnls_bo):.1%}")

    if v2z_only:
        pnls_vo = [t["raw_pnl"] for t in v2z_only]
        print(f"\n  v2z_bar-only trades ({len(v2z_only)}): WR={sum(1 for p in pnls_vo if p>0)/len(pnls_vo):.1%}")

    print(f"\n[5] Summary:")
    print(f"  {'Metric':<20}  {'hfdf_m1':>12}  {'v2z_bar':>12}  {'Match':>10}")
    print(f"  {'-' * 20}  {'-' * 12}  {'-' * 12}  {'-' * 10}")
    print(f"  {'Trades':<20}  {n_bar:>12}  {n_v2z:>12}  {'✓' if n_bar == n_v2z else '✗':>10}")
    print(f"  {'Win Rate':<20}  {wr_bar:>11.1%}  {wr_v2z:>11.1%}  {'✓' if abs(wr_bar - wr_v2z) < 0.001 else '✗':>10}")
    match_pct = 100 * len(common) / max(n_bar, n_v2z) if max(n_bar, n_v2z) > 0 else 0
    print(f"  {'Trade match':<20}  {len(common):>12}  {match_pct:>11.1f}%  {'✓' if match_pct > 99.5 else '≈':>10}")

    if common:
        avg_pnl_diff = np.mean(pnl_diffs)
        print(f"  {'Avg |PnL diff|':<20}  {'':>12}  {avg_pnl_diff:>11.8f}  {'✓' if avg_pnl_diff < 1e-8 else '≈':>10}")

    verdict = "PASS" if len(common) == n_bar == n_v2z else "MINOR DIFF"
    print(f"\n{'=' * 70}")
    print(f"VERDICT: {verdict}")
    if bar_only:
        print(f"  hfdf_m1-only trades: {len(bar_only)} (missed by v2z_bar)")
        for t in bar_only:
            print(f"    Bar {t['entry_bar']}: dir={t['dir']:+d} z={t['z']:+.2f} entry={t['entry']:.6f}")
    if v2z_only:
        print(f"  v2z_bar-only trades: {len(v2z_only)} (extra)")
        for t in v2z_only:
            print(f"    Bar {t['entry_bar']}: dir={t['dir']:+d} z={t['z']:+.2f} entry={t['entry']:.6f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
