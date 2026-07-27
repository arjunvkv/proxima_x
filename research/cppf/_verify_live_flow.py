"""Verify the live run.py flow matches the verification_replay.
Tests generate_signal (with MT5 mocked) + entry_queue + BarStopManager.check_stops
against hfdf_m1 reference backtest."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

PAIR = "GBPNZD"
Z_THRESH = 2.5
STOP_A, TRIG_A, GAP_A, MAX_BARS = 0.15, 0.20, 0.10, 54


def hfdf_m1(b):
    """Reference backtest (same as _verify_v2z_bar.py)."""
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


# ─── Mock MT5 for live generate_signal ───

class MockMT5:
    """Mocks _mt5.copy_rates_from_pos to return pre-loaded parquet bars.
    
    Time-aware: uses _current_time (set before each generate_signal call)
    to resolve positions. Position N = bar starting at (current_minute - N) * 60.
    This matches real MT5 semantics across gaps.
    """
    TIMEFRAME_M1 = 1
    _current_time = None

    def __init__(self, bars_by_pair):
        self._bars = {}
        for pair, df in bars_by_pair.items():
            self._bars[pair.upper()] = df

    def _bar_at_minute(self, df, minute_ts):
        """Find bar with timestamp at or before minute_ts (in seconds)."""
        if isinstance(df.index, pd.DatetimeIndex):
            target = pd.to_datetime(minute_ts, unit="s")
            mask = df.index <= target
        else:
            mask = df.index <= minute_ts
        if not mask.any():
            return None
        return df.iloc[mask.sum() - 1]

    def copy_rates_from_pos(self, pair, timeframe, start_pos, count):
        pair = pair.upper()
        if pair not in self._bars or self._current_time is None:
            return None
        df = self._bars[pair]
        target_ts = (self._current_time // 60 - start_pos) * 60
        row = self._bar_at_minute(df, target_ts)
        if row is None:
            return None
        ts = int(row["timestamp"]) if "timestamp" in row else int(row.name.timestamp()) if hasattr(row.name, "timestamp") else int(target_ts)
        arr = np.array([(ts, float(row["open"]), float(row["high"]),
                         float(row["low"]), float(row["close"]), 0, 0, 0)],
                       dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"),
                              ("low", "f8"), ("close", "f8"), ("tick_volume", "i8"),
                              ("spread", "i8"), ("real_volume", "i8")])
        return arr

    def copy_rates_from(self, pair, timeframe, date_from, count):
        pair = pair.upper()
        if pair not in self._bars:
            return None
        df = self._bars[pair]
        minute_ts = date_from
        row = self._bar_at_minute(df, minute_ts)
        if row is None:
            return None
        ts = int(row["timestamp"]) if "timestamp" in row else int(row.name.timestamp()) if hasattr(row.name, "timestamp") else int(minute_ts)
        arr = np.array([(ts, float(row["open"]), float(row["high"]),
                         float(row["low"]), float(row["close"]), 0, 0, 0)],
                       dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"),
                              ("low", "f8"), ("close", "f8"), ("tick_volume", "i8"),
                              ("spread", "i8"), ("real_volume", "i8")])
        return arr


# ─── Live flow simulation ───

def run_live_flow(b, mock_mt5):
    """Simulate the EXACT run.py + generate_signal flow.

    Uses seed_history to prime PerPairState, then the MT5 mock to feed generate_signal
    one bar per minute boundary (matching live MT5 behavior).
    """
    from paper_trade.strategies.v2z_bar.strategy import (
        generate_signal, seed_history, BarStopManager, CONFIG, _states
    )

    # Reset module state
    _states.clear()
    import paper_trade.strategies.v2z_bar.strategy as st
    st._last_minute = None
    st._last_bars = {}

    # Build mock feed for seed_history (reads bars 0..SEED_N-1)
    class MockFeed:
        def __init__(self, bars):
            self._bars = bars
        def copy_m1_history(self, pair, count=100):
            pair = pair.upper()
            if pair not in self._bars:
                return None
            df = self._bars[pair]
            n = min(count, len(df))
            out = []
            for i in range(n):
                row = df.iloc[i]
                out.append({
                    "time": int(row["timestamp"]) if "timestamp" in row else int(row.name.timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                })
            return out

    SEED_N = 51
    PAIR_KEY = "GBPNZD"
    mock_feed = MockFeed(mock_mt5._bars)
    # Override count to match SEED_N
    orig_copy = mock_feed.copy_m1_history
    def seeded_copy(*a, **kw):
        return orig_copy(*a, **{**kw, "count": SEED_N})
    mock_feed.copy_m1_history = seeded_copy
    seed_history(mock_feed)

    stop_mgr = BarStopManager({"stop_a": STOP_A, "trig_a": TRIG_A, "gap_a": GAP_A, "max_hold_min": MAX_BARS})
    entry_queue = []
    trades = []
    _signal_info = {}
    _entry_times = {}

    n = len(b)
    for i in range(SEED_N, n):
        bt = int(b.index[i].timestamp()) if hasattr(b.index[i], "timestamp") else int(b.index[i])
        # Simulate tick data for this bar (minimal)
        data = {PAIR_KEY: {"bid": float(b.iloc[i]["close"]), "ask": float(b.iloc[i]["close"]),
                           "time": bt, "spread": 0}}

        # Step 1: Generate signal (uses mock MT5 internally)
        mock_mt5._current_time = bt
        signals, bar_data = generate_signal(data, current_time=bt)

        # Step 2: Process entry queue
        remaining = []
        for eq in entry_queue:
            if bt >= eq["fire_at"]:
                pair_e = eq["pair"]
                direction = eq["direction"]
                if stop_mgr.pair_count(pair_e) >= 1:
                    continue
                ep = eq["signal"].get("entry_price")
                if ep is None:
                    continue
                atr_v = eq["signal"].get("atr", 1)
                entry_bar_time = eq["signal"].get("bar_time", bt)
                _signal_info[pair_e] = {"z": eq["signal"].get("z_score", 0), "atr": atr_v}
                _entry_times[pair_e] = entry_bar_time
                stop_mgr.add(pair_e, direction, ep, atr_v, entry_time=entry_bar_time)
            else:
                remaining.append(eq)
        entry_queue = remaining

        # Step 3: Queue new signals
        for signal in signals:
            if signal.get("confidence", 0) >= 0.30 and stop_mgr.pair_count(signal["pair"]) == 0:
                entry_queue.append({
                    "pair": signal["pair"], "direction": signal["direction"],
                    "fire_at": bt, "signal": signal,
                })

        # Step 4: Stop checks
        closed = stop_mgr.check_stops(bar_data, bt)
        for ct in closed:
            si = _signal_info.pop(ct["pair"], {"z": 0, "atr": 0})
            et = _entry_times.pop(ct["pair"], 0)
            trades.append({
                "dir": ct["direction"], "entry": ct["entry"], "exit": ct["exit"],
                "entry_bar": 0, "exit_bar": 0,
                "entry_time": et, "exit_time": ct["exit_time"],
                "raw_pnl": (ct["exit"] - ct["entry"]) * ct["direction"],
                "z": si["z"], "atr": si["atr"], "exit_reason": ct["exit_reason"],
            })

    return trades


def main():
    print(f"{'=' * 70}")
    print(f"LIVE FLOW VERIFICATION: run.py flow vs hfdf_m1")
    print(f"Pair: {PAIR}  |  Z>{Z_THRESH}")
    print(f"{'=' * 70}")

    print(f"\n[1] Loading parquet data...")
    df = pd.read_parquet(f"research/phase_dislocation/dukascopy_data/{PAIR.lower()}.parquet")
    # The parquet may have 'timestamp' column or be indexed
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index)
    df["range"] = df["high"] - df["low"]
    print(f"  {len(df)} M1 bars")

    print(f"\n[2] Running hfdf_m1 (reference backtest)...")
    t_bar = hfdf_m1(df)
    pnls_bar = [t["raw_pnl"] for t in t_bar if t["raw_pnl"] is not None]
    n_bar = len(pnls_bar)
    wr_bar = sum(1 for p in pnls_bar if p > 0) / n_bar if n_bar > 0 else 0
    net_bar = sum(pnls_bar)
    print(f"  Trades: {n_bar}  WR: {wr_bar:.1%}  Net: {net_bar:.6f} ({net_bar * 10000:.0f} MP)")

    print(f"\n[3] Running live flow simulation (generate_signal + entry_queue + check_stops)...")
    # Create mock MT5 with the parquet data keyed by the strategy pair
    mock_mt5 = MockMT5({"GBPNZD": df})
    # Stub the _mt5 module
    import paper_trade.strategies.v2z_bar.strategy as st
    st._mt5 = mock_mt5

    t_live = run_live_flow(df, mock_mt5)
    pnls_live = [t["raw_pnl"] for t in t_live if t["raw_pnl"] is not None]
    n_live = len(pnls_live)
    wr_live = sum(1 for p in pnls_live if p > 0) / n_live if n_live > 0 else 0
    net_live = sum(pnls_live)
    print(f"  Trades: {n_live}  WR: {wr_live:.1%}  Net: {net_live:.6f} ({net_live * 10000:.0f} MP)")

    print(f"\n[4] Trade-by-trade comparison (matched by entry_time)...")
    bar_by_et = {t["entry_time"]: t for t in t_bar}
    live_by_et = {t["entry_time"]: t for t in t_live}
    common = []
    bar_only = []
    live_only = []
    for et in sorted(set(list(bar_by_et.keys()) + list(live_by_et.keys()))):
        tb = bar_by_et.get(et)
        tl = live_by_et.get(et)
        if tb and tl:
            common.append((tb, tl))
        elif tb:
            bar_only.append(tb)
        else:
            live_only.append(tl)

    print(f"  Common: {len(common)} trades")
    print(f"  hfdf_m1 only: {len(bar_only)} trades")
    print(f"  Live only: {len(live_only)} trades")

    if common:
        entry_diffs = [abs(tb["entry"] - tl["entry"]) for tb, tl in common]
        exit_diffs = [abs(tb["exit"] - tl["exit"]) for tb, tl in common]
        pnl_diffs = [abs(tb["raw_pnl"] - tl["raw_pnl"]) for tb, tl in common]
        dir_match = sum(1 for tb, tl in common if tb["dir"] == tl["dir"])
        reason_match = sum(1 for tb, tl in common if tb["exit_reason"] == tl["exit_reason"])
        nonzero_exit = [(tb, tl) for tb, tl in common if abs(tb["exit"] - tl["exit"]) > 1e-10]
        reason_mismatch = [(tb, tl) for tb, tl in common if tb["exit_reason"] != tl["exit_reason"]]
        print(f"  Entry diff:  mean={np.mean(entry_diffs):.8f}  median={np.median(entry_diffs):.8f}")
        print(f"  Exit diff:   mean={np.mean(exit_diffs):.8f}  median={np.median(exit_diffs):.8f}")
        print(f"  PnL diff:    mean={np.mean(pnl_diffs):.8f}  median={np.median(pnl_diffs):.8f}")
        print(f"  Direction agreement: {dir_match}/{len(common)} = {dir_match / len(common):.0%}")
        print(f"  Exit reason agreement: {reason_match}/{len(common)} = {reason_match / len(common):.0%}")
        if nonzero_exit:
            print(f"\n  Trades with exit diff > 1e-10 ({len(nonzero_exit)}):")
            for tb, tl in nonzero_exit[:10]:
                print(f"    ET={tl['entry_time']}: live_exit={tl['exit']:.8f} bt_exit={tb['exit']:.8f} diff={tb['exit']-tl['exit']:.10f} reason={tl['exit_reason']} dir={tl['dir']:+d}")
        if reason_mismatch:
            print(f"\n  Exit reason mismatches ({len(reason_mismatch)}):")
            for tb, tl in reason_mismatch:
                print(f"    ET={tl['entry_time']}: live={tl['exit_reason']} bt={tb['exit_reason']} exit_diff={tb['exit']-tl['exit']:.10f}")

    if bar_only:
        print(f"  hfdf_m1-only trades ({len(bar_only)}):")
        for t in bar_only:
            print(f"    Bar {t['entry_bar']}: dir={t['dir']:+d} z={t['z']:+.2f}")
    if live_only:
        print(f"  Live-only trades ({len(live_only)}):")
        for t in live_only:
            print(f"    Bar {t['entry_bar']}: dir={t['dir']:+d} z={t['z']:+.2f}")

    print(f"\n{'=' * 70}")
    match_pct = 100 * len(common) / max(n_bar, n_live) if max(n_bar, n_live) > 0 else 0
    if common:
        avg_pnl_diff = np.mean(pnl_diffs)
    else:
        avg_pnl_diff = 1.0
    print(f"VERDICT: {'PASS' if match_pct > 99.0 and avg_pnl_diff < 1e-8 else 'FAIL'}")
    print(f"  {len(common)}/{max(n_bar, n_live)} trades matched ({match_pct:.1f}%)")
    if bar_only:
        print(f"  {len(bar_only)} hfdf_m1-only trades: 1-position-per-pair limit")
    if live_only:
        print(f"  {len(live_only)} live-only trades (unexpected)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
