"""Run live flow simulation on MT5-sourced M1 data to verify trade matching."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
import numpy as np
import datetime

PAIR = "GBPNZD"
Z_THRESH = 2.5
STOP_A, TRIG_A, GAP_A, MAX_BARS = 0.15, 0.20, 0.10, 54

def hfdf_m1(df):
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    ret = np.diff(c)
    z_arr = np.full(n, np.nan)
    for i in range(51, n):
        rw = ret[i - 51:i - 1]
        mu = rw.mean()
        sig = rw.std(ddof=1)
        z_arr[i] = (ret[i - 1] - mu) / sig if sig > 1e-10 else 0
    atr_arr = np.full(n, np.nan)
    for i in range(21, n):
        atr_arr[i] = np.mean(df["range"].values[i - 20:i])
    valid = np.where((~np.isnan(z_arr)) & (~np.isnan(atr_arr)) & (np.abs(z_arr) >= Z_THRESH))[0]
    out = []
    for pos in valid:
        if pos + 2 >= n: continue
        direction = -1 if z_arr[pos] > 0 else 1
        entry = c[pos]
        s = STOP_A * atr_arr[pos]; tg = TRIG_A * atr_arr[pos]; gp = GAP_A * atr_arr[pos]
        best = entry
        for j in range(1, min(MAX_BARS + 1, n - pos)):
            bp = pos + j
            if direction == 1:
                if h[bp] > best: best = h[bp]
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if l[bp] <= sl:
                    out.append({"dir": direction, "entry": entry, "exit": sl,
                                "entry_bar": pos, "exit_bar": bp,
                                "entry_time": int(df.index[pos].timestamp()),
                                "exit_time": int(df.index[bp].timestamp()),
                                "raw_pnl": (sl - entry) * direction,
                                "z": z_arr[pos], "atr": atr_arr[pos], "exit_reason": "stop"})
                    break
            else:
                if l[bp] < best: best = l[bp]
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if h[bp] >= sl:
                    out.append({"dir": direction, "entry": entry, "exit": sl,
                                "entry_bar": pos, "exit_bar": bp,
                                "entry_time": int(df.index[pos].timestamp()),
                                "exit_time": int(df.index[bp].timestamp()),
                                "raw_pnl": (sl - entry) * direction,
                                "z": z_arr[pos], "atr": atr_arr[pos], "exit_reason": "stop"})
                    break
        else:
            eb = min(pos + MAX_BARS, n - 1)
            out.append({"dir": direction, "entry": entry, "exit": c[eb],
                        "entry_bar": pos, "exit_bar": eb,
                        "entry_time": int(df.index[pos].timestamp()),
                        "exit_time": int(df.index[eb].timestamp()),
                        "raw_pnl": (c[eb] - entry) * direction,
                        "z": z_arr[pos], "atr": atr_arr[pos], "exit_reason": "expiry"})
    return out

def run_live_flow(b, mock_mt5):
    from paper_trade.strategies.v2z_bar.strategy import (
        generate_signal, seed_history, BarStopManager, CONFIG, _states
    )
    _states.clear()
    import paper_trade.strategies.v2z_bar.strategy as st
    st._last_minute = None
    st._last_bars = {}

    class MockFeed:
        def __init__(self, bars):
            self._bars = {k.upper(): v for k, v in bars.items()}
        def copy_m1_history(self, pair, count=100):
            pair = pair.upper()
            if pair not in self._bars: return None
            df = self._bars[pair]
            n = min(count, len(df))
            out = []
            for i in range(n):
                ts = int(df.index[i].timestamp()) if hasattr(df.index[i], "timestamp") else int(df.index[i])
                out.append({"time": ts, "open": float(df.iloc[i]["open"]),
                            "high": float(df.iloc[i]["high"]), "low": float(df.iloc[i]["low"]),
                            "close": float(df.iloc[i]["close"])})
            return out

    SEED_N = 51
    PAIR_KEY = "GBPNZD"
    mock_feed = MockFeed(mock_mt5._bars)
    orig_copy = mock_feed.copy_m1_history
    def seeded_copy(pair, count=100):
        return orig_copy(pair, count=SEED_N)
    mock_feed.copy_m1_history = seeded_copy
    seed_history(mock_feed)

    stop_mgr = BarStopManager({"stop_a": STOP_A, "trig_a": TRIG_A, "gap_a": GAP_A, "max_hold_min": MAX_BARS})
    entry_queue = []
    trades = []
    _signal_info = {}
    _entry_times = {}

    n = len(b)
    for i in range(SEED_N, n):
        bt = int(b.index[i].timestamp())
        data = {PAIR_KEY: {"bid": float(b.iloc[i]["close"]), "ask": float(b.iloc[i]["close"]),
                           "time": bt, "spread": 0}}

        mock_mt5._current_time = bt
        signals, bar_data = generate_signal(data, current_time=bt)

        remaining = []
        for eq in entry_queue:
            if bt >= eq["fire_at"]:
                pair_e = eq["pair"]
                direction = eq["direction"]
                if stop_mgr.pair_count(pair_e) >= 1: continue
                ep = eq["signal"].get("entry_price")
                if ep is None: continue
                atr_v = eq["signal"].get("atr", 1)
                entry_bar_time = eq["signal"].get("bar_time", bt)
                stop_mgr.add(pair_e, direction, ep, atr_v, entry_time=entry_bar_time)
            else:
                remaining.append(eq)
        entry_queue = remaining

        for signal in signals:
            if signal.get("confidence", 0) >= 0.30 and stop_mgr.pair_count(signal["pair"]) == 0:
                entry_queue.append({"pair": signal["pair"], "direction": signal["direction"],
                                    "fire_at": bt, "signal": signal})

        closed = stop_mgr.check_stops(bar_data, bt)
        for ct in closed:
            trades.append({
                "dir": ct["direction"], "entry": ct["entry"], "exit": ct["exit"],
                "entry_time": ct.get("entry_time", 0), "exit_time": ct["exit_time"],
                "raw_pnl": (ct["exit"] - ct["entry"]) * ct["direction"],
                "exit_reason": ct["exit_reason"],
            })

    return trades


# ─── Main ───
print("=" * 70)
print("LIVE FLOW VERIFICATION on MT5 DATA")
print(f"Pair: {PAIR}  |  Z>{Z_THRESH}")
print("=" * 70)

df = pd.read_parquet(f"research/cppf/_mt5_data/{PAIR.lower()}.parquet")
df.index = pd.to_datetime(df.index, utc=True)
df["range"] = df["high"] - df["low"]
print(f"\n{len(df)} M1 bars loaded")

print(f"\nRunning reference backtest (hfdf_m1)...")
t_bar = hfdf_m1(df)
pnls_bar = [t["raw_pnl"] for t in t_bar if t["raw_pnl"] is not None]
print(f"  Trades: {len(pnls_bar)}  WR: {sum(1 for p in pnls_bar if p > 0)/len(pnls_bar):.1%}  Net: {sum(pnls_bar)*10000:.0f}pips")

print(f"\nRunning live flow simulation...")
class MockMT5:
    TIMEFRAME_M1 = 1
    _current_time = None
    def __init__(self, bars_by_pair):
        self._bars = {k.upper(): v for k, v in bars_by_pair.items()}
    def _bar_at_minute(self, df, minute_ts):
        if isinstance(df.index, pd.DatetimeIndex):
            target = pd.Timestamp(minute_ts, unit="s")
            if df.index.tz is not None:
                target = target.tz_localize("UTC")
            mask = df.index <= target
        else:
            mask = df.index <= minute_ts
        if not mask.any(): return None
        return df.iloc[mask.sum() - 1]
    def copy_rates_from(self, pair, timeframe, date_from, count):
        pair = pair.upper()
        if pair not in self._bars: return None
        df = self._bars[pair]
        row = self._bar_at_minute(df, date_from)
        if row is None: return None
        ts = int(row.name.timestamp()) if hasattr(row.name, "timestamp") else int(date_from)
        arr = np.array([(ts, float(row["open"]), float(row["high"]),
                         float(row["low"]), float(row["close"]), 0, 0, 0)],
                       dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"),
                              ("low", "f8"), ("close", "f8"), ("tick_volume", "i8"),
                              ("spread", "i8"), ("real_volume", "i8")])
        return arr

    def copy_rates_from_pos(self, pair, timeframe, start_pos, count):
        pair = pair.upper()
        if pair not in self._bars or self._current_time is None: return None
        df = self._bars[pair]
        target_ts = (self._current_time // 60 - start_pos) * 60
        row = self._bar_at_minute(df, target_ts)
        if row is None: return None
        ts = int(row.name.timestamp()) if hasattr(row.name, "timestamp") else int(target_ts)
        arr = np.array([(ts, float(row["open"]), float(row["high"]),
                         float(row["low"]), float(row["close"]), 0, 0, 0)],
                       dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"),
                              ("low", "f8"), ("close", "f8"), ("tick_volume", "i8"),
                              ("spread", "i8"), ("real_volume", "i8")])
        return arr

mock = MockMT5({PAIR: df})
import paper_trade.strategies.v2z_bar.strategy as st
st._mt5 = mock

t_live = run_live_flow(df, mock)
pnls_live = [t["raw_pnl"] for t in t_live if t["raw_pnl"] is not None]
print(f"  Trades: {len(pnls_live)}  WR: {sum(1 for p in pnls_live if p > 0)/len(pnls_live):.1%}  Net: {sum(pnls_live)*10000:.0f}pips")

print(f"\nTrade-by-trade comparison...")
bar_by_et = {t["entry_time"]: t for t in t_bar}
live_by_et = {t["entry_time"]: t for t in t_live}
common = [(bar_by_et[et], live_by_et[et]) for et in sorted(set(bar_by_et) & set(live_by_et))]
bar_only = [t for et, t in bar_by_et.items() if et not in live_by_et]
live_only = [t for et, t in live_by_et.items() if et not in bar_by_et]

print(f"  Common: {len(common)} trades")
print(f"  Backtest only: {len(bar_only)} trades")
print(f"  Live only: {len(live_only)} trades")

if common:
    entry_diffs = [abs(tb["entry"] - tl["entry"]) for tb, tl in common]
    exit_diffs = [abs(tb["exit"] - tl["exit"]) for tb, tl in common]
    dir_match = sum(1 for tb, tl in common if tb["dir"] == tl["dir"])
    reason_match = sum(1 for tb, tl in common if tb["exit_reason"] == tl["exit_reason"])
    print(f"  Entry diff: mean={np.mean(entry_diffs):.8f}  max={max(entry_diffs):.8f}")
    print(f"  Exit diff:  mean={np.mean(exit_diffs):.8f}  max={max(exit_diffs):.8f}")
    print(f"  Direction agreement: {dir_match}/{len(common)} = {100*dir_match/len(common):.0f}%")
    print(f"  Exit reason agreement: {reason_match}/{len(common)} = {100*reason_match/len(common):.0f}%")
    
    nonzero = [(tb, tl) for tb, tl in common if abs(tb["entry"] - tl["entry"]) > 1e-10 or abs(tb["exit"] - tl["exit"]) > 1e-10]
    if nonzero:
        print(f"\n  Trades with any diff > 1e-10 ({len(nonzero)}):")
        for tb, tl in nonzero[:5]:
            ed = tb["entry"] - tl["entry"]
            xd = tb["exit"] - tl["exit"]
            print(f"    ET={tl['entry_time']}: entry_diff={ed:.10f} exit_diff={xd:.10f}")

print(f"\n{'=' * 70}")
match_pct = 100 * len(common) / max(len(t_bar), len(t_live)) if max(len(t_bar), len(t_live)) > 0 else 0
avg_ed = np.mean(entry_diffs) if common else 1.0
print(f"VERDICT: {'PASS' if match_pct > 99.0 and avg_ed < 1e-8 else 'FAIL'}")
print(f"  {len(common)}/{max(len(t_bar), len(t_live))} trades matched ({match_pct:.1f}%)")
if bar_only:
    print(f"  {len(bar_only)} backtest-only trades (1-position-per-pair limit)")
if live_only:
    print(f"  {len(live_only)} live-only trades (unexpected)")
print(f"{'=' * 70}")
