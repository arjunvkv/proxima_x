"""Debug P0 — trace why sweeps are not being found."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import numpy as np
from proxima_honest_backtest.strategies.london_h7.sweep import load_and_align, ALL_PAIRS, _pip_value

raw, pre_align = load_and_align()
print(f"Total aligned rows: {len(pre_align)}")
print(f"Date range: {pre_align[0]['time']} to {pre_align[-1]['time']}")

filters = {"total_07bars": 0, "has_ar": 0, "count_ok": 0, "range_ok": 0, "sw_detected": 0, "ratio_ok": 0, "close_in": 0, "fwd_ok": 0}
asian_ranges = {}

for row_idx, row in enumerate(pre_align):
    ts = row["time"]
    hour = ts.hour
    minute = ts.minute
    today = ts.strftime("%Y-%m-%d")

    if hour == 0 and minute == 0:
        asian_ranges = {}

    # Track Asian range
    if hour < 7:
        for pair in ["EURUSD", "GBPUSD", "USDJPY"]:
            val = row.get(pair)
            if val is None or np.isnan(val):
                continue
            high = row.get(f"{pair}_high", val)
            low = row.get(f"{pair}_low", val)
            ar = asian_ranges.setdefault(pair, {"high": -1e9, "low": 1e9, "date": today, "count": 0})
            if ar["date"] != today:
                ar["high"] = -1e9; ar["low"] = 1e9
                ar["date"] = today; ar["count"] = 0
            ar["high"] = max(ar["high"], high)
            ar["low"] = min(ar["low"], low)
            ar["count"] += 1

    # At 07:00 bar: detect sweep
    if hour == 7 and minute == 0:
        for pair in ["EURUSD", "GBPUSD", "USDJPY"]:
            filters["total_07bars"] += 1
            ar = asian_ranges.get(pair)
            if not ar or ar["date"] != today:
                continue
            filters["has_ar"] += 1
            if ar["count"] < 70:
                continue
            filters["count_ok"] += 1
            pv = _pip_value(pair)
            range_pips = (ar["high"] - ar["low"]) / pv
            if range_pips < 10 or range_pips > 60:
                continue
            filters["range_ok"] += 1

            high_val = row.get(f"{pair}_high")
            low_val = row.get(f"{pair}_low")
            close_val = row.get(pair)
            if any(v is None or np.isnan(v) for v in (high_val, low_val, close_val)):
                continue

            high_sweep = high_val - ar["high"]
            low_sweep = ar["low"] - low_val
            max_sweep = max(high_sweep, low_sweep)
            sweep_pips = max_sweep / pv
            if sweep_pips < 5 or sweep_pips > 30:
                continue
            filters["sw_detected"] += 1

            if max_sweep / max(ar["high"] - ar["low"], 1e-10) < 0.3:
                continue
            filters["ratio_ok"] += 1
            is_high = high_sweep > low_sweep

            # Store for 07:05 check
            sweep_07_entry = {
                "pair": pair, "is_high_sweep": is_high, "sweep_pips": sweep_pips,
                "asian_high": ar["high"], "asian_low": ar["low"],
                "bar_close": close_val, "range_pips": range_pips,
                "row_idx": row_idx, "today": today,
            }

    # At 07:05 bar: check close back inside
    if hour == 7 and minute == 5:
        for pair in ["EURUSD", "GBPUSD", "USDJPY"]:
            close_val = row.get(pair)
            if close_val is None or np.isnan(close_val):
                continue
            ar = asian_ranges.get(pair)
            if not ar or ar["date"] != today:
                continue
            close_in_range = ar["low"] <= close_val <= ar["high"]
            if close_in_range:
                filters["close_in"] += 1

print(f"\nFilter chain (3 pairs: EURUSD, GBPUSD, USDJPY):")
for k, v in filters.items():
    print(f"  {k:>15s}: {v:>6d}")
print(f"\nClose-back-inside rate: {filters['close_in']}/{filters['count_ok']} = {filters['close_in']/max(filters['count_ok'],1)*100:.1f}%")
print(f"Sweep detection rate: {filters['sw_detected']}/{filters['count_ok']} = {filters['sw_detected']/max(filters['count_ok'],1)*100:.1f}%")
