import zipfile, os, sys, csv, io
from datetime import datetime, timezone, timedelta
from collections import OrderedDict
import numpy as np

DATA_DIR = r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks"
OUT_DIR = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\tick_data"
os.makedirs(OUT_DIR, exist_ok=True)

PAIRS = ["EURJPY", "EURUSD", "GBPJPY"]
MONTHS = ["10", "11", "12"]

ZIP_NAMES = {}
for p in PAIRS:
    ZIP_NAMES[p] = [os.path.join(DATA_DIR, f"{p}_Raw_Spread_2025_{m}.zip") for m in MONTHS]

def parse_timestamp(ts_str):
    ts_str = ts_str.strip('"')
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1]
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    dt = dt.replace(tzinfo=timezone.utc)
    return dt

def parse_tick_row(row):
    if not row or len(row) < 5:
        return None
    try:
        ts = parse_timestamp(row[2])
        bid = float(row[3])
        return ts, bid
    except (ValueError, IndexError):
        return None

def ticks_to_m1(ticks):
    if not ticks:
        return np.array([]), np.array([]), np.array([])
    m1_bars = []
    m1_times = []
    m1_volumes = []
    current_minute = None
    o = h = l = c = v = None
    for ts, bid in ticks:
        minute_key = ts.replace(second=0, microsecond=0)
        if minute_key != current_minute:
            if current_minute is not None:
                m1_bars.append([o, h, l, c])
                m1_times.append(int(current_minute.timestamp()))
                m1_volumes.append(v)
            current_minute = minute_key
            o = h = l = c = bid
            v = 1
        else:
            h = max(h, bid)
            l = min(l, bid)
            c = bid
            v += 1
    if current_minute is not None:
        m1_bars.append([o, h, l, c])
        m1_times.append(int(current_minute.timestamp()))
        m1_volumes.append(v)
    return np.array(m1_bars, dtype=np.float64), np.array(m1_times, dtype=np.int64), np.array(m1_volumes, dtype=np.int64)

def find_gaps(times):
    if len(times) < 2:
        return []
    gaps = []
    expected = 60
    for i in range(1, len(times)):
        diff = times[i] - times[i-1]
        if diff > expected + 5:
            missing = int(diff // 60)
            gaps.append((times[i-1], times[i], missing - 1))
    return gaps

def process_pair(pair):
    all_ticks = []
    skipped = 0
    total_lines = 0
    for zp in ZIP_NAMES[pair]:
        with zipfile.ZipFile(zp, "r") as z:
            for csv_name in z.namelist():
                with z.open(csv_name) as f:
                    text_io = io.TextIOWrapper(f, encoding="utf-8")
                    reader = csv.reader(text_io)
                    for row in reader:
                        total_lines += 1
                        parsed = parse_tick_row(row)
                        if parsed is None:
                            skipped += 1
                            continue
                        all_ticks.append(parsed)
    print(f"  Total lines read: {total_lines}, skipped: {skipped}, valid ticks: {len(all_ticks)}")
    pct = (skipped / total_lines * 100) if total_lines else 0
    if pct > 1:
        print(f"  WARNING: {pct:.2f}% of lines were skipped!")
    all_ticks.sort(key=lambda x: x[0])
    m1_bars, m1_times, m1_vol = ticks_to_m1(all_ticks)
    prefix = pair.lower()
    np.save(os.path.join(OUT_DIR, f"{prefix}_m1_prices.npy"), m1_bars)
    np.save(os.path.join(OUT_DIR, f"{prefix}_m1_times.npy"), m1_times)
    np.save(os.path.join(OUT_DIR, f"{prefix}_m1_volume.npy"), m1_vol)
    gaps = find_gaps(m1_times)
    return m1_bars, m1_times, m1_vol, gaps

def fmt_gap(g):
    t1 = datetime.fromtimestamp(g[0], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    t2 = datetime.fromtimestamp(g[1], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    return f"    {t1} -> {t2}: {g[2]} missing minute(s)"

def fmt_dt(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

print("=" * 70)
print("TICK DATA EXTRACTION — Exness Raw Spread → M1 Bars")
print("=" * 70)

for pair in PAIRS:
    print(f"\n{'─' * 50}")
    print(f"Processing: {pair}")
    bars, times, vol, gaps = process_pair(pair)
    n = len(bars)
    print(f"\n  RESULTS:")
    print(f"  M1 bars: {n:,}")
    if n > 0:
        print(f"  Date range: {fmt_dt(times[0])} → {fmt_dt(times[-1])}")
        total_ticks = int(vol.sum())
        print(f"  Total ticks: {total_ticks:,}")
        print(f"  Avg ticks/bar: {total_ticks / n:.1f}")
        print(f"  Expected bars (approx): {((times[-1] - times[0]) / 60):.0f}")
        print(f"  Coverage: {n / ((times[-1] - times[0]) / 60) * 100:.2f}%")
    if gaps:
        print(f"\n  DATA GAPS ({len(gaps)} gap(s)):")
        for g in gaps:
            print(fmt_gap(g))
    else:
        print(f"\n  No gaps detected.")

print(f"\n{'═' * 70}")
print("FILES WRITTEN:")
for pair in PAIRS:
    prefix = pair.lower()
    for suffix, desc in [("prices.npy", "O,H,L,C"), ("times.npy", "unix_seconds"), ("volume.npy", "tick_count")]:
        fp = os.path.join(OUT_DIR, f"{prefix}_m1_{suffix}")
        if os.path.exists(fp):
            arr = np.load(fp)
            print(f"  {fp}  shape={arr.shape}  dtype={arr.dtype}")
print(f"{'═' * 70}")
