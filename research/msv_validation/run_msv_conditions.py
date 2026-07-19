"""MSV Regime Robustness — how the signal behaves in different market conditions.

Tests:
  A. Volatility quintiles (low-vol → high-vol regimes)
  B. Dispersion regimes (calm vs stressed)
  C. Intra-Asia timing (early/mid/late)
  D. Day-of-week patterns
  E. Concrete best/worst examples with timestamps
  F. Loss analysis — what do failures look like?
  G. Rolling 5-day Sharpe stability
  H. Noise regime vs signal regime
"""

import sys, os, time, hashlib, random, json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque

project_root = str(Path(__file__).resolve().parents[2])
os.chdir(project_root)
cd_root = os.path.join(project_root, "currency_decomposition")
sys.path.insert(0, cd_root)

import MetaTrader5 as mt5
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

from state.market_state import MarketStateVector
from config.settings import BASE_CURRENCY_MAP

PAIRS = list(BASE_CURRENCY_MAP.keys())[:16]

def load_data():
    end = datetime.now()
    start = end - timedelta(days=120)
    all_data = {}
    for pair in PAIRS:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    return all_data

def tstat(vals):
    if len(vals) < 3: return 0.0
    mu, s = float(np.mean(vals)), float(np.std(vals))
    return mu / (s / np.sqrt(len(vals))) if s > 0 else 0.0

def compute_pct(disp, history, window):
    h = history[-min(window, len(history)):]
    if len(h) < 10: return 0.5
    return sum(1 for x in h if x < disp) / len(h)

def build_records(all_data):
    N = min(len(v) for v in all_data.values())
    print(f"Processing {N} bars...")
    ms = MarketStateVector(history_size=50)
    dh = deque(maxlen=1500)
    records = []
    t0 = time.time()

    # Also build ATR history
    atr_history = deque(maxlen=500)

    for idx in range(N):
        rets = {}
        for p in all_data:
            if idx == 0:
                rets[p] = 0.0
            else:
                c = float(all_data[p][idx]["close"])
                pv = float(all_data[p][idx - 1]["close"])
                rets[p] = max(min((c / pv - 1) if pv > 0 else 0.0, 0.05), -0.05)
        now = float(all_data[list(all_data.keys())[0]][idx]["time"])
        snap = ms.update(rets, timestamp=now)
        dh.append(snap.network.dispersion)

        # Compute ATR (average true range across basket)
        if idx > 0:
            atr_val = 0.0
            for p in all_data:
                hi = float(all_data[p][idx]["high"])
                lo = float(all_data[p][idx]["low"])
                pc = float(all_data[p][idx - 1]["close"])
                tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
                atr_val += tr / float(all_data[p][idx]["close"])
            atr_val /= len(all_data)
            atr_history.append(atr_val)

        pre60 = 0.0
        if idx >= 12:
            for p in all_data:
                cur = float(all_data[p][idx]["close"])
                p60 = float(all_data[p][idx - 12]["close"])
                pre60 += (cur / p60 - 1) if p60 > 0 else 0.0
            pre60 /= len(all_data)

        dt = datetime.fromtimestamp(now, tz=timezone.utc)

        # Forward 30m
        fwd_30 = None
        if idx + 30 < N:
            vals = []
            for p in all_data:
                cur = float(all_data[p][idx]["close"])
                fut = float(all_data[p][idx + 30]["close"])
                vals.append((fut / cur - 1) if cur > 0 else 0.0)
            fwd_30 = float(np.mean(vals))

        records.append({
            "idx": idx, "ts": now, "dt": dt, "hour": dt.hour, "wd": dt.weekday(),
            "year": dt.year, "month": dt.month, "day": dt.day,
            "disp": snap.network.dispersion, "pre60": pre60,
            "fwd_30": fwd_30, "dh_snapshot": list(dh),
            "atr": atr_val if atr_history else 0,
        })

        if (idx + 1) % 5000 == 0:
            print(f"  {idx+1}/{N} ({(time.time()-t0):.0f}s)")

    return records

def get_events(records, pct_thresh=0.95, decl_thresh=-0.0002, window=500):
    events = []
    for r in records:
        if r["fwd_30"] is None: continue
        if r["hour"] >= 7: continue
        dp = compute_pct(r["disp"], r["dh_snapshot"], window)
        if dp < pct_thresh: continue
        if r["pre60"] > decl_thresh: continue
        events.append(r)
    return events

def line():
    print(f"  {'-'*65}")

def print_table(rows, headers):
    """rows is list of lists of strings. headers is list of strings."""
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    header = "  " + "  ".join(f"{h:>{col_widths[i]}}" for i, h in enumerate(headers))
    print(header)
    print("  " + "-" * (sum(col_widths) + 2 * (len(headers) - 1)))
    for row in rows:
        print("  " + "  ".join(f"{r:>{col_widths[i]}}" for i, r in enumerate(row)))
    print()

def main():
    all_data = load_data()
    records = build_records(all_data)
    events = get_events(records)
    print(f"\nMSV events: {len(events)}")

    # ──────────────────────────────────────────────────────────
    # REGIME DECOMPOSITION
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("MARKET CONDITION ANALYSIS")
    print("=" * 70)

    # Compute ATR percentiles for regime binning
    atr_vals = sorted([r["atr"] for r in records if r["atr"] > 0])
    atr_p20 = atr_vals[len(atr_vals)//5] if atr_vals else 0
    atr_p40 = atr_vals[2*len(atr_vals)//5] if atr_vals else 0
    atr_p60 = atr_vals[3*len(atr_vals)//5] if atr_vals else 0
    atr_p80 = atr_vals[4*len(atr_vals)//5] if atr_vals else 0

    print(f"\n1. VOLATILITY REGIMES (ATR quintiles)")
    print(f"   P20={atr_p20:.6f}  P40={atr_p40:.6f}  P60={atr_p60:.6f}  P80={atr_p80:.6f}")

    regimes = [
        ("Calm (0-20pct)", lambda r: r["atr"] <= atr_p20),
        ("Low (20-40pct)", lambda r: atr_p20 < r["atr"] <= atr_p40),
        ("Mid (40-60pct)", lambda r: atr_p40 < r["atr"] <= atr_p60),
        ("High (60-80pct)", lambda r: atr_p60 < r["atr"] <= atr_p80),
        ("Extreme (80-100pct)", lambda r: r["atr"] > atr_p80),
    ]

    print(f"\n   {'Regime':>20s} {'Events':>7s} {'Mean(bp)':>10s} {'Pos%':>6s} {'t':>8s} {'Freq':>8s}")
    line()
    for label, cond in regimes:
        evts = [e for e in events if cond(e)]
        if len(evts) < 2: continue
        vals = [e["fwd_30"] for e in evts]
        t = tstat(vals)
        mu = float(np.mean(vals))
        pos = sum(1 for v in vals if v > 0) / len(vals) * 100
        freq = len(evts) / len([r for r in records if cond(r) and r["hour"] < 7]) * 100 if len([r for r in records if cond(r) and r["hour"] < 7]) > 0 else 0
        print(f"   {label:>20s}  {len(evts):5d}  {mu*10000:>+9.2f}  {pos:>5.1f}%  {t:>+7.2f}  {freq:>7.2f}%")

    # ──────────────────────────────────────────────────────────
    # 2. DISPERSION REGIMES
    # ──────────────────────────────────────────────────────────
    print(f"\n2. DISPERSION INTENSITY REGIMES")
    disp_vals = sorted([r["disp"] for r in events])
    d_med = disp_vals[len(disp_vals)//2] if disp_vals else 0
    d_high = disp_vals[3*len(disp_vals)//4] if disp_vals else 0

    d_regimes = [
        ("Low dispersion", lambda e: e["disp"] <= d_med),
        ("Medium dispersion", lambda e: d_med < e["disp"] <= d_high),
        ("High dispersion", lambda e: e["disp"] > d_high),
    ]
    print(f"   {'Regime':>20s} {'Events':>7s} {'Mean(bp)':>10s} {'Pos%':>6s} {'t':>8s}")
    line()
    for label, cond in d_regimes:
        evts = [e for e in events if cond(e)]
        if len(evts) < 2: continue
        vals = [e["fwd_30"] for e in evts]
        mu = float(np.mean(vals))
        t = tstat(vals)
        pos = sum(1 for v in vals if v > 0) / len(vals) * 100
        print(f"   {label:>20s}  {len(evts):5d}  {mu*10000:>+9.2f}  {pos:>5.1f}%  {t:>+7.2f}")

    # ──────────────────────────────────────────────────────────
    # 3. DECLINE INTENSITY REGIMES (pre60 magnitude)
    # ──────────────────────────────────────────────────────────
    print(f"\n3. PRE-EVENT DECLINE MAGNITUDE")
    pre_vals = sorted([e["pre60"] for e in events])
    p_med = pre_vals[len(pre_vals)//2]
    p_big = pre_vals[len(pre_vals)//4]  # 25th percentile = most negative

    p_regimes = [
        ("Moderate decline", lambda e: e["pre60"] > p_med),
        ("Significant decline", lambda e: p_big < e["pre60"] <= p_med),
        ("Severe decline", lambda e: e["pre60"] <= p_big),
    ]
    print(f"   {'Regime':>20s} {'Events':>7s} {'Mean(bp)':>10s} {'Pos%':>6s} {'t':>8s}")
    line()
    for label, cond in p_regimes:
        evts = [e for e in events if cond(e)]
        if len(evts) < 2: continue
        vals = [e["fwd_30"] for e in evts]
        mu = float(np.mean(vals))
        t = tstat(vals)
        pos = sum(1 for v in vals if v > 0) / len(vals) * 100
        print(f"   {label:>20s}  {len(evts):5d}  {mu*10000:>+9.2f}  {pos:>5.1f}%  {t:>+7.2f}")

    # ──────────────────────────────────────────────────────────
    # 4. INTRA-ASIA TIMING
    # ──────────────────────────────────────────────────────────
    print(f"\n4. INTRA-ASIA TIMING (hour of day)")
    hours = sorted(set(e["hour"] for e in events))
    print(f"   {'Hour':>6s} {'Events':>7s} {'Mean(bp)':>10s} {'Pos%':>6s} {'t':>8s}")
    line()
    for h in hours:
        evts = [e for e in events if e["hour"] == h]
        if len(evts) < 2: continue
        vals = [e["fwd_30"] for e in evts]
        mu = float(np.mean(vals))
        t = tstat(vals)
        pos = sum(1 for v in vals if v > 0) / len(vals) * 100
        print(f"   {h:2d}:00    {len(evts):5d}  {mu*10000:>+9.2f}  {pos:>5.1f}%  {t:>+7.2f}")

    # ──────────────────────────────────────────────────────────
    # 5. DAY OF WEEK
    # ──────────────────────────────────────────────────────────
    print(f"\n5. DAY OF WEEK")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"   {'Day':>6s} {'Events':>7s} {'Mean(bp)':>10s} {'Pos%':>6s} {'t':>8s}")
    line()
    for wd in range(7):
        evts = [e for e in events if e["wd"] == wd]
        if len(evts) < 2: continue
        vals = [e["fwd_30"] for e in evts]
        mu = float(np.mean(vals))
        t = tstat(vals)
        pos = sum(1 for v in vals if v > 0) / len(vals) * 100
        print(f"   {days[wd]:>6s}  {len(evts):5d}  {mu*10000:>+9.2f}  {pos:>5.1f}%  {t:>+7.2f}")

    # ──────────────────────────────────────────────────────────
    # 6. CONCRETE BEST/WORST EXAMPLES
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CONCRETE EXAMPLES")
    print("=" * 70)

    by_return = sorted(events, key=lambda e: e["fwd_30"])

    print(f"\n  TOP 5 BEST TRADES (highest forward return):")
    print(f"  {'Date':>20s} {'Time':>8s} {'Disp':>10s} {'Pre60':>10s} {'Fwd30':>10s}")
    line()
    for e in by_return[-5:]:
        print(f"  {e['dt'].strftime('%Y-%m-%d'):>20s} {e['dt'].strftime('%H:%M'):>8s}  {e['disp']:>10.6f}  {e['pre60']*10000:>+9.2f}bp  {e['fwd_30']*10000:>+9.2f}bp")

    print(f"\n  BOTTOM 5 WORST TRADES (lowest forward return):")
    print(f"  {'Date':>20s} {'Time':>8s} {'Disp':>10s} {'Pre60':>10s} {'Fwd30':>10s}")
    line()
    for e in by_return[:5]:
        print(f"  {e['dt'].strftime('%Y-%m-%d'):>20s} {e['dt'].strftime('%H:%M'):>8s}  {e['disp']:>10.6f}  {e['pre60']*10000:>+9.2f}bp  {e['fwd_30']*10000:>+9.2f}bp")

    # ──────────────────────────────────────────────────────────
    # 7. LOSS ANALYSIS — what do losing trades look like?
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("LOSS ANALYSIS — what do failures look like?")
    print("=" * 70)

    winners = [e for e in events if e["fwd_30"] > 0]
    losers = [e for e in events if e["fwd_30"] <= 0]

    print(f"\n  {'Metric':>25s} {'Winners':>12s} {'Losers':>12s} {'Diff':>12s}")
    line()

    for metric, fn in [
        ("Count", lambda e: len(e)),
        ("Mean disp", lambda e: float(np.mean([x["disp"] for x in e]))*10000),
        ("Mean pre60 (bp)", lambda e: float(np.mean([x["pre60"] for x in e]))*10000),
        ("Mean ATR", lambda e: float(np.mean([x["atr"] for x in e]))*10000),
        ("Hour", lambda e: float(np.mean([x["hour"] for x in e]))),
    ]:
        wv = fn(winners)
        lv = fn(losers)
        diff = wv - lv
        print(f"  {metric:>25s}  {wv:>12.2f}  {lv:>12.2f}  {diff:>+12.2f}")

    # Dispersion percentile comparison
    print(f"\n  Dispersion percentile distribution during events:")
    dp_w = []
    dp_l = []
    for e in winners:
        dp_w.append(compute_pct(e["disp"], e["dh_snapshot"], 500))
    for e in losers:
        dp_l.append(compute_pct(e["disp"], e["dh_snapshot"], 500))
    if dp_w:
        print(f"  Winners median disp%:  {np.median(dp_w):.4f}")
    if dp_l:
        print(f"  Losers  median disp%:  {np.median(dp_l):.4f}")

    # What % of losers had rising market before?
    losers_rising = sum(1 for e in losers if e["pre60"] > 0)
    print(f"  Losers with pre60>0:  {losers_rising}/{len(losers)} ({(losers_rising/len(losers)*100) if losers else 0:.0f}%)")

    # ──────────────────────────────────────────────────────────
    # 8. ROLLING 5-DAY SHARPE
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ROLLING STABILITY — event density over time")
    print("=" * 70)

    events_by_date = {}
    for e in events:
        d = e["dt"].strftime("%Y-%m-%d")
        if d not in events_by_date:
            events_by_date[d] = []
        events_by_date[d].append(e["fwd_30"])

    dates = sorted(events_by_date.keys())
    print(f"\n  Event distribution across {len(dates)} trading days:")
    print(f"  {'Metric':>30s} {'Value':>12s}")
    line()
    daily_counts = [len(events_by_date[d]) for d in dates]
    daily_means = [float(np.mean(events_by_date[d]))*10000 for d in dates]
    print(f"  {'Days with events':>30s}  {len(dates):>12d}")
    print(f"  {'Events/day (mean)':>30s}  {np.mean(daily_counts):>12.2f}")
    print(f"  {'Events/day (max)':>30s}  {max(daily_counts):>12d}")
    print(f"  {'Daily mean (avg)':>30s}  {np.mean(daily_means):>+12.2f}bp")
    print(f"  {'Days with mean>0':>30s}  {sum(1 for m in daily_means if m > 0):>12d}/{len(daily_means)}")

    # Weekly
    weekly_data = {}
    for e in events:
        wk = e["dt"].strftime("%Y-W%W")
        if wk not in weekly_data:
            weekly_data[wk] = []
        weekly_data[wk].append(e["fwd_30"])

    weeks = sorted(weekly_data.keys())
    print(f"\n  Weekly performance:")
    print(f"  {'Week':>12s} {'Events':>7s} {'Mean(bp)':>10s} {'Pos%':>6s}")
    line()
    for wk in weeks:
        vals = weekly_data[wk]
        mu = float(np.mean(vals))
        pos = sum(1 for v in vals if v > 0) / len(vals) * 100
        print(f"  {wk:>12s}  {len(vals):5d}  {mu*10000:>+9.2f}  {pos:>5.1f}%")

    all_weekly_means = [float(np.mean(weekly_data[wk]))*10000 for wk in weeks]
    print(f"\n  Total weeks: {len(weeks)}")
    print(f"  Weeks with mean>0: {sum(1 for m in all_weekly_means if m > 0)}/{len(all_weekly_means)} ({sum(1 for m in all_weekly_means if m > 0)/len(all_weekly_means)*100:.0f}%)")

    # ──────────────────────────────────────────────────────────
    # 9. FRIDAY vs OTHER DAYS (weekend effect)
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("WEEKEND / LIQUIDITY EDGE CASES")
    print("=" * 70)

    for label, day_list in [("Monday", [0]), ("Tuesday-Thursday", [1,2,3]), ("Friday", [4])]:
        evts = [e for e in events if e["wd"] in day_list]
        if len(evts) < 2: continue
        vals = [e["fwd_30"] for e in evts]
        t = tstat(vals)
        mu = float(np.mean(vals))
        pos = sum(1 for v in vals if v > 0) / len(vals) * 100
        print(f"  {label:>20s}: n={len(evts):3d}  mean={mu*10000:>+8.2f}bp  pos%={pos:5.1f}%  t={t:>+7.2f}")

    # ──────────────────────────────────────────────────────────
    # 10. NOISE REGIME — what if we look at events that FIRED but dispersion was low?
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("NOISE vs SIGNAL — what if we remove dispersion filter?")
    print("=" * 70)

    print(f"\n  All ASIA bars with decline only (NO dispersion filter):")
    decl_only_vals = [r["fwd_30"] for r in records if r["hour"] < 7 and r["fwd_30"] is not None and r["pre60"] < -0.0002]
    t_dec = tstat(decl_only_vals)
    mu_dec = float(np.mean(decl_only_vals)) if decl_only_vals else 0
    pos_dec = sum(1 for v in decl_only_vals if v > 0) / len(decl_only_vals) * 100 if decl_only_vals else 0
    print(f"    n={len(decl_only_vals):4d}  mean={mu_dec*10000:>+8.2f}bp  pos%={pos_dec:5.1f}%  t={t_dec:>+7.2f}")

    print(f"\n  All ASIA bars with dispersion filter only (NO decline):")
    disp_only_vals = []
    for r in records:
        if r["hour"] >= 7 or r["fwd_30"] is None: continue
        dp = compute_pct(r["disp"], r["dh_snapshot"], 500)
        if dp >= 0.95:
            disp_only_vals.append(r["fwd_30"])
    t_disp = tstat(disp_only_vals)
    mu_disp = float(np.mean(disp_only_vals)) if disp_only_vals else 0
    pos_disp = sum(1 for v in disp_only_vals if v > 0) / len(disp_only_vals) * 100 if disp_only_vals else 0
    print(f"    n={len(disp_only_vals):4d}  mean={mu_disp*10000:>+8.2f}bp  pos%={pos_disp:5.1f}%  t={t_disp:>+7.2f}")

    print(f"\n  ASIA — NO filters (raw):")
    raw_vals = [r["fwd_30"] for r in records if r["hour"] < 7 and r["fwd_30"] is not None]
    t_raw = tstat(raw_vals)
    mu_raw = float(np.mean(raw_vals)) if raw_vals else 0
    pos_raw = sum(1 for v in raw_vals if v > 0) / len(raw_vals) * 100 if raw_vals else 0
    print(f"    n={len(raw_vals):4d}  mean={mu_raw*10000:>+8.2f}bp  pos%={pos_raw:5.1f}%  t={t_raw:>+7.2f}")

    print(f"\n  MSV v1 (both filters):")
    msv_vals = [e["fwd_30"] for e in events]
    t_msv = tstat(msv_vals)
    mu_msv = float(np.mean(msv_vals))
    pos_msv = sum(1 for v in msv_vals if v > 0) / len(msv_vals) * 100
    print(f"    n={len(msv_vals):4d}  mean={mu_msv*10000:>+8.2f}bp  pos%={pos_msv:5.1f}%  t={t_msv:>+7.2f}")

    print(f"\n  SUMMARY — filter contribution:")
    print(f"  {'Filter':>25s} {'n':>6s} {'Mean(bp)':>10s} {'t':>8s} {'vs MSV':>8s}")
    line()
    print(f"  {'No filters (raw ASIA)':>25s}  {len(raw_vals):4d}  {mu_raw*10000:>+9.2f}  {t_raw:>+7.2f}  {t_raw-t_msv:>+7.2f}")
    print(f"  {'Decline only':>25s}  {len(decl_only_vals):4d}  {mu_dec*10000:>+9.2f}  {t_dec:>+7.2f}  {t_dec-t_msv:>+7.2f}")
    print(f"  {'Dispersion only':>25s}  {len(disp_only_vals):4d}  {mu_disp*10000:>+9.2f}  {t_disp:>+7.2f}  {t_disp-t_msv:>+7.2f}")
    print(f"  {'MSV (both)':>25s}  {len(msv_vals):4d}  {mu_msv*10000:>+9.2f}  {t_msv:>+7.2f}  {'-':>8s}")

    mt5.shutdown()

if __name__ == "__main__":
    main()
