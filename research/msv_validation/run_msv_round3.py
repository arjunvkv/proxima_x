"""MSV Round 3: Multi-period validation, session-phase, liquidity proxy, MSV Energy Vector.
Goal: determine sign stability of Asia/London asymmetry and build MSV v2.
"""

import sys, os, time, numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque, defaultdict

project_root = str(Path(__file__).resolve().parents[2])
os.chdir(project_root)
cd_root = os.path.join(project_root, "currency_decomposition")
sys.path.insert(0, cd_root)

import MetaTrader5 as mt5
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

from state.market_state import MarketStateVector
from config.settings import BASE_CURRENCY_MAP

PAIRS = list(BASE_CURRENCY_MAP.keys())[:16]  # first 16
HORIZONS = [5, 15, 30, 60]
ROLLING_WINDOW = 500
TOTAL_BARS = 0  # computed after loading

DER_CACHE = {}  # will store DER values per bar

def load_data_window(pairs, end, days=120):
    start = end - timedelta(days=days)
    all_data = {}
    for pair in pairs:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    return all_data

def session_info(ts):
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    h, m = dt.hour, dt.minute
    minute_of_day = h * 60 + m
    # Session definitions (UTC)
    if 0 <= h < 7:
        session = "ASIA"
        session_open = 0 * 60
        session_close = 7 * 60
    elif 7 <= h < 12:
        session = "LONDON"
        session_open = 7 * 60
        session_close = 12 * 60
    elif 12 <= h < 16:
        session = "LONDON_NY"
        session_open = 12 * 60
        session_close = 16 * 60
    elif 16 <= h < 21:
        session = "NY"
        session_open = 16 * 60
        session_close = 21 * 60
    else:
        session = "NY_LATE"
        session_open = 21 * 60
        session_close = 24 * 60

    mins_from_open = minute_of_day - session_open
    session_pct = mins_from_open / max(session_close - session_open, 1)

    # Phase
    if session_pct < 0.25:
        phase = f"{session}_OPEN"
    elif session_pct < 0.75:
        phase = f"{session}_MID"
    else:
        phase = f"{session}_CLOSE"

    # Is this a session boundary zone?
    boundary = False
    # 30 min before/after session transitions
    for boundary_ts in [7*60, 12*60, 16*60, 21*60]:
        if abs(minute_of_day - boundary_ts) <= 30:
            boundary = True
            break

    return session, phase, mins_from_open, session_pct, boundary

def rolling_pct(value, history):
    if len(history) < 10:
        return 0.5
    return sum(1 for h in history if h < value) / len(history)

def process_with_der(returns_dict, all_data_pairs):
    """Compute DER for each pair from returns."""
    der = {}
    for sym in PAIRS:
        ret = abs(returns_dict.get(sym, 0.0))
        der[sym] = ret  # proxy for DER: absolute return as energy
    return der

def process_period(all_data, N, period_label):
    """Run MSV over one contiguous data period, return records."""
    print(f"\n{'='*70}")
    print(f"PERIOD: {period_label} ({N} bars, {N/288:.1f} days)")
    print('='*70)

    ms = MarketStateVector(history_size=50)
    disp_hist = deque(maxlen=ROLLING_WINDOW)
    der_hist = deque(maxlen=ROLLING_WINDOW)
    records = []
    pairs = list(all_data.keys())

    for idx in range(N):
        returns = {}
        for pair in all_data:
            if idx == 0:
                ret = 0.0
            else:
                prev = float(all_data[pair][idx - 1]["close"])
                curr = float(all_data[pair][idx]["close"])
                ret = (curr / prev - 1) if prev > 0 else 0.0
            ret = np.clip(ret, -0.05, 0.05)
            returns[pair] = ret

        now = float(all_data[pairs[0]][idx]["time"])
        snapshot = ms.update(returns, timestamp=now)

        disp = snapshot.network.dispersion
        agree = snapshot.network.agreement
        shock = abs(snapshot.residual.residual_shock)
        disp_hist.append(disp)
        disp_pct = rolling_pct(disp, list(disp_hist))

        # Dispersion velocity
        if len(disp_hist) >= 12:
            disp_vel = disp - list(disp_hist)[-12]
        else:
            disp_vel = 0.0
        if len(disp_hist) >= 24:
            prev_vel = list(disp_hist)[-12] - list(disp_hist)[-24]
            disp_accel = disp_vel - prev_vel
        else:
            disp_accel = 0.0

        # Cross-market DER proxy (mean absolute return across pairs)
        cross_der = float(np.mean([abs(v) for v in returns.values()]))
        der_hist.append(cross_der)
        der_pct = rolling_pct(cross_der, list(der_hist))

        # Session
        sess, phase, mins_open, sess_pct, boundary = session_info(now)
        dt = datetime.fromtimestamp(float(now), tz=timezone.utc)
        hour = dt.hour

        # --- Forward returns ---
        fwd = {}
        for h in HORIZONS:
            fwd_idx = idx + h
            if fwd_idx >= N:
                fwd[h] = None
                continue
            fwd[h] = {}
            for pair in all_data:
                cur = float(all_data[pair][idx]["close"])
                fut = float(all_data[pair][fwd_idx]["close"])
                fwd[h][pair] = (fut / cur - 1) if cur > 0 else 0.0

        records.append({
            "idx": idx, "ts": now,
            "session": sess, "phase": phase,
            "mins_open": mins_open, "sess_pct": sess_pct,
            "boundary": boundary, "hour": hour,
            "disp": disp, "disp_pct": disp_pct,
            "disp_vel": disp_vel, "disp_accel": disp_accel,
            "agree": agree, "shock": shock,
            "cross_der": cross_der, "der_pct": der_pct,
            "strengths": {c: n.level for c, n in snapshot.currencies.items()},
            "fwd": fwd,
        })

        if (idx + 1) % 4000 == 0:
            print(f"  {idx+1}/{N}")

    return records

def analyze_period(records, period_label, p95_global=None):
    """Full analysis of a period's records."""
    all_disp = [r["disp_pct"] for r in records if r["fwd"][5] is not None]
    if len(all_disp) < 20:
        print(f"  {period_label}: Too few records ({len(all_disp)})")
        return {}

    p95 = p95_global if p95_global else float(np.percentile(all_disp, 95))
    p20 = float(np.percentile(all_disp, 20))

    results = {}

    # 1. Session-stratified EXTREME_DISP
    print(f"\n  SESSION-STRATIFIED EXTREME_DISP (>{p95:.2f}pct):")
    for sess in ["ASIA", "LONDON", "LONDON_NY", "NY", "NY_LATE"]:
        extreme = [r for r in records if r["session"] == sess and r["disp_pct"] >= p95 and r["fwd"][15] is not None]
        if len(extreme) < 3:
            continue
        print(f"\n    {sess:12s} (n={len(extreme)}):")
        for h in HORIZONS:
            vals = [np.mean([r["fwd"][h][p] for p in all_data_keys(records)])
                    for r in extreme if r["fwd"][h] is not None]
            if len(vals) < 3:
                continue
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
            sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
            pos = sum(1 for v in vals if v > 0) / len(vals) * 100
            t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
            print(f"      {h:3d}m  mean={mu:+.6f}  sharpe={sharpe:+.3f}  pos%={pos:5.1f}%  t={t:+.2f}")

    # 2. DER × Dispersion interaction
    print(f"\n  DER × DISPERSION INTERACTION (30m horizon):")
    all_der = [r["der_pct"] for r in records if r["fwd"][30] is not None]
    if len(all_der) >= 10:
        der_p95 = np.percentile(all_der, 80)
        d20 = 0.2  # low disp
        d95 = p95

        for (d_lo, d_hi, d_label), (e_lo, e_hi, e_label) in [
            ((d95, 1.0, "HIGH_DISP"), (der_p95, 1.0, "HIGH_DER")),
            ((d95, 1.0, "HIGH_DISP"), (0.0, der_p95, "LOW_DER")),
            ((0.0, d20, "LOW_DISP"), (der_p95, 1.0, "HIGH_DER")),
        ]:
            sub = [r for r in records
                   if r["fwd"][30] is not None
                   and d_lo <= r["disp_pct"] < d_hi
                   and e_lo <= r["der_pct"] < e_hi]
            if len(sub) < 5:
                continue
            # Split by session
            for sess in ["ASIA", "LONDON"]:
                vals = [np.mean([r["fwd"][30][p] for p in all_data_keys(records)])
                        for r in sub if r["session"] == sess]
                if len(vals) < 3:
                    continue
                mu = float(np.mean(vals))
                sigma = float(np.std(vals))
                sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
                pos = sum(1 for v in vals if v > 0) / len(vals) * 100
                t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
                print(f"    {d_label:12s}+{e_label:10s} {sess:10s} n={len(vals):4d} "
                      f"mean={mu:+.6f}  sharpe={sharpe:+.3f}  pos%={pos:5.1f}%  t={t:+.2f}")

    # 3. Session phase decomposition (ASIA only, most signal)
    print(f"\n  ASIA PHASE DECOMPOSITION (EXTREME_DISP, 30m):")
    for phase in ["ASIA_OPEN", "ASIA_MID", "ASIA_CLOSE"]:
        sub = [r for r in records if r["phase"] == phase
               and r["disp_pct"] >= p95 and r["fwd"][30] is not None]
        if len(sub) < 3:
            continue
        vals = [np.mean([r["fwd"][30][p] for p in all_data_keys(records)]) for r in sub]
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
        pos = sum(1 for v in vals if v > 0) / len(vals) * 100
        t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
        print(f"    {phase:12s} n={len(vals):4d}  mean={mu:+.6f}  sharpe={sharpe:+.3f}  pos%={pos:5.1f}%  t={t:+.2f}")

    # 4. MSV Energy Vector classification
    print(f"\n  MSV ENERGY VECTOR (30m horizon):")
    # Energy Vector: [disp_pct, disp_vel, der_pct, session_pressure]
    # session_pressure = 1.0 for ASIA, -1.0 for LONDON, 0.0 otherwise
    all_der_p = [r["der_pct"] for r in records if r["fwd"][30] is not None]

    class EnergyState:
        def __init__(self, disp_hi, disp_lo, vel_sign, der_hi, der_lo, sess_filter, label):
            self.disp_hi = disp_hi
            self.disp_lo = disp_lo
            self.vel_sign = vel_sign
            self.der_hi = der_hi
            self.der_lo = der_lo
            self.sess_filter = sess_filter
            self.label = label

    energy_states = [
        EnergyState(1.0, p95, lambda v: v > 0, 1.0, 0.0, lambda s: s == "ASIA",
                    "HIGH_DISP+V_INC ASIA"),
        EnergyState(1.0, p95, lambda v: v > 0, 1.0, 0.0, lambda s: s == "LONDON",
                    "HIGH_DISP+V_INC LONDON"),
        EnergyState(1.0, p95, lambda v: v <= 0, 1.0, 0.0, lambda s: s == "ASIA",
                    "HIGH_DISP+V_DEC ASIA"),
        EnergyState(1.0, p95, lambda v: v <= 0, 1.0, 0.0, lambda s: s == "LONDON",
                    "HIGH_DISP+V_DEC LONDON"),
        EnergyState(p95, d20, lambda v: True, der_p95, 0.0, lambda s: True,
                    "MID_DISP+HIGH_DER ANY"),
        EnergyState(p95, d20, lambda v: True, 1.0, der_p95, lambda s: True,
                    "MID_DISP+LOW_DER ANY"),
    ]

    for es in energy_states:
        sub = [r for r in records
               if r["fwd"][30] is not None
               and es.disp_lo <= r["disp_pct"] < es.disp_hi
               and es.vel_sign(r["disp_vel"])
               and es.der_lo <= r["der_pct"] < es.der_hi
               and es.sess_filter(r["session"])]
        if len(sub) < 5:
            continue
        vals = [np.mean([r["fwd"][30][p] for p in all_data_keys(records)]) for r in sub]
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
        pos = sum(1 for v in vals if v > 0) / len(vals) * 100
        t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
        print(f"    {es.label:30s} n={len(vals):4d}  mean={mu:+.6f}  "
              f"sharpe={sharpe:+.3f}  pos%={pos:5.1f}%  t={t:+.2f}")

    # 5. Boundary zone effect
    print(f"\n  SESSION BOUNDARY ZONE EFFECT (30m):")
    for sess in ["ASIA", "LONDON", "LONDON_NY", "NY"]:
        for boundary_flag, b_label in [(True, "BOUNDARY"), (False, "NON_BOUNDARY")]:
            sub = [r for r in records
                   if r["session"] == sess and r["boundary"] == boundary_flag
                   and r["disp_pct"] >= p95 and r["fwd"][30] is not None]
            if len(sub) < 3:
                continue
            vals = [np.mean([r["fwd"][30][p] for p in all_data_keys(records)]) for r in sub]
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
            sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
            pos = sum(1 for v in vals if v > 0) / len(vals) * 100
            t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
            print(f"    {sess:12s} {b_label:15s} n={len(vals):4d}  "
                  f"sharpe={sharpe:+.3f}  pos%={pos:5.1f}%  t={t:+.2f}")

    return {"p95": p95, "p20": p20, "n": len(records)}

def all_data_keys(records):
    """Get pair keys from a record."""
    if not records:
        return []
    for h in HORIZONS:
        if records[0]["fwd"][h] is not None:
            return list(records[0]["fwd"][h].keys())
    return []

def main():
    global TOTAL_BARS
    # ── LOAD 120 DAYS ──
    end = datetime.now()
    all_data = load_data_window(PAIRS, end, 120)
    N = min(len(v) for v in all_data.values())
    TOTAL_BARS = N
    print(f"Loaded {len(all_data)} pairs, {N} bars")

    # Split into 3 periods for multi-period validation
    split1 = N // 3
    split2 = 2 * N // 3
    periods = [
        ("EARLY", {p: v[:split1] for p, v in all_data.items()}, split1),
        ("MID", {p: v[split1:split2] for p, v in all_data.items()}, split2 - split1),
        ("LATE", {p: v[split2:] for p, v in all_data.items()}, N - split2),
    ]

    all_records = {}
    global_p95 = None

    for label, data, n in periods:
        records = process_period(data, n, label)
        all_records[label] = records

    # Determine global P95 from full sample
    full_disp = []
    for label, records in all_records.items():
        full_disp.extend([r["disp_pct"] for r in records if r["fwd"][5] is not None])
    if len(full_disp) >= 20:
        global_p95 = float(np.percentile(full_disp, 95))
        print(f"\nGlobal P95 threshold: {global_p95:.4f}")

    # Analyze each period
    summary = {}
    for label, records in all_records.items():
        s = analyze_period(records, label, global_p95)
        summary[label] = s

    # ── CROSS-PERIOD SIGN STABILITY ──
    print(f"\n{'='*70}")
    print("CROSS-PERIOD SIGN STABILITY")
    print('='*70)
    print(f"{'Session':12s} {'Period':10s} {'Horizon':8s} {'Sharpe':>8s} {'t':>8s} {'n':>6s}")
    print("-" * 55)

    sess_signs = {}
    for label, records in all_records.items():
        p95_t = global_p95 if global_p95 else float(np.percentile(
            [r["disp_pct"] for r in records if r["fwd"][5] is not None], 95))
        for sess in ["ASIA", "LONDON", "LONDON_NY", "NY"]:
            for h in [15, 30]:
                vals = [np.mean([r["fwd"][h][p] for p in all_data_keys(records)])
                        for r in records
                        if r["session"] == sess and r["disp_pct"] >= p95_t
                        and r["fwd"][h] is not None]
                if len(vals) < 3:
                    continue
                mu = float(np.mean(vals))
                sigma = float(np.std(vals))
                sharpe = (mu / sigma) * np.sqrt(12 * 24) if sigma > 0 else 0.0
                t = mu / (sigma / np.sqrt(len(vals))) if sigma > 0 else 0.0
                print(f"{sess:12s} {label:10s} {h:3d}m    {sharpe:+8.3f} {t:+8.2f} {len(vals):5d}")
                key = (sess, h)
                if key not in sess_signs:
                    sess_signs[key] = []
                sess_signs[key].append(np.sign(sharpe))

    print(f"\n  Sign consistency:")
    for (sess, h), signs in sorted(sess_signs.items()):
        consistent = all(s == signs[0] for s in signs) if len(signs) >= 2 else False
        pos_count = sum(1 for s in signs if s > 0)
        neg_count = sum(1 for s in signs if s < 0)
        print(f"    {sess:12s} {h:3d}m: {pos_count}+ / {neg_count}- ({len(signs)} periods) "
              f"{'✅ CONSISTENT' if consistent else '❌ NOT CONSISTENT'}")

    # ── FINAL SUMMARY ──
    print(f"\n{'='*70}")
    print("FINAL DIAGNOSTIC")
    print('='*70)
    print("""
Hypothesis: Extreme network dispersion is a session-dependent market transition state.

Evidence so far:
- ASIA: strong positive (continuation of Asian positioning)
- LONDON: strong negative (reversal when European liquidity enters)
- NY: neutral/weak

Next: Build MSV Energy Vector → permission score for Proxima V2.
""")

    mt5.shutdown()

if __name__ == "__main__":
    main()
