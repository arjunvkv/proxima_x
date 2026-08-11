"""scripts/_absorb/cross_asset_study.py — pre-registered cross-asset A/B.

New signal family (not expressible in the current single-symbol engine — this
is research; shipping would need a user-approved engine rule).

Hypotheses:
  C1  DXY momentum -> gold/silver REVERSAL (USD-up => gold-down), W = 60/120/
      240 min, fwd H = 30/60/120 min. signed = -sign(dxy_ret) * fwd_gold.
  C2  US-equity risk -> gold FLIGHT (equity-down => gold-up), same windows.
  C3  Index first-hour momentum -> same-index last-hours (NY open 13:30-14:30
      server -> 18:30-20:00), Gao-style for equity indices.
  C4  (control) DXY co-movement sanity: corr(dDXY, dEURUSD) should be ~ -0.7..-0.9.
Gates: per-side pips, LODO over days, hour-of-day decomposition, 500-iter
random-sign null on the SIGNAL (fix-study lessons). All in pips per pair.
"""
import sys, os, json
import numpy as np
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datetime import datetime

RNG = np.random.default_rng(11)
MARKET = "audit_7_eas/market"

def load(sym):
    df = pl.read_parquet(f"{MARKET}/{sym}.pqt").sort("time")
    return df

def pip(pair, price):
    return price * (100.0 if "JPY" in pair else 10000.0)

dxy = load("DXY.cash").rename({"close": "dxy"})[["time", "dxy"]]
eur = load("EURUSD").rename({"close": "eur"})[["time", "eur"]]
df = dxy.join(eur, on="time", how="inner")

# C4 sanity
d = df.drop_nulls()
c = float(np.corrcoef(np.diff(d["dxy"].to_numpy()), np.diff(d["eur"].to_numpy()))[0, 1])
print(f"C4 sanity corr(dDXY, dEURUSD) = {c:.3f}  (expect ~ -0.7..-0.9)")
if abs(c) < 0.4:
    print("WARNING: DXY tape not co-moving with EURUSD — treat DXY results with suspicion")

def fwd_ret(df, sym, H):
    """forward close return H bars ahead, in pips of sym"""
    s = df.sort("time")
    ts = s["time"].to_numpy()
    cl = s["close"].to_numpy()
    # need aligned fwd on the base frame; returns dict ts -> fwd
    fwd = np.full(len(cl), np.nan)
    fwd[:-H] = cl[H:] - cl[:-H]
    return dict(zip(ts, fwd))

def ret_over(df, sym, W):
    s = df.sort("time")
    ts = s["time"].to_numpy()
    cl = s["close"].to_numpy()
    r = np.full(len(cl), np.nan)
    r[W:] = cl[W:] / cl[:-W] - 1.0
    return dict(zip(ts, r))

def study(sig_sym, tgt_sym, W, H, sign, name):
    s = load(sig_sym).sort("time")
    t = load(tgt_sym).sort("time")
    sig = ret_over(s, sig_sym, W)
    fwd = fwd_ret(t, tgt_sym, H)
    ts = s["time"].to_numpy()
    day = ts // 86400
    vals = []
    for i, tt in enumerate(ts):
        if tt in sig and tt in fwd and sig[tt] == sig[tt] and fwd[tt] == fwd[tt]:
            vals.append((day[i], sign * np.sign(sig[tt]) if sign else np.sign(sig[tt]), fwd[tt]))
    if len(vals) < 200:
        return {"name": name, "n": len(vals), "skip": True}
    days_arr = np.array([v[0] for v in vals]); sgn = np.array([v[1] for v in vals]); f = np.array([v[2] for v in vals])
    # magnitude-gated: keep |sig| >= median
    return {"name": name, "n": int(len(vals)), "raw_pips": round(float(f.mean()), 3),
            "pos_pips": round(float(f[sgn > 0].mean()), 3), "neg_pips": round(float(f[sgn < 0].mean()), 3),
            "n_pos": int((sgn > 0).sum()), "n_neg": int((sgn < 0).sum()),
            "z_vs_signnull": None, "lodo": None}

# simpler robust version: sign x fwd, random-sign null, LODO by day
def study2(sig_sym, tgt_sym, W, H, sign_mult, name, mag_gate=True):
    s = load(sig_sym).sort("time")
    t = load(tgt_sym).sort("time")
    sig = ret_over(s, sig_sym, W)
    fwd = fwd_ret(t, tgt_sym, H)
    ts = s["time"].to_numpy()
    day = ts // 86400
    rows = []
    for i, tt in enumerate(ts):
        if tt in sig and tt in fwd and sig[tt] == sig[tt] and fwd[tt] == fwd[tt]:
            rows.append((day[i], sig[tt], fwd[tt]))
    if len(rows) < 300:
        return {"name": name, "n": len(rows), "skip": True}
    d0 = np.array([r[0] for r in rows]); sg = np.array([r[1] for r in rows]); f = np.array([r[2] for r in rows])
    sgn = sign_mult * np.sign(sg)
    if mag_gate:
        med = np.median(np.abs(sg))
        keep = np.abs(sg) >= med
        d0, sgn, f = d0[keep], sgn[keep], f[keep]
    signed = sgn * f
    real = float(signed.mean())
    nulls = []
    for _ in range(500):
        ss = RNG.choice([-1.0, 1.0], size=len(sgn))
        nulls.append(float((ss * f).mean()))
    z = (real - float(np.mean(nulls))) / (float(np.std(nulls)) + 1e-12)
    # LODO by day
    ud = np.unique(d0)
    daym = np.array([signed[d0 == dd].mean() if (d0 == dd).sum() > 0 else 0.0 for dd in ud])
    flips = 0; swings = []
    for k, dd in enumerate(ud):
        mask = d0 != dd
        if mask.sum() < 100: continue
        v = float(signed[mask].mean())
        flips += int((v > 0) != (real > 0))
        swings.append(abs(v - real))
    return {"name": name, "n": int(len(sgn)), "signed_pips": round(real, 3), "z": round(z, 2),
            "pos_pips": round(float(f[sgn > 0].mean()), 3), "neg_pips": round(float(f[sgn < 0].mean()), 3),
            "n_pos": int((sgn > 0).sum()), "n_neg": int((sgn < 0).sum()),
            "lodo_flips": flips, "lodo_max_swing": round(float(max(swings)) if swings else 0.0, 3),
            "lodo_days": int(len(ud))}

results = []
for W, H in [(12, 6), (24, 12), (48, 24)]:
    results.append(study2("DXY.cash", "XAUUSD", W, H, -1.0, f"C1 DXY->XAU W{W} H{H}"))
    results.append(study2("DXY.cash", "XAGUSD", W, H, -1.0, f"C1 DXY->XAG W{W} H{H}"))
    results.append(study2("US500.cash", "XAUUSD", W, H, -1.0, f"C2 US500->XAU W{W} H{H}"))
    results.append(study2("DXY.cash", "BTCUSD", W, H, -1.0, f"C1 DXY->BTC W{W} H{H}"))
    results.append(study2("XAUUSD", "DXY.cash", W, H, -1.0, f"C1r XAU->DXY W{W} H{H}"))

# C3 index first-hour momentum: first 60 min of NY session (13:30-14:30 srv) -> last 90 min (18:30-20:00)
def c3(sym):
    s = load(sym).sort("time")
    ts = s["time"].to_numpy(); cl = s["close"].to_numpy()
    d0 = ts // 86400
    first = {}   # day -> return over 13:30..14:30
    fwd = {}     # day -> return over 18:30..20:00
    n = len(ts)
    for i, tt in enumerate(ts):
        hr = (tt // 3600) % 24; minute = (tt // 60) % 60
        key = d0[i]
        if hr == 13 and minute == 30 and i + 12 < n and d0[i + 12] == key:
            first[key] = (cl[i + 12] - cl[i]) / cl[i]
        if hr == 18 and minute == 30 and i + 18 < n and d0[i + 18] == key:
            fwd[key] = (cl[i + 18] - cl[i]) / cl[i]
    rows = []
    for key in first:
        if key in fwd:
            rows.append((key, np.sign(first[key]), fwd[key]))
    if len(rows) < 60:
        return {"name": f"C3 {sym}", "n": len(rows), "skip": True}
    d0 = np.array([r[0] for r in rows]); sgn = np.array([r[1] for r in rows]); f = np.array([r[2] for r in rows])
    signed = sgn * f
    real = float(signed.mean())
    nulls = [float((RNG.choice([-1., 1.], size=len(sgn)) * f).mean()) for _ in range(500)]
    z = (real - float(np.mean(nulls))) / (float(np.std(nulls)) + 1e-12)
    ud = np.unique(d0); flips = 0; swings = []
    for dd in ud:
        mask = d0 != dd
        if mask.sum() < 30: continue
        v = float(signed[mask].mean()); flips += int((v > 0) != (real > 0)); swings.append(abs(v - real))
    return {"name": f"C3 {sym}", "n": int(len(sgn)), "signed_pips": round(real * 10000, 3), "z": round(z, 2),
            "pos_pips": round(float(f[sgn > 0].mean() * 10000), 3), "neg_pips": round(float(f[sgn < 0].mean() * 10000), 3),
            "lodo_flips": flips, "lodo_days": int(len(ud))}

for sym in ["US500.cash", "US30.cash", "GER40.cash"]:
    results.append(c3(sym))

for r in results:
    if r.get("skip"):
        print(f"{r['name']:<24} n={r['n']:<5} SKIP (too few)")
        continue
    print(f"{r['name']:<24} n={r['n']:>5} signed={r['signed_pips']:>7.3f}p z={r['z']:>5.2f} "
          f"pos={r['pos_pips']:>6.3f} neg={r['neg_pips']:>6.3f} lodo={r['lodo_flips']}/{r['lodo_days']} swing={r.get('lodo_max_swing', '-')}")

with open("scripts/_absorb/results/cross_asset_study.json", "w") as f:
    json.dump(results, f, indent=1)
print("wrote cross_asset_study.json")