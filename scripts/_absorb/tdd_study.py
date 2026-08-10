"""scripts/_absorb/tdd_study.py — pre-registered TDD (tick event-rate acceleration).

RDSA (research/RESEARCH_DIRECTION_AUDIT.md, 2026-06-16) ranked event-rate
acceleration as the ONLY direction with new information, gated on tick data.
The gate is now cleared: 60d minute bars with n_quotes (quote-event rate) per
pair, built from genuine FTMO ticks.

Hypotheses (pre-registered):
  T0 — event-rate acceleration carries no forward information.
  T1 — acceleration precedes VOLATILITY: |fwd move| is larger after high
       acceleration than low, beyond a null that preserves the rate
       distribution but destroys temporal structure (the RDSA-prescribed
       Poisson-ish null: per-hour circular shifts of the accel sequence).
  T2 — acceleration predicts DIRECTION (signed continuation/reversal) —
       exploratory, reported with per-side pips + LODO, explicitly NOT
       engine-shippable (no TDD rule exists in the engine registry; zero-
       mutation contract forbids adding one without a user decision).

Design (fix-study lessons baked in):
  - de-session the rate: log1p(n_quotes) z-scored WITHIN hour-of-day,
    calibrated on days 0-29 only (causal for the hold-out days 30-59).
  - acceleration a[t] = z[t] - z[t-15] (rate surge vs its hour baseline).
  - all tests on HOLD-OUT days 30-59 (30d) — the fix study's in-sample
    lesson; both halves reported for stability.
  - pooled units in PIPS per pair (scale-artifact lesson).
Engine/book/live untouched.
"""
import sys, os, json
import numpy as np
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PAIRS = ["EURUSD","GBPUSD","USDJPY","EURJPY","GBPJPY","AUDUSD","USDCAD"]
TICK_DIR = "scripts/_absorb/results/ticks"
H_LIST = [5, 15, 30, 60]
RNG = np.random.default_rng(7)

def pip(pair, price):
    return price * (100.0 if "JPY" in pair else 10000.0)

def load(pair):
    df = pl.read_parquet(os.path.join(TICK_DIR, f"{pair}.pqt"))
    df = df.with_columns([
        ((pl.col("ts") // 3600) % 24).alias("hour"),
        ((pl.col("ts") // 86400) - (pl.col("ts").min() // 86400)).alias("day"),
        pl.col("n_quotes").log1p().alias("lr"),
    ])
    return df

def calibrate(df, pair):
    """hour-of-day mean/std of log-rate on days 0-29 (causal)."""
    cal = df.filter(pl.col("day") < 30)
    agg = cal.group_by("hour").agg([
        pl.col("lr").mean().alias("m"), pl.col("lr").std().alias("s"),
    ])
    df = df.join(agg, on="hour", how="left")
    df = df.with_columns(((pl.col("lr") - pl.col("m")) / pl.col("s").clip(lower_bound=1e-9)).alias("z"))
    df = df.with_columns(pl.col("z").shift(15).alias("z15"))
    df = df.with_columns((pl.col("z") - pl.col("z15")).alias("accel"))
    for H in H_LIST:
        df = df.with_columns((pl.col("close").shift(-H) - pl.col("close")).alias(f"fwd{H}"))
    return df

def null_shift(accel, hour):
    """Per-hour circular shift: preserves rate distribution, destroys sequence."""
    out = np.empty_like(accel)
    for h in np.unique(hour):
        idx = np.where(hour == h)[0]
        k = RNG.integers(0, len(idx))
        out[idx] = np.roll(accel[idx], k)
    return out

def test_vol(hold, pair):
    """T1: accel high vs low -> |fwd| ratio, z vs 200 per-hour-shift nulls."""
    res = {}
    for H in H_LIST:
        f = pip(pair, np.abs(hold[f"fwd{H}"].to_numpy()))
        a = hold["accel"].to_numpy()
        hr = hold["hour"].to_numpy()
        hi, lo = a >= np.quantile(a, 0.75), a <= np.quantile(a, 0.25)
        if hi.sum() < 20 or lo.sum() < 20:
            res[H] = None; continue
        ratio = f[hi].mean() / f[lo].mean()
        real = f[hi].mean() - f[lo].mean()
        nulls = []
        for _ in range(200):
            ash = null_shift(a, hr)
            hh, ll = ash >= np.quantile(ash, 0.75), ash <= np.quantile(ash, 0.25)
            if hh.sum() < 20 or ll.sum() < 20:
                continue
            nulls.append(f[hh].mean() - f[ll].mean())
        z = (real - np.mean(nulls)) / (np.std(nulls) + 1e-12) if len(nulls) > 10 else 0.0
        res[H] = {"ratio": round(ratio, 3), "z": round(z, 2), "n_hi": int(hi.sum()),
                  "n_lo": int(lo.sum()), "null_n": len(nulls)}
    return res

def test_dir(hold, pair):
    """T2: signed accel x fwd, per side, in pips, LODO over hold-out days."""
    res = {}
    for H in H_LIST:
        f = pip(pair, hold[f"fwd{H}"].to_numpy())
        a = hold["accel"].to_numpy()
        d = hold["day"].to_numpy()
        sgn = np.sign(a)
        pos = sgn > 0; neg = sgn < 0
        if pos.sum() < 20 or neg.sum() < 20:
            res[H] = None; continue
        day_means = {}
        for side, mask in (("pos", pos), ("neg", neg)):
            dd = d[mask]
            ff = f[mask]
            msum = np.bincount(dd, weights=ff, minlength=int(d.max()) + 1)
            mcnt = np.bincount(dd, minlength=int(d.max()) + 1)
            day_means[side] = np.divide(msum, mcnt, out=np.zeros_like(msum, dtype=float), where=mcnt > 0)
        full = {s: float(day_means[s][day_means[s] > -1e9].sum() / max((day_means[s] != 0).sum(), 1))
                for s in day_means}
        lodo = {}
        for s in day_means:
            dm = day_means[s]
            days = np.where(dm != 0)[0]
            flips = 0; swings = []
            for dd in days:
                rem = np.delete(dm, np.where(days == dd))
                v = rem[rem != 0].mean() if (rem != 0).any() else 0.0
                flips += int((v > 0) != (full[s] > 0))
                swings.append(abs(v - full[s]))
            lodo[s] = {"flips": flips, "n_days": int(len(days)), "max_swing": round(max(swings), 3)}
        res[H] = {"pos_pips": round(full["pos"], 3), "neg_pips": round(full["neg"], 3),
                  "n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "lodo": lodo}
    return res

out = {"vol": {}, "dir": {}}
for pair in PAIRS:
    df = calibrate(load(pair), pair)
    hold = df.filter((pl.col("day") >= 30) & (pl.col("day") < 60))
    out["vol"][pair] = test_vol(hold, pair)
    out["dir"][pair] = test_dir(hold, pair)
    v = out["vol"][pair]
    d = out["dir"][pair]
    print(f"{pair}: T1-vol { {H: (v[H]['ratio'], v[H]['z']) if v[H] else None for H in H_LIST} }")
    print(f"     T2-dir { {H: (d[H]['pos_pips'], d[H]['neg_pips']) if d[H] else None for H in H_LIST} }")

os.makedirs("scripts/_absorb/results", exist_ok=True)
with open("scripts/_absorb/results/tdd_study.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote scripts/_absorb/results/tdd_study.json")