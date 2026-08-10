"""Absorption -> price-impact transition probe (brief sections 3-5, 8, 24).

Runs on the REAL FTMO quote tape already archived in data/ticks/ (bid/ask ticks,
zero trade volume - so all "flow" here is QUOTE-DRIVEN PROXY, stated honestly).

Design principles (brief §24): distribution-relative thresholds only, no magic
constants; 3 horizons; rare-state detection from trailing regime distribution;
non-overlapping windows so samples are independent (no auto-correlation).

State definition (fully causal, non-repainting):
  window W (s):  aggregate quote ticks into 1-sec bars; sum signed quote pressure
  P = (ask_up + bid_up) - (ask_dn + bid_dn); mid move dM over W; impact-per-flow
  lambda = |dM| / (|P|+1).
  ABSORPTION state at window endpoint t:
    |P|  >= q90(P over trailing T window of the same symbol)   (unusually strong
                                                               directional pressure)
    AND  lambda <= q25(lambda over trailing T)                 (unusually LOW price
                                                               response per unit flow)
  TRANSITION state: ABSORPTION at t, and by t+W the flow continues same-sign while
  lambda rises above its q75 -> the absorption capacity is degrading.

Hypothesis test (§8): after ABSORPTION at t, forward mid return over H in
{60, 300, 900}s - reversal (Hyp A: sign opposite to P) or continuation (Hyp B:
same sign)? Also the TRANSITION-gated variant (continued same-sign pressure +
impact rise = the "candidate trade" moment).

Everything is computed on information available at time t (trailing windows only).

Usage: unset PYTHONPATH && ./.venv/Scripts/python.exe research/absorption_probe.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

SYM = "EURJPY"
ARCHIVE = Path("data/ticks/EURJPY/2026")
W = int(os.environ.get("PROBE_W", "60"))          # aggregation window, seconds
TRAIL = int(os.environ.get("PROBE_TRAIL", "14400"))  # trailing regime window (4h)
HORIZONS = (60, 300, 900)
P_QUANT = float(os.environ.get("PROBE_PQ", "0.90"))   # pressure threshold quantile
L_QUANT = float(os.environ.get("PROBE_LQ", "0.25"))   # impact-per-flow quantile
MIN_PRESSURE = int(os.environ.get("PROBE_MINP", "10"))


def load_1s(sym: str, archive: Path) -> pl.DataFrame:
    df = pl.read_parquet(archive)
    df = df.filter((pl.col("bid") > 0) & (pl.col("ask") > 0)).sort("time_msc")
    # per-tick directional moves (quote pressure: which side is being hit/improved)
    df = df.with_columns(
        bid_chg=pl.col("bid") - pl.col("bid").shift(1),
        ask_chg=pl.col("ask") - pl.col("ask").shift(1),
    )
    df = df.with_columns(
        ask_up=(pl.col("ask_chg") > 0).cast(pl.Int32),
        ask_dn=(pl.col("ask_chg") < 0).cast(pl.Int32),
        bid_up=(pl.col("bid_chg") > 0).cast(pl.Int32),
        bid_dn=(pl.col("bid_chg") < 0).cast(pl.Int32),
    )
    # 1-second bars
    s = df.group_by("time_sec").agg(
        [
            pl.col("ask_up").sum().alias("ask_up"),
            pl.col("ask_dn").sum().alias("ask_dn"),
            pl.col("bid_up").sum().alias("bid_up"),
            pl.col("bid_dn").sum().alias("bid_dn"),
            pl.col("spread").mean().alias("spread"),
            ((pl.col("bid") + pl.col("ask")) / 2).last().alias("mid_last"),
            pl.len().alias("n_ticks"),
        ]
    ).sort("time_sec")
    s = s.with_columns(
        (# signed quote pressure: demand (up) minus supply (down) pressure
            (pl.col("ask_up") + pl.col("bid_up"))
            - (pl.col("ask_dn") + pl.col("bid_dn"))
        ).alias("pressure"),
        mid_prev=pl.col("mid_last").shift(1),
    )
    s = s.with_columns(dM=pl.col("mid_last") - pl.col("mid_prev"))
    return s


def main() -> None:
    print(f"== absorption probe: {SYM} ==")
    s = load_1s(SYM, ARCHIVE)
    print(f"1-sec bars: {s.height}, span: {s['time_sec'].min()} -> {s['time_sec'].max()} "
          f"({(s['time_sec'].max()-s['time_sec'].min())/86400:.1f} days)")

    t = s["time_sec"].to_numpy()
    P = s["pressure"].to_numpy().astype(float)
    dM = s["dM"].to_numpy()
    mid = s["mid_last"].to_numpy()

    # ---- non-overlapping W-second windows: signed pressure & mid move per window
    n_win = int(np.floor(len(t) / W))
    print(f"non-overlapping {W}s windows: {n_win}")
    Pw = P[: n_win * W].reshape(n_win, W).sum(axis=1)
    dMw = dM[: n_win * W].reshape(n_win, W).sum(axis=1)
    tw = t[: n_win * W].reshape(n_win, W)[:, -1]  # window endpoint ts
    lam = np.abs(dMw) / (np.abs(Pw) + 1.0)

    # ---- trailing distribution of |P| and lambda (window i uses data before i)
    n_trail = TRAIL // W
    print(f"trailing regime window: {n_trail} windows ({TRAIL}s)")
    valid = np.zeros(n_win, bool)
    p90_absP = np.full(n_win, np.nan)
    q25_lam = np.full(n_win, np.nan)
    q75_lam = np.full(n_win, np.nan)
    for i in range(n_win):
        if i < n_trail:
            continue
        a = max(0, i - n_trail)
        hist_absP = np.abs(Pw[a:i])
        hist_lam = lam[a:i]
        hist_absP = hist_absP[~np.isnan(hist_absP)]
        hist_lam = hist_lam[~np.isnan(hist_lam)]
        if len(hist_absP) < 50 or len(hist_lam) < 50:
            continue
        p90_absP[i] = np.quantile(hist_absP, P_QUANT)
        q25_lam[i] = np.quantile(hist_lam, L_QUANT)
        q75_lam[i] = np.quantile(hist_lam, 0.75)
        valid[i] = True

    abs_state = valid & (np.abs(Pw) >= p90_absP) & (np.abs(Pw) >= MIN_PRESSURE) & (lam <= q25_lam)
    print(f"\nABSORPTION states: {int(abs_state.sum())} of {int(valid.sum())} valid windows "
          f"({100*abs_state.sum()/max(1,valid.sum()):.2f}%)")

    # ---- forward returns from window endpoint vs H
    fwd = {}
    idx0 = np.searchsorted(t, tw)  # index of window endpoint bar
    for H in HORIZONS:
        fwd[H] = np.full(n_win, np.nan)
        idx = np.searchsorted(t, tw + H)
        ok = idx < len(mid)
        fwd[H][ok] = mid[idx[ok]] - mid[idx0[ok]]

    print(f"\n{'state':<14}{'H(s)':>5}{'n':>7}{'fwd(pts)':>10}{'t':>7}{'hit%':>7}"
          f"{'hyp':>6}")
    # pts scale (EURJPY point = 0.001, 1 pt = 0.001 quote units)
    pt = 0.001
    for H in HORIZONS:
        f = fwd[H]
        for name, mask in (("abs", abs_state), ("all", valid)):
            m = mask & ~np.isnan(f)
            n = int(m.sum())
            if n < 30:
                print(f"{name:<14}{H:>5}{n:>7}   (insufficient)")
                continue
            signed = f[m] * np.sign(Pw[m])  # + = continuation (Hyp B), - = reversal (Hyp A)
            mean_s = signed.mean() / pt
            se = signed.std(ddof=1) / np.sqrt(n) / pt
            tstat = mean_s / se if se > 0 else 0.0
            hit = 100 * (signed > 0).mean()
            hyp = "B" if tstat > 1.645 else ("A" if tstat < -1.645 else "?")
            print(f"{name:<14}{H:>5}{n:>7}{mean_s:>10.2f}{tstat:>7.2f}{hit:>7.1f}{hyp:>6}")

    # ---- transition-gated: absorption at t, same-sign pressure & impact rise by t+W
    print("\n== transition-gated (absorb@t, same-sign pressure, lambda>=q75@t+W) ==")
    P_next = np.full(n_win, np.nan)
    P_next[:-1] = Pw[1:]
    lam_next = np.full(n_win, np.nan)
    lam_next[:-1] = lam[1:]
    trans = abs_state & (np.sign(P_next) == np.sign(Pw)) & (lam_next >= q75_lam)
    print(f"transition states: {int(trans.sum())}")
    for H in HORIZONS:
        f = fwd[H][trans] if H in fwd else np.full(int(trans.sum()), np.nan)
        m = trans & ~np.isnan(fwd[H])
        n = int(m.sum())
        if n < 30:
            print(f"  H={H:>4}s n={n} (insufficient)")
            continue
        signed = fwd[H][m] * np.sign(Pw[m])
        mean_s = signed.mean() / pt
        se = signed.std(ddof=1) / np.sqrt(n) / pt
        tstat = mean_s / se if se > 0 else 0.0
        hit = 100 * (signed > 0).mean()
        hyp = "B" if tstat > 1.645 else ("A" if tstat < -1.645 else "?")
        print(f"  H={H:>4}s n={n} mean_s={mean_s:8.2f}pts t={tstat:6.2f} hit={hit:5.1f}% {hyp}")

    # ---- cost check: expectancy must beat 1 spread + commission at H-horizon entry
    spread_pts = s["spread"].quantile(0.5) / 1e-3  # median spread in points
    tick_usd = 0.63097  # EURJPY USD per point per 1.0 lot (broker-measured)
    comm = 3.0 * 2      # $3/lot/side
    print(f"\n== cost gate (median spread {spread_pts:.1f} pts = "
          f"${spread_pts*tick_usd:.2f}/lot + ${comm} comm = "
          f"${spread_pts*tick_usd+comm:.2f}/lot round trip) ==")
    for H in HORIZONS:
        m = abs_state & ~np.isnan(fwd[H])
        if int(m.sum()) < 30:
            continue
        signed_pts = (fwd[H][m] * np.sign(Pw[m])).mean() / pt
        net = signed_pts * tick_usd - (spread_pts * tick_usd + comm)
        print(f"  H={H:>4}s abs-states net expectancy ${net:+.2f}/lot "
              f"({signed_pts:+.1f} pts vs {spread_pts:.1f} pt breakeven)")

    # ---- session/hour distribution of absorption states (is it localized?)
    hrs = (tw[abs_state] // 3600) % 24
    c = Counter(hrs)
    top = c.most_common(6)
    print(f"\nhour-of-day distribution of absorption states (top 6): {top}")
    days = Counter(tw[abs_state] // 86400)
    print(f"trading days covered: {len(days)}, states/day: "
          f"{abs_state.sum()/max(1,len(days)):.1f}")


if __name__ == "__main__":
    main()