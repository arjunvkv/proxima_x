#!/usr/bin/env python3
"""
Dark Research — All 5 analyses in one script.
numpy vectorization + numba, runs < 15 seconds.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from numba import jit
from scipy import stats as sp_stats
from numpy.lib.stride_tricks import sliding_window_view as swv

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Trading\Agentic_Trading\proxima_x\data\temp\mt5_m1_9day.parquet"
DIR_PATH  = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research"

# ── FX constants ─────────────────────────────────────────────────────────────
CURRENCIES = ["EUR", "USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]
CUR_MAP = {c: i for i, c in enumerate(CURRENCIES)}
PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "NZDUSD", "EURJPY", "GBPJPY"]
PAIR_IDX = {p: i for i, p in enumerate(PAIRS)}
WLS_REG = 0.01
ES_WINDOW = 50

BASE_CURRENCY_MAP = {
    "EURUSD": ("EUR", "USD"), "USDJPY": ("USD", "JPY"), "GBPUSD": ("GBP", "USD"),
    "AUDUSD": ("AUD", "USD"), "NZDUSD": ("NZD", "USD"), "EURJPY": ("EUR", "JPY"),
    "GBPJPY": ("GBP", "JPY"), "EURGBP": ("EUR", "GBP"), "EURAUD": ("EUR", "AUD"),
    "EURNZD": ("EUR", "NZD"), "EURCHF": ("EUR", "CHF"), "GBPAUD": ("GBP", "AUD"),
    "GBPNZD": ("GBP", "NZD"), "GBPCHF": ("GBP", "CHF"), "AUDJPY": ("AUD", "JPY"),
    "NZDJPY": ("NZD", "JPY"), "AUDNZD": ("AUD", "NZD"), "AUDCHF": ("AUD", "CHF"),
    "NZDCAD": ("NZD", "CAD"), "AUDCAD": ("AUD", "CAD"), "CADJPY": ("CAD", "JPY"),
    "CHFJPY": ("CHF", "JPY"), "USDCAD": ("USD", "CAD"), "USDCHF": ("USD", "CHF"),
    "GBPCAD": ("GBP", "CAD"), "EURCAD": ("EUR", "CAD"), "CADCHF": ("CAD", "CHF"),
    "NZDCHF": ("NZD", "CHF"),
}

# ── WLS pre-computation ──────────────────────────────────────────────────────
def build_wls():
    n_p, n_c = len(PAIRS), len(CURRENCIES)
    X = np.zeros((n_p, n_c))
    for i, p in enumerate(PAIRS):
        b, q = BASE_CURRENCY_MAP[p]
        X[i, CUR_MAP[b]] = 1.0
        X[i, CUR_MAP[q]] = -1.0
    A = X.T @ X + WLS_REG * np.eye(n_c)
    pinv = np.linalg.solve(A, X.T)
    return pinv, X

PINV, XMAT = build_wls()

# ── Data loading ─────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_parquet(DATA_PATH)
    times = pd.DatetimeIndex(df["time"].unique()).sort_values()
    close = df.pivot_table(index="time", columns="pair", values="close").values.astype(np.float64)
    high  = df.pivot_table(index="time", columns="pair", values="high").values.astype(np.float64)
    low   = df.pivot_table(index="time", columns="pair", values="low").values.astype(np.float64)
    open_ = df.pivot_table(index="time", columns="pair", values="open").values.astype(np.float64)
    return close, high, low, open_, times

# ── H1 resampling ────────────────────────────────────────────────────────────
def to_h1(close, high, low, open_, times):
    idx = times
    c = pd.DataFrame(close, index=idx, columns=PAIRS).resample("1h").last()
    h  = pd.DataFrame(high,  index=idx, columns=PAIRS).resample("1h").max()
    l  = pd.DataFrame(low,   index=idx, columns=PAIRS).resample("1h").min()
    o  = pd.DataFrame(open_, index=idx, columns=PAIRS).resample("1h").first()
    return c.values.astype(np.float64), h.values.astype(np.float64), \
           l.values.astype(np.float64), o.values.astype(np.float64), c.index

# ── Energy Storage ───────────────────────────────────────────────────────────
def compute_es(rets_m1, times):
    sq = rets_m1 ** 2
    T, P = sq.shape
    cum = np.zeros((T + 1, P))
    cum[1:] = np.cumsum(sq, axis=0)
    es = cum[50:] - cum[:-50]
    es_t = times[50:]
    es_df = pd.DataFrame(es, index=es_t, columns=PAIRS)
    es_h1 = es_df.resample("1h").last()
    return es_h1.values.astype(np.float64), es_h1.index

# ── Session helper ───────────────────────────────────────────────────────────
def classify_session(hour):
    asia  = (hour >= 0) & (hour < 7)
    eu    = (hour >= 7) & (hour < 13)
    us    = (hour >= 13) | (hour < 0)
    return asia, eu, us

# ═══════════════════════════════════════════════════════════════════════════════
#  NUMBA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
@jit(nopython=True)
def find_reversion_events(resid, sigma120):
    n = len(resid)
    ev = []
    rt = []
    for i in range(n):
        if abs(resid[i]) > 2.0 * sigma120[i]:
            rev = -1
            for j in range(i, min(i + 100, n)):
                if abs(resid[j]) < 1.0 * sigma120[i]:
                    rev = j - i
                    break
            if rev >= 0:
                ev.append(i)
                rt.append(rev)
    return np.array(ev, dtype=np.int64), np.array(rt, dtype=np.int64)

@jit(nopython=True)
def find_release_events_scalar(delta_es, sigma_scalar):
    n = len(delta_es)
    out = []
    for i in range(1, n):
        if delta_es[i] < -1.5 * sigma_scalar:
            out.append(i)
    return np.array(out, dtype=np.int64)

@jit(nopython=True)
def cascade_counts(releases, delta_other, max_lag):
    n = len(delta_other)
    cascade, indep = 0, 0
    for idx in releases:
        if idx >= n: break
        end = min(idx + max_lag, n)
        s = 0.0
        for j in range(idx, end):
            s += delta_other[j]
        if s > 0:
            cascade += 1
        else:
            indep += 1
    return cascade, indep

@jit(nopython=True)
def xcorr_numba(a, b, max_lag):
    n = len(a)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.empty(len(lags))
    for ki, k in enumerate(lags):
        s, c = 0.0, 0
        if k < 0:
            for i in range(-k, n):
                s += a[i + k] * b[i]
                c += 1
        else:
            for i in range(n - k):
                s += a[i] * b[i + k]
                c += 1
        out[ki] = s / c if c > 0 else 0.0
    return lags, out

@jit(nopython=True)
def xcorr_lead_lag(a, b, max_lag):
    """Find best non-zero lag (if any) and whether a leads b."""
    n = len(a)
    best_lag, best_val = 0, 0.0
    for k in range(1, max_lag + 1):
        # a leads b: a[t-k] vs b[t], i.e., a at earlier time correlates with b
        s1, c1 = 0.0, 0
        for i in range(k, n):
            s1 += a[i - k] * b[i]
            c1 += 1
        r1 = s1 / c1 if c1 > 0 else 0.0
        if abs(r1) > abs(best_val):
            best_val = r1
            best_lag = -k
        # b leads a: a[t] vs b[t-k]
        s2, c2 = 0.0, 0
        for i in range(k, n):
            s2 += a[i] * b[i - k]
            c2 += 1
        r2 = s2 / c2 if c2 > 0 else 0.0
        if abs(r2) > abs(best_val):
            best_val = r2
            best_lag = k
    return best_lag, best_val

# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1 — Triangular Residual Reversion Dynamics (TRRD)
# ═══════════════════════════════════════════════════════════════════════════════
def analysis1(ret_valid, orig_idx, h1_close):
    n_valid = ret_valid.shape[0]
    strengths = ret_valid @ PINV.T
    residuals = ret_valid - strengths @ XMAT.T
    eurjpy_res = residuals[:, PAIR_IDX["EURJPY"]]

    if n_valid < 121:
        print("  [TRRD] Too few valid H1 returns (<121), skipping.\n")
        return

    # Rolling 120-bar σ of EURJPY residual on valid returns
    windows = swv(eurjpy_res, 120)
    sigma120 = np.std(windows, axis=1, ddof=0)
    resid_trim = eurjpy_res[119:]
    n_trim = len(resid_trim)

    ev_idx, rev_times = find_reversion_events(resid_trim, sigma120)
    if len(ev_idx) < 5:
        print("  [TRRD] Too few events, skipping.\n")
        return

    persistent = rev_times >= 4
    transient  = rev_times < 2
    p_filt = ev_idx[persistent]
    t_filt = ev_idx[transient]

    # Map filtered event index → original H1 return index → h1_close index
    # resid_trim[i] corresponds to original H1 return at orig_idx[119 + i]
    def forward_rets(filt_idxs, fwd):
        res = []
        for fi in filt_idxs:
            orig_ret_idx = orig_idx[119 + fi]
            start = orig_ret_idx + 1
            end   = start + fwd
            if end >= h1_close.shape[0]:
                continue
            r = np.log(h1_close[end, PAIR_IDX["EURJPY"]] / h1_close[start, PAIR_IDX["EURJPY"]])
            res.append(r)
        return np.array(res)

    p_h5  = forward_rets(p_filt, 5)
    p_h20 = forward_rets(p_filt, 20)
    t_h5  = forward_rets(t_filt, 5)
    t_h20 = forward_rets(t_filt, 20)

    print("=" * 72)
    print("  ANALYSIS 1: Triangular Residual Reversion Dynamics (TRRD)")
    print("=" * 72)
    print(f"  Total H1 returns: {ret_valid.shape[0]} (filtered, no gaps)")
    print(f"  EURJPY residual events (>2σ): {len(ev_idx)}")
    print(f"    Persistent (rev ≥ 4 bars):  {len(p_filt)}  Transient (rev < 2 bars): {len(t_filt)}")
    print()

    def print_grp(label, h5, h20):
        if len(h5):
            print(f"  [{label}] H5  — mean={np.mean(h5):+.6f}  win%={np.mean(h5>0)*100:.1f}%  n={len(h5)}")
        else:
            print(f"  [{label}] H5  — no data")
        if len(h20):
            print(f"  [{label}] H20 — mean={np.mean(h20):+.6f}  win%={np.mean(h20>0)*100:.1f}%  n={len(h20)}")

    print_grp("Persistent", p_h5, p_h20)
    print_grp("Transient",  t_h5, t_h20)
    if len(p_h5) >= 2 and len(t_h5) >= 2:
        tst, pv = sp_stats.ttest_ind(p_h5, t_h5, equal_var=False)
        print(f"  H5  t-test:  t={tst:.3f}  p={pv:.4f}")
    elif len(p_h5) < 2:
        print(f"  H5 t-test:  insufficient persistent events ({len(p_h5)})")
    if len(p_h20) >= 2 and len(t_h20) >= 2:
        tst, pv = sp_stats.ttest_ind(p_h20, t_h20, equal_var=False)
        print(f"  H20 t-test:  t={tst:.3f}  p={pv:.4f}")
    elif len(p_h20) < 2:
        print(f"  H20 t-test: insufficient persistent events ({len(p_h20)})")
    # Reversion time distribution
    if len(rev_times) > 0:
        print(f"  Reversion time distribution:  mean={np.mean(rev_times):.1f}  median={np.median(rev_times):.0f}  min={rev_times.min()}  max={rev_times.max()}")
        hist_vals = [f"{b}:{np.sum((rev_times >= b-2) & (rev_times < b) if b>2 else (rev_times < b))}" for b in [2,4,8,16,32]]
        print(f"    Hist:  rev<2:{np.sum(rev_times<2)}  2≤rev<4:{np.sum((rev_times>=2)&(rev_times<4))}  4≤rev<8:{np.sum((rev_times>=4)&(rev_times<8))}  ≥8:{np.sum(rev_times>=8)}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2 — Cross-Pair Energy Divergence (CPED)
# ═══════════════════════════════════════════════════════════════════════════════
def analysis2(es_h1, h1_close):
    n = es_h1.shape[0]
    if n < 5:
        print("  [CPED] Not enough H1 data, skipping.\n")
        return

    print("=" * 72)
    print("  ANALYSIS 2: Cross-Pair Energy Divergence (CPED)")
    print("=" * 72)

    pairsets = [("EURJPY", "GBPJPY"), ("EURUSD", "GBPUSD")]
    for pA, pB in pairsets:
        iA, iB = PAIR_IDX[pA], PAIR_IDX[pB]
        ESa = es_h1[:, iA]
        ESb = es_h1[:, iB]
        en_div = np.abs(ESa - ESb) / (ESa + ESb + 1e-10)
        high_idx = np.where(en_div > 0.5)[0]
        n_high = len(high_idx)

        if n_high < 5:
            print(f"  [{pA}/{pB}] Too few high-divergence events ({n_high}), skip.")
            continue

        def event_stats(fwd):
            hi_more = 0; rsum = 0.0; cnt = 0
            for idx in high_idx:
                if idx + fwd >= n: break
                ra = np.log(h1_close[idx+fwd, iA] / h1_close[idx, iA])
                rb = np.log(h1_close[idx+fwd, iB] / h1_close[idx, iB])
                hi = 0 if ESa[idx] >= ESb[idx] else 1
                hm = abs(ra) if hi == 0 else abs(rb)
                lm = abs(rb) if hi == 0 else abs(ra)
                if hm >= lm: hi_more += 1
                rsum += hm / (lm + 1e-10)
                cnt += 1
            return hi_more, rsum, cnt

        h5_h, h5_r, h5_c = event_stats(5)
        h20_h, h20_r, h20_c = event_stats(20)

        def en_div_corr(fwd):
            xs, ys = [], []
            for idx in range(n - fwd):
                ra = np.log(h1_close[idx+fwd, iA] / h1_close[idx, iA])
                rb = np.log(h1_close[idx+fwd, iB] / h1_close[idx, iB])
                hi = 0 if ESa[idx] >= ESb[idx] else 1
                rm = (abs(ra)/abs(rb)) if hi == 0 else (abs(rb)/abs(ra))
                xs.append(en_div[idx]); ys.append(rm)
            xa = np.array(xs); ya = np.array(ys)
            m = np.isfinite(xa) & np.isfinite(ya) & (xa > 0) & (ya > 0)
            if m.sum() < 5: return 0.0, 0.0
            return sp_stats.pearsonr(xa[m], ya[m])

        r5, p5 = en_div_corr(5)
        r20, p20 = en_div_corr(20)

        print(f"  [{pA}/{pB}]  events: {n_high}")
        print(f"    H5  — high-energy releases more: {h5_h}/{h5_c} ({h5_h/max(h5_c,1)*100:.1f}%)  mean ratio={h5_r/max(h5_c,1):.3f}")
        print(f"    H20 — high-energy releases more: {h20_h}/{h20_c} ({h20_h/max(h20_c,1)*100:.1f}%)  mean ratio={h20_r/max(h20_c,1):.3f}")
        print(f"    en_div vs rel-mag: H5 r={r5:.3f} p={p5:.4f} | H20 r={r20:.3f} p={p20:.4f}")
        print()

# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 3 — Session-Conditional ES Hysteresis
# ═══════════════════════════════════════════════════════════════════════════════
def analysis3(es_h1, h1_close, h1_times):
    n = es_h1.shape[0]
    if n < 5:
        print("  [ES Hysteresis] Not enough data, skipping.\n")
        return
    hour = pd.DatetimeIndex(h1_times).hour.values
    delta_es = np.diff(es_h1, axis=0)
    nc = es_h1.shape[0] - 1

    print("=" * 72)
    print("  ANALYSIS 3: Session-Conditional ES Hysteresis")
    print("=" * 72)

    for pair in PAIRS:
        i = PAIR_IDX[pair]
        es = es_h1[:-1, i]
        de = delta_es[:, i]
        p70 = np.percentile(es, 70)
        p90 = np.percentile(es, 90)

        acc = (es > p70) & (de > 0)
        dec = (es > p70) & (de < 0)
        acc90 = (es > p90) & (de > 0)
        dec90 = (es > p90) & (de < 0)

        def fwd_abs_ret(mask, fwd):
            idxs = np.where(mask)[0]
            vals = []
            for idx in idxs:
                if idx + fwd >= h1_close.shape[0]: continue
                r = np.log(h1_close[idx+fwd, i] / h1_close[idx, i])
                vals.append(abs(r))
            return np.array(vals)

        asia, eu, us = classify_session(hour[:nc])

        print(f"  [{pair}]")
        # 90th percentile breakdown
        for lbl, mk in [("ACCELERATING (90th)", acc90), ("DECELERATING (90th)", dec90)]:
            print(f"    {lbl}: {mk.sum()} events")
            for fw, fwl in [(5, "H5"), (20, "H20")]:
                mg = fwd_abs_ret(mk, fw)
                if len(mg):
                    print(f"      {fwl} mean |ret|={np.mean(mg):.6f}")

        # 70th percentile comparison
        na, nd = acc.sum(), dec.sum()
        if na >= 5 and nd >= 5:
            print(f"    ES > 70th pctile — ACCEL: {na}  DECEL: {nd}")
            for fw, fwl in [(5, "H5"), (20, "H20")]:
                ma = fwd_abs_ret(acc, fw)
                md = fwd_abs_ret(dec, fw)
                if len(ma) >= 2 and len(md) >= 2:
                    tst, _ = sp_stats.ttest_ind(ma, md, equal_var=False)
                    print(f"      {fwl} mean |ret| — ACCEL: {np.mean(ma):.6f}  DECEL: {np.mean(md):.6f}  t={tst:.3f}")

        # Session-specific: predictive power for ES > p90 (both accel+decel)
        p90_mask = (es > p90)
        print(f"    Session-specific ES > 90th pctile:")
        any_s = False
        for sname, sm in [("ASIAN", asia[:nc]), ("EUROPEAN", eu[:nc]), ("US", us[:nc])]:
            subm = p90_mask & sm
            cnt = subm.sum()
            if cnt < 3: continue
            mg = fwd_abs_ret(subm, 5)
            if len(mg) < 3: continue
            any_s = True
            median_base = np.median(mg)
            pp = np.mean(mg > median_base)
            print(f"      {sname:10s}: n={cnt:3d}  pp={pp:.2f}  mean|H5|={np.mean(mg):.6f}")
        if not any_s:
            print(f"      (insufficient data in all sessions)")
        print()

# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 4 — Multi-Pair Energy Cascade (MPEC)
# ═══════════════════════════════════════════════════════════════════════════════
def analysis4(es_h1):
    n = es_h1.shape[0]
    if n < 10:
        print("  [MPEC] Not enough data, skipping.\n")
        return
    delta_es = np.diff(es_h1, axis=0)
    npairs = len(PAIRS)
    max_lag = 10

    print("=" * 72)
    print("  ANALYSIS 4: Multi-Pair Energy Cascade (MPEC)")
    print("=" * 72)

    # Cross-correlation matrix (best |corr| per pair)
    print("  ΔES cross-correlation matrix (best |r| at |lag|≤10):")
    header = "".join(f"{p:>10}" for p in PAIRS)
    print(f"  {'':>8}{header}")
    for i, pA in enumerate(PAIRS):
        row = f"  {pA:>8}"
        for j, pB in enumerate(PAIRS):
            if i == j:
                row += f"{'  —':>10}"
                continue
            a = (delta_es[:, i] - np.mean(delta_es[:, i])) / (np.std(delta_es[:, i]) + 1e-10)
            b = (delta_es[:, j] - np.mean(delta_es[:, j])) / (np.std(delta_es[:, j]) + 1e-10)
            lags, corr = xcorr_numba(a, b, max_lag)
            best = np.argmax(np.abs(corr))
            row += f"{corr[best]:+9.3f} "
        print(row)

    # EURJPY release → cascade
    eur_i = PAIR_IDX["EURJPY"]
    gbp_i = PAIR_IDX["GBPJPY"]
    de_eur = delta_es[:, eur_i]
    de_gbp = delta_es[:, gbp_i]
    sigma_eur = np.std(de_eur)
    releases = find_release_events_scalar(de_eur, sigma_eur)
    print(f"\n  EURJPY release events (ΔES < -1.5σ): {len(releases)}")
    if len(releases):
        casc, ind = cascade_counts(releases, de_gbp, 5)
        print(f"    GBPJPY cascade   (ΔES rises next 5 bars): {casc}/{casc+ind} ({casc/max(casc+ind,1)*100:.1f}%)")
        print(f"    GBPJPY independent (ΔES falls next 5 bars): {ind}/{casc+ind} ({ind/max(casc+ind,1)*100:.1f}%)")

    # Lead-lag: non-zero lags only, detect asymmetry
    print(f"\n  Lead-lag structure (non-zero lags, |k| ≤ {max_lag}):")
    for i, pA in enumerate(PAIRS):
        best_r, best_lag, best_p = 0.0, 0, ""
        for j, pB in enumerate(PAIRS):
            if i == j: continue
            a = (delta_es[:, i] - np.mean(delta_es[:, i])) / (np.std(delta_es[:, i]) + 1e-10)
            b = (delta_es[:, j] - np.mean(delta_es[:, j])) / (np.std(delta_es[:, j]) + 1e-10)
            lag, r = xcorr_lead_lag(a, b, max_lag)
            if abs(r) > abs(best_r):
                best_r, best_lag, best_p = r, lag, pB
        if best_p and best_lag != 0:
            dir_ = "leads" if best_lag < 0 else "lags"
            print(f"    {pA:>8} {dir_:>6} {best_p:>8} by {abs(best_lag):2d} bars (r={best_r:.3f})")
        elif best_p:
            print(f"    {pA:>8} synchronous with {best_p:>8} (all |k|>0 ≤ r at k=0)")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 5 — Dispersion Component Analysis (DCA)
# ═══════════════════════════════════════════════════════════════════════════════
def analysis5(ret_valid, orig_idx, h1_close):
    n_valid = ret_valid.shape[0]
    if n_valid < 10:
        print("  [DCA] Not enough data, skipping.\n")
        return

    strengths = ret_valid @ PINV.T
    dispersion = np.std(strengths, axis=1)
    centered = np.abs(strengths - np.mean(strengths, axis=1, keepdims=True))
    driver_idx = np.argmax(centered, axis=1)

    commodity_set = {"AUD", "NZD", "CAD"}
    def classify(d):
        if d == "JPY": return "JPY-driven"
        if d == "USD": return "USD-driven"
        if d == "EUR": return "EUR-driven"
        if d in commodity_set: return "commodity-driven"
        return "mixed"

    classifications = np.array([classify(CURRENCIES[d]) for d in driver_idx])

    # Forward mean return across all pairs from each valid H1 bar
    T_h5 = 5
    n_close = h1_close.shape[0]
    fwd_h5 = np.full(n_valid, np.nan)
    for vi, oi in enumerate(orig_idx):
        end = oi + T_h5
        if end >= n_close: continue
        rets = np.log(h1_close[end] / h1_close[oi])
        fwd_h5[vi] = np.mean(rets)

    print("=" * 72)
    print("  ANALYSIS 5: Dispersion Component Analysis (DCA)")
    print("=" * 72)
    print(f"  Total valid H1 returns: {n_valid}")

    p80 = np.percentile(dispersion, 80)
    hi_disp = dispersion > p80
    print(f"  High-dispersion events (>80th pctile): {hi_disp.sum()}")

    cats = ["JPY-driven", "USD-driven", "EUR-driven", "commodity-driven", "mixed"]
    print(f"\n  Dispersion driver distribution (high-dispersion only):")
    for cat in cats:
        mask = hi_disp & (classifications == cat)
        cnt = mask.sum()
        if cnt == 0: continue
        pct = cnt / hi_disp.sum() * 100
        mf = np.nanmean(fwd_h5[mask]) if cnt > 0 else 0.0
        wr = np.nanmean(fwd_h5[mask] > 0) * 100 if cnt > 0 else 0.0
        hi_idx = np.where(mask)[0]
        p70 = np.percentile(dispersion, 70)
        resolves = []
        for idx in hi_idx:
            for j in range(idx, min(idx + 100, n_valid)):
                if dispersion[j] < p70:
                    resolves.append(j - idx)
                    break
        avg_res = np.mean(resolves) if resolves else np.nan
        if cnt < 5:
            print(f"    {cat:20s}: {cnt:4d} ({pct:5.1f}%)  (too few for stats)")
        else:
            print(f"    {cat:20s}: {cnt:4d} ({pct:5.1f}%)  mean H5={mf:+.6f}  WR={wr:5.1f}%  resolve={avg_res:.1f} bars")

    # JPY-driven vs USD-driven comparison
    print(f"\n  JPY-driven vs USD-driven exhaustion:")
    for lbl, drv in [("JPY-driven", "JPY"), ("USD-driven", "USD")]:
        mask = hi_disp & (classifications == f"{drv}-driven")
        cnt = mask.sum()
        if cnt < 3:
            print(f"    {lbl:12s}: too few events ({cnt})")
            continue
        mf = np.nanmean(fwd_h5[mask])
        wr = np.nanmean(fwd_h5[mask] > 0) * 100
        hi_idx = np.where(mask)[0]
        resolves = []
        p70 = np.percentile(dispersion, 70)
        for idx in hi_idx:
            for j in range(idx, min(idx + 100, n_valid)):
                if dispersion[j] < p70:
                    resolves.append(j - idx)
                    break
        avg_res = np.mean(resolves) if resolves else np.nan
        # H20 Sharpe
        rets20 = []
        for idx in hi_idx:
            orig = orig_idx[idx]
            if orig + 20 >= n_close: continue
            r = np.mean(np.log(h1_close[orig+20] / h1_close[orig]))
            rets20.append(r)
        if len(rets20) >= 5:
            sharpe = (np.mean(rets20) / (np.std(rets20)+1e-10)) * np.sqrt(252*24/20)
        else:
            sharpe = np.nan
        sharpe_str = f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A (n<5)"
        print(f"    {lbl:12s}: n={cnt:4d}  H5 ret={mf:+.6f}  WR={wr:5.1f}%  resolve={avg_res:.1f} bars  H20 Sharpe={sharpe_str}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(DIR_PATH, exist_ok=True)

    close, high, low, open_, times = load_data()
    T = len(close)
    print(f"Loaded {T} M1 bars x {len(PAIRS)} pairs")

    # M1 log returns
    log_rets_m1 = np.diff(np.log(close), axis=0)

    # ── H1 resample ──
    h1_close, h1_high, h1_low, h1_open, h1_times = to_h1(close, high, low, open_, times)
    # Drop weekend H1 bars (NaN rows created by resample for empty bins)
    h1_idx = pd.DatetimeIndex(h1_times)
    h1_close_df = pd.DataFrame(h1_close, index=h1_idx, columns=PAIRS)
    h1_close_df = h1_close_df.replace(0, np.nan).dropna(how="any")
    h1_close = h1_close_df.values
    h1_times = h1_close_df.index
    N_h1 = h1_close.shape[0]
    h1_ret = np.diff(np.log(h1_close), axis=0)
    print(f"H1 bars (no weekend NaN): {N_h1}, H1 returns: {h1_ret.shape[0]}")

    # Drop returns that span weekend gaps (time diff > 1.5 hours)
    h1_gap_ns = np.diff(h1_times.asi8)
    valid_ret = h1_gap_ns <= int(1.5 * 3600 * 1e9)
    valid_indices = np.where(valid_ret)[0]
    h1_ret_valid = h1_ret[valid_ret]
    n_valid = len(h1_ret_valid)
    print(f"Valid H1 returns (no weekend gaps): {n_valid}  (dropped {h1_ret.shape[0] - n_valid})")

    # ── ES at H1 ──
    es_h1, es_h1_times = compute_es(log_rets_m1, times)
    # Align ES to valid H1 bars only
    es_df = pd.DataFrame(es_h1, index=pd.DatetimeIndex(es_h1_times), columns=PAIRS)
    es_full = es_df.reindex(h1_idx, method='ffill').fillna(0).values.astype(np.float64)
    # Also align to the filtered h1_times
    es_filtered = es_df.reindex(h1_times, method='ffill').fillna(0).values.astype(np.float64)
    print(f"ES at H1: {es_filtered.shape[0]} values (aligned to {len(h1_times)} H1 bars)")

    # ── Run all analyses ──
    print("\n" + "#" * 72)
    print("#  RUNNING ALL 5 DARK RESEARCH ANALYSES")
    print("#" * 72)

    if n_valid >= 121:
        analysis1(h1_ret_valid, valid_indices, h1_close)
    else:
        print("\n  [TRRD] Too few valid H1 returns, skipping.\n")

    nh1 = es_filtered.shape[0]
    if nh1 >= 5:
        analysis2(es_filtered, h1_close)
    else:
        print("\n  [CPED] Not enough ES data, skipping.\n")

    if nh1 >= 5:
        analysis3(es_filtered, h1_close, h1_times)
    else:
        print("\n  [ES Hysteresis] Not enough data, skipping.\n")

    if nh1 >= 10:
        analysis4(es_filtered)
    else:
        print("\n  [MPEC] Not enough data, skipping.\n")

    if n_valid >= 10:
        analysis5(h1_ret_valid, valid_indices, h1_close)
    else:
        print("\n  [DCA] Not enough data, skipping.\n")

    print("All analyses complete.")


if __name__ == "__main__":
    main()
