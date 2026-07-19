#!/usr/bin/env python3
"""
M1 Cross-Pair Engine v2 — Proper index alignment.
es_aligned[t] = energy as of close[t] → predicts close[t+fwd] / close[t].
"""
import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from numba import jit, prange
from scipy import stats as sp_stats
from scipy.stats import binomtest
from numpy.lib.stride_tricks import sliding_window_view as swv

DATA_PATH = r"C:\Trading\Agentic_Trading\proxima_x\data\temp\mt5_m1_9day.parquet"
DIR_PATH  = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research"
os.makedirs(DIR_PATH, exist_ok=True)

CURRENCIES = ["EUR", "USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]
CUR_MAP = {c: i for i, c in enumerate(CURRENCIES)}
PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "NZDUSD", "EURJPY", "GBPJPY"]
PIDX = {p: i for i, p in enumerate(PAIRS)}
WLS_REG = 0.01

BCM = {  # BASE_CURRENCY_MAP
    "EURUSD": ("EUR","USD"),"USDJPY":("USD","JPY"),"GBPUSD":("GBP","USD"),
    "AUDUSD":("AUD","USD"),"NZDUSD":("NZD","USD"),"EURJPY":("EUR","JPY"),
    "GBPJPY":("GBP","JPY"),"EURGBP":("EUR","GBP"),"EURAUD":("EUR","AUD"),
    "EURNZD":("EUR","NZD"),"EURCHF":("EUR","CHF"),"GBPAUD":("GBP","AUD"),
    "GBPNZD":("GBP","NZD"),"GBPCHF":("GBP","CHF"),"AUDJPY":("AUD","JPY"),
    "NZDJPY":("NZD","JPY"),"AUDNZD":("AUD","NZD"),"AUDCHF":("AUD","CHF"),
    "NZDCAD":("NZD","CAD"),"AUDCAD":("AUD","CAD"),"CADJPY":("CAD","JPY"),
    "CHFJPY":("CHF","JPY"),"USDCAD":("USD","CAD"),"USDCHF":("USD","CHF"),
    "GBPCAD":("GBP","CAD"),"EURCAD":("EUR","CAD"),"CADCHF":("CAD","CHF"),
    "NZDCHF":("NZD","CHF"),
}

def build_wls():
    n_p, n_c = len(PAIRS), len(CURRENCIES)
    X = np.zeros((n_p, n_c))
    for i, p in enumerate(PAIRS):
        b, q = BCM[p]
        X[i, CUR_MAP[b]] = 1.0; X[i, CUR_MAP[q]] = -1.0
    return np.linalg.solve(X.T @ X + WLS_REG * np.eye(n_c), X.T), X

PINV, XMAT = build_wls()

# ── Load & precompute ────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
times = pd.DatetimeIndex(df["time"].unique()).sort_values()
close = df.pivot_table(index="time", columns="pair", values="close").values.astype(np.float64)
T = close.shape[0]
hour = pd.DatetimeIndex(times).hour.values  # hour for each close

rets = np.diff(np.log(close), axis=0)  # (T-1, 7)

# ES aligned: es_aligned[t] = energy ending at close[t]
sq = rets ** 2
es_aligned = np.zeros((T, 7))
cum = np.zeros((T, 7))
cum[1:] = np.cumsum(sq, axis=0)
es_aligned[51:] = cum[51:] - cum[1:T-50]  # 50-bar rolling sum ending at bar t
es_aligned[:51] = np.nan
for i in range(1, T):  # Forward fill NaN
    if np.isnan(es_aligned[i, 0]):
        es_aligned[i] = es_aligned[i-1]

delta_aligned = np.diff(es_aligned, axis=0)  # (T-1, 7), delta[t] ≈ es_aligned[t+1] - es_aligned[t]

# ── Helpers ───────────────────────────────────────────────────────────────────
def asia_mask(h): return (h >= 0) & (h < 7)
def eu_mask(h):   return (h >= 7) & (h < 13)
def us_mask(h):   return (h >= 13) | (h < 0)

def fwd_ret(t, fwd, pi):
    """Log return from close[t] to close[t+fwd] for pair pi."""
    if t + fwd >= T: return np.nan
    return np.log(close[t+fwd, pi] / close[t, pi])

def fwd_mag(t, fwd, pi):
    return abs(fwd_ret(t, fwd, pi))

def basket_fwd_ret(t, fwd):
    if t + fwd >= T: return np.nan
    return np.mean(np.log(close[t+fwd] / close[t]))

def basket_fwd_mag(t, fwd):
    if t + fwd >= T: return np.nan
    return np.mean(np.abs(np.log(close[t+fwd] / close[t])))

def session_name(h):
    if asia_mask(h): return "ASIAN"
    if eu_mask(h): return "EUROPEAN"
    return "US"

# ═══════════════════════════════════════════════════════════════════════════════
#  A1: Triangular Residual Reversion Dynamics
# ═══════════════════════════════════════════════════════════════════════════════
@jit(nopython=True)
def find_events(resid, sigma, thresh=2.0):
    n = len(resid)
    ev, rt = [], []
    for i in range(n):
        if abs(resid[i]) > thresh * sigma[i]:
            for j in range(i, min(i + 200, n)):
                if abs(resid[j]) < 1.0 * sigma[i]:
                    rt.append(j - i); ev.append(i); break
    return np.array(ev, dtype=np.int64), np.array(rt, dtype=np.int64)

def a1():
    print("=" * 72)
    print("  A1: TRIANGULAR RESIDUAL (EURJPY - EURUSD - USDJPY) [M1]")
    print("=" * 72)
    # rets[t] = log return from close[t] to close[t+1]
    # Triangular residual at rets[t]: does EURJPY deviate from EURUSD+USDJPY?
    ei, usi, udi = PIDX["EURJPY"], PIDX["EURUSD"], PIDX["USDJPY"]
    tri_res = rets[:, ei] - rets[:, usi] - rets[:, udi]  # (T-1,)
    n = len(tri_res)
    if n < 501: print("  Insufficient data\n"); return
    
    sigma = np.std(swv(tri_res, 500), axis=1)
    ev, rt = find_events(tri_res[499:], sigma, 2.0)
    print(f"  Bars: {n}, events (>2σ): {len(ev)} ({len(ev)/n*1440:.2f}/day)")
    if len(ev) == 0: print("  No events\n"); return
    
    persistent = rt >= 10; transient = rt < 3
    pi, ti = ev[persistent], ev[transient]
    print(f"  Persistent (rev≥10): {len(pi)}  Transient (rev<3): {len(ti)}")
    
    for lbl, idxs in [("Persistent", pi), ("Transient", ti)]:
        for fw, fn in [(5,"M5"),(30,"M30"),(60,"H1")]:
            v = np.array([basket_fwd_mag(499+idx+1, fw) for idx in idxs if 499+idx+1+fw < T])
            if len(v) >= 2: print(f"    {lbl:10s} {fn}: mag={np.mean(v):.6f} n={len(v)}")
    
    for fw, fn in [(5,"M5"),(60,"H1")]:
        pv = np.array([basket_fwd_mag(499+idx+1, fw) for idx in pi if 499+idx+1+fw < T])
        tv = np.array([basket_fwd_mag(499+idx+1, fw) for idx in ti if 499+idx+1+fw < T])
        if len(pv) >= 2 and len(tv) >= 2:
            tst, p = sp_stats.ttest_ind(pv, tv, equal_var=False)
            print(f"    P/T {fn} ratio={np.mean(pv)/(np.mean(tv)+1e-10):.3f} t={tst:.3f} p={p:.4f}")
    
    print(f"  Rev time mean={np.mean(rt):.1f} median={np.median(rt):.0f} max={rt.max()}")
    print(f"  Hist: <3:{np.sum(rt<3)} 3-9:{np.sum((rt>=3)&(rt<10))} >=10:{np.sum(rt>=10)}")
    
    # Session breakdown
    ev_hour = hour[499:][ev] if len(hour) > 499 else hour[min(len(hour)-1, 499):]
    for sn in ["ASIAN","EUROPEAN","US"]:
        sm = asia_mask(ev_hour) if sn=="ASIAN" else (eu_mask(ev_hour) if sn=="EUROPEAN" else us_mask(ev_hour))
        sr = rt[sm]
        if len(sr) > 0: print(f"  {sn} rev_time: {np.mean(sr):.1f} (n={len(sr)})")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
#  A2: Cross-Pair Energy Divergence
# ═══════════════════════════════════════════════════════════════════════════════
def a2():
    print("=" * 72)
    print("  A2: CROSS-PAIR ENERGY DIVERGENCE [M1]")
    print("=" * 72)
    psets = [("EURJPY","GBPJPY"),("EURUSD","GBPUSD"),("AUDUSD","NZDUSD"),("EURJPY","EURUSD")]
    for pA, pB in psets:
        iA, iB = PIDX[pA], PIDX[pB]
        div = np.abs(es_aligned[:, iA] - es_aligned[:, iB]) / (es_aligned[:, iA] + es_aligned[:, iB] + 1e-10)
        for th in [0.3, 0.5, 0.7]:
            idxs = np.where(div > th)[0]
            n_ev = len(idxs)
            if n_ev < 10: continue
            hi_more = 0
            for t in idxs:
                if t + 5 >= T: continue
                ma = abs(fwd_ret(t, 5, iA)); mb = abs(fwd_ret(t, 5, iB))
                if (es_aligned[t,iA] >= es_aligned[t,iB] and ma >= mb) or \
                   (es_aligned[t,iB] > es_aligned[t,iA] and mb >= ma):
                    hi_more += 1
            wr = hi_more / n_ev
            pv = binomtest(hi_more, n_ev, 0.5, alternative='two-sided').pvalue
            sig = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
            rate = n_ev / T * 1440
            print(f"  [{pA}/{pB}] th={th:.1f} n={n_ev:5d} WR={wr:.1%} {rate:.1f}/d p={pv:.4f} {sig}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
#  A3: ES Hysteresis — Path-Dependent Magnitude
# ═══════════════════════════════════════════════════════════════════════════════
def a3():
    print("=" * 72)
    print("  A3: ES HYSTERESIS [M1]")
    print("=" * 72)
    N = T - 1  # es_aligned[1:] → N elements
    for pair in PAIRS:
        i = PIDX[pair]
        es_1 = es_aligned[1:, i]  # (T-1,) aligns: es_1[t] corresponds to close[t+1]
        de_1 = delta_aligned[:, i]  # (T-1,)
        p70, p90 = np.percentile(es_1, 70), np.percentile(es_1, 90)
        
        acc = (es_1 > p70) & (de_1 > 0)
        dec = (es_1 > p70) & (de_1 < 0)
        acc90 = (es_1 > p90) & (de_1 > 0)
        dec90 = (es_1 > p90) & (de_1 < 0)
        
        def get_mag(mask, fwd):
            # mask[t] → energy as of close[t+1] → predict from close[t+1]
            idxs = np.where(mask)[0]
            vals = []
            for idx in idxs:
                t_pred = idx + 1  # close index for prediction
                if t_pred + fwd >= T: continue
                vals.append(abs(np.log(close[t_pred+fwd, i] / close[t_pred, i])))
            return np.array(vals)
        
        sigs = []
        for fw, fn in [(5,"M5"),(30,"M30"),(60,"H1")]:
            ma = get_mag(acc90, fw); md = get_mag(dec90, fw)
            if len(ma) >= 2 and len(md) >= 2:
                tst, pv = sp_stats.ttest_ind(ma, md, equal_var=False)
                ratio = np.mean(ma) / (np.mean(md)+1e-10)
                if abs(tst) > 1.5:
                    sigs.append((fn, ratio, tst, pv, len(ma), len(md)))
        if sigs:
            print(f"  [{pair}] ACCEL/DECEL mag ratio at 90th pctile ES:")
            for fn, ratio, tst, pv, na, nd in sigs:
                d = "ACCEL>" if ratio > 1 else "DECEL>"
                print(f"    {fn}: {d}{abs(ratio):.3f}x t={tst:.3f} p={pv:.4f} na={na} nd={nd}")
        
        # Session-specific ES > p90 predictive power
        for sn, sm in [("ASIAN", asia_mask(hour[1:])), ("EUROPEAN", eu_mask(hour[1:])), ("US", us_mask(hour[1:]))]:
            sub = (es_1 > p90) & sm
            cnt = sub.sum()
            if cnt < 5: continue
            mg5 = get_mag(sub, 5)
            base = get_mag((es_1 <= p90), 5)
            if len(mg5) >= 2 and len(base) >= 2:
                tst, pv = sp_stats.ttest_ind(mg5, base, equal_var=False)
                print(f"    {sn:8s} ES>p90 n={cnt:4d} M5_mag={np.mean(mg5):.6f} vs_base={np.mean(base):.6f} t={tst:.3f} p={pv:.4f}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
#  A4: Multi-Pair Energy Cascade
# ═══════════════════════════════════════════════════════════════════════════════
@jit(nopython=True)
def xcorr(a, b, ml):
    n, lags = len(a), np.arange(-ml, ml+1)
    out = np.empty(len(lags))
    for ki, k in enumerate(lags):
        s, c = 0.0, 0
        if k < 0:
            for i in range(-k, n): s += a[i+k]*b[i]; c += 1
        else:
            for i in range(n-k): s += a[i]*b[i+k]; c += 1
        out[ki] = s / c if c > 0 else 0.0
    return lags, out

@jit(nopython=True, parallel=True)
def xcorr_mat(data, ml):
    P = data.shape[1]; best = np.zeros((P,P))
    for i in prange(P):
        for j in range(P):
            if i == j: continue
            a = (data[:,i]-np.mean(data[:,i]))/(np.std(data[:,i])+1e-10)
            b = (data[:,j]-np.mean(data[:,j]))/(np.std(data[:,j])+1e-10)
            _, c = xcorr(a,b,ml); best[i,j] = c[np.argmax(np.abs(c))]
    return best

@jit(nopython=True)
def lead_lag(a, b, ml):
    best_l, best_v = 0, 0.0
    for k in range(1, ml+1):
        s1=c1=0.0
        for i in range(k, len(a)): s1 += a[i-k]*b[i]; c1 += 1
        r1 = s1/c1 if c1>0 else 0.0
        if abs(r1) > abs(best_v): best_v, best_l = r1, -k
        s2=c2=0.0
        for i in range(k, len(a)): s2 += a[i]*b[i-k]; c2 += 1
        r2 = s2/c2 if c2>0 else 0.0
        if abs(r2) > abs(best_v): best_v, best_l = r2, k
    return best_l, best_v

@jit(nopython=True, parallel=True)
def lead_lag_mat(data, ml):
    P = data.shape[1]; lm = np.zeros((P,P)); vm = np.zeros((P,P))
    for i in prange(P):
        for j in range(P):
            if i == j: continue
            a = (data[:,i]-np.mean(data[:,i]))/(np.std(data[:,i])+1e-10)
            b = (data[:,j]-np.mean(data[:,j]))/(np.std(data[:,j])+1e-10)
            lm[i,j], vm[i,j] = lead_lag(a,b,ml)
    return lm, vm

def a4():
    print("=" * 72)
    print("  A4: MULTI-PAIR ENERGY CASCADE [M1]")
    print("=" * 72)
    de = delta_aligned[52:].copy()  # (T-53, 7) strip NaN, make contiguous
    ml = 20
    print("  ΔES cross-correlation (best |r| at |lag|≤20):")
    bc = xcorr_mat(de, ml)
    h = "".join(f"{p:>10}" for p in PAIRS)
    print(f"  {'':>8}{h}")
    for i, pA in enumerate(PAIRS):
        r = f"  {pA:>8}"
        for j in range(len(PAIRS)):
            r += f"{'  —':>10}" if i==j else f"{bc[i,j]:+9.3f} "
        print(r)
    
    lm, vm = lead_lag_mat(de, ml)
    print(f"\n  ΔES lead-lag (|lag|≤{ml} M1, non-zero):")
    for i, pA in enumerate(PAIRS):
        br, bl, bp = 0.0, 0, ""
        for j, pB in enumerate(PAIRS):
            if i==j: continue
            if abs(vm[i,j]) > abs(br): br, bl, bp = vm[i,j], lm[i,j], pB
        if bp and bl != 0:
            d = "leads" if bl < 0 else "lags"
            print(f"    {pA:>8} {d:>6} {bp:>8} by {abs(int(bl)):2d} bars (r={br:.3f})")
    
    # Energy release cascade
    print("\n  Energy releases (ΔES < -2σ):")
    for src in ["EURJPY","USDJPY","GBPJPY","EURUSD"]:
        si = PIDX[src]; s = de[:,si]; sigma = np.std(s)
        rel = np.where(s < -2.0*sigma)[0]
        if len(rel) < 3: continue
        print(f"    {src}: {len(rel)} releases")
        for tgt in [p for p in PAIRS if p != src]:
            ti = PIDX[tgt]; casc = ind = 0
            for idx in rel:
                end = min(idx+10, de.shape[0])
                if np.sum(de[idx:end,ti]) > 0: casc += 1
                else: ind += 1
            pct = casc/(casc+ind)*100
            print(f"      → {tgt:8s}: casc={casc:3d} ind={ind:3d} ({pct:.0f}%)")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
#  A5: Dispersion Component Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def basket_ret_from(t, fwd):
    cp = t + 1 + fwd
    if cp >= T: return np.nan
    return np.mean(np.log(close[cp] / close[t+1]))

def a5():
    print("=" * 72)
    print("  A5: DISPERSION COMPONENT ANALYSIS [M1]")
    print("=" * 72)
    # WLS on rets (T-1) → currency strengths
    strg = rets @ PINV.T  # (T-1, 8)
    disp = np.std(strg, axis=1)
    cent = np.abs(strg - np.mean(strg, axis=1, keepdims=True))
    drv_i = np.argmax(cent, axis=1)
    comm = {"AUD","NZD","CAD"}
    drv = np.array([("JPY" if CURRENCIES[d]=="JPY" else "USD" if CURRENCIES[d]=="USD" else 
                      "EUR" if CURRENCIES[d]=="EUR" else "COMMODITY" if CURRENCIES[d] in comm else "MIXED") 
                     for d in drv_i])
    
    p80 = np.percentile(disp, 80)
    hi = disp > p80
    print(f"  Bars: {T-1}, hi-disp events (>80th): {hi.sum()} ({hi.sum()/(T-1)*1440:.1f}/day)")
    
    for cat in ["EUR","USD","JPY","COMMODITY","MIXED"]:
        m = hi & (drv == cat)
        cnt = m.sum()
        if cnt == 0: continue
        m5 = np.array([basket_ret_from(t, 5) for t in np.where(m)[0] if t+1+5 < T])
        m60 = np.array([basket_ret_from(t, 60) for t in np.where(m)[0] if t+1+60 < T])
        wr5 = np.nanmean(m5 > 0) * 100 if len(m5) > 0 else 0
        resolves = []
        for t in np.where(m)[0]:
            p70 = np.percentile(disp, 70)
            for j in range(t, min(t+500, T-1)):
                if disp[j] < p70: resolves.append(j-t); break
        avgr = np.mean(resolves) if resolves else np.nan
        
        pv = 1.0
        if cnt >= 10 and wr5 > 55:
            pv = binomtest(int(np.sum(m5>0)), len(m5), 0.5, alternative='two-sided').pvalue
        
        sig = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
        
        # H20 Sharpe
        r20 = np.array([basket_ret_from(t, 20) for t in np.where(m)[0] if t+1+20 < T])
        shrp20 = (np.nanmean(r20)/(np.nanstd(r20)+1e-10))*np.sqrt(1440/20) if len(r20)>=5 else np.nan
        
        print(f"    {cat:10s}: n={cnt:4d} M5_WR={wr5:5.1f}% M5_ret={np.nanmean(m5):+.6f} H1_ret={np.nanmean(m60):+.6f} resolve={avgr:.1f} H20_Sharpe={shrp20:.2f} p={pv:.4f} {sig}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
#  A6: Cross-Pair Return Dispersion — Vol Predictor
# ═══════════════════════════════════════════════════════════════════════════════
def a6():
    print("=" * 72)
    print("  A6: CROSS-PAIR RETURN DISPERSION → VOL [M1]")
    print("=" * 72)
    cs_disp = np.std(rets, axis=1)  # (T-1,)
    avg_abs = np.mean(np.abs(rets), axis=1)
    p80, p20 = np.percentile(cs_disp, 80), np.percentile(cs_disp, 20)
    
    for fw, fn in [(5,"M5"),(15,"M15"),(30,"M30"),(60,"H1")]:
        hi = np.where(cs_disp[:-fw] > p80)[0]
        lo = np.where(cs_disp[:-fw] < p20)[0]
        hv = np.array([np.mean(np.abs(rets[i:i+fw])) for i in hi if i+fw < T-1])
        lv = np.array([np.mean(np.abs(rets[i:i+fw])) for i in lo if i+fw < T-1])
        if len(hv) >= 5 and len(lv) >= 5:
            r = np.mean(hv)/(np.mean(lv)+1e-10)
            tst, pv = sp_stats.ttest_ind(hv, lv, equal_var=False)
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            print(f"    {fn} hi_vol={np.mean(hv):.6f} lo_vol={np.mean(lv):.6f} ratio={r:.3f}x t={tst:.2f} p={pv:.4f} {sig}")
    
    # By session
    print(f"\n  Session → H1 vol ratio (hi/lo disp):")
    for sn, sm in [("ASIAN",asia_mask(hour[1:])),("EUROPEAN",eu_mask(hour[1:])),("US",us_mask(hour[1:]))]:
        sm1 = sm[:-60]
        hi = np.where((cs_disp[:-60] > p80) & sm1)[0]
        lo = np.where((cs_disp[:-60] < p20) & sm1)[0]
        if len(hi) < 3 or len(lo) < 3: continue
        hv = np.array([np.mean(np.abs(rets[i:i+60])) for i in hi if i+60 < T-1])
        lv = np.array([np.mean(np.abs(rets[i:i+60])) for i in lo if i+60 < T-1])
        print(f"    {sn:8s} ratio={np.mean(hv)/(np.mean(lv)+1e-10):.2f}x n_hi={len(hi)} n_lo={len(lo)}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
#  A7: Return Dispersion Asymmetry
# ═══════════════════════════════════════════════════════════════════════════════
def a7():
    print("=" * 72)
    print("  A7: RETURN DISPERSION ASYMMETRY [M1]")
    print("=" * 72)
    n_up = np.sum(rets > 0, axis=1)
    
    for th in [5, 6]:
        maj = n_up >= th
        min_ = n_up <= 7 - th
        if maj.sum() < 10 or min_.sum() < 10:
            print(f"    {th}/7 maj: insufficient events (maj={maj.sum()} min={min_.sum()})")
            continue
        for fw, fn in [(5,"M5"),(30,"M30"),(60,"H1")]:
            mi = np.where(maj[:-fw])[0]; ni = np.where(min_[:-fw])[0]
            mv = np.array([basket_ret_from(t, fw) for t in mi if t+1+fw < T])
            nv = np.array([basket_ret_from(t, fw) for t in ni if t+1+fw < T])
            if len(mv) >= 5 and len(nv) >= 5:
                tst, pv = sp_stats.ttest_ind(mv, nv, equal_var=False)
                sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
                print(f"    {fn} {th}/7 maj={np.nanmean(mv):+.6f} min={np.nanmean(nv):+.6f} diff={np.nanmean(mv)-np.nanmean(nv):+.6f} t={tst:.3f} p={pv:.4f} {sig}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
#  BONUS: Cross-Session Energy Memory
# ═══════════════════════════════════════════════════════════════════════════════
def bonus():
    print("=" * 72)
    print("  BONUS: CROSS-SESSION ENERGY MEMORY [H1]")
    print("=" * 72)
    h1c = pd.DataFrame(close, index=times, columns=PAIRS).resample("1h").last().values.astype(np.float64)
    h1r = np.diff(np.log(h1c), axis=0)
    h1t = pd.date_range(times[0], times[-1], freq="1h")[:h1c.shape[0]]
    h1h = h1t.hour.values[1:]
    Th1 = h1r.shape[0]
    
    for pair in PAIRS:
        i = PIDX[pair]
        res = []
        for lag in [1, 2, 3]:
            am = (h1h >= 0) & (h1h < 7)
            ai = np.where(am)[0]
            ai = ai[ai + lag < Th1]
            for tn, th_s, th_e in [("EU_open",7,10),("US_open",13,16)]:
                tm = (h1h >= th_s) & (h1h < th_e)
                ti = np.where(tm)[0]
                pairs_found = []
                for a in ai:
                    for tgt in ti:
                        if 0 <= tgt - (a + lag) <= 2:
                            pairs_found.append((a, tgt, min(tgt+3, Th1)))
                            break
                if len(pairs_found) < 10: continue
                ar = np.array([np.mean(np.abs(h1r[a:a+lag, i])) for a,_,_ in pairs_found])
                er = np.array([np.mean(np.abs(h1r[ts:te, i])) for _,ts,te in pairs_found])
                if len(ar) >= 10:
                    corr = sp_stats.pearsonr(ar, er)[0]
                    med = np.median(ar)
                    hi = er[ar > med]; lo = er[ar <= med]
                    if len(hi) >= 5 and len(lo) >= 5:
                        tst, pv = sp_stats.ttest_ind(hi, lo, equal_var=False)
                        res.append((tn, lag, corr, np.mean(hi)/(np.mean(lo)+1e-10), tst, pv, len(pairs_found)))
        if res:
            print(f"  [{pair}]")
            for tn, lag, corr, ratio, tst, pv, n in res:
                sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
                print(f"    {tn:10s} lag={lag} corr={corr:.3f} ratio={ratio:.3f}x t={tst:.3f} p={pv:.4f} n={n} {sig}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
t_start = time.time()
print(f"Loaded {T} M1 bars x {len(PAIRS)} pairs")
print(f"Range: {times[0]} to {times[-1]}")
print(f"rets: {rets.shape}, es_aligned: {es_aligned.shape}, delta: {delta_aligned.shape}\n")

for name, fn in [("A1: Triangular Residual", a1),
                 ("A2: Cross-Pair Energy Divergence", a2),
                 ("A3: ES Hysteresis", a3),
                 ("A4: Energy Cascade", a4),
                 ("A5: Dispersion Components", a5),
                 ("A6: Return Dispersion → Vol", a6),
                 ("A7: Return Asymmetry", a7),
                 ("Bonus: Cross-Session Memory", bonus)]:
    t0 = time.time()
    fn()
    print(f"  [{name}] {time.time()-t0:.1f}s\n")

print(f"Total: {time.time()-t_start:.1f}s")
