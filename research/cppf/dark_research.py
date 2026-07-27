"""Dark Research: Triangular decomposition, session filters, multi-TF z-score.

Usage:
    python research/cppf/dark_research.py                     # Full analysis
    python research/cppf/dark_research.py --quick              # Baseline only
    python research/cppf/dark_research.py --triangular         # Triangular only
    python research/cppf/dark_research.py --sessions           # Session filter only
    python research/cppf/dark_research.py --multitf            # Multi-TF only
    python research/cppf/dark_research.py --param-sweep        # Parameter sweep
"""

import argparse, time, sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

DATA_DIR = Path(__file__).parent.parent.parent / "research" / "phase_dislocation" / "dukascopy_data"

# Cross pairs and their constituent legs for triangular decomposition
# cross = base/quote, so cross ~ base_leg / quote_leg
TRIANGULAR = {
    "gbpnzd": {"base": "gbpusd", "quote": "nzdusd"},
    "eurnzd": {"base": "eurusd", "quote": "nzdusd"},
    "gbpaud": {"base": "gbpusd", "quote": "audusd"},
    "euraud": {"base": "eurusd", "quote": "audusd"},
    "gbpcad": {"base": "gbpusd", "quote": "usdcad"},
    "audnzd": {"base": "audusd", "quote": "nzdusd"},
}

# Session hours (UTC)
SESSIONS = {
    "asian":  {"hours": range(0, 8),   "label": "Asian (00-07 UTC)"},
    "london": {"hours": range(8, 16),  "label": "London (08-15 UTC)"},
    "ny":     {"hours": range(13, 21), "label": "NY (13-20 UTC)"},
    "sydney": {"hours": range(21, 24), "label": "Sydney (21-23 UTC)"},
}

SPREAD_PIPS = {
    "gbpnzd": 5.0, "eurnzd": 4.0, "gbpaud": 4.0,
    "euraud": 3.0, "gbpcad": 4.0, "audnzd": 3.0,
}

STOP_A, TRIG_A, GAP_A, MAX_BARS = 3.0, 0.5, 0.1, 54
Z_WINDOW, ATR_WINDOW = 50, 20


# ── Data loading ──────────────────────────────────────────────────────

def load_all_pairs():
    """Load all pairs into a dict of DataFrames."""
    pairs = list(TRIANGULAR.keys())
    legs = set()
    for v in TRIANGULAR.values():
        legs.add(v["base"])
        legs.add(v["quote"])
    all_p = pairs + list(legs)
    dfs = {}
    for p in all_p:
        df = pd.read_parquet(DATA_DIR / f"{p}.parquet")
        df = df.set_index("timestamp").astype(np.float64)
        dfs[p] = df
    # Combine into single DataFrame with multi-index columns: (pair, metric)
    combined = pd.concat(dfs.values(), axis=1, keys=dfs.keys())
    combined.sort_index(axis=1, inplace=True)
    return combined


def z_score(close, window=Z_WINDOW):
    """z-score of close returns using historical mean/std (shift=1 style)."""
    ret = close.diff()
    mu = ret.shift(1).rolling(window).mean()
    sigma = ret.shift(1).rolling(window).std().clip(1e-10)
    z = (ret - mu) / sigma
    return z, ret


def compute_atr(df, pair, atr_window=ATR_WINDOW):
    """ATR for a given pair."""
    h = df[pair, "high"]
    l_ = df[pair, "low"]
    atr = (h - l_).shift(1).rolling(atr_window).mean().clip(1e-10)
    return atr


# ── Core backtest logic ────────────────────────────────────────────────

def backtest_core(df, pair, z_values, session_filter=None, z_thresh=2.5,
                  stop_a=STOP_A, trig_a=TRIG_A, gap_a=GAP_A,
                  max_bars=MAX_BARS, spread_pips=None):
    """Core backtest engine. Returns DataFrame of trades."""
    c = df[pair, "close"]
    h = df[pair, "high"]
    l_ = df[pair, "low"]
    z = z_values  # already computed
    atr = compute_atr(df, pair)

    mask = z.notna() & atr.notna() & (z.abs() >= z_thresh)
    idxs = np.where(mask.values)[0] if hasattr(mask, 'values') else np.where(mask)[0]

    trades = []
    in_trade_until = -1
    for pos in idxs:
        if pos <= in_trade_until:
            continue
        if pos + 2 >= len(df):
            continue

        hour = df.index[pos].hour
        if session_filter and hour not in SESSIONS[session_filter]["hours"]:
            continue

        direction = -1 if z.iloc[pos] > 0 else 1
        entry = c.iloc[pos]
        atr_v = atr.iloc[pos]
        s = stop_a * atr_v
        tg = trig_a * atr_v
        gp = gap_a * atr_v

        best = entry
        exited = False
        for j in range(1, max_bars + 1):
            bp = pos + j
            if bp >= len(df):
                break

            if direction == 1:  # long
                if h.iloc[bp] > best:
                    best = h.iloc[bp]
                sl = entry - s
                if best - entry > tg:
                    sl = best - gp
                if l_.iloc[bp] <= sl:
                    pnl = (sl - entry) * direction
                    if spread_pips:
                        pnl -= spread_pips * 0.0001
                    trades.append({
                        "entry_bar": int(pos), "exit_bar": int(bp),
                        "direction": direction, "entry": float(entry),
                        "exit": float(sl), "pnl": float(pnl),
                        "z": float(z.iloc[pos]), "atr": float(atr_v),
                        "exit_reason": "stop", "hour": int(hour),
                        "timestamp": df.index[bp],
                    })
                    in_trade_until = bp
                    exited = True
                    break
            else:  # short
                if l_.iloc[bp] < best:
                    best = l_.iloc[bp]
                sl = entry + s
                if entry - best > tg:
                    sl = best + gp
                if h.iloc[bp] >= sl:
                    pnl = (sl - entry) * direction
                    # For short: (sl - entry) * direction = (sl - entry) * (-1) = entry - sl
                    # Wait, let me re-check. direction = -1 for short.
                    # pnl = (sl - entry) * direction = (sl - entry) * (-1) = entry - sl
                    # When stop hit on short: sl = entry + s > entry, so sl - entry > 0
                    # pnl = (positive) * (-1) = negative. Correct, stop loss is negative.
                    if spread_pips:
                        pnl -= spread_pips * 0.0001
                    trades.append({
                        "entry_bar": int(pos), "exit_bar": int(bp),
                        "direction": direction, "entry": float(entry),
                        "exit": float(sl), "pnl": float(pnl),
                        "z": float(z.iloc[pos]), "atr": float(atr_v),
                        "exit_reason": "stop", "hour": int(hour),
                        "timestamp": df.index[bp],
                    })
                    in_trade_until = bp
                    exited = True
                    break

        if not exited:
            eb = min(pos + max_bars, len(df) - 1)
            exit_px = c.iloc[eb]
            pnl = (exit_px - entry) * direction
            if spread_pips:
                pnl -= spread_pips * 0.0001
            trades.append({
                "entry_bar": int(pos), "exit_bar": int(eb),
                "direction": direction, "entry": float(entry),
                "exit": float(exit_px), "pnl": float(pnl),
                "z": float(z.iloc[pos]), "atr": float(atr_v),
                "exit_reason": "expiry", "hour": int(hour),
                "timestamp": df.index[eb],
            })
            in_trade_until = eb

    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ── Triangular decomposition ──────────────────────────────────────────

def compute_triangular_z(df, pair):
    """Compute 'pure cross' z-score: residual after removing leg contribution.

    For GBPNZD = GBPUSD / NZDUSD:
      r_cross ≈ r_base - r_quote  (in log returns)
      z_base = z-score of r_base
      z_quote = z-score of r_quote
      z_cross = z-score of r_cross
      residual_z = z_cross - β₁·z_base - β₂·z_quote
      where β₁, β₂ are rolling regression coefficients

    Returns: (z_cross_raw, z_residual, z_base, z_quote)
    """
    info = TRIANGULAR[pair]

    z_cross_raw, r_cross = z_score(df[pair, "close"])
    z_base, r_base = z_score(df[info["base"], "close"])
    z_quote, r_quote = z_score(df[info["quote"], "close"])

    # Rolling regression: r_cross ~ β₁·r_base + β₂·r_quote
    # Use expanding window of 500 bars at start, then rolling 500
    window = 500
    half = window // 2
    n = len(df)
    residual_z = pd.Series(np.nan, index=df.index)

    for i in range(window, n):
        r_base_slice = r_base.iloc[i - window:i].fillna(0).values
        r_quote_slice = r_quote.iloc[i - window:i].fillna(0).values
        X = np.column_stack([r_base_slice, r_quote_slice])
        y = r_cross.iloc[i - window:i].fillna(0).values
        # Remove any remaining NaNs
        valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        if valid.sum() < half:
            continue
        Xv, yv = X[valid], y[valid]
        try:
            beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
            predicted = beta[0] * r_base.iloc[i] + beta[1] * r_quote.iloc[i]
            residual = r_cross.iloc[i] - predicted
            # z-score the residual against its own history
            res_history = yv - Xv @ beta
            res_mu = np.mean(res_history)
            res_std = np.std(res_history, ddof=1)
            if res_std > 1e-10:
                residual_z.iloc[i] = (residual - res_mu) / res_std
        except np.linalg.LinAlgError:
            continue

    return z_cross_raw, residual_z, z_base, z_quote


# ── Session filter ─────────────────────────────────────────────────────

def get_session(hour_utc):
    for name, info in SESSIONS.items():
        if hour_utc in info["hours"]:
            return name
    return "closed"


# ── Multi-TF z-score filter ───────────────────────────────────────────

def compute_mtf_z(df, pair):
    """Compute z-scores on M5 and M15 by resampling."""
    c = df[pair, "close"]
    # M5
    c5 = c.resample("5min").last().dropna()
    z5, r5 = z_score(c5)
    # Map back to M1 index
    z5_m1 = z5.reindex(c.index, method="ffill")
    # M15
    c15 = c.resample("15min").last().dropna()
    z15, r15 = z_score(c15)
    z15_m1 = z15.reindex(c.index, method="ffill")
    return z5_m1, z15_m1


# ── Report helpers ─────────────────────────────────────────────────────

def print_trade_stats(trades, label="", pair=""):
    if len(trades) == 0:
        print(f"  {label:30s}: 0 trades")
        return
    wr = (trades["pnl"] > 0).mean()
    net_pips = trades["pnl"].sum() * 10000
    avg_pips = trades["pnl"].mean() * 10000
    n_stop = (trades["exit_reason"] == "stop").sum()
    n_exp = (trades["exit_reason"] == "expiry").sum()
    avg_w = trades.loc[trades["pnl"] > 0, "pnl"].mean() * 10000 if (trades["pnl"] > 0).any() else 0
    avg_l = trades.loc[trades["pnl"] <= 0, "pnl"].mean() * 10000 if (trades["pnl"] <= 0).any() else 0
    gross_w = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_l = trades.loc[trades["pnl"] <= 0, "pnl"].sum()
    pf = abs(gross_w / max(gross_l, 1e-12))
    print(f"  {label:30s}: n={len(trades):>5d}  WR={wr:>6.2%}  "
          f"PnL={net_pips:>+9.1f}p  avg={avg_pips:>+7.1f}p  "
          f"PF={pf:>5.2f}  stop={n_stop} exp={n_exp}")
    if avg_w != 0 and avg_l != 0:
        print(f"  {'':30s}  avg_win={avg_w:>+7.1f}p  avg_loss={avg_l:>+7.1f}p  "
              f"payoff={abs(avg_w/avg_l):>5.2f}")


# ── Main analyses ──────────────────────────────────────────────────────

def run_baseline(df, pair):
    """Baseline V2+z backtest with and without spread."""
    print(f"\n{'='*60}")
    print(f"BASELINE: {pair.upper()}")
    print(f"{'='*60}")

    z, _ = z_score(df[pair, "close"])
    spread = SPREAD_PIPS.get(pair, 5.0)

    for sp, label in [(None, "No spread"), (spread, f"With {spread:.0f}p spread")]:
        trades = backtest_core(df, pair, z, spread_pips=sp)
        print_trade_stats(trades, label)

    return z


def run_triangular(df, pair, z_raw):
    """Triangular decomposition test."""
    print(f"\n{'='*60}")
    print(f"TRIANGULAR DECOMPOSITION: {pair.upper()}")
    print(f"{'='*60}")

    info = TRIANGULAR[pair]
    z_cross_raw, z_residual, z_base, z_quote = compute_triangular_z(df, pair)
    spread = SPREAD_PIPS.get(pair, 5.0)

    # 1. Baseline on raw z (for reference)
    trades_raw = backtest_core(df, pair, z_raw, spread_pips=spread)
    print_trade_stats(trades_raw, f"Raw z (baseline)")

    # 2. Trade only on residual z
    trades_resid = backtest_core(df, pair, z_residual, spread_pips=spread)
    print_trade_stats(trades_resid, f"Residual z (pure cross)")

    # 3. Trade only when both raw z AND residual z are extreme
    z_combined = pd.Series(np.nan, index=df.index)
    valid = z_raw.notna() & z_residual.notna()
    z_combined[valid] = np.where(
        (z_raw[valid].abs() >= 2.5) & (z_residual[valid].abs() >= 2.5),
        z_raw[valid], np.nan
    )
    trades_both = backtest_core(df, pair, z_combined, spread_pips=spread)
    print_trade_stats(trades_both, f"Raw AND residual extreme")

    # 4. Trade when residual is extreme but raw is NOT (pure cross dislocation)
    z_cross_only = pd.Series(np.nan, index=df.index)
    valid = z_raw.notna() & z_residual.notna()
    z_cross_only[valid] = np.where(
        (z_residual[valid].abs() >= 2.5) & (z_raw[valid].abs() < 2.5),
        z_residual[valid], np.nan
    )
    trades_cross = backtest_core(df, pair, z_cross_only, spread_pips=spread)
    print_trade_stats(trades_cross, f"Residual extreme, raw not")

    # 5. Trade when raw is extreme but residual is NOT (broad flow only)
    z_flow_only = pd.Series(np.nan, index=df.index)
    valid = z_raw.notna() & z_residual.notna()
    z_flow_only[valid] = np.where(
        (z_raw[valid].abs() >= 2.5) & (z_residual[valid].abs() < 2.5),
        z_raw[valid], np.nan
    )
    trades_flow = backtest_core(df, pair, z_flow_only, spread_pips=spread)
    print_trade_stats(trades_flow, f"Raw extreme, residual not (broad flow)")

    return z_residual


def run_sessions(df, pair, z_values):
    """Test strategy per session."""
    print(f"\n{'='*60}")
    print(f"SESSION FILTERS: {pair.upper()}")
    print(f"{'='*60}")

    spread = SPREAD_PIPS.get(pair, 5.0)

    # All hours
    trades_all = backtest_core(df, pair, z_values, spread_pips=spread)
    print_trade_stats(trades_all, "All hours")

    for sess_name in SESSIONS:
        trades = backtest_core(df, pair, z_values,
                               session_filter=sess_name, spread_pips=spread)
        label = SESSIONS[sess_name]["label"]
        print_trade_stats(trades, label)


def run_multitf(df, pair, z_values):
    """Multi-timeframe z-score filter test."""
    print(f"\n{'='*60}")
    print(f"MULTI-TF Z-SCORE FILTER: {pair.upper()}")
    print(f"{'='*60}")

    spread = SPREAD_PIPS.get(pair, 5.0)
    z5, z15 = compute_mtf_z(df, pair)

    # Baseline
    trades_all = backtest_core(df, pair, z_values, spread_pips=spread)
    print_trade_stats(trades_all, "Baseline (M1 only)")

    # Only trade when M5 does NOT confirm (noise filter)
    z_noise = pd.Series(np.nan, index=df.index)
    valid = z_values.notna() & z5.notna()
    z_noise[valid] = np.where(
        (z_values[valid].abs() >= 2.5) & (z5[valid].abs() < 1.5),
        z_values[valid], np.nan
    )
    trades_noise = backtest_core(df, pair, z_noise, spread_pips=spread)
    print_trade_stats(trades_noise, "M1 extreme, M5 quiet (noise)")

    # Only trade when M5 and M1 agree (trend filter)
    z_trend = pd.Series(np.nan, index=df.index)
    valid = z_values.notna() & z5.notna()
    z_trend[valid] = np.where(
        (z_values[valid].abs() >= 2.5) & (z5[valid].abs() >= 1.5),
        z_values[valid], np.nan
    )
    trades_trend = backtest_core(df, pair, z_trend, spread_pips=spread)
    print_trade_stats(trades_trend, "M1+M5 both extreme (trend skip)")


def run_param_sweep(df, pair):
    """Sweep parameters to find positive expectancy at given spread."""
    print(f"\n{'='*60}")
    print(f"PARAMETER SWEEP: {pair.upper()}")
    print(f"{'='*60}")

    z, _ = z_score(df[pair, "close"])
    spread = SPREAD_PIPS.get(pair, 5.0)

    results = []
    configs = list(product(
        [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],  # z_thresh
        [1.0, 2.0, 3.0, 4.0, 5.0],          # stop_a
        [0.3, 0.5, 0.7, 1.0],                # trig_a
        [0.05, 0.1, 0.15, 0.2],              # gap_a
    ))

    print(f"  Sweeping {len(configs)} parameter combinations...")
    for zt, sa, ta, ga in configs:
        trades = backtest_core(df, pair, z, z_thresh=zt,
                               stop_a=sa, trig_a=ta, gap_a=ga,
                               spread_pips=spread)
        if len(trades) < 10:
            continue
        net = trades["pnl"].sum()
        wr = (trades["pnl"] > 0).mean()
        results.append({
            "z_thresh": zt, "stop_a": sa, "trig_a": ta, "gap_a": ga,
            "n": len(trades), "wr": wr, "net_pnl": net,
        })

    df_r = pd.DataFrame(results)
    if len(df_r) == 0:
        print("  No valid configs found.")
        return

    # Top by net PnL
    top = df_r.sort_values("net_pnl", ascending=False).head(10)
    print(f"\n  Top 10 configurations by net PnL:")
    print(f"  {'z':>5s} {'stop':>5s} {'trig':>5s} {'gap':>5s}  "
          f"{'n':>5s} {'WR':>6s} {'PnL':>8s}")
    print(f"  {'-'*5} {'-'*5} {'-'*5} {'-'*5}  {'-'*5} {'-'*6} {'-'*8}")
    for _, row in top.iterrows():
        print(f"  {row['z_thresh']:>5.1f} {row['stop_a']:>5.1f} {row['trig_a']:>5.1f} "
              f"{row['gap_a']:>5.2f}  {row['n']:>5d} {row['wr']:>6.1%} "
              f"{row['net_pnl']:>+8.2f}")

    # Also show by z_thresh
    print(f"\n  By z_thresh (best config each):")
    for zt in sorted(df_r["z_thresh"].unique()):
        sub = df_r[df_r["z_thresh"] == zt]
        best = sub.loc[sub["net_pnl"].idxmax()]
        print(f"  z>={zt:3.1f}: n={best['n']:>5d}  WR={best['wr']:>5.1%}  "
              f"PnL={best['net_pnl']:>+8.2f}p  "
              f"[stop={best['stop_a']:.1f} trig={best['trig_a']:.1f} gap={best['gap_a']:.2f}]")

    return df_r


def run_combined_analysis(df, pair):
    """Run combined triangular + session filter with parameter sweep."""
    print(f"\n{'='*60}")
    print(f"COMBINED ANALYSIS: {pair.upper()}")
    print(f"{'='*60}")

    spread = SPREAD_PIPS.get(pair, 5.0)
    z_raw, _ = z_score(df[pair, "close"])
    _, z_resid, _, _ = compute_triangular_z(df, pair)

    # Build combined signal: raw AND residual extreme
    z_combined = pd.Series(np.nan, index=df.index)
    valid = z_raw.notna() & z_resid.notna()
    z_combined[valid] = np.where(
        (z_raw[valid].abs() >= 2.5) & (z_resid[valid].abs() >= 2.5),
        z_raw[valid], np.nan
    )

    # Test Sydney session only
    for sess_name in ["sydney"]:
        trades = backtest_core(df, pair, z_combined,
                               session_filter=sess_name, spread_pips=spread)
        label = f"Triangular + {SESSIONS[sess_name]['label']}"
        print_trade_stats(trades, label)

    # Test all hours
    trades_all = backtest_core(df, pair, z_combined, spread_pips=spread)
    print_trade_stats(trades_all, "Triangular + all hours")

    # Parameter sweep on combined signal
    print(f"\n  Parameter sweep (combined signal, all hours):")
    results = []
    configs = list(product(
        [2.0, 2.5, 3.0, 3.5],      # z_thresh
        [2.0, 3.0, 4.0, 5.0],       # stop_a
        [0.3, 0.5, 0.7],             # trig_a
        [0.05, 0.1, 0.15, 0.2],     # gap_a
    ))
    for zt, sa, ta, ga in configs:
        # Rebuild combined signal with new z_thresh
        z_c = pd.Series(np.nan, index=df.index)
        valid = z_raw.notna() & z_resid.notna()
        z_c[valid] = np.where(
            (z_raw[valid].abs() >= zt) & (z_resid[valid].abs() >= zt),
            z_raw[valid], np.nan
        )
        trades = backtest_core(df, pair, z_c, z_thresh=zt,
                               stop_a=sa, trig_a=ta, gap_a=ga,
                               spread_pips=spread)
        if len(trades) < 5:
            continue
        net = trades["pnl"].sum() * 10000
        wr = (trades["pnl"] > 0).mean()
        results.append({
            "z": zt, "stop": sa, "trig": ta, "gap": ga,
            "n": len(trades), "wr": wr, "pnl_pips": net,
        })

    df_r = pd.DataFrame(results)
    if len(df_r) == 0:
        print("  No valid configs found.")
        return

    top = df_r.sort_values("pnl_pips", ascending=False).head(10)
    print(f"\n  Top 10 configs by net PnL:")
    print(f"  {'z':>4s} {'stop':>5s} {'trig':>5s} {'gap':>5s}  "
          f"{'n':>5s} {'WR':>6s} {'PnL':>8s}")
    print(f"  {'-'*4} {'-'*5} {'-'*5} {'-'*5}  {'-'*5} {'-'*6} {'-'*8}")
    for _, row in top.iterrows():
        print(f"  {row['z']:>4.1f} {row['stop']:>5.1f} {row['trig']:>5.1f} "
              f"{row['gap']:>5.2f}  {int(row['n']):>5d} {row['wr']:>6.1%} "
              f"{row['pnl_pips']:>+8.1f}")


def run_all_pairs(quick=False, triangular=False, sessions=False,
                  multitf=False, param_sweep=False):
    """Run selected analyses for all cross pairs."""
    print("Loading all pair data...")
    t0 = time.time()
    df = load_all_pairs()
    print(f"  {len(df)} bars loaded in {time.time()-t0:.1f}s")

    pairs = list(TRIANGULAR.keys())

    for pair in pairs:
        if quick:
            run_baseline(df, pair)
        elif triangular:
            run_baseline(df, pair)  # baseline first
            z_raw, _ = z_score(df[pair, "close"])
            run_triangular(df, pair, z_raw)
        elif sessions:
            z, _ = z_score(df[pair, "close"])
            run_sessions(df, pair, z)
        elif multitf:
            z, _ = z_score(df[pair, "close"])
            run_multitf(df, pair, z)
        elif param_sweep:
            run_param_sweep(df, pair)
        else:
            # Full analysis
            run_baseline(df, pair)
            z_raw, _ = z_score(df[pair, "close"])
            run_triangular(df, pair, z_raw)
            run_sessions(df, pair, z_raw)
            run_multitf(df, pair, z_raw)
            run_combined_analysis(df, pair)
            print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dark Research: V2+z analysis")
    ap.add_argument("--quick", action="store_true", help="Baseline only")
    ap.add_argument("--triangular", action="store_true", help="Triangular only")
    ap.add_argument("--sessions", action="store_true", help="Session filter only")
    ap.add_argument("--multitf", action="store_true", help="Multi-TF only")
    ap.add_argument("--param-sweep", action="store_true", help="Parameter sweep")
    ap.add_argument("--combined", action="store_true", help="Combined triangular + session + sweep")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="Specific pairs (default: all cross pairs)")
    args = ap.parse_args()

    mode = "full"
    if args.quick: mode = "quick"
    elif args.triangular: mode = "triangular"
    elif args.sessions: mode = "sessions"
    elif args.multitf: mode = "multitf"
    elif args.param_sweep: mode = "param_sweep"
    elif args.combined: mode = "combined"

    run_all_pairs(
        quick=(mode == "quick"),
        triangular=(mode == "triangular"),
        sessions=(mode == "sessions"),
        multitf=(mode == "multitf"),
        param_sweep=(mode == "param_sweep"),
    )
