"""MarketStateVector predictive validation.
Tests whether MSV-derived features (regime, risk score, entry gating)
have any predictive power for forward returns.
"""

import sys, os, time, numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parents[2])
cd_root = os.path.join(project_root, "currency_decomposition")
if cd_root not in sys.path:
    sys.path.insert(0, cd_root)
os.chdir(project_root)

import MetaTrader5 as mt5
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

from state.market_state import MarketStateVector

ALL_PAIRS = [
    "EURUSD", "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPUSD", "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "USDJPY", "USDCHF", "USDCAD",
    "AUDCAD", "AUDNZD",
]

HORIZONS = [5, 15, 30, 60]
DAYS = 14

def load_returns(pairs, days=DAYS):
    end = datetime.now()
    start = end - timedelta(days=days)
    print(f"Loading M5 data: {start.date()} to {end.date()} ({days} days)")

    all_data = {}
    for pair in pairs:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is None or len(rates) == 0:
            print(f"  [SKIP] {pair} -> no data")
            continue
        all_data[pair] = rates
        print(f"  [{len(all_data)}/{len(pairs)}] {pair} -> {len(rates)} bars")

    if len(all_data) < 3:
        raise RuntimeError(f"Too few pairs loaded: {len(all_data)}")
    print(f"\nLoaded {len(all_data)} pairs")
    return all_data

def main():
    pairs = [p for p in ALL_PAIRS]
    all_data = load_returns(pairs)

    N = min(len(v) for v in all_data.values())
    print(f"Common bar count: {N}")

    ms = MarketStateVector(history_size=50)
    records = []

    for idx in range(N):
        returns = {}
        for pair in all_data:
            bar = all_data[pair][idx]
            if idx == 0:
                ret = 0.0
            else:
                prev = all_data[pair][idx - 1]["close"]
                curr = all_data[pair][idx]["close"]
                ret = (curr / prev - 1) if prev > 0 else 0.0
            ret = np.clip(ret, -0.05, 0.05)
            returns[pair] = ret

        now = float(bar["time"])
        snapshot = ms.update(returns, timestamp=now)

        reg = ms.regime(snapshot)
        rs = ms.risk_score(snapshot)
        entry_ok, entry_reason = ms.entry_allowed(snapshot)
        disp = snapshot.network.dispersion
        agree = snapshot.network.agreement
        shock = abs(snapshot.residual.residual_shock)
        energy = snapshot.residual.residual_energy

        fwd_ret = {}
        for h in HORIZONS:
            fwd_idx = idx + h
            if fwd_idx < N:
                fwd_pnl = []
                for pair in all_data:
                    fwd_close = all_data[pair][fwd_idx]["close"]
                    cur_close = all_data[pair][idx]["close"]
                    r = (fwd_close / cur_close - 1) if cur_close > 0 else 0.0
                    fwd_pnl.append(r)
                fwd_ret[h] = float(np.mean(fwd_pnl))
            else:
                fwd_ret[h] = None

        records.append({
            "ts": now, "regime": reg, "risk": rs,
            "entry_ok": entry_ok, "disp": disp,
            "agree": agree, "shock": shock, "energy": energy,
            "fwd_ret": fwd_ret,
        })

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{N}")

    print(f"\nTotal records: {len(records)}")

    # --- REGIME ---
    regimes = sorted(set(r["regime"] for r in records))
    print(f"\nRegimes: {regimes}")
    print(f"{'Regime':16s} {'Hrzm':4s} {'n':6s} {'MeanRet':>10s} {'Sharpe':>7s} {'Pos%':>6s} {'t':>7s}")
    print("-" * 60)
    for h in HORIZONS:
        for reg in regimes:
            vals = [r["fwd_ret"][h] for r in records
                    if r["fwd_ret"][h] is not None and r["regime"] == reg]
            if len(vals) < 5:
                continue
            mean_ret = float(np.mean(vals))
            std_ret = float(np.std(vals))
            sharpe = (mean_ret / std_ret) * np.sqrt(12 * 24) if std_ret > 0 else 0.0
            pos_pct = sum(1 for v in vals if v > 0) / len(vals) * 100
            t_stat = mean_ret / (std_ret / np.sqrt(len(vals))) if std_ret > 0 else 0.0
            print(f"{reg:16s} {h:4d}m {len(vals):6d} {mean_ret:+10.6f} {sharpe:+7.3f} {pos_pct:5.1f}% {t_stat:+7.2f}")

    # --- RISK SCORE BINS ---
    print(f"\n--- Risk Score Bins ---")
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    print(f"{'Risk':16s} {'Hrzm':4s} {'n':6s} {'MeanRet':>10s} {'Sharpe':>7s} {'Pos%':>6s} {'t':>7s}")
    print("-" * 60)
    for h in HORIZONS:
        for lo, hi in bins:
            vals = [r["fwd_ret"][h] for r in records
                    if r["fwd_ret"][h] is not None and lo <= r["risk"] < hi]
            if len(vals) < 5:
                continue
            mean_ret = float(np.mean(vals))
            std_ret = float(np.std(vals))
            sharpe = (mean_ret / std_ret) * np.sqrt(12 * 24) if std_ret > 0 else 0.0
            pos_pct = sum(1 for v in vals if v > 0) / len(vals) * 100
            t_stat = mean_ret / (std_ret / np.sqrt(len(vals))) if std_ret > 0 else 0.0
            print(f"[{lo:.1f},{hi:.1f})  {h:4d}m {len(vals):6d} {mean_ret:+10.6f} {sharpe:+7.3f} {pos_pct:5.1f}% {t_stat:+7.2f}")

    # --- ENTRY GATE ---
    print(f"\n--- Entry Gate ---")
    print(f"{'Gate':16s} {'Hrzm':4s} {'n':6s} {'MeanRet':>10s} {'Sharpe':>7s} {'Pos%':>6s} {'t':>7s}")
    print("-" * 60)
    for h in HORIZONS:
        for gate in [True, False]:
            vals = [r["fwd_ret"][h] for r in records
                    if r["fwd_ret"][h] is not None and r["entry_ok"] == gate]
            if len(vals) < 5:
                continue
            mean_ret = float(np.mean(vals))
            std_ret = float(np.std(vals))
            sharpe = (mean_ret / std_ret) * np.sqrt(12 * 24) if std_ret > 0 else 0.0
            pos_pct = sum(1 for v in vals if v > 0) / len(vals) * 100
            t_stat = mean_ret / (std_ret / np.sqrt(len(vals))) if std_ret > 0 else 0.0
            label = "ALLOWED" if gate else "BLOCKED"
            print(f"{label:16s} {h:4d}m {len(vals):6d} {mean_ret:+10.6f} {sharpe:+7.3f} {pos_pct:5.1f}% {t_stat:+7.2f}")

    # --- DISPERSION QUANTILES ---
    print(f"\n--- Dispersion Quantiles ---")
    disp_vals = sorted([r["disp"] for r in records])
    if disp_vals:
        qs = [np.percentile(disp_vals, p) for p in [25, 50, 75]]
        edges = [-1e-6] + list(qs) + [1.0]
    print(f"{'Disp':16s} {'Hrzm':4s} {'n':6s} {'MeanRet':>10s} {'Sharpe':>7s} {'Pos%':>6s} {'t':>7s}")
    print("-" * 60)
    for h in HORIZONS:
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            vals = [r["fwd_ret"][h] for r in records
                    if r["fwd_ret"][h] is not None and lo <= r["disp"] < hi]
            if len(vals) < 5:
                continue
            mean_ret = float(np.mean(vals))
            std_ret = float(np.std(vals))
            sharpe = (mean_ret / std_ret) * np.sqrt(12 * 24) if std_ret > 0 else 0.0
            pos_pct = sum(1 for v in vals if v > 0) / len(vals) * 100
            t_stat = mean_ret / (std_ret / np.sqrt(len(vals))) if std_ret > 0 else 0.0
            print(f"[{lo:.4f},{hi:.4f}) {h:4d}m {len(vals):6d} {mean_ret:+10.6f} {sharpe:+7.3f} {pos_pct:5.1f}% {t_stat:+7.2f}")

    mt5.shutdown()

if __name__ == "__main__":
    main()
