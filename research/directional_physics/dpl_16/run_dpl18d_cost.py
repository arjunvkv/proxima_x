"""DPL-18D: Cost Survivability Audit.
Final pre-deployment test: does high-confidence TPI survive spread + slippage?

Run: python -u proxima_x/research/directional_physics/dpl_16/run_dpl18d_cost.py
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, "C:/Trading/Agentic_Trading/proxima_x")
from research.directional_physics.dpl_16.core.msl_engine import (
    load_ticks, load_m5, align_ticks_to_bars,
    compute_tpi, aggregate_to_bars_vectorized,
    compute_directional_labels_vect,
)

REPORT_DIR = "C:/Trading/Agentic_Trading/research/directional_physics/dpl_16/reports"
os.makedirs(REPORT_DIR, exist_ok=True)
SYMBOLS = ["EURJPY", "USDJPY", "EURUSD", "GBPUSD"]

SESSION = {"asia": (0, 9), "london": (9, 17), "ny": (13, 22)}

# Estimated broker spreads in pips (conservative)
SPREAD_PIPS = {"EURJPY": 1.2, "USDJPY": 1.0, "EURUSD": 0.8, "GBPUSD": 1.4}

def load_data(symbol):
    ticks = load_ticks(symbol)
    m5 = load_m5(symbol)
    close = m5["close"]
    ts_m5 = m5["timestamp"]
    n = len(close)
    starts, ends, tick_sec = align_ticks_to_bars(ticks["timestamp"], ts_m5)
    tick_bar_idx = np.searchsorted(ts_m5, tick_sec, side="right") - 1
    tick_bar_idx = np.clip(tick_bar_idx, 0, n - 1)
    tpi_feats = compute_tpi(ticks["mid"])
    bar_tpi = aggregate_to_bars_vectorized(tpi_feats["tpi_200"], tick_bar_idx, n)

    # Get spread data from ticks aggregated to bars
    tick_spread = ticks["spread"]
    bar_spread = aggregate_to_bars_vectorized(tick_spread, tick_bar_idx, n)

    first_valid = np.where(~np.isnan(bar_tpi))[0]
    if len(first_valid) == 0: return None
    s = first_valid[0]; e = min(first_valid[-1] + 1, n - 1)
    return {"symbol": symbol, "close": close[s:e], "tpi": bar_tpi[s:e],
            "ts": ts_m5[s:e], "spread": bar_spread[s:e], "n": e - s}

def session_hour(ts_sec):
    return (ts_sec // 3600) % 24

def pips_to_log(pips, price):
    """Convert pips to log return. 1 pip = 0.0001 for most pairs, 0.01 for JPY pairs."""
    pip_size = np.where(np.array(price) > 10, 0.01, 0.0001)
    return pips * pip_size / np.array(price)

def log_to_pips(log_ret, price):
    """Convert log return to pips."""
    pip_size = 0.01 if price > 10 else 0.0001
    return log_ret * price / pip_size

if __name__ == "__main__":
    print("=" * 65)
    print("DPL-18D: Cost Survivability Audit")
    print("=" * 65)

    all_data = {}
    for sym in SYMBOLS:
        d = load_data(sym)
        if d: all_data[sym] = d

    results = {}
    for sym, data in all_data.items():
        tpi = data["tpi"]
        close = data["close"]
        ts = data["ts"]
        spread_bar = data["spread"]
        n = len(tpi)
        tpi_mag = np.abs(tpi)
        tpi_sign = np.where(tpi > 0, 1, np.where(tpi < 0, -1, 0)).astype(float)
        labels = compute_directional_labels_vect(close)
        hours = session_hour(ts)

        # Forward return over 1 bar
        log_close = np.log(close)
        fwd_ret_1 = np.full(n, np.nan)
        fwd_ret_1[:-1] = log_close[1:] - log_close[:-1]

        # Valid bars
        valid = (tpi != 0) & ~np.isnan(tpi) & ~np.isnan(labels) & ~np.isnan(fwd_ret_1) & ~np.isnan(spread_bar)
        if np.sum(valid) < 10:
            continue

        # Cost: fixed spread + slippage per trade in pips, converted to log-return per bar
        spread_pips_est = SPREAD_PIPS[sym]
        cost_per_trade_pips = spread_pips_est * 1.5  # spread + 50% slippage
        # Convert to log-return using avg close price
        avg_close = np.mean(close[valid])
        cost_per_trade_log = pips_to_log(cost_per_trade_pips, avg_close)

        # Conf thresholds
        mag_v = tpi_mag[valid]
        q75 = np.percentile(mag_v, 75)
        q90 = np.percentile(mag_v, 90)

        sym_res = {}

        for filter_name, thresh in [("Q4", q75), ("p90", q90)]:
            for session_name, session_filter in [
                ("all", slice(None)),
                ("asia", hours[valid] < 9),
                ("london", (hours[valid] >= 9) & (hours[valid] < 17)),
                ("ny", (hours[valid] >= 13) & (hours[valid] < 22)),
            ]:
                # Apply both filters
                if session_name == "all":
                    mask = tpi_mag[valid] >= thresh
                else:
                    mask = (tpi_mag[valid] >= thresh) & session_filter

                n_trades = int(np.sum(mask))
                if n_trades < 5: continue

                tpi_v = tpi_sign[valid][mask]
                fwd_v = fwd_ret_1[valid][mask]
                label_v = labels[valid][mask]
                pred_v = np.where(tpi_v > 0, 1, -1)
                spread_v = spread_bar[valid][mask]

                # Gross
                gross_ret = fwd_v * pred_v  # positive = win
                gross_hit = float(np.mean(pred_v == label_v))
                gross_mean = float(np.mean(gross_ret))
                gross_win = float(np.mean(gross_ret[gross_ret > 0])) if np.sum(gross_ret > 0) > 0 else 0.0
                gross_loss = float(np.mean(gross_ret[gross_ret < 0])) if np.sum(gross_ret < 0) > 0 else 0.0

                # Net after spread
                net_ret = gross_ret - cost_per_trade_log
                net_mean = float(np.mean(net_ret))
                net_win = float(np.mean(net_ret[net_ret > 0])) if np.sum(net_ret > 0) > 0 else 0.0
                net_loss = float(np.mean(net_ret[net_ret < 0])) if np.sum(net_ret < 0) > 0 else 0.0
                net_hit = float(np.mean(net_ret > 0))
                net_pf = float(abs(np.sum(net_ret[net_ret > 0]) / max(np.sum(net_ret[net_ret < 0]), 1e-12)))
                net_sharpe = float(np.mean(net_ret) / max(np.std(net_ret), 1e-12)) * np.sqrt(288 * 90) if np.std(net_ret) > 0 else 0.0

                # Convert to pips
                avg_price = float(np.mean(close[valid][mask]))
                gross_mean_pips = float(log_to_pips(gross_mean, avg_price))
                net_mean_pips = float(log_to_pips(net_mean, avg_price))
                cost_pips = spread_pips_est * 1.5  # spread + slippage

                sym_res[f"{filter_name}_{session_name}"] = {
                    "n": n_trades,
                    "gross_hit": gross_hit,
                    "gross_mean_log": gross_mean,
                    "gross_mean_pips": gross_mean_pips,
                    "net_hit": net_hit,
                    "net_mean_log": net_mean,
                    "net_mean_pips": net_mean_pips,
                    "net_pf": net_pf,
                    "net_sharpe": net_sharpe,
                    "cost_pips": cost_pips,
                    "avg_price": avg_price,
                }

                # Print
                pips_str = f"gross={gross_mean_pips:+.4f} net={net_mean_pips:+.4f}"
                print(f"  {sym:8s} {filter_name:4s} {session_name:8s}  "
                      f"gross_hit={gross_hit:.4f} net_hit={net_hit:.4f}  "
                      f"{pips_str}  PF={net_pf:.3f}  Sharpe={net_sharpe:.4f}  n={n_trades}")

        results[sym] = sym_res

    # Summary
    print(f"\n{'='*65}")
    print(f"SURVIVABILITY SUMMARY — NET EXPECTANCY IN PIPS")
    print(f"{'='*65}")
    for sym in SYMBOLS:
        r = results.get(sym, {})
        print(f"\n  {sym}")
        for key in ["Q4_all", "Q4_asia", "p90_asia"]:
            if key in r:
                d = r[key]
                print(f"    {key:12s}  net_pips={d['net_mean_pips']:+.4f}  net_hit={d['net_hit']:.4f}  "
                      f"PF={d['net_pf']:.3f}  Sharpe={d['net_sharpe']:.4f}  n={d['n']}")

    # Deployment decision
    print(f"\n{'='*65}")
    print(f"DEPLOYMENT DECISION")
    print(f"{'='*65}")
    deploy = True
    for sym in SYMBOLS:
        r = results.get(sym, {})
        for key in ["Q4_asia", "p90_asia"]:
            if key in r:
                net = r[key]["net_mean_pips"]
                status = "SURVIVES" if net > 0 else "FAILS"
                if net <= 0: deploy = False
                print(f"  {sym:8s} {key:12s}  net={net:+.4f} pips  -> {status}")
    print(f"\n  Overall: {'DEPLOY' if deploy else 'DO NOT DEPLOY'}")
    if not deploy:
        print(f"  Some filters/combinations failed survivability.")

    with open(os.path.join(REPORT_DIR, "dpl18d_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDPL-18D -> dpl18d_results.json")
