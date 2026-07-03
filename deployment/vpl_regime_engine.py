"""
VPL-1 Regime Engine — Live Deployment Module

Deploys SAF + SIT + VEM as volatility regime infrastructure for forex trading.

Symbols:
  FX (full stack): EURJPY, USDJPY, GBPJPY
  XAUUSD (SAF-only): isolated due to inverted SIT dynamics

Deployment layers:
  A (hard gate): SAF regime, SIT persistence gate, SAF×SIT state machine
  B (soft): VEM dampener, sequence memory sizing boost
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from research.volatility_physics.vpl_1.core.target_engine import (
    load_m5, compute_returns, compute_crf, compute_crf_deciles,
    compute_expansion_labels, realized_variance_from_log_returns,
)
from research.volatility_physics.vpl_1.core.sit_engine import compute_sit
from research.volatility_physics.vpl_1.core.vcm_engine import compute_vcm


FX_SYMBOLS = ["EURJPY", "USDJPY", "GBPJPY"]
XAU_SYMBOL = "XAUUSD"


def compute_vpl_state(symbol):
    """Compute full VPL state for a single symbol."""
    data = load_m5(symbol)
    c, h, l, ts = data["close"], data["high"], data["low"], data["timestamp"]
    r = compute_returns(c)
    n_close = len(c)

    rv = realized_variance_from_log_returns(r, 24)
    sav = compute_crf(c, h, l)
    saf = sav["crf"]

    sit_out = compute_sit(c, h, l, r, rv, saf)
    sit = sit_out["instability"]

    vcm_out = compute_vcm(r, rv)
    vem = vcm_out["vcm"]

    # SIT and SAF are already close-aligned
    saf_arr = saf.copy()
    sit_arr = sit.copy()

    # VEM needs padding (returns-based)
    vem_arr = np.full(n_close, np.nan)
    vem_arr[1:] = vem

    # Use dynamic thresholds: SAF median (should be ~0), SIT 66th percentile (turbulence threshold)
    saf_threshold = np.nanmedian(saf_arr)
    sit_threshold = np.nanpercentile(sit_arr, 66)

    return {
        "symbol": symbol,
        "timestamp": ts,
        "close": c,
        "saf": saf_arr,
        "sit": sit_arr,
        "vem": vem_arr,
        "saf_threshold": float(saf_threshold),
        "sit_threshold": float(sit_threshold),
    }


def classify_regime(saf, sit, saf_threshold, sit_threshold):
    """
    Classify bar into one of 4 regimes based on SAF and SIT.

    SAF > threshold = high SAF (absorbed/contraction regime)
    SIT > threshold = high SIT (unstable regime)
    """
    saf_high = saf > saf_threshold if not np.isnan(saf) else False
    sit_high = sit > sit_threshold if not np.isnan(sit) else False

    if not saf_high and not sit_high:
        return "SMOOTH_TREND", 1.0
    elif not saf_high and sit_high:
        return "ACTIVE_INSTABILITY", 1.5
    elif saf_high and not sit_high:
        return "LOCKED", 0.5
    else:
        return "COMPRESSED_CHAOS", 0.5


def compute_persistence(series, above=True, threshold=0.0, min_streak=2):
    """Count consecutive bars meeting condition."""
    streak = np.zeros(len(series))
    count = 0
    for i in range(len(series)):
        val = series[i]
        if np.isnan(val):
            count = 0
            streak[i] = 0
            continue
        if (above and val > threshold) or (not above and val < threshold):
            count += 1
        else:
            count = 0
        streak[i] = count
    return streak


def compute_risk_multiplier(regime, sit_persist, vem_val, vem_threshold=0.5):
    """Adjust risk multiplier based on regime and soft dampeners."""
    base = {"SMOOTH_TREND": 1.0, "ACTIVE_INSTABILITY": 1.5,
            "LOCKED": 0.5, "COMPRESSED_CHAOS": 0.5}.get(regime, 1.0)

    # Persistence boost: active instability with 3+ consecutive high SIT = 2.0x
    if regime == "ACTIVE_INSTABILITY" and sit_persist >= 3:
        base = 2.0

    # VEM dampener: if vem is high, reduce multiplier
    if not np.isnan(vem_val) and vem_val > vem_threshold:
        base *= 0.7

    return round(base, 2)


def build_bar_records(state, sit_persist_arr, saf_persist_arr):
    """Build per-bar regime records."""
    n = len(state["saf"])
    saf_th = state["saf_threshold"]
    sit_th = state["sit_threshold"]
    records = []
    for i in range(n):
        saf_val, sit_val, vem_val = state["saf"][i], state["sit"][i], state["vem"][i]
        if np.isnan(saf_val) or np.isnan(sit_val):
            continue

        regime, base_risk = classify_regime(saf_val, sit_val, saf_th, sit_th)
        sit_pers = sit_persist_arr[i]
        saf_pers = saf_persist_arr[i]

        risk_mult = compute_risk_multiplier(regime, sit_pers, vem_val)

        records.append({
            "timestamp": int(state["timestamp"][i]),
            "close": float(state["close"][i]),
            "regime": regime,
            "saf": round(float(saf_val), 4),
            "sit": round(float(sit_val), 4),
            "vem": round(float(vem_val), 4) if not np.isnan(vem_val) else None,
            "sit_persistence": int(sit_pers),
            "saf_persistence": int(saf_pers),
            "risk_multiplier": risk_mult,
            "trade_permission": "FULL" if risk_mult >= 1.0 else "REDUCED" if risk_mult >= 0.5 else "RESTRICTED",
        })
    return records


def run_fx(fx_symbols=FX_SYMBOLS):
    """Run full VPL-1 regime engine on forex pairs."""
    all_records = {}
    for sym in fx_symbols:
        state = compute_vpl_state(sym)
        sit_persist = compute_persistence(state["sit"], above=True, threshold=state["sit_threshold"], min_streak=2)
        saf_persist = compute_persistence(state["saf"], above=False, threshold=state["saf_threshold"], min_streak=2)
        records = build_bar_records(state, sit_persist, saf_persist)
        all_records[sym] = records
        print(f"  {sym}: {len(records)} bars processed")
    return all_records


def run_xau():
    """Run SAF-only logic for XAUUSD."""
    state = compute_vpl_state(XAU_SYMBOL)
    n = len(state["saf"])
    saf_th = state["saf_threshold"]
    records = []
    for i in range(n):
        saf_val = state["saf"][i]
        if np.isnan(saf_val):
            continue
        regime = "LOCKED" if saf_val > saf_th else "ACTIVE"
        risk_mult = 0.5 if regime == "LOCKED" else 1.0
        records.append({
            "timestamp": int(state["timestamp"][i]),
            "close": float(state["close"][i]),
            "regime": regime + "_SAF_ONLY",
            "saf": round(float(saf_val), 4),
            "risk_multiplier": risk_mult,
            "trade_permission": "REDUCED" if risk_mult < 1.0 else "FULL",
        })
    print(f"  XAUUSD (SAF-only): {len(records)} bars processed")
    return {XAU_SYMBOL: records}


def run_all(output_path=None):
    """Run VPL regime engine for all symbols and save signals."""
    print("=== VPL-1 Regime Engine — Live Deployment ===\n")

    print("FX pairs (full stack):")
    fx = run_fx()

    print("\nXAUUSD (SAF-only):")
    xau = run_xau()

    all_data = {**fx, **xau}

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        json.dump(all_data, open(output_path, "w"), indent=2)

    # Summary stats
    print("\n=== Summary ===")
    for sym, recs in all_data.items():
        regimes = {}
        for r in recs:
            reg = r["regime"]
            regimes[reg] = regimes.get(reg, 0) + 1
        total = len(recs)
        print(f"\n{sym} ({total} bars):")
        for reg, count in sorted(regimes.items()):
            pct = count / total * 100
            print(f"  {reg}: {count} ({pct:.1f}%)")
        n_high_risk = sum(1 for r in recs if r.get("risk_multiplier", 1.0) >= 1.5)
        n_low_risk = sum(1 for r in recs if r.get("risk_multiplier", 1.0) <= 0.5)
        print(f"  High-risk bars: {n_high_risk} ({n_high_risk/total*100:.1f}%)")
        print(f"  Low-risk bars: {n_low_risk} ({n_low_risk/total*100:.1f}%)")

    return all_data


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "vpl_regime_signals.json")
    run_all(output_path=out)
    print(f"\nSignals saved to {out}")
