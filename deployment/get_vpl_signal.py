"""
VPL-1 Live Signal Interface

Quick integration: get current regime signal for any symbol.

Usage:
    from deployment.get_vpl_signal import get_current_signal

    signal = get_current_signal("EURJPY")
    # Returns: { "regime": "ACTIVE_INSTABILITY", "risk_mult": 2.0, "trade_perm": "FULL", ... }
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import json

from deployment.vpl_regime_engine import (
    compute_vpl_state, classify_regime, compute_persistence,
    compute_risk_multiplier, FX_SYMBOLS, XAU_SYMBOL,
)


def get_current_signal(symbol):
    """Get the latest VPL regime signal for a symbol."""
    state = compute_vpl_state(symbol)
    i = -1  # last bar

    saf_val = float(state["saf"][i])
    sit_val = float(state["sit"][i])
    vem_val = float(state["vem"][i]) if not np.isnan(state["vem"][i]) else None

    if np.isnan(saf_val) or np.isnan(sit_val):
        return None

    saf_th = state["saf_threshold"]
    sit_th = state["sit_threshold"]

    is_xau = symbol == XAU_SYMBOL

    if is_xau:
        regime = "LOCKED" if saf_val > saf_th else "ACTIVE"
        risk_mult = 0.5 if regime == "LOCKED" else 1.0
        return {
            "symbol": symbol,
            "regime": regime + "_SAF_ONLY",
            "saf": round(float(saf_val), 4),
            "saf_threshold": round(float(saf_th), 4),
            "risk_multiplier": risk_mult,
            "trade_permission": "REDUCED" if risk_mult < 1.0 else "FULL",
            "bars_processed": int(len(state["saf"])),
        }

    # Compute persistence on full arrays, then take last value
    sit_persist_full = compute_persistence(state["sit"], above=True, threshold=sit_th, min_streak=2)
    saf_persist_full = compute_persistence(state["saf"], above=False, threshold=saf_th, min_streak=2)
    sit_persist = int(sit_persist_full[i])
    saf_persist = int(saf_persist_full[i])

    regime, base_risk = classify_regime(saf_val, sit_val, saf_th, sit_th)
    risk_mult = compute_risk_multiplier(regime, sit_persist, vem_val if vem_val is not None else 0.0)

    return {
        "symbol": symbol,
        "regime": regime,
        "saf": round(saf_val, 4),
        "sit": round(sit_val, 4),
        "vem": round(vem_val, 4) if vem_val is not None else None,
        "saf_threshold": round(float(saf_th), 4),
        "sit_threshold": round(float(sit_th), 4),
        "sit_persistence": sit_persist,
        "saf_persistence": saf_persist,
        "risk_multiplier": round(float(risk_mult), 2),
        "trade_permission": "FULL" if risk_mult >= 1.0 else "REDUCED" if risk_mult >= 0.5 else "RESTRICTED",
        "bars_processed": int(len(state["saf"])),
    }


def print_signal(symbol):
    sig = get_current_signal(symbol)
    if sig is None:
        print(f"{symbol}: No signal (insufficient data)")
        return
    print(f"\n{symbol} VPL Signal:")
    for k, v in sig.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    for sym in FX_SYMBOLS + [XAU_SYMBOL]:
        print_signal(sym)
