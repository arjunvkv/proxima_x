"""
PHASE 4 — STOP DISTANCE REALITY

Audit catastrophic stop distances against historical adverse moves using H1 data.

For each asset, compute:
  median_H1_adverse_move (pips)
  95th percentile adverse move
  99th percentile adverse move

Compare to configured catastrophic stops.

Output: STOP_DISTANCE_AUDIT.md
"""

import os
import sys
import math
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from proxima_ops.config.settings import SETTINGS
from proxima_ops.risk.catastrophic_stop import CATASTROPHIC_STOP_PIPS

try:
    import MetaTrader5 as mt5
    MT5_OK = True
except ImportError:
    MT5_OK = False


SYMBOLS = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD", "EURUSD"]
LOOKBACK_HOURS = 720  # 30 days of H1 data


def get_h1_rates(symbol: str, count: int = 720) -> list:
    """Fetch H1 bars from MT5."""
    if not MT5_OK:
        return []
    try:
        if not mt5.initialize():
            return []
        tf = mt5.TIMEFRAME_H1
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None:
            # Try broker mapping
            mappings = {"NAS100": "USTEC", "XAUUSD": "GOLD"}
            sym2 = mappings.get(symbol, symbol)
            rates = mt5.copy_rates_from_pos(sym2, tf, 0, count)
        if rates is None:
            return []
        result = [{"high": float(r["high"]), "low": float(r["low"]), "close": float(r["close"])} for r in rates]
        return result
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return []


def pip_diff(symbol: str, price_diff: float) -> float:
    """Convert price difference to pips."""
    if "JPY" in symbol:
        return price_diff / 0.01
    elif "XAU" in symbol or "XAG" in symbol:
        return price_diff / 0.01
    elif "NAS" in symbol:
        return price_diff / 0.01
    else:
        return price_diff / 0.0001


def compute_adverse_moves(symbol: str, rates: list) -> dict:
    """
    For a BUY position entered at close of bar N, the adverse move
    is the maximum adverse excursion: close - min(low of bar N+1).
    For SELL: max(high) - close.

    We compute both directions for every sequential bar pair.
    """
    if len(rates) < 100:
        return {"n_samples": 0}

    buy_adverse = []
    sell_adverse = []

    for i in range(len(rates) - 1):
        entry = rates[i]["close"]
        next_low = rates[i + 1]["low"]
        next_high = rates[i + 1]["high"]

        buy_adv = entry - next_low if entry > next_low else 0.0
        sell_adv = next_high - entry if next_high > entry else 0.0

        if buy_adv > 0:
            buy_adverse.append(pip_diff(symbol, buy_adv))
        if sell_adv > 0:
            sell_adverse.append(pip_diff(symbol, sell_adv))

    if not buy_adverse:
        return {"n_samples": len(rates) - 1}

    bp = np.percentile(buy_adverse, [50, 95, 99])
    sp = np.percentile(sell_adverse, [50, 95, 99])

    return {
        "n_samples": len(rates) - 1,
        "buy_median_pips": round(float(bp[0]), 1),
        "buy_95th_pips": round(float(bp[1]), 1),
        "buy_99th_pips": round(float(bp[2]), 1),
        "sell_median_pips": round(float(sp[0]), 1),
        "sell_95th_pips": round(float(sp[1]), 1),
        "sell_99th_pips": round(float(sp[2]), 1),
    }


def classify_stop(config_pips: float, adverse_95th: float) -> str:
    """Classify if stop is too tight, reasonable, or too wide."""
    if adverse_95th <= 0:
        return "INSUFFICIENT_DATA"
    ratio = config_pips / adverse_95th
    if ratio < 0.5:
        return "TOO_TIGHT"
    elif ratio < 1.5:
        return "REASONABLE"
    elif ratio < 3.0:
        return "WIDE"
    else:
        return "TOO_WIDE"


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(output_dir, exist_ok=True)

    print("Fetching H1 data and computing adverse moves...")
    results = []
    for sym in SYMBOLS:
        print(f"  {sym}...", end=" ", flush=True)
        rates = get_h1_rates(sym, LOOKBACK_HOURS)
        if not rates:
            print(f"No data")
            continue
        print(f"{len(rates)} bars", end=" ", flush=True)
        adv = compute_adverse_moves(sym, rates)
        results.append((sym, adv))
        print(f"| Buy 95th={adv.get('buy_95th_pips', 'N/A')}p Sell 95th={adv.get('sell_95th_pips', 'N/A')}p")

    # Generate report
    lines = []
    lines.append("# Stop Distance Audit")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Data:** {LOOKBACK_HOURS} hours of H1 bars (~30 days)")
    lines.append("")
    lines.append("## Per-Asset Adverse Move Analysis")
    lines.append("")
    lines.append("| Asset | Config Stop | N | Buy Med | Buy 95th | Buy 99th | Sell Med | Sell 95th | Sell 99th | Classification |")
    lines.append("|-------|-------------|---|---------|----------|----------|----------|-----------|-----------|----------------|")
    for sym, adv in results:
        config = CATASTROPHIC_STOP_PIPS.get(sym, 50)
        cls = classify_stop(config, adv.get("buy_95th_pips", 0))
        lines.append(
            f"| {sym:<6} "
            f"| {config:<11} "
            f"| {adv.get('n_samples', 0):<1} "
            f"| {adv.get('buy_median_pips', 'N/A'):<7} "
            f"| {adv.get('buy_95th_pips', 'N/A'):<8} "
            f"| {adv.get('buy_99th_pips', 'N/A'):<8} "
            f"| {adv.get('sell_median_pips', 'N/A'):<7} "
            f"| {adv.get('sell_95th_pips', 'N/A'):<8} "
            f"| {adv.get('sell_99th_pips', 'N/A'):<8} "
            f"| {cls:<14} |"
        )
    lines.append("")

    lines.append("## Classification Legend")
    lines.append("")
    lines.append("- **TOO_TIGHT**: Stop is less than 50% of the 95th percentile adverse move")
    lines.append("- **REASONABLE**: Stop is between 50% and 150% of the 95th percentile")
    lines.append("- **WIDE**: Stop is between 150% and 300% of the 95th percentile")
    lines.append("- **TOO_WIDE**: Stop exceeds 300% of the 95th percentile adverse move")
    lines.append("")

    lines.append("## Note")
    lines.append("")
    lines.append("These are 1-H1-bar adverse moves, which are shorter than the typical")
    lines.append("research holding period (H20 = 20 bars). Actual holding-period adverse")
    lines.append("moves will be LARGER than these per-bar estimates.")
    lines.append("The catastrophic stop exists only for broker/VPS crash survival,")
    lines.append("not as the research stop-loss. These are disaster-protection stops.")

    report = "\n".join(lines)
    report_path = os.path.join(output_dir, "STOP_DISTANCE_AUDIT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
