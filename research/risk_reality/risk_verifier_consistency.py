"""
RISK VERIFIER CONSISTENCY CHECK — Phase 3 / V2.4 Risk Engine Repair

For each symbol, computes:
  1) Volume via OrderManager.calculate_volume()
  2) Dollar risk via TradeRiskVerifier.verify() internal logic
  3) Dollar risk via correct pip-value formula: volume × pip_value_per_lot × stop_pips

Requirement: |verifier_risk − correct_risk| ÷ correct_risk < 0.001
"""

import sys

sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")

from proxima_ops.config.settings import SETTINGS
from proxima_ops.risk.catastrophic_stop import catastrophic_sl, CATASTROPHIC_STOP_PIPS

SYMBOLS = {
    "EURJPY": 185,
    "USDJPY": 160,
    "GBPJPY": 214,
    "XAUUSD": 4325,
    "EURUSD": 1.15,
    "NAS100": 19500,
}
BALANCE = 25000.0
RISK_PCT = 0.0025


def calc_volume(symbol, price, balance=BALANCE, risk_pct=RISK_PCT):
    """Standalone reproduction of OrderManager.calculate_volume()."""
    rp = risk_pct if risk_pct is not None else SETTINGS.risk_per_trade
    if price is None or price <= 0 or balance is None or balance <= 0:
        return 0.01
    risk_amount = float(balance) * float(rp)
    if "JPY" in symbol:
        pv = max(float(price), 1.0)
        pv = 1000.0 / pv
    else:
        pv = 10.0
    sl_pts = max(SETTINGS.max_spread_points.get(symbol, 50), 50)
    lots = risk_amount / max(sl_pts * pv, 1.0)
    lots = max(0.01, round(lots, 2))
    return min(lots, 1.0)


def verifier_risk(symbol, volume, entry, sl):
    """Reproduce TradeRiskVerifier.verify() dollar-risk logic.

    point_value_per_lot is always 1.0 regardless of symbol.
    """
    pv = 1.0
    pip_dist = abs(entry - sl)
    if "JPY" in symbol:
        stop_pts = pip_dist / 0.001
    elif "XAU" in symbol or "XAG" in symbol:
        stop_pts = pip_dist / 0.01
    else:
        stop_pts = pip_dist / 0.0001
    return volume * pv * stop_pts


def correct_risk(symbol, volume, entry, sl):
    """Compute dollar risk using: volume × pip_value_per_lot × stop_pips.

    Pip sizes are aligned with catastrophic_sl() conventions:
      - JPY / XAU / XAG / NAS : 1 pip = 0.01
      - others (EURUSD)        : 1 pip = 0.0001

    Pip values per lot are aligned with calculate_volume() conventions:
      - JPY pairs             : 1000.0 / price  (USD per pip per standard lot)
      - non-JPY pairs         : $10.00          (100k units × 0.0001)
    """
    pip_dist = abs(entry - sl)
    if "JPY" in symbol or "XAU" in symbol or "XAG" in symbol or "NAS" in symbol:
        pip_size = 0.01
    else:
        pip_size = 0.0001
    stop_pips = pip_dist / pip_size

    if "JPY" in symbol:
        pip_value_per_lot = 1000.0 / max(float(entry), 1.0)
    else:
        pip_value_per_lot = 10.0

    return volume * pip_value_per_lot * stop_pips


if __name__ == "__main__":
    results = []
    header = (
        f"{'Symbol':>8}  {'Price':>9}  {'SL(pips)':>9}  "
        f"{'Volume':>8}  {'Verifier$':>11}  {'Correct$':>11}  "
        f"{'Ratio':>11}  {'Status':>7}"
    )
    sep = "-" * len(header)

    all_pass = True
    for symbol, price in SYMBOLS.items():
        sl_pips = CATASTROPHIC_STOP_PIPS.get(symbol, 50)
        sl_price = catastrophic_sl(symbol, price, "BUY")
        vol = calc_volume(symbol, price, BALANCE, RISK_PCT)
        v_risk = verifier_risk(symbol, vol, price, sl_price)
        c_risk = correct_risk(symbol, vol, price, sl_price)
        if c_risk > 0:
            ratio = abs(v_risk - c_risk) / c_risk
        else:
            ratio = float("inf")
        passed = ratio < 0.001
        all_pass = all_pass and passed
        results.append(
            (symbol, price, sl_pips, vol, v_risk, c_risk, ratio,
             "PASS" if passed else "FAIL")
        )

    print(header)
    print(sep)
    for r in results:
        print(
            f"{r[0]:>8}  {r[1]:>9.2f}  {r[2]:>9}  "
            f"{r[3]:>8.4f}  {r[4]:>11.2f}  {r[5]:>11.2f}  "
            f"{r[6]:>11.6f}  {r[7]:>7}"
        )
    print(sep)
    verdict = "ALL PASS — Verifier IS Consistent" if all_pass else "SOME FAIL — Verifier IS Inconsistent"
    print(f"\n  Verdict: {verdict}")
