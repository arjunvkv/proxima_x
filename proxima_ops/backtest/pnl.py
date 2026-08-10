"""Trade PnL with tick-value-correct USD conversion + costs.

LESSONS ENCODED (Phase 8/9.2 + LIVE micro-batch 2026-08-07):
  * The legacy JPY formula `ppts x volume x (1000/entry)` priced *per-pip* against
    *machine-point* increments and converted through the instrument's own quote —
    inflating EURJPY ~8.7x.
  * tick_value from MT5 symbol_info is USD per POINT per 1.0 lot (e.g. EURJPY
    0.625 USD per 0.001-point). So the engine must FIRST convert pnl_pts from
    PRICE units to POINTS:  points = pnl_pts / point_size  (point=0.001 JPY,
    0.00001 non-JPY). Multiplying price-units directly by tick_value understates
    by the point size (live micro-batch caught this: EURJPY engine -0.0001 vs
    live -0.05).
  * COMMISSION: live FTMO-Demo broker charges $3.0/lot/side (verified: -0.06 on
    a 0.01-lot round trip). The audit gate used $3.5 as a conservative bound;
    the ENGINE default is the broker-true 3.0. net = gross - both-leg comm.
  * Compare NET-to-NET vs MT5 history_deals (profit + commission), tolerance
    +-0.01 (MT5 rounds deal profit to cents).
"""
from __future__ import annotations
from typing import Optional

COMMISSION_PER_LOT = 3.0   # $/lot per side — live FTMO-Demo broker rate (verified 0.06/0.01 RT)

# Broker-authoritative tick values: USD per machine POINT per 1.0 lot, read live
# from FTMO-Demo symbol_info.trade_tick_value (2026-08-10). Point = 0.001 on JPY
# quotes, 0.00001 everywhere else (see point_size); every pip = 10 points, so
# pip value = tick_value * 10 (EURUSD 1.0 -> $10/pip; EURNZD 0.58816 -> $5.88/pip).
# REFRESH: run the symbol_info.trade_tick_value probe on the box and update.
FTMO_TICK_VALUES = {
    "EURUSD": 1.0, "USDJPY": 0.628895, "GBPUSD": 1.0, "AUDUSD": 1.0,
    "EURJPY": 0.628895, "GBPJPY": 0.628895, "AUDJPY": 0.628895,
    "EURAUD": 0.70554, "EURNZD": 0.58816, "GBPAUD": 0.70554,
    "GBPNZD": 0.58816, "GBPCAD": 0.716759, "USDCAD": 0.716759,
    "NZDUSD": 1.0, "AUDNZD": 0.58816, "EURGBP": 1.35015,
    "EURCHF": 1.234751, "USDCHF": 1.234751,
}


def pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol else 0.0001


def point_size(symbol: str) -> float:
    """Machine point of the symbol's price (tick size). JPY pairs quote 3-digit
    (point 0.001), all others 5-digit (point 0.00001) on FTMO."""
    return 0.001 if "JPY" in symbol else 0.00001


def pip_value_usd(symbol: str, entry_price: float) -> float:
    """USD per pip per 1.0 lot (legacy/audit convention)."""
    if "JPY" in symbol:
        return 1000.0 / entry_price
    return 10.0


def trade_to_usd(t: dict, volume: float,
                 tick_value_map: Optional[dict] = None,
                 commission_per_lot: Optional[float] = None,
                 spread_pips_map: Optional[dict] = None) -> dict:
    """Convert a port trade {symbol, pnl_pts} to USD gross/commission/net.

    pnl_pts is in PRICE UNITS (e.g. 0.008 for EURJPY 8 points). With a
    tick_value_map (broker per-point value) we first convert to POINTS via
    point_size(symbol), then multiply by tick_value and volume:
        gross = pnl_pts / point_size * tick_value * volume
    Without a map, falls back to audit-classic pip math (parity with the
    validated audit curve). commission_per_lot defaults to the live broker 3.0.
    spread_pips_map: optional {symbol: spread in pips}; one full spread is
    charged per round trip (a fill at bar-open + spread/2 on entry, - spread/2
    at exit) — the engine models offers as open-price so without this it is
    optimistic by exactly one spread. Default None = zero spread (legacy parity).
    """
    sym = t["symbol"]
    if tick_value_map and sym in tick_value_map:
        pts = t["pnl_pts"] / point_size(sym)
        gross = pts * tick_value_map[sym] * volume
    else:
        pip = pip_size(sym)
        gross = (t["pnl_pts"] / pip) * pip_value_usd(sym, t.get("entry") or 1.0) * volume
    rate = commission_per_lot if commission_per_lot is not None else COMMISSION_PER_LOT
    comm = round(2 * rate * volume, 8)
    spread = 0.0
    if spread_pips_map and sym in spread_pips_map:
        if tick_value_map and sym in tick_value_map:
            # broker-true pip value: every pip = 10 machine points on FTMO quotes
            pip_usd = tick_value_map[sym] * 10.0
        else:
            pip_usd = pip_value_usd(sym, t.get("entry") or 1.0)
        spread = round(spread_pips_map[sym] * pip_usd * volume, 8)
    net = gross - comm - spread
    # keep gross intact (P&L before costs); add explicit spread for audit trail
    return {**{"symbol": sym, "pnl_pts": t["pnl_pts"], "entry": t.get("entry"),
               "entry_ts": t.get("entry_ts"), "exit_ts": t.get("exit_ts"),
               "side": t.get("side", "BUY"), "reason": t.get("reason"),
               "gross_usd": round(gross, 8), "commission": comm,
               "spread": spread,
               "net": round(net, 8)}}