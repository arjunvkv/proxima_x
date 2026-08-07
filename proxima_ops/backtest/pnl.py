"""Trade PnL with tick-value-correct USD conversion + costs.

LESSONS ENCODED (Phase 8/9.2, pnl-alignment-tick-value.md):
  * The legacy JPY formula `ppts x volume x (1000/entry)` priced *per-pip* against
    *machine-point* increments and converted through the instrument's own quote —
    inflating EURJPY ~8.7x. Basis is the broker's authoritative per-point value:
      net = pnl_pts x tick_value x volume - (both-leg commission / spread / slippage)
  * Prefer a tick_value_map from the live engine's symbol_info; net paths mirror
    MT5 history_deals (profit - commission). Compare NET-to-NET.
  * No-map fallback preserves audit-classic pip pricing for parity checks.
"""
from __future__ import annotations
from typing import Optional

COMMISSION_PER_LOT = 3.5   # $ / lot per side (matches live ExecutionCost rate)


def pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol else 0.0001


def pip_value_usd(symbol: str, entry_price: float) -> float:
    """USD per pip per 1.0 lot (legacy/audit convention)."""
    if "JPY" in symbol:
        return 1000.0 / entry_price
    return 10.0


def trade_to_usd(t: dict, volume: float,
                 tick_value_map: Optional[dict] = None) -> dict:
    """Convert a port trade {symbol, pnl_pts, gross pts} to USD gross/commission/net.

    pnl_pts is in PRICE UNITS at the instrument's POINT (JPY machine point = 0.001,
    else 0.00001). With a tick_value_map the conversion uses the broker's actual
    per-point USD value per lot (the live-engine truth, validated 0.63097.. USD /
    0.001 pt / 1.0 lot for EURJPY); without it, falls back to audit-classic pip math.
    """
    if tick_value_map and t.get("symbol") in tick_value_map:
        gross = t["pnl_pts"] * tick_value_map[t["symbol"]] * volume
    else:
        pip = pip_size(t["symbol"])
        gross = (t["pnl_pts"] / pip) * pip_value_usd(t["symbol"], t["entry"]) * volume
    comm = round(2 * COMMISSION_PER_LOT * volume, 8)
    return {**{"symbol": t["symbol"], "pnl_pts": t["pnl_pts"], "entry": t.get("entry"),
               "entry_ts": t.get("entry_ts"), "exit_ts": t.get("exit_ts"),
               "side": t.get("side", "BUY"),
               "reason": t.get("reason"),
               "gross_usd": round(gross, 8), "commission": comm, "net": round(gross - comm, 8)}}