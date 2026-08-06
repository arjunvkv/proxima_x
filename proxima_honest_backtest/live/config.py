"""Ship registry — validated configurations of honest-backtest strategies.

These are the ONLY strategies permitted in the live engine. Params are the
VALIDATED ship values (from honest backtests), NOT the registry defaults which
exist only as cheap placeholders for the battle-royale report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

from proxima_honest_backtest.strategies import V2zStrategy, TokyoH0Strategy

ALL_18 = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "EURJPY",
    "GBPJPY", "EURAUD", "EURNZD", "GBPAUD", "GBPNZD",
    "GBPCAD", "AUDNZD", "USDCAD", "NZDUSD", "EURGBP",
    "EURCHF", "USDCHF", "AUDJPY",
]

CPPF_CROSS = ["EURAUD", "GBPAUD"]


@dataclass
class ShipConfig:
    key: str
    name: str
    description: str
    pairs: List[str]
    factory: Callable[[], Any]
    mode: str      # "multi" (on_bars kernel) only path that is live-ready
    magic_base: int
    base_lot: float
    live_ready: bool = True


SHIP_REGISTRY: List[ShipConfig] = [
    ShipConfig(
        key="tokyo_h0",
        name="Tokyo H0 (LB=6/H=12/N=5)",
        description=(
            "UTC-midnight cross-pair mean reversion. Entry at session bar open, "
            "60-min hold, top-5 most-declined pairs. VALIDATED: 77.3% WR, PF 10.62 "
            "on 3-mo M5 slice (open-exit)."
        ),
        pairs=ALL_18,
        factory=lambda: TokyoH0Strategy({
            "top_n": 5,
            "lookback_bars": 6,
            "lookback_confirm_bars": 3,
            "hold_bars": 12,
            "session_hour": 0,
            "min_pairs": 8,
            "min_confidence": 0.30,
            "require_decline_persistence": True,
        }),
        mode="multi",
        magic_base=400000,
        base_lot=0.15,
        live_ready=True,
    ),
    ShipConfig(
        key="v2z_z6_long",
        name="CPPF Z>=6 LONG (EURAUD+GBPAUD)",
        description=(
            "Cross-pair volatility dislocation mean reversion. LONG-only at z<=-6 "
            "with z-exit 2.0. VALIDATED: 90.0% WR, PF 11.34 on 3-mo slice (open-exit). "
            "NOT live-ready yet: single-pair on_bar path has no kernel parity gate. "
            "Follow-up planned after Phase 7 (option (a) per GPT)."
        ),
        pairs=CPPF_CROSS,
        factory=lambda: V2zStrategy({
            "lookback": 200,
            "z_entry": 6.0,
            "z_exit": 2.0,
            "direction": "LONG",
            "stop_a": 0.0,      # pure z-exit; no trailing stop
            "trig_a": 0.0,
            "gap_a": 0.05,
        }),
        mode="single",
        magic_base=400100,
        base_lot=0.15,
        live_ready=False,
    ),
]


def get_ship(key: str, require_live: bool = True) -> ShipConfig:
    """Look up a ship config. By default refuses non-live-ready strategies
    (a strategy is live-ready only if it implements the interface path that
    passed replay parity)."""
    for c in SHIP_REGISTRY:
        if c.key == key:
            if require_live and not c.live_ready:
                raise ValueError(
                    f"{c.key} is not live-ready (no dedicated kernel parity gate)"
                )
            return c
    raise KeyError(f"ship strategy not found: {key}")


def list_live_ready() -> List[ShipConfig]:
    return [c for c in SHIP_REGISTRY if c.live_ready]
