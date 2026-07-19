"""TEMPLATE STRATEGY — Copy this folder to create a new strategy.

Plug-and-play contract:
- Implement generate_signal(data) -> dict | None
- Set CONFIG dict with metadata
- run.py will call your generate_signal each tick
"""
from paper_trade.core.config import register

STRATEGY_NAME = "template"

CONFIG = {
    "name": STRATEGY_NAME,
    "mt5_account": 0,
    "mt5_path": None,
    "pairs": ["EURUSD", "GBPUSD"],
    "hold_bars": 3,
    "session_start": 7,
    "session_end": 21,
    "max_concurrent": 2,
    "max_spread_mult": 1.5,
    "max_daily_loss": 500,
    "lot_size": 1.0,
    "mag_thresh": 0.002,
}

register(STRATEGY_NAME, CONFIG)

def generate_signal(data):
    """data: {pair: {bid, ask, time, ...}} from feed.current_bar()

    Returns: dict | None
        {
            "pair": "EURUSD",
            "direction": 1,       # 1 = LONG, -1 = SHORT
            "confidence": 0.8,    # 0-1, for filtering
            "metadata": {}        # optional debug info
        }
    """
    # --- YOUR SIGNAL LOGIC HERE ---
    # No lookahead: only use data['bid'], data['ask'], data['time']
    return None
