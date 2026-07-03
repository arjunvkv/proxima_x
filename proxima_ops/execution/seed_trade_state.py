import sys, os, json, time, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from proxima_ops.execution.mt5_connector import MT5Connector
from proxima_ops.execution.order_manager import OrderManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_trade_state")

TRADE_STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "state", "trade_state_persistence.json")

def seed():
    mt5 = MT5Connector()
    if not mt5.connect():
        logger.error("Cannot connect to MT5")
        return False

    positions = mt5.get_positions()
    if not positions:
        logger.info("No open positions found")
        return False

    om = OrderManager(mt5)
    now = time.time()
    state = {}

    for pos in positions:
        ticket = pos["ticket"]
        pos_time = pos["time"]

        symbol = pos["symbol"]
        tick = mt5.get_tick(symbol) if symbol else None
        if tick:
            server_now = tick["time"]
        else:
            # Fallback: use time.time() + estimated server offset
            server_now = now + 10800  # assume UTC+3
        elapsed_sec = max(0, server_now - pos_time)
        estimated_cycles = max(1, int(elapsed_sec / 3))

        entry_atr = om._compute_atr(symbol) if symbol else 0.0

        entry_price = float(pos["price_open"])
        side = pos["type"]
        current_price = float(pos["price_current"])

        adv = (entry_price - current_price) if side == "SELL" else (current_price - entry_price)
        mfe = max(0.0, adv)
        mae = min(0.0, adv)

        state[str(ticket)] = {
            "hold_cycle_count": estimated_cycles,
            "entry_timestamp": float(pos_time),
            "last_cycle_timestamp": now,
            "mfe": round(mfe, 6),
            "mae": round(mae, 6),
            "entry_atr": round(entry_atr, 6),
            "entry_price": round(entry_price, 5),
            "side": side,
        }

        logger.info(f"SEED ticket={ticket} symbol={symbol} side={side}")
        logger.info(f"  entry_price={entry_price} current_price={current_price}")
        logger.info(f"  elapsed={elapsed_sec:.0f}s estimated_cycles={estimated_cycles}")
        logger.info(f"  mfe={mfe:.6f} mae={mae:.6f} entry_atr={entry_atr:.6f}")

    os.makedirs(os.path.dirname(TRADE_STATE_PATH), exist_ok=True)
    with open(TRADE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

    logger.info(f"Trade state seeded to {TRADE_STATE_PATH}")
    logger.info(json.dumps(state, indent=2))
    return True

if __name__ == "__main__":
    seed()
