"""
Ultra Monster NautilusTrader Live Test — Quick One-Trade Match with Local FTMO MT5
Run: python run_live_test.py
Connects to local FTMO MT5 terminal, runs strategy for one signal match.
"""
import os
import time
import signal
import sys
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from nautilus_trader.live.node import TradingNode
from nautilus_trader.config import LoggingConfig, LiveRiskEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.objects import Money

# Import mt5-connector factories
from mt5connect.config import MT5Config
from mt5connect.factories import build_mt5_node_config

# Import our strategy
sys.path.insert(0, str(Path(__file__).parent / "strategies"))
from strategies.ultra_monster_nt import UltraMonsterStrategy, UltraMonsterConfig


# Load environment
load_dotenv(Path(__file__).parent / ".env")

FTMO_ACCOUNT = int(os.getenv("FTMO_ACCOUNT", "0"))
FTMO_PASSWORD = os.getenv("FTMO_PASSWORD", "")
FTMO_SERVER = os.getenv("FTMO_SERVER", "FTMO-Demo")
MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe")

UNIVERSE = [
    "EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm", "USDCADm", "USDCHFm",
    "EURJPYm", "GBPJPYm", "EURAUDm",
]


def main():
    if FTMO_ACCOUNT == 0:
        print("ERROR: Set FTMO_ACCOUNT, FTMO_PASSWORD, FTMO_SERVER in .env file")
        print("Copy .env.example to .env and fill in your credentials")
        return

    print("=" * 60)
    print("ULTRA MONSTER NAUTILUS LIVE TEST — FTMO MT5")
    print("=" * 60)
    print(f"Account: {FTMO_ACCOUNT}")
    print(f"Server:  {FTMO_SERVER}")
    print(f"Symbols: {', '.join(UNIVERSE)}")
    print()

    # Build MT5 config
    mt5_config = MT5Config(
        account=FTMO_ACCOUNT,
        password=FTMO_PASSWORD,
        server=FTMO_SERVER,
        symbols=UNIVERSE,
    )

    # Build Nautilus node config via mt5-connector
    node_config = build_mt5_node_config(
        mt5_config=mt5_config,
        risk_engine_config=LiveRiskEngineConfig(
            starting_balances=[Money(10000, USD)],
        ),
        logging_config=LoggingConfig(log_level="INFO"),
    )

    # Create TradingNode
    node = TradingNode(config=node_config)

    # Add strategy instance (Nautilus 1.224+ pattern)
    strategy_config = UltraMonsterConfig(
        instrument_ids=[f"{s.replace('m','')}.MT5" for s in UNIVERSE],
        bar_type="5-MINUTE-LAST",
        lookback_bars=12,
        min_range_pips=Decimal("6.0"),
        lot_size=Decimal("1.20"),
        hold_bars=3,
        triggers=[0, 30],
    )
    strategy = UltraMonsterStrategy(config=strategy_config)
    node.trader.add_strategy(strategy)

    # Graceful shutdown handler
    def shutdown_handler(signum, frame):
        print("\nShutdown signal received...")
        node.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Run the node
    print("Starting NautilusTrader live node...")
    print("Waiting for Ultra Monster signal at :00 or :30 (half-hourly)...")
    print("Press Ctrl+C to stop\n")

    try:
        node.run()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        node.stop()
        node.dispose()
        print("Node stopped cleanly")


if __name__ == "__main__":
    main()