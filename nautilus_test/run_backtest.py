"""
Ultra Monster NautilusTrader Backtest — Local FTMO MT5 Data
Run: python run_backtest.py
"""
import os
import sys
from pathlib import Path
from decimal import Decimal

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.core.datetime import dt_to_unix_nanos

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone, timedelta

# Import our strategy
sys.path.insert(0, str(Path(__file__).parent / "strategies"))
from strategies.ultra_monster_nt import UltraMonsterStrategy, UltraMonsterConfig


# FTMO MT5 Config from environment
FTMO_ACCOUNT = int(os.getenv("FTMO_ACCOUNT", "0"))
FTMO_PASSWORD = os.getenv("FTMO_PASSWORD", "")
FTMO_SERVER = os.getenv("FTMO_SERVER", "FTMO-Demo")
MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe")

UNIVERSE = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
    "EURJPY", "GBPJPY", "EURAUD",
]

# Strategy params from settings.py
LOOKBACK_BARS = 12
MIN_RANGE_PIPS = Decimal("6.0")
LOT_SIZE = Decimal("1.20")
HOLD_BARS = 3
TRIGGERS = [0, 30]


def download_mt5_data(symbols, count=3000):
    """Download M5 bars from FTMO MT5 and convert to Nautilus Parquet catalog."""
    # Terminal already running — initialize without path
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    catalog_path = Path("catalog")
    catalog_path.mkdir(exist_ok=True)
    catalog = ParquetDataCatalog(catalog_path)

    for sym in symbols:
        print(f"Downloading {sym}...")
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, count)
        if rates is None or len(rates) == 0:
            print(f"  No data for {sym}")
            continue

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Convert to Nautilus Bar objects
        base = sym.replace("m", "")
        instrument_id = InstrumentId(Symbol(base), Venue("MT5"))
        bar_type = BarType(instrument_id, BarAggregation.MINUTE, 5, PriceType.LAST)

        bars = []
        for idx, row in df.iterrows():
            bars.append(
                Bar(
                    bar_type=bar_type,
                    open=Price(row['open'], 5),
                    high=Price(row['high'], 5),
                    low=Price(row['low'], 5),
                    close=Price(row['close'], 5),
                    volume=Quantity(int(row['tick_volume']), 0),
                    ts_event=dt_to_unix_nanos(idx.to_pydatetime()),
                    ts_init=dt_to_unix_nanos(idx.to_pydatetime()),
                )
            )

        # Write to catalog
        catalog.write_data(bars)
        print(f"  Wrote {len(bars)} bars for {sym}")

    mt5.shutdown()
    return catalog


def run_backtest(catalog):
    """Run backtest with Ultra Monster strategy."""
    # Engine config - simplified
    engine_config = BacktestEngineConfig(
        logging=LoggingConfig(log_level="INFO"),
    )

    engine = BacktestEngine(config=engine_config)

    # Add data
    for sym in UNIVERSE:
        base = sym.replace("m", "")
        instrument_id = InstrumentId(Symbol(base), Venue("MT5"))
        bar_type = BarType(instrument_id, BarAggregation.MINUTE, 5, PriceType.LAST)
        data = catalog.get_data(bar_type)
        if data:
            engine.add_data(data)
            print(f"Added {len(data)} bars for {base}")

    # Add strategy
    config = UltraMonsterConfig(
        instrument_ids=[f"{s.replace('m','')}.MT5" for s in UNIVERSE],
        bar_type="5-MINUTE-LAST",
        lookback_bars=LOOKBACK_BARS,
        min_range_pips=MIN_RANGE_PIPS,
        lot_size=LOT_SIZE,
        hold_bars=HOLD_BARS,
        triggers=TRIGGERS,
    )
    engine.add_strategy(UltraMonsterStrategy, config)

    # Run
    print("\nRunning backtest...")
    engine.run()
    result = engine.get_result()
    print("\n" + "=" * 60)
    print("BACKTEST RESULT")
    print("=" * 60)
    print(result)
    
    # Print trade details
    trades = engine.cache.trades()
    if trades:
        print(f"\nTotal trades: {len(trades)}")
        for t in trades[:10]:
            print(f"  {t.side} {t.instrument_id} @ {t.price} qty={t.quantity} pnl={t.pnl}")


if __name__ == "__main__":
    if FTMO_ACCOUNT == 0:
        print("ERROR: Set FTMO_ACCOUNT, FTMO_PASSWORD, FTMO_SERVER in .env file")
        sys.exit(1)

    # Step 1: Download data (run once)
    print("=" * 60)
    print("DOWNLOADING FTMO MT5 DATA FOR BACKTEST")
    print("=" * 60)
    catalog = download_mt5_data(UNIVERSE, count=3000)

    # Step 2: Run backtest
    print("\n" + "=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)
    run_backtest(catalog)