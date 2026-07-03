"""
Environment factory for Proxima V6 Tick Time-Machine.
Binds adapters (TickSource, Clock, Broker) for live/paper/replay modes.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import polars as pl

from core.adapters.tick_source import TickSource, MT5TickSource, ReplayTickSource
from core.adapters.clock import Clock, RealClock, ReplayClock
from core.adapters.broker import Broker, MT5Broker, PaperBroker
from replay.tick_archive import TickArchive
from replay.replay_feed import ReplayFeed
from replay.replay_clock import ReplayClock as ReplayClockImpl
from replay.execution_model import ExecutionModel
from replay.sampler import Sampler
from replay.tsv import TemporalShuffleValidator
from replay.metrics import ReplayMetrics
from replay.parity import ParityLedger

logger = logging.getLogger("proxima.environment")


@dataclass
class ReplayConfig:
    symbols: list[str] = field(default_factory=lambda: sorted(
        ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD", "EURUSD"],
        key=lambda s: {"EURJPY": 0, "USDJPY": 1, "GBPJPY": 2, "XAUUSD": 3, "EURUSD": 4}.get(s, 99),
    ))
    start: str = "2025-01-01"
    end: str = "2025-03-01"
    speed: float = 1000.0
    mode: str = "ACCELERATED"
    burst: bool = False
    latency: bool = True
    slippage: bool = True
    seed: int = 42
    sampling: str = "exact"
    random_window_days: int = 7
    regime_sampling: bool = False
    tsv_enabled: bool = False
    tsv_chunk_days: int = 5
    warmup_ticks: int = 5000


@dataclass
class Environment:
    tick_source: TickSource
    clock: Clock
    broker: Broker
    replay_feed: Optional[ReplayFeed] = None
    replay_clock_impl: Optional[ReplayClockImpl] = None
    execution_model: Optional[ExecutionModel] = None
    archive: Optional[TickArchive] = None
    sampler: Optional[Sampler] = None
    tsv: Optional[TemporalShuffleValidator] = None
    metrics: Optional[ReplayMetrics] = None
    config: Optional[ReplayConfig] = None
    ledger: Optional[ParityLedger] = None


def build_live_environment(mt5_connector) -> Environment:
    clock = RealClock()
    tick_source = MT5TickSource(mt5_connector)
    broker = MT5Broker(mt5_connector)
    return Environment(
        tick_source=tick_source,
        clock=clock,
        broker=broker,
        metrics=ReplayMetrics(),
    )


def _prewarm_bars(archive: TickArchive, broker: PaperBroker, symbols: list[str],
                  start: datetime, end: datetime, min_bars: int = 550):
    """Pre-synthesize H1 bars using direct Polars aggregation (fast)."""
    import polars as pl
    for sym in sorted(symbols):
        logger.info(f"Prewarming bars for {sym}...")
        df = archive.load_range(sym, start, end)
        if df is None:
            logger.warning(f"No data for {sym}")
            continue
        try:
            collected = df.collect()
            n = len(collected)
            if n == 0:
                logger.warning(f"No ticks for {sym}")
                continue
            logger.info(f"Read {n} ticks for {sym}")
            # Build H1 bars via polars aggregation
            bars_df = collected.with_columns(
                (pl.col("time_sec") // 3600 * 3600).alias("hour")
            ).group_by("hour").agg([
                pl.col("bid").first().alias("open"),
                pl.col("bid").max().alias("high"),
                pl.col("bid").min().alias("low"),
                pl.col("bid").last().alias("close"),
                pl.len().alias("tick_volume"),
            ]).sort("hour")
            bar_records = bars_df.to_dicts()
            # Feed each bar into broker's bar buffer
            for br in bar_records:
                ts = br["hour"]
                broker._feed_tick_for_bars_manual(sym, {
                    "time_sec": ts,
                    "bid": br["open"],
                    "ask": br["open"],
                })
                broker._feed_tick_for_bars_manual(sym, {
                    "time_sec": ts + 1800,
                    "bid": br["high"],
                    "ask": br["high"],
                })
                broker._feed_tick_for_bars_manual(sym, {
                    "time_sec": ts + 3599,
                    "bid": br["close"],
                    "ask": br["close"],
                })
            bars = broker._build_h1_bars(sym)
            logger.info(f"Prewarmed {len(bars)} H1 bars for {sym} via polars agg")
            if len(bars) < min_bars:
                logger.warning(f"Only {len(bars)} bars for {sym} (need {min_bars})")
        except Exception as e:
            logger.warning(f"Prewarm failed for {sym}: {e}")


def build_replay_environment(config: ReplayConfig) -> Environment:
    archive = TickArchive()
    start = datetime.strptime(config.start, "%Y-%m-%d")
    end = datetime.strptime(config.end, "%Y-%m-%d")

    # Deterministic symbol ordering — always sorted by canonical rank
    config.symbols = sorted(config.symbols, key=lambda s: ReplayFeed._symbol_rank.get(s, 99))

    # Determine first tick timestamp for clock initialization
    first_ts = 0.0
    for sym in config.symbols:
        df = archive.load_range(sym, start, end)
        if df is not None:
            try:
                first = df.select(pl.col("time_sec")).collect().item(0, 0)
                first_ts = min(first_ts, float(first)) if first_ts else float(first)
            except Exception:
                pass

    replay_clock = ReplayClockImpl(speed_factor=config.speed, start_ts=first_ts)
    feed = ReplayFeed(clock=replay_clock, speed=config.speed, mode=config.mode)

    for sym in config.symbols:
        logger.info(f"Loading {sym} from {config.start} to {config.end}")
        feed.load_symbol_from_archive(sym, archive, start, end)

    feed.prepare()

    execution = ExecutionModel(seed=config.seed)
    execution.latency_enabled = config.latency
    execution.slippage_enabled = config.slippage

    tick_source = ReplayTickSource(feed)
    clock = replay_clock
    broker = PaperBroker(tick_source=tick_source, clock=clock, execution_model=execution)

    # Parity ledger
    ledger = ParityLedger(symbol=",".join(config.symbols), seed=config.seed)
    feed.set_ledger(ledger)
    broker.set_ledger(ledger)
    tick_source.set_ledger(ledger)

    # Pre-synthesize H1 bars from all loaded ticks
    _prewarm_bars(archive, broker, config.symbols, start, end)

    sampler = Sampler(archive, seed=config.seed)
    tsv = TemporalShuffleValidator(seed=config.seed)
    metrics = ReplayMetrics()

    if config.tsv_enabled:
        tsv._chunk_size_days = config.tsv_chunk_days

    env = Environment(
        tick_source=tick_source,
        clock=clock,
        broker=broker,
        replay_feed=feed,
        replay_clock_impl=replay_clock,
        execution_model=execution,
        archive=archive,
        sampler=sampler,
        tsv=tsv,
        metrics=metrics,
        config=config,
        ledger=ledger,
    )

    if config.burst:
        feed.mode = ReplayFeed.MODE_BURST

    return env


def build_paper_environment(mt5_connector, config: ReplayConfig = None) -> Environment:
    clock = RealClock()
    tick_source = MT5TickSource(mt5_connector)
    broker = PaperBroker(tick_source=tick_source, clock=clock)
    return Environment(
        tick_source=tick_source,
        clock=clock,
        broker=broker,
        metrics=ReplayMetrics(),
    )
