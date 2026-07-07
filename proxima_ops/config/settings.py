import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    # MT5
    mt5_path: str = os.environ.get("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    mt5_account: int = int(os.environ.get("MT5_ACCOUNT", "5051788806"))
    mt5_password: str = os.environ.get("MT5_PASSWORD", "A!J7DyMk")
    mt5_server: str = os.environ.get("MT5_SERVER", "MetaQuotes-Demo")
    mt5_timeout_ms: int = 5000

    # research-validated execution symbols (full weight)
    execution_symbols: list[str] = field(default_factory=lambda: [
        "EURJPY", "USDJPY"])

    # candidate symbols (provisional weight, eligible for promotion)
    candidate_symbols: list[str] = field(default_factory=lambda: [
        "EURUSD","GBPUSD","AUDUSD","USDCAD","USDCHF","NZDUSD",
        "GBPJPY","AUDJPY","CADJPY","CHFJPY","NZDJPY",
        "EURGBP","EURCHF","EURAUD","EURCAD","EURNZD",
        "GBPCHF","GBPAUD","GBPCAD","GBPNZD",
        "AUDNZD","AUDCAD","AUDCHF","NZDCAD","NZDCHF",
        "CADCHF"])

    # all symbols (execution + candidate)
    symbols: list[str] = field(default_factory=lambda: [
        "EURJPY", "USDJPY",
        "EURUSD","GBPUSD","AUDUSD","USDCAD","USDCHF","NZDUSD",
        "GBPJPY","AUDJPY","CADJPY","CHFJPY","NZDJPY",
        "EURGBP","EURCHF","EURAUD","EURCAD","EURNZD",
        "GBPCHF","GBPAUD","GBPCAD","GBPNZD",
        "AUDNZD","AUDCAD","AUDCHF","NZDCAD","NZDCHF",
        "CADCHF"])

    # observation-only shadow symbols (tick capture, ECDF, spread stats, no execution)
    shadow_symbols: list[str] = field(default_factory=lambda: [
        "EURUSD","GBPUSD","AUDUSD","USDCAD","USDCHF","NZDUSD",
        "GBPJPY","AUDJPY","CADJPY","CHFJPY","NZDJPY",
        "EURGBP","EURCHF","EURAUD","EURCAD","EURNZD",
        "GBPCHF","GBPAUD","GBPCAD","GBPNZD",
        "AUDNZD","AUDCAD","AUDCHF","NZDCAD","NZDCHF",
        "CADCHF"])

    # V6 validated parameters (FROZEN — do not modify)
    # Bootstrap/demo mode uses lower threshold for immediate activation
    threshold: float = 0.55
    frequency_target: int = 30
    risk_per_trade: float = 0.0025  # 0.25%

    # Default per-symbol threshold (fallback for unconfigured symbols)
    default_symbol_threshold: float = 0.55

    # Default max spread points (fallback for unconfigured symbols)
    default_max_spread_points: int = 25

    # V6 Portfolio — Top-3 Rotation with 3-bar Position Lock
    portfolio_mode: str = "TOP_3_ROTATION"  # TOP_3_ROTATION | H20_ONLY
    max_positions_active: int = 6           # expanded for 26-symbol universe
    position_lock_bars: int = 12            # 12-bar lock (was 1) — aligns with MIN_HOLD_TICKS_FLIP=12 to prevent premature re-entry

    # Per-symbol ECDF rank thresholds — set to each symbol's own P80-P90 distribution
    # (was: fixed high values that structurally excluded EURJPY, EURUSD, GBPJPY)
    # Computed from 504-bar trailing ECDF on market parquet data
    # (see PROXIMA_NORMALIZATION_BIAS_AUDIT.md for per-asset distribution analysis)
    symbol_thresholds: dict = field(default_factory=lambda: {
        "EURJPY": 0.55,
        "GBPJPY": 0.78,
        "USDJPY": 0.88,
        "EURUSD": 0.65,
    })

    # Activation mode: "FIXED_GLOBAL" | "PER_SYMBOL_ECDF" (normalization bias fix)
    activation_mode: str = "PER_SYMBOL_ECDF"

    # Fallback spread when tick returns 0 (stale read during low liquidity)
    min_spread_fallback: dict = field(default_factory=lambda: {
        "EURUSD": 1, "GBPUSD": 1, "USDCHF": 1, "USDCAD": 1,
        "AUDUSD": 1, "NZDUSD": 1, "EURJPY": 2, "USDJPY": 2,
        "GBPJPY": 3, "EURGBP": 1, "EURCHF": 1, "AUDJPY": 2,
        "CADJPY": 2, "CHFJPY": 3, "NZDJPY": 2, "GBPCHF": 3})

    # Execution
    max_spread_points: dict = field(default_factory=lambda: {
        "EURJPY": 30, "USDJPY": 30, "GBPJPY": 40,
        "XAUUSD": 100, "EURUSD": 20})
    max_slippage_points: int = 10
    max_positions_total: int = 5
    max_positions_per_symbol: int = 1

    # Telegram (paused — set TELEGRAM_TOKEN env var to enable)
    telegram_token: str = os.environ.get("TELEGRAM_TOKEN", "")
    telegram_chat_id: str = os.environ.get("TELEGRAM_CHAT_ID", "")
    telegram_auth_users: list[int] = field(default_factory=lambda: [
        int(x) for x in os.environ.get("TELEGRAM_AUTH_USERS", "").split(",") if x])

    # DuckDB
    db_path: str = os.environ.get("OPS_DB_PATH",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "proxima_ops.duckdb"))

    # Monitoring
    deployment_score_target: float = 0.75
    min_pp: float = 0.55
    min_sharpe: float = 1.0
    max_frequency_cv: float = 0.50
    max_drawdown_pct: float = 0.12

    # Deployment mode
    deployment_mode: str = "LOCAL_CURRENT"  # LOCAL_CURRENT | GLOBAL_TOP1 | GLOBAL_ALL_QUALIFIED
    global_rank_threshold: float = 80.0

    # Demo validation duration
    demo_days_target: int = 30

    # Fixed volume override — if > 0, all trades use this exact volume instead of risk-based calculation
    fixed_volume: float = 0.7

    # Profit target — when combined unrealized PnL of all active positions reaches this, close all
    profit_target_close: float = 50.0

    # Controlled exposure / exploration mode — randomly samples low-conviction trades for data collection
    exploration_mode: bool = True
    exploration_max_per_cycle: int = 1
    exploration_min_lot: float = 0.01
    exploration_per_symbol_cooldown: int = 10  # cycles between exploration trades per symbol

    @classmethod
    def load(cls) -> "Settings":
        return cls()


SETTINGS = Settings.load()
