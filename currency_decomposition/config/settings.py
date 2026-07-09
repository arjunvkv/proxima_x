from typing import Dict, Tuple

CURRENCY_LIST = ["EUR", "USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]

BASE_CURRENCY_MAP: Dict[str, Tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"),
    "EURGBP": ("EUR", "GBP"),
    "EURJPY": ("EUR", "JPY"),
    "EURCHF": ("EUR", "CHF"),
    "EURAUD": ("EUR", "AUD"),
    "EURCAD": ("EUR", "CAD"),
    "EURNZD": ("EUR", "NZD"),
    "GBPUSD": ("GBP", "USD"),
    "GBPJPY": ("GBP", "JPY"),
    "GBPCHF": ("GBP", "CHF"),
    "GBPAUD": ("GBP", "AUD"),
    "GBPCAD": ("GBP", "CAD"),
    "GBPNZD": ("GBP", "NZD"),
    "USDJPY": ("USD", "JPY"),
    "USDCHF": ("USD", "CHF"),
    "USDAUD": ("USD", "AUD"),
    "USDCAD": ("USD", "CAD"),
    "USDNZD": ("USD", "NZD"),
    "JPYCHF": ("JPY", "CHF"),
    "JPYAUD": ("JPY", "AUD"),
    "JPYCAD": ("JPY", "CAD"),
    "JPYNZD": ("JPY", "NZD"),
    "CHFAUD": ("CHF", "AUD"),
    "CHFCAD": ("CHF", "CAD"),
    "CHFNZD": ("CHF", "NZD"),
    "AUDCAD": ("AUD", "CAD"),
    "AUDNZD": ("AUD", "NZD"),
    "CADNZD": ("CAD", "NZD"),
}

WLS_REGULARIZATION: float = 0.01
MIN_SOLVE_PAIRS = 10

SYMBOLS = list(BASE_CURRENCY_MAP.keys())

MAX_POSITIONS = 3
MAX_EXPOSURE_PER_CURRENCY = 200_000
STOP_LOSS_PIPS = 30
TAKE_PROFIT_PIPS = 60
MAX_HOLD_HOURS = 12
MIN_CONFIDENCE = 0.10
SIGNAL_EXPIRY_SECONDS = 300

MAX_TICK_AGE_SECONDS = 120
MIN_SYMBOL_COVERAGE = 0.70
MAX_SPREAD_MULTIPLE = 5

INITIAL_CAPITAL = 10_000
LOT_SIZE = 0.7
MAX_TOTAL_LOTS = 2.1
PROFIT_TARGET = 50.0
PROFIT_COOLDOWN = 300

EXECUTION_MODE = "live"  # "paper" for simulated, "live" for real MT5 order placement
WLS_DIRECT_MODE = True  # bypass SWPS, GATE, and other WLS-blocking layers

# Burst conflict threshold (normalized participation spread × direction).
# Hypotheses with conflict_score < -BURST_CONFLICT_THRESHOLD are rejected.
# 0.30 = reject when participation strongly contradicts WLS direction.
# Higher = more aggressive rejection, lower = more permissive.

BURST_CONFLICT_THRESHOLD = 0.30
MAX_DAILY_LOSS = 100
MAX_DRAWDOWN = 500
MAX_CURRENCY_POSITIONS = 2
DRS_LAMBDA_DECAY = 0.05
DRS_REPLACE_MARGIN = 0.10
DRS_SLOT_INERTIA = [1.0, 0.85, 0.7]
MAX_CURRENCY_FACTOR_EXPOSURE = 2
DRS_CANDIDATE_POOL_SIZE = 10
MIN_GRAPH_CONNECTIVITY = 0.45
DIRECTION_PERSISTENCE_CYCLES = 3

# Short Window Persistence Scanner
# Number of WLS solve snapshots needed before SWPS activates.
# With 5s solve interval:
#   SWPS_WINDOW_SIZE=5  → ~25s capture → usable at first 30s decision
#   SWPS_WINDOW_SIZE=10 → ~50s capture → higher confidence, delayed entry

SWPS_WINDOW_SIZE = 5

# Minimum persistence score threshold.
# Only pairs with score >= this value trigger SWPS override.
# 0.75 = strong directional consistency across all snapshots.

SWPS_MIN_SCORE = 0.75
