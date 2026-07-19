"""Historical M5 data loader for WLS walk-forward validation.

Auto-detects pairs available in MT5 (exotic JPY/CHF crosses are often missing).
"""

import numpy as np
import MetaTrader5 as mt5


ALL_PAIRS = [
    "EURUSD", "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPUSD", "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "USDJPY", "USDCHF", "USDAUD", "USDCAD", "USDNZD",
    "JPYCHF", "JPYAUD", "JPYCAD", "JPYNZD",
    "CHFAUD", "CHFCAD", "CHFNZD",
    "AUDCAD", "AUDNZD", "CADNZD",
]

ALL_PAIR_MAP = {
    "EURUSD": ("EUR", "USD"), "EURGBP": ("EUR", "GBP"), "EURJPY": ("EUR", "JPY"),
    "EURCHF": ("EUR", "CHF"), "EURAUD": ("EUR", "AUD"), "EURCAD": ("EUR", "CAD"),
    "EURNZD": ("EUR", "NZD"), "GBPUSD": ("GBP", "USD"), "GBPJPY": ("GBP", "JPY"),
    "GBPCHF": ("GBP", "CHF"), "GBPAUD": ("GBP", "AUD"), "GBPCAD": ("GBP", "CAD"),
    "GBPNZD": ("GBP", "NZD"), "USDJPY": ("USD", "JPY"), "USDCHF": ("USD", "CHF"),
    "USDAUD": ("USD", "AUD"), "USDCAD": ("USD", "CAD"), "USDNZD": ("USD", "NZD"),
    "JPYCHF": ("JPY", "CHF"), "JPYAUD": ("JPY", "AUD"), "JPYCAD": ("JPY", "CAD"),
    "JPYNZD": ("JPY", "NZD"), "CHFAUD": ("CHF", "AUD"), "CHFCAD": ("CHF", "CAD"),
    "CHFNZD": ("CHF", "NZD"), "AUDCAD": ("AUD", "CAD"), "AUDNZD": ("AUD", "NZD"),
    "CADNZD": ("CAD", "NZD"),
}

CURRENCIES = ["EUR", "USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]


def discover_available_pairs() -> list[str]:
    available = []
    for sym in ALL_PAIRS:
        info = mt5.symbol_info(sym)
        if info is not None:
            available.append(sym)
    return available


def load_m5_bars(symbol: str, date_from, date_to) -> np.ndarray | None:
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, date_from, date_to)
    if rates is None or len(rates) == 0:
        return None
    return rates


def build_return_matrix(
    rates_map: dict[str, np.ndarray],
    symbols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    valid_symbols = [s for s in symbols if s in rates_map and rates_map[s] is not None and len(rates_map[s]) >= 2]
    if not valid_symbols:
        return np.array([]), np.array([])
    min_len = min(len(rates_map[s]) for s in valid_symbols)
    T = min_len
    n = len(valid_symbols)
    returns = np.zeros((T, n))
    timestamps = np.zeros(T)
    for j, sym in enumerate(valid_symbols):
        rates = rates_map[sym]
        close = np.array([float(r[4]) for r in rates[:T]], dtype=np.float64)
        log_rets = np.diff(np.log(np.maximum(close, 1e-12)))
        if len(log_rets) > 0:
            n_to_copy = min(len(log_rets), T - 1)
            returns[1:n_to_copy+1, j] = log_rets[:n_to_copy]
        for i in range(T):
            timestamps[i] = float(rates[i][0])
    return returns, timestamps, valid_symbols


def build_design_matrix(symbols: list[str]) -> tuple[np.ndarray, list[str]]:
    n = len(CURRENCIES)
    A = np.zeros((len(symbols), n))
    for i, sym in enumerate(symbols):
        base, quote = ALL_PAIR_MAP[sym]
        A[i, CURRENCIES.index(base)] = 1.0
        A[i, CURRENCIES.index(quote)] = -1.0
    return A, symbols
