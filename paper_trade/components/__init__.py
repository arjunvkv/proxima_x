"""Reusable metric calculations. Same formulas used across all strategies."""
import numpy as np

def sharpe(pnls, periods_per_year=1440, hold_bars=3):
    if len(pnls) < 2:
        return 0.0
    s = np.std(pnls)
    if s < 1e-10:
        return 0.0
    return np.mean(pnls) / s * np.sqrt(periods_per_year / hold_bars)

def win_rate(pnls):
    if len(pnls) == 0:
        return 0.0
    return float(np.mean(pnls > 0) * 100)

def profit_factor(pnls):
    gross_win = np.sum(pnls[pnls > 0])
    gross_loss = abs(np.sum(pnls[pnls < 0]))
    if gross_loss < 1e-10:
        return float("inf") if gross_win > 0 else 0.0
    return float(gross_win / gross_loss)

def max_drawdown(cumulative_pnls):
    if len(cumulative_pnls) == 0:
        return 0.0
    running_max = np.maximum.accumulate(cumulative_pnls)
    dd = cumulative_pnls - running_max
    return float(abs(np.min(dd)))

def var(pnls, percentile=5):
    if len(pnls) == 0:
        return 0.0
    return float(np.percentile(pnls, percentile))

def consecutive_streak(pnls, win=True):
    streaks = []
    current = 0
    for p in pnls:
        if (p > 0) == win:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return max(streaks) if streaks else 0

_QUOTE_TO_USD_FALLBACK = {
    "AUD": 6.7, "NZD": 5.8, "GBP": 12.5, "EUR": 10.5, "CHF": 10.0, "CAD": 7.8,
}

def pip_value_usd(pair, rate=None):
    """Return USD per pip for 1 standard lot.
    rate param: quote/USD rate for cross pairs (e.g. AUDUSD for XXXAUD)."""
    base, quote = pair[:3], pair[-3:]
    if pair in ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"):
        return 10.0
    if quote == "JPY":
        return 1000.0 / (rate or 100)
    if quote == "USD":
        return 10.0
    if base == "USD":
        return 10.0 / (rate or 1.0)
    if rate:
        return 10.0 * rate if quote in ("EUR", "GBP", "AUD", "NZD") else 10.0 / rate
    return _QUOTE_TO_USD_FALLBACK.get(quote, 10.0)
