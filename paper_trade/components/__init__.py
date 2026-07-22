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

def pip_value_usd(pair, rate=None):
    """Return USD per pip for 1 standard lot."""
    if pair in ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"):
        return 10.0
    elif pair in ("USDJPY",):
        return 1000.0 / (rate or 100)
    elif pair.endswith("JPY"):
        return 1000.0 / (rate or 140)
    elif pair in ("USDCHF", "USDCAD"):
        return 10.0 / (rate or 1.0)
    elif pair.endswith("CHF") or pair.endswith("CAD"):
        return 10.0 / (rate or 1.0)
    else:
        return 10.0
