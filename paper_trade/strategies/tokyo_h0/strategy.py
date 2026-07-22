"""Tokyo Hour 0 — cross-pair mean reversion at 00:00 UTC.

Picks 3 most-declined pairs over last 15 min from an 18-pair universe, goes LONG.
Hold 15 min. Fires once daily at 00:00 UTC.

No volatility filter — tested across 9 months and 5 non-overlapping windows:
Oct 2025–Jun 2026: 401 trades, 79.3% avg WR, +4.74bp mean, all windows positive.
120-day stress test: 228 trades, 82.0% WR, +4.99bp mean.
Vol filter (66th-pct ATR) was tested and rejected — it blocks 72% of trades
while discarding profitable opportunities (filtered trades average +2.99bp).
"""
from paper_trade.core.config import register
from collections import deque
import time
import numpy as np

STRATEGY_NAME = "tokyo_h0"

CONFIG = {
    "name": STRATEGY_NAME,
    "mt5_account": None,
    "magic": 202401,
    "max_spread_pips": 2.5,
    "mt5_path": None,
    "pairs": [
        "EURUSD", "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
        "GBPUSD", "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
        "USDJPY", "USDCHF", "USDCAD",
        "AUDCAD", "AUDNZD",
    ],
    "hold_bars": 15,
    "session_start": 0,
    "session_end": 23,
    "max_concurrent": 3,
    "max_spread_mult": 2.0,
    "max_daily_loss": 1250,
    "lot_size": 0.25,
    "min_pairs": 8,
    "lookback_seconds": 900,
    "top_n": 3,
    "gap_threshold_pct": 0.5,
    "stop_loss_pips": 20,
}

register(STRATEGY_NAME, CONFIG)

_price_history = {}
_last_entry_date = None
_CACHE_SIZE = 500


def seed_history(feed):
    """Seed initial 15-minute price history from MT5 M1 bars."""
    for pair in CONFIG["pairs"]:
        rates = feed.copy_m1_history(pair, count=30)
        if rates:
            _price_history[pair] = deque(rates, maxlen=_CACHE_SIZE)


def _return_15m(pair, now):
    hist = _price_history.get(pair)
    if hist is None or len(hist) < 10:
        return None
    cutoff = now - CONFIG["lookback_seconds"]
    old_price = None
    for ts, p in hist:
        if ts <= cutoff:
            old_price = p
        else:
            break
    if old_price is None or old_price <= 0:
        return None
    cur_price = hist[-1][1]
    return (cur_price - old_price) / old_price


def _gap_check(pair, now):
    hist = list(_price_history.get(pair, []))
    if len(hist) < 5:
        return True
    cur_price = hist[-1][1]
    prev_close = hist[-2][1]
    pct_change = abs(cur_price - prev_close) / prev_close * 100
    return pct_change < CONFIG["gap_threshold_pct"]


def generate_signal(data):
    global _last_entry_date

    now = int(time.time())
    today = time.strftime("%Y-%m-%d", time.gmtime(now))

    for pair, values in data.items():
        if pair not in _price_history:
            _price_history[pair] = deque(maxlen=_CACHE_SIZE)
        mid = (values.get("bid", 0) + values.get("ask", 0)) / 2
        if mid > 0:
            _price_history[pair].append((now, mid))

    hm = time.gmtime(now)
    if hm.tm_hour != 0 or hm.tm_min != 0 or hm.tm_sec > 50:
        return None

    if _last_entry_date == today:
        return None
    _last_entry_date = today

    pairs_returns = []
    for pair in CONFIG["pairs"]:
        if pair not in _price_history:
            continue
        ret = _return_15m(pair, now)
        if ret is None:
            continue
        if not _gap_check(pair, now):
            continue
        pairs_returns.append((pair, ret))

    if len(pairs_returns) < CONFIG["min_pairs"]:
        return None

    pairs_returns.sort(key=lambda x: x[1])

    signals = []
    for pair, ret in pairs_returns[:CONFIG["top_n"]]:
        if ret >= 0:
            break
        confidence = min(0.95, abs(ret) * 200)
        if confidence < 0.30:
            continue
        signals.append({
            "pair": pair,
            "direction": 1,
            "confidence": round(confidence, 4),
            "metadata": {"ret_15m_bp": round(ret * 10000, 1)},
        })

    return signals if signals else None
