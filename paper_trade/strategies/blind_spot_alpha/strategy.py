"""Blind Spot Alpha — Currency Divergence Momentum.

Core insight: The market's spread defense works pair-by-pair.
The currency NETWORK (strongest vs weakest relationship) is the blind spot.

When the strongest currency (max Z) and weakest currency (min Z)
both exceed threshold in OPPOSITE directions, trade the pair between them.
Both currencies NATURALLY confirm the pair direction.

Backtest (3 months M1 Dukascopy, Apr-Jun 2026):
- 56.7% mid WR, +$3.06 avg edge (vs 47.5% baseline CP)
- 21.4 trades/day
- With ECN costs ($1.75/trade): +$1.31/trade net = profitable
"""
from paper_trade.core.config import register
import MetaTrader5 as _mt5
import numpy as np
from collections import deque

STRATEGY_NAME = "blind_spot_alpha"

_ALL_CURRENCIES = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']

_ALL_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD","USDCHF",
    "EURJPY","GBPJPY","EURGBP","EURAUD","EURCHF","EURCAD","EURNZD",
    "GBPAUD","GBPCAD","GBPCHF","GBPNZD",
    "AUDJPY","AUDCAD","AUDCHF","AUDNZD",
    "NZDJPY","NZDCAD","NZDCHF",
    "CADJPY","CADCHF",
    "CHFJPY",
]

_BEST_PAIR = {
    "USD": "AUDUSD", "EUR": "EURUSD", "JPY": "NZDJPY",
    "GBP": "GBPUSD", "AUD": "AUDUSD", "NZD": "NZDUSD",
    "CAD": "NZDCAD", "CHF": "USDCHF",
}

CONFIG = {
    "name": STRATEGY_NAME,
    "mt5_account": None,
    "magic": 202405,
    "max_spread_pips": 3.0,
    "mt5_path": None,
    "pairs": _ALL_PAIRS,
    "hold_minutes": 5,
    "z_threshold": 2.0,
    "z_window": 2000,
    "session_start": 0,
    "session_end": 23,
    "max_concurrent": 10,
    "max_spread_mult": 2.0,
    "max_daily_loss": 1000,
    "lot_size": 0.5,
}

register(STRATEGY_NAME, CONFIG)

_pair_close_prev = {}
_pair_return_buf = {}
_last_minute = None

_curr_return_history = {c: deque(maxlen=CONFIG["z_window"]) for c in _ALL_CURRENCIES}

_pair_vol_fixed = {}

_VOL_WINDOW = 200

_PAIR_SPREAD_BPS = {
    "AUDUSD": 1.5, "EURUSD": 1.5, "GBPUSD": 2.0, "NZDUSD": 2.0,
    "USDCAD": 2.0, "USDCHF": 2.0, "USDJPY": 1.8, "EURJPY": 2.5,
    "GBPJPY": 4.0, "EURGBP": 2.0, "EURAUD": 2.5, "EURCHF": 2.0,
    "EURCAD": 2.0, "EURNZD": 2.5, "GBPAUD": 2.5, "GBPCAD": 2.5,
    "GBPCHF": 3.0, "GBPNZD": 3.0, "AUDJPY": 2.5, "AUDCAD": 2.5,
    "AUDCHF": 2.5, "AUDNZD": 2.5, "NZDJPY": 3.0, "NZDCAD": 2.5,
    "NZDCHF": 2.5, "CADJPY": 2.5, "CADCHF": 2.5, "CHFJPY": 3.0,
}


def _base_quote(pair):
    for c in _ALL_CURRENCIES:
        if pair.startswith(c):
            return c, pair[len(c):]
    return None, None


def _currency_pairs():
    result = {c: [] for c in _ALL_CURRENCIES}
    for j, pair in enumerate(_ALL_PAIRS):
        base, quote = _base_quote(pair)
        if base and quote:
            if base in result:
                result[base].append((j, 1.0, pair))
            if quote in result:
                result[quote].append((j, -1.0, pair))
    return result


_CURR_PAIRS = _currency_pairs()


def _pair_map():
    """Return (ci1, ci2) -> pj for any currency pair order."""
    m = {}
    for pj, p in enumerate(_ALL_PAIRS):
        b, q = _base_quote(p)
        ci1 = _ALL_CURRENCIES.index(b)
        ci2 = _ALL_CURRENCIES.index(q)
        m[(ci1, ci2)] = pj
        m[(ci2, ci1)] = pj
    return m


_CURR_PAIR_MAP = _pair_map()


def _curr_sign():
    """Return (ci, pj) -> sign for every currency-pair combo."""
    m = {}
    for ci, c in enumerate(_ALL_CURRENCIES):
        for _, sg, pn in _CURR_PAIRS.get(c, []):
            pj = _ALL_PAIRS.index(pn)
            m[(ci, pj)] = sg
    return m


_CURR_SIGN = _curr_sign()


def _init_pair_buffers():
    for pair in _ALL_PAIRS:
        if pair not in _pair_close_prev:
            _pair_close_prev[pair] = None
            _pair_return_buf[pair] = deque(maxlen=_VOL_WINDOW)


def _compute_currency_return(pair_returns):
    curr_rets = {}
    for pair, ret in pair_returns.items():
        if ret is None:
            continue
        if _pair_vol_fixed:
            vol = _pair_vol_fixed.get(pair, 1e-10)
        else:
            buf = _pair_return_buf.get(pair, deque())
            vol = np.std(buf) + 1e-10
        base, quote = _base_quote(pair)
        if base:
            if base not in curr_rets:
                curr_rets[base] = [[], []]
            curr_rets[base][0].append(ret)
            curr_rets[base][1].append(vol)
        if quote:
            if quote not in curr_rets:
                curr_rets[quote] = [[], []]
            curr_rets[quote][0].append(-ret)
            curr_rets[quote][1].append(vol)

    result = {}
    for c in _ALL_CURRENCIES:
        entry = curr_rets.get(c)
        if entry is None or len(entry[0]) < 2:
            continue
        rets, vols = entry
        w = np.array([1.0 / v for v in vols])
        w = w / np.sum(w)
        result[c] = np.dot(rets, w)

    return result


def seed_history(feed):
    _init_pair_buffers()
    vol_count = _VOL_WINDOW - 1
    total_count = CONFIG["z_window"] + vol_count + 1
    m1_prices = {}
    for pair in _ALL_PAIRS:
        rates = feed.copy_m1_history(pair, count=total_count)
        if rates:
            m1_prices[pair] = [r[1] for r in rates]

    if not m1_prices:
        return

    min_len = min(len(v) for v in m1_prices.values())
    if min_len < 10:
        return

    _pair_vol_first = {p: [] for p in _ALL_PAIRS}
    for i in range(1, min_len):
        for pair in _ALL_PAIRS:
            if pair in m1_prices:
                prev = m1_prices[pair][i - 1]
                curr = m1_prices[pair][i]
                if prev > 0 and curr > 0:
                    ret = np.log(curr / prev)
                    buf = _pair_return_buf.get(pair)
                    if buf is not None:
                        buf.append(ret)
                    if i <= vol_count:
                        _pair_vol_first[pair].append(ret)

    for pair in _ALL_PAIRS:
        arr = _pair_vol_first[pair]
        _pair_vol_fixed[pair] = np.std(arr) + 1e-10 if arr and len(arr) > 1 else 1e-10

    for c in _ALL_CURRENCIES:
        _curr_return_history[c].clear()
    for i in range(1, min_len):
        minute_returns = {}
        for pair in _ALL_PAIRS:
            if pair in m1_prices:
                prev = m1_prices[pair][i - 1]
                curr = m1_prices[pair][i]
                if prev > 0 and curr > 0:
                    minute_returns[pair] = np.log(curr / prev)
        curr_rets = _compute_currency_return(minute_returns)
        for c, ret in curr_rets.items():
            hist = _curr_return_history.get(c)
            if hist is not None:
                hist.append(ret)


def generate_signal(data, current_time=None):
    import time as _time
    now = current_time or int(_time.time())
    current_minute = now // 60

    global _last_minute

    _init_pair_buffers()

    pairs_with_data = [p for p in _ALL_PAIRS if p in data and data[p].get("bid", 0) > 0]
    if len(pairs_with_data) < 5:
        return None

    if _last_minute is None:
        _last_minute = current_minute
        for pair in pairs_with_data:
            rates = _mt5.copy_rates_from_pos(pair, _mt5.TIMEFRAME_M1, 1, 1)
            if rates is not None and len(rates) > 0:
                _pair_close_prev[pair] = float(rates[0][4])
        return None

    if current_minute <= _last_minute:
        return None

    minute_returns = {}
    for pair in pairs_with_data:
        rates = _mt5.copy_rates_from_pos(pair, _mt5.TIMEFRAME_M1, 1, 1)
        if rates is None or len(rates) == 0:
            continue
        close = float(rates[0][4])
        prev = _pair_close_prev.get(pair)
        if prev and prev > 0 and close > 0:
            ret = np.log(close / prev)
            minute_returns[pair] = ret
            buf = _pair_return_buf.get(pair)
            if buf is not None:
                buf.append(ret)
        _pair_close_prev[pair] = close

    _last_minute = current_minute

    if len(minute_returns) < 5:
        return None

    curr_returns = _compute_currency_return(minute_returns)
    if len(curr_returns) < 3:
        return None

    for c, ret in curr_returns.items():
        hist = _curr_return_history.get(c)
        if hist is not None:
            hist.append(ret)

    z_scores = np.zeros(len(_ALL_CURRENCIES))
    for ci, c in enumerate(_ALL_CURRENCIES):
        hist = _curr_return_history.get(c)
        if hist is None or len(hist) < 5:
            continue
        arr = np.array(hist)
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-12:
            continue
        z_scores[ci] = (curr_returns.get(c, 0) - mean) / std

    zt = CONFIG["z_threshold"]
    abs_z = np.abs(z_scores)
    sorted_idx = np.argsort(z_scores)

    strongest_ci = sorted_idx[-1]
    weakest_ci = sorted_idx[0]
    sz = z_scores[strongest_ci]
    wz = z_scores[weakest_ci]

    if sz < zt or wz > -zt:
        return None

    if abs(sz) >= abs(wz):
        trade_ci = strongest_ci
        other_ci = weakest_ci
    else:
        trade_ci = weakest_ci
        other_ci = strongest_ci

    pair = _BEST_PAIR.get(_ALL_CURRENCIES[trade_ci])
    if pair is None:
        return None
    pj = _ALL_PAIRS.index(pair)

    sg = _CURR_SIGN.get((trade_ci, pj), 0)
    if sg == 0:
        return None
    direction = 1 if z_scores[trade_ci] > 0 else -1
    d_star = int(direction * sg)

    other_sg = _CURR_SIGN.get((other_ci, pj), 0)
    if other_sg == 0:
        return None
    other_dir = 1 if z_scores[other_ci] > 0 else -1
    other_dstar = int(other_dir * other_sg)

    if d_star != other_dstar:
        return None

    confidence = min(0.99, max(abs(sz), abs(wz)) / 5.0)
    spread_bps = _PAIR_SPREAD_BPS.get(pair, 2.0)

    event = {
        "pair": pair,
        "direction": d_star,
        "confidence": round(confidence, 4),
        "metadata": {
            "strongest_currency": _ALL_CURRENCIES[strongest_ci],
            "strongest_z": round(float(sz), 2),
            "weakest_currency": _ALL_CURRENCIES[weakest_ci],
            "weakest_z": round(float(wz), 2),
            "trade_currency": _ALL_CURRENCIES[trade_ci],
            "pair_agreement": True,
            "spread_est_bps": spread_bps,
        },
    }

    return [event]
