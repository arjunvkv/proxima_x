"""Combined DC + 10sMR signal for paper trading.

DC: Dealer Capitulation — 1-min bar z-score + spread-widen + recovery.
     EURUSD z>1.5, GBPJPY z>1.75, EURJPY z>1.75, hold 10 min.
10sMR: Mean reversion on extreme 10s bar moves. EURUSD only, z>3.5, hold 3 min.
"""
from paper_trade.core.config import register
from collections import deque
import numpy as np
import time

STRATEGY_NAME = "combined_dc_mr"

CONFIG = {
    "name": STRATEGY_NAME,
    "mt5_account": None,
    "magic": 202409,
    "max_spread_pips": 2.0,
    "mt5_path": None,
    "pairs": ["EURUSD", "EURJPY", "GBPJPY"],
    "session_start": 0,
    "session_end": 23,
    "max_concurrent": 5,
    "max_spread_mult": 1.5,
    "max_daily_loss": 750,
    "lot_size": 0.5,
    "dc_z_thresholds": {"EURUSD": 1.5, "EURJPY": 1.75, "GBPJPY": 1.75},
    "dc_spread_ratio": 2.0,
    "dc_recovery_mult": 1.3,
    "dc_hold_seconds": 600,
    "mr_z_threshold": 3.5,
    "mr_hold_seconds": 180,
}

register(STRATEGY_NAME, CONFIG)

class _MRTracker:
    def __init__(self):
        self.open = None
        self.close = None
        self.last_key = None
        self.returns = deque(maxlen=100)

class _DCTracker:
    def __init__(self):
        self.open = None
        self.close = None
        self.last_key = None
        self.returns = deque(maxlen=60)
        self.spreads = deque(maxlen=60)
        self.active = False
        self.z = 0.0
        self.spike_time = 0
        self.median_spread = 0.0

_mr = {}
_dc = {}

def seed_history(feed):
    for pair in CONFIG["pairs"]:
        _mr[pair] = _MRTracker()
        _dc[pair] = _DCTracker()

def generate_signal(data):
    now = int(time.time())
    signals = []

    for pair, values in data.items():
        if pair not in CONFIG["pairs"]:
            continue

        bid = values.get("bid", 0)
        ask = values.get("ask", 0)
        if bid <= 0 or ask <= 0:
            continue

        mid = (bid + ask) / 2
        spread = ask - bid

        mr = _mr[pair]
        dc = _dc[pair]

        key_10s = int(now // 10)
        key_1m = int(now // 60)

        # --- 10s MR (EURUSD only — pair spread cost requirement) ---
        if key_10s != mr.last_key:
            if mr.open is not None and mr.last_key is not None:
                ret = mr.close - mr.open
                mr.returns.append(ret)
            mr.open = mid
            mr.close = mid
            mr.last_key = key_10s

            if pair == "EURUSD" and len(mr.returns) >= 50:
                arr = np.array(list(mr.returns))
                latest = arr[-1]
                mean = arr[:-1].mean()
                std = arr[:-1].std(ddof=0)
                if std > 1e-10:
                    z = (latest - mean) / std
                    thr = CONFIG["mr_z_threshold"]
                    if abs(z) > thr:
                        direction = -1 if z > 0 else 1
                        conf = min(0.95, (abs(z) - thr) * 0.12)
                        signals.append({
                            "pair": pair, "direction": direction,
                            "confidence": round(conf + 0.1, 4),
                            "signal_type": "MR",
                            "hold_seconds": CONFIG["mr_hold_seconds"],
                            "metadata": {"z": round(float(z), 2)},
                        })
        else:
            mr.close = mid

        # --- 1m DC ---
        dc.spreads.append(spread)

        if key_1m != dc.last_key:
            if dc.open is not None and dc.last_key is not None:
                ret = dc.close - dc.open
                dc.returns.append(ret)
            dc.open = mid
            dc.close = mid
            dc.last_key = key_1m

            if len(dc.returns) >= 20 and len(dc.spreads) >= 20:
                med_sp = float(np.median(list(dc.spreads)))
                if med_sp > 1e-10:
                    arr = np.array(list(dc.returns))
                    latest = arr[-1]
                    mean = arr[:-1].mean()
                    std = arr[:-1].std(ddof=0)
                    if std > 1e-10:
                        z = (latest - mean) / std
                        sp_ratio_val = spread / med_sp
                        thr = CONFIG["dc_z_thresholds"].get(pair, 1.75)

                        if abs(z) > thr and sp_ratio_val > CONFIG["dc_spread_ratio"]:
                            dc.active = True
                            dc.z = z
                            dc.spike_time = now
                            dc.median_spread = med_sp
        else:
            dc.close = mid

        # --- DC recovery check (fires between bar boundaries) ---
        if dc.active:
            if len(dc.spreads) >= 20:
                if dc.median_spread <= 1e-10:
                    dc.active = False
                else:
                    recovery_thr = CONFIG["dc_recovery_mult"] * dc.median_spread
                    if spread < recovery_thr:
                        dc.active = False
                        direction = -1 if dc.z > 0 else 1
                        conf = min(0.95, abs(dc.z) * 0.15)
                        signals.append({
                            "pair": pair, "direction": direction,
                            "confidence": round(conf + 0.2, 4),
                            "signal_type": "DC",
                            "hold_seconds": CONFIG["dc_hold_seconds"],
                            "metadata": {"z": round(float(dc.z), 2)},
                        })
                    elif now - dc.spike_time > 120:
                        dc.active = False

    return signals if signals else None
