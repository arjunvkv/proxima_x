"""audit_7_eas/ea_ports.py — faithful M5-bar ports of the 7 v106 EAs.

ANTI-LOOKAHEAD CONTRACT (the core of this audit):
  * A signal is computed ONLY from bars CLOSED at or before bar i (the EA
    reads CopyRates(...,0) inside OnTick at a new-bar boundary, so rates[0] is
    the just-closed bar). The signal therefore becomes actionable at bar i's
    close.
  * FILL happens at the OPEN of bar i+1 — never inside the signal bar. Entry
    index is ALWAYS signal bar + 1.
  * HOLD exits mirror the EA: held increments on each new bar; when held >= N
    the EA closes at that boundary, i.e. our position fills at entry bar E's
    open and closes at the open of bar E+N (no SL/TP) or intrabar on
    SL/TP touch during bars E..E+N-1.
  * SL/TP: intrabar touches use bar high/low. If both SL and TP touch in the
    same bar we assume the WORSE outcome (SL first) — MT5's stop-first
    convention; conservative for the strategy.
  * Blocked-conflict: one position per symbol at a time (EA refuses a second).

Each port returns list of TRADE dicts. Prices are MID; the harness applies
spread + commission via ExecutionCost-equivalent math.
"""
from __future__ import annotations
import math
from datetime import datetime, timezone


# ---------------------------------------------------------------- helpers ---
def _sl_tp_dist(symbol: str) -> tuple[float, float]:
    """JPY pairs: 0.35/0.45; others: 0.0035/0.0045 (EA constant)."""
    if "JPY" in symbol:
        return 0.35, 0.45
    return 0.0035, 0.0045


def _simulate_exit(bars: list[dict], entry_idx: int, side: str, hold_n: int,
                   sl_dist: float | None, tp_dist: float | None, symbol: str) -> dict:
    """Walk bars from entry (fill at entry_idx's OPEN); SL/TP + hold exits.

    entry_idx is the bar whose OPEN is the fill price (signal bar + 1).
    sl_dist/tp_dist are DISTANCES from entry (EA uses 0.35/0.45 JPY, else
    0.0035/0.0045). Absolute levels are derived from entry and side:
      BUY:  SL = entry - sl_dist, TP = entry + tp_dist
      SELL: SL = entry + sl_dist, TP = entry - tp_dist
    SL/TP monitored intrabar on bars entry_idx .. entry_idx+hold_n-1; if no
    touch, position closes at the open of bar entry_idx+hold_n.
    """
    entry_idx = min(entry_idx, len(bars) - 1)
    entry = bars[entry_idx]["open"]
    dirn = 1 if side == "BUY" else -1
    sl = (entry - sl_dist) if (sl_dist is not None and side == "BUY") else \
         (entry + sl_dist if sl_dist is not None else None)
    tp = (entry + tp_dist) if (tp_dist is not None and side == "BUY") else \
         (entry - tp_dist if tp_dist is not None else None)
    last_i = min(entry_idx + hold_n, len(bars) - 1)
    for k in range(entry_idx, last_i + 1):
        b = bars[k]
        hi, lo = b["high"], b["low"]
        touch_sl = sl is not None and ((lo <= sl) if dirn == 1 else (hi >= sl))
        touch_tp = tp is not None and ((hi >= tp) if dirn == 1 else (lo <= tp))
        if touch_sl and touch_tp:
            exit_price, reason = sl, "SL"
        elif touch_sl:
            exit_price, reason = sl, "SL"
        elif touch_tp:
            exit_price, reason = tp, "TP"
        else:
            continue
        pnl_pts = (exit_price - entry) * dirn
        return {"symbol": symbol, "side": side, "entry_ts": bars[entry_idx]["ts"],
                "entry": entry, "exit_ts": b["ts"], "exit": exit_price,
                "pnl_pts": pnl_pts, "sl": sl, "tp": tp, "reason": reason,
                "hold_bars": k - entry_idx}
    # hold expired: close at open of entry_idx+hold_n (EA closes at that bar
    # boundary, fill at next tick ~ open of the following bar)
    exit_i = min(entry_idx + hold_n, len(bars) - 1)
    exit_price = bars[exit_i]["open"]
    pnl_pts = (exit_price - entry) * dirn
    return {"symbol": symbol, "side": side, "entry_ts": bars[entry_idx]["ts"],
            "entry": entry, "exit_ts": bars[exit_i]["ts"], "exit": exit_price,
            "pnl_pts": pnl_pts, "sl": sl, "tp": tp, "reason": "HOLD",
            "hold_bars": exit_i - entry_idx}


def _find_bar_idx(bars: list[dict], ts: int) -> int | None:
    """Binary search for exact bar start ts."""
    lo, hi = 0, len(bars) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if bars[mid]["ts"] == ts:
            return mid
        if bars[mid]["ts"] < ts:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


def _close_stragglers(trades: list[dict], open_pos: dict[str, tuple[int, str]],
                      bars_map: dict[str, list[dict]], hold_n: int,
                      sl_tp: bool) -> None:
    for sym, (entry_idx, side) in open_pos.items():
        bars = bars_map[sym]
        sl, tp = _sl_tp_dist(sym) if sl_tp else (None, None)
        trades.append(_simulate_exit(bars, entry_idx, side, hold_n, sl, tp, sym))


# ------------------------------------------------------------- Ultra Monster --
def ultra_monster(bars_map: dict[str, list[dict]], hold_bars: int = 3,
                  min_range_pips: float = 6.0) -> list[dict]:
    """Rolling ORB: at :00/:30 M5 closes, break of prior 12-bar (60min) range.

    EA: trigger when dt.min == 0 or 30 -> M5 bar closes at minute 0/30. The
    signal bar's close is the breakout close; fill at next bar open. Exit:
    held >= HOLD_BARS(3) -> close (no SL/TP in Ultra v106).
    """
    trades: list[dict] = []
    for symbol, bars in bars_map.items():
        open_pos: dict[str, tuple[int, str]] = {}
        mult = 100.0 if "JPY" in symbol else 10000.0
        for i in range(13, len(bars)):
            b = bars[i]
            # exits at every new bar (EA CheckExits runs before CheckEntry)
            for sym in list(open_pos):
                entry_idx, side = open_pos[sym]
                if i - entry_idx >= hold_bars:
                    trades.append(_simulate_exit(bars, entry_idx, side, hold_bars,
                                                 None, None, sym))
                    del open_pos[sym]
            if symbol in open_pos:
                continue
            if b["ts"] % 1800 != 0:   # :00 or :30 M5 close
                continue
            c_closed = b["close"]
            h_prev = max(x["high"] for x in bars[i - 12:i])
            l_prev = min(x["low"] for x in bars[i - 12:i])
            range_pips = (h_prev - l_prev) * mult
            if range_pips < min_range_pips:
                continue
            if c_closed > h_prev and i + 1 < len(bars):
                open_pos[symbol] = (i + 1, "BUY")
            elif c_closed < l_prev and i + 1 < len(bars):
                open_pos[symbol] = (i + 1, "SELL")
        _close_stragglers(trades, open_pos, bars_map, hold_bars, sl_tp=False)
    return trades


# ---------------------------------------------------------------- Tokyo H0 ---
def tokyo_h0(bars_map: dict[str, list[dict]], lookback: int = 6,
             hold_bars: int = 12, top_n: int = 5, session_hour: int = 0) -> list[dict]:
    """At UTC 00:00 (first M5 bar of session hour), buy top-N most-declined.

    Entry at the first M5 bar of the session hour (fill = next bar open after
    the ranking bar). Exit: held >= 12 bars, SL/TP 0.35/0.45 (JPY) else
    0.0035/0.0045.
    """
    trades: list[dict] = []
    open_pos: dict[str, tuple[int, str]] = {}
    all_syms = list(bars_map)
    # per symbol: [(day_epoch, session_idx, ts)] first bar whose hour==session
    def first_session(sym):
        bars = bars_map[sym]
        out = []
        day0 = None
        for i in range(lookback, len(bars)):
            d = bars[i]["ts"] // 86400
            h = (bars[i]["ts"] // 3600) % 24
            if h == session_hour and d != day0:
                out.append(i); day0 = d
        return out
    firsts = {s: first_session(s) for s in all_syms}
    # iterate the union of session bar indices in chronological day order
    days = sorted({bars_map[s][i]["ts"] // 86400 for s in all_syms for i in firsts[s]})
    for dnum, day in enumerate(days):
        # exit management for positions opened during a prior session of this day
        for sym in list(open_pos):
            entry_idx, side = open_pos[sym]
            bars = bars_map[sym]
            # find first bar of this day (any hour) to advance hold count
            idx = None
            for i in firsts.get(sym, []):
                if bars[i]["ts"] // 86400 == day:
                    idx = i; break
            if idx is None:
                continue
            if idx - entry_idx >= hold_bars:
                trades.append(_simulate_exit(bars, entry_idx, side, hold_bars,
                                             *_sl_tp_dist(sym), sym))
                del open_pos[sym]
        # ranking entry at this day's session bar
        idx_by_sym = {}
        for sym in all_syms:
            cand = firsts[sym]
            got = None
            for i in cand:
                if bars_map[sym][i]["ts"] // 86400 == day:
                    got = i; break
            if got is not None:
                idx_by_sym[sym] = got
        rets: list[tuple[float, str]] = []
        for sym, idx in idx_by_sym.items():
            bars = bars_map[sym]
            r = (bars[idx]["close"] - bars[idx - lookback]["close"]) / bars[idx - lookback]["close"]
            rets.append((r, sym))
        rets.sort()
        opened = 0
        for _r, sym in rets:
            if sym in open_pos or opened >= top_n:
                continue
            idx = idx_by_sym[sym]
            if idx + 1 >= len(bars_map[sym]):
                continue
            open_pos[sym] = (idx + 1, "BUY")
            opened += 1
    _close_stragglers(trades, open_pos, bars_map, hold_bars, sl_tp=True)
    return trades


# --------------------------------------------------------- CPPF Z (reversion) --
def cppf_z(bars_map: dict[str, list[dict]], z_thresh: float = 6.0,
           lookback: int = 200, hold_bars: int = 18) -> list[dict]:
    """6-sigma 4-bar return shock vs 200-bar M5 rolling distribution."""
    return _z_engine(bars_map, z_thresh, lookback, hold_bars, reversion=True)


# --------------------------------------------------- CPMC (momentum cont.) --
def cpmc_z(bars_map: dict[str, list[dict]], z_thresh: float = 4.5,
           lookback: int = 200, hold_bars: int = 9) -> list[dict]:
    """4.5-sigma momentum continuation on the SAME 1-bar return."""
    return _z_engine(bars_map, z_thresh, lookback, hold_bars, reversion=False)


def _z_engine(bars_map: dict[str, list[dict]], z_thresh: float, lookback: int,
              hold_bars: int, reversion: bool) -> list[dict]:
    trades: list[dict] = []
    for symbol, bars in bars_map.items():
        open_pos: dict[str, tuple[int, str]] = {}
        sl, tp = _sl_tp_dist(symbol)
        for i in range(lookback + 10, len(bars)):
            # exits at each new bar
            for sym in list(open_pos):
                entry_idx, side = open_pos[sym]
                if i - entry_idx >= hold_bars:
                    trades.append(_simulate_exit(bars, entry_idx, side, hold_bars,
                                                 sl, tp, sym))
                    del open_pos[sym]
            if symbol in open_pos:
                continue
            # EA: skip Sunday 00:00 (fresh week)
            if datetime.fromtimestamp(bars[i]["ts"], tz=timezone.utc).weekday() == 6 \
               and (bars[i]["ts"] // 3600) % 24 == 0:
                continue
            n = lookback
            closes = [bars[k]["close"] for k in range(i - n, i)]
            rets = [(closes[k - 1] - closes[k]) / closes[k] for k in range(1, n)]
            mean = sum(rets) / len(rets)
            var = sum(r * r for r in rets) / len(rets) - mean * mean
            std = math.sqrt(var) if var > 0 else 0.0001
            if reversion:
                curr = (bars[i]["close"] - bars[i - 3]["close"]) / bars[i - 3]["close"]
            else:
                curr = (bars[i]["close"] - bars[i - 1]["close"]) / bars[i - 1]["close"]
            z = (curr - mean) / std
            if z >= z_thresh and i + 1 < len(bars):
                open_pos[symbol] = (i + 1, "BUY")
            elif z <= -z_thresh and i + 1 < len(bars):
                open_pos[symbol] = (i + 1, "SELL")
        _close_stragglers(trades, open_pos, bars_map, hold_bars, sl_tp=True)
    return trades


# -------------------------------------------------------------- NY H21 ---------
def ny_h21(bars_map: dict[str, list[dict]], lookback: int = 12,
           hold_bars: int = 12, session_hour: int = 21) -> list[dict]:
    """21:00 mean reversion on 12-bar M5 return (EURJPY, GBPJPY)."""
    trades: list[dict] = []
    for symbol, bars in bars_map.items():
        open_pos: dict[str, tuple[int, str]] = {}
        sl, tp = _sl_tp_dist(symbol)
        # build the per-day first-session-hour index list
        session_idxs = []
        day0 = None
        for i in range(lookback, len(bars)):
            d = bars[i]["ts"] // 86400
            h = (bars[i]["ts"] // 3600) % 24
            if h == session_hour and d != day0:
                session_idxs.append(i); day0 = d
        for rank_i in session_idxs:
            # exit management (hold count vs this day's session bar)
            idxr = _find_bar_idx(bars, bars[rank_i]["ts"])
            for sym in list(open_pos):
                entry_idx, side = open_pos[sym]
                if idxr is not None and idxr - entry_idx >= hold_bars:
                    trades.append(_simulate_exit(bars, entry_idx, side, hold_bars,
                                                 sl, tp, sym))
                    del open_pos[sym]
            if symbol in open_pos:
                continue
            idx = rank_i
            if idx < lookback or idx + 1 >= len(bars):
                continue
            ret = (bars[idx]["close"] - bars[idx - lookback]["close"]) / bars[idx - lookback]["close"]
            if ret < -0.0001:
                open_pos[symbol] = (idx + 1, "BUY")
            elif ret > 0.0001:
                open_pos[symbol] = (idx + 1, "SELL")
        _close_stragglers(trades, open_pos, bars_map, hold_bars, sl_tp=True)
    return trades


# ------------------------------------------------------------ MSV Asian ------
def msv_asian(bars_map: dict[str, list[dict]], hold_bars: int = 12,
              window_hours: tuple = (0, 6), ret_thresh: float = 0.0002) -> list[dict]:
    """Asian-session (0-6h UTC) mean reversion on 12-bar M5 return, 18 pairs."""
    trades: list[dict] = []
    for symbol, bars in bars_map.items():
        open_pos: dict[str, tuple[int, str]] = {}
        sl, tp = _sl_tp_dist(symbol)
        for i in range(12, len(bars)):
            ts = bars[i]["ts"]
            hour = (ts // 3600) % 24
            for sym in list(open_pos):
                entry_idx, side = open_pos[sym]
                if i - entry_idx >= hold_bars:
                    trades.append(_simulate_exit(bars, entry_idx, side, hold_bars,
                                                 sl, tp, sym))
                    del open_pos[sym]
            if symbol in open_pos:
                continue
            if hour < window_hours[0] or hour > window_hours[1]:
                continue
            ret = (bars[i]["close"] - bars[i - 12]["close"]) / bars[i - 12]["close"]
            if ret < -ret_thresh and i + 1 < len(bars):
                open_pos[symbol] = (i + 1, "BUY")
            elif ret > ret_thresh and i + 1 < len(bars):
                open_pos[symbol] = (i + 1, "SELL")
        _close_stragglers(trades, open_pos, bars_map, hold_bars, sl_tp=True)
    return trades
